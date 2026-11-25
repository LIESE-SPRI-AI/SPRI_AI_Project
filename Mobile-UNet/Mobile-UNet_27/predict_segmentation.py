#!/usr/bin/env python3
"""
predict_segmentation.py
=======================
Predicción de área quemada sobre escenas completas de cualquier tamaño.

Toda la configuración está en el bloque CONFIGURACIÓN (no hay argumentos por terminal).
Bandas, índices (NDVI, BAI...), normalización, cuantización, tamaño de parche y umbral se
leen del checkpoint, así que la entrada se prepara exactamente igual que en el entrenamiento.

Cada entrada puede ser:
  * un TIFF ya apilado (bandas en el orden ORDEN_BANDAS_TIFF),
  * un zip de EO Browser (se buscan las bandas que pide el modelo),
  * una carpeta con esas bandas.

Con EVAL = True, si el zip/carpeta trae la capa Burned Area Detection (o se da MASCARA_REF),
se calculan IoU, F1, precisión y recall sobre la escena completa.

Salidas por escena (en DIR_SALIDA, o junto a SALIDA):
  <nombre>_mascara.tif   uint8 georreferenciado: 0 = no quemado, 1 = quemado, 255 = sin dato
  <nombre>_prob.tif      (GUARDAR_PROB) probabilidad * 254, 255 = sin dato
  <nombre>_overlay.png   (VISUALIZACION) color verdadero con lo detectado en rojo
  <nombre>_prob.png      (VISUALIZACION) mapa de probabilidad
  metricas.csv           (EVAL) métricas por escena y globales

inferencias.py importa preparar_modelo() y predecir_archivo() para procesar muchos modelos.
"""

import csv
import glob
import os
import sys
import tempfile
import time

import numpy as np

import wildfire_common as wc

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
MODELO = "auto"              # "auto" = best.pth de la corrida más reciente en DIR_RUNS, o ruta a un .pth
DIR_RUNS = os.path.join(wc.DIR_CODIGO, "runs")
ENTRADAS = [                 # rutas o patrones glob (TIFF, zip de EO Browser o carpeta)
    wc.ruta_proyecto("Inferencias", "Input", "*_Merged.tif"),
]
SALIDA = None                # archivo de máscara (.tif o .png); solo con una entrada
DIR_SALIDA = wc.ruta_proyecto("Inferencias", "Predicciones")
ORDEN_BANDAS_TIFF = ("B04", "B03", "B02", "B08")   # orden de las bandas dentro de los TIFF de entrada

GPU = "auto"                 # "auto" | "true" | "false"
UMBRAL = None                # None = el guardado en el checkpoint
TILE = None                  # None = el del entrenamiento
OVERLAP = None               # None = TILE / 2
BATCH_SIZE = 16
TTA = False                  # promediar con volteos (4x más lento, más estable)
MIN_AREA = 0                 # eliminar manchas de menos de N píxeles
VISUALIZACION = True
GUARDAR_PROB = False
EVAL = False                 # métricas contra la capa Burned Area
MASCARA_REF = None           # capa Burned Area de referencia (para una sola entrada TIFF)
ESCALA = None                # conversión a DN; None = "auto" ("dn" con checkpoints antiguos)
# =============================================================================


def opciones_por_defecto():
    return {"salida": SALIDA, "dir_salida": DIR_SALIDA, "batch_size": BATCH_SIZE, "min_area": MIN_AREA,
            "visualizacion": VISUALIZACION, "guardar_prob": GUARDAR_PROB, "eval": EVAL,
            "mascara_ref": MASCARA_REF, "orden_tiff": tuple(ORDEN_BANDAS_TIFF)}


def expandir_entradas(entradas):
    rutas = []
    for e in entradas:
        if any(c in e for c in "*?["):
            encontradas = sorted(glob.glob(e))
            if not encontradas:
                print(f"⚠ El patrón no encontró archivos: {e}")
            rutas += encontradas
        else:
            rutas.append(e)
    return rutas


def ultimo_best(dir_runs=DIR_RUNS):
    cands = glob.glob(os.path.join(dir_runs, "*", "best.pth"))
    return max(cands, key=os.path.getmtime) if cands else None


# ---------------------------------------------------------------------------
# Modelo (torch se importa aquí para que el resto del módulo no dependa de él)
# ---------------------------------------------------------------------------

def elegir_dispositivo(opcion):
    import torch
    opcion = str(opcion).lower()
    if opcion not in ("false", "cpu", "0", "no") and torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


def cargar_modelo(ruta, device):
    import torch
    import Wildfire_models as wm
    try:
        ck = torch.load(ruta, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 1.13
        ck = torch.load(ruta, map_location=device)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        sd = ck["model_state_dict"]
        meta = ck.get("meta") or {}
        formato = "nuevo" if "norm" in meta else "antiguo"
    else:
        sd, meta, formato = ck, {}, "antiguo"
    if any(k.startswith("module.") for k in sd):          # guardado desde DataParallel
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model = wm.modelo_desde_meta(meta)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, meta, formato


def crear_fn_inferencia(model, device, tta=False):
    import torch

    def correr(x):
        return torch.softmax(model(x), dim=1)[:, 1]

    @torch.no_grad()
    def fn(batch):
        x = torch.from_numpy(batch).to(device)
        p = correr(x)
        if tta:
            p = p + correr(x.flip(-1)).flip(-1)
            p = p + correr(x.flip(-2)).flip(-2)
            p = p + correr(x.flip(-1, -2)).flip(-1, -2)
            p = p / 4.0
        return p.float().cpu().numpy()

    return fn


def preparar_modelo(ruta_modelo, gpu=GPU, tta=TTA, umbral=UMBRAL, tile=TILE, overlap=OVERLAP, escala=ESCALA):
    """Carga el modelo y arma la configuración de predicción (todo lo del entrenamiento sale del checkpoint)."""
    import Wildfire_models as wm
    device = elegir_dispositivo(gpu)
    model, meta, formato = cargar_modelo(ruta_modelo, device)
    tile = tile or int(meta.get("image_size", 128))
    cfg = {
        "modelo": ruta_modelo, "formato": formato, "device": device,
        "bandas_modelo": tuple(meta.get("bandas_modelo") or meta.get("orden_bandas") or wc.BAND_ORDER),
        "indices": tuple(meta.get("indices") or ()),
        "norm": meta.get("norm", {"tipo": "div", "valor": 12500.0}),
        "tile": tile,
        "overlap": overlap if overlap is not None else tile // 2,
        "umbral": umbral if umbral is not None else float(meta.get("umbral") or 0.5),
        "escala": escala or ("auto" if formato == "nuevo" else "dn"),
        "quant": model.config["quant"], "tta": tta,
        "fn_inferencia": crear_fn_inferencia(model, device, tta=tta),
    }
    cfg["bandas_lectura"] = wc.bandas_requeridas(cfg["bandas_modelo"], cfg["indices"])
    if not 0 <= cfg["overlap"] < tile:
        raise ValueError("OVERLAP debe estar entre 0 y TILE-1")
    canales = len(cfg["bandas_modelo"]) + len(cfg["indices"])
    if canales != model.config["in_channels"]:
        raise ValueError(f"El checkpoint declara {model.config['in_channels']} canales pero bandas+índices "
                         f"dan {canales}")

    print(f"Modelo: {ruta_modelo} ({formato}) | dispositivo {device}")
    print(f"Entrada: {wc.nombres_canales(cfg['bandas_modelo'], cfg['indices'])} | norm {cfg['norm']['tipo']}")
    print(f"Parche {tile}px, traslape {cfg['overlap']}px | umbral {cfg['umbral']} | TTA {'sí' if tta else 'no'}")
    print(f"Cuantización: {wm.resumen_quant(cfg['quant'])}")
    if formato == "antiguo":
        print("⚠ Checkpoint antiguo (sin metadatos): se asume R,G,B,NIR, DN/12500, umbral 0.5 y cuantización "
              "de 8 bits. Si el modelo se entrenó con el dataset estirado por min/max, las predicciones sobre "
              "zips crudos pueden no coincidir.")
    return cfg


# ---------------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------------

def _escena_desde_carpeta(raiz, escala, bandas, con_mascara):
    hallados = wc.buscar_archivos_escena(raiz, bandas=bandas)
    faltan = [b for b in bandas if b not in hallados["bandas"]]
    if faltan:
        raise ValueError(f"Faltan bandas {faltan}. Archivos: {hallados['archivos'][:12]}")
    dn, meta, nodata, info = wc.leer_stack_bandas(hallados["bandas"], escala=escala, orden=bandas)
    gt = None
    if con_mascara and hallados["mascara"]:
        gt, _ = wc.extraer_mascara_quemado(hallados["mascara"], nodata.shape, meta)
    return dn, meta, nodata, info, gt


def cargar_entrada(ruta, escala, bandas, orden_tiff, con_mascara=False, ruta_mascara=None):
    """Devuelve dict(nombre, dn (C,H,W) float32, orden, meta, nodata, info, gt o None)."""
    nombre = wc.nombre_seguro(os.path.splitext(os.path.basename(os.path.normpath(ruta)))[0])
    ext = os.path.splitext(ruta)[1].lower()
    orden = tuple(bandas)
    if os.path.isdir(ruta):
        dn, meta, nodata, info, gt = _escena_desde_carpeta(ruta, escala, bandas, con_mascara)
    elif ext == ".zip":
        with tempfile.TemporaryDirectory(prefix="wf_pred_") as tmp:
            wc.extraer_zip_seguro(ruta, tmp)
            dn, meta, nodata, info, gt = _escena_desde_carpeta(tmp, escala, bandas, con_mascara)
    elif ext in (".tif", ".tiff"):
        faltan = [b for b in bandas if b not in orden_tiff]
        if faltan:
            raise ValueError(f"El modelo necesita {faltan}, que no están en ORDEN_BANDAS_TIFF = {list(orden_tiff)}")
        dn, meta, nodata, info = wc.leer_tiff_bandas(ruta, len(orden_tiff), escala=escala)
        orden, gt = tuple(orden_tiff), None
    else:
        raise ValueError(f"Entrada no soportada: {ruta}")
    if ruta_mascara:
        gt, _ = wc.extraer_mascara_quemado(ruta_mascara, nodata.shape, meta)
    if gt is not None:
        gt = gt.copy()
        gt[nodata] = wc.IGNORE_INDEX
    return {"nombre": nombre, "dn": dn, "orden": orden, "meta": meta, "nodata": nodata, "info": info, "gt": gt}


# ---------------------------------------------------------------------------
# Post-proceso y salidas
# ---------------------------------------------------------------------------

def quitar_manchas(pred, min_area):
    if min_area <= 0:
        return pred
    import cv2
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    conservar = np.zeros(n, dtype=bool)
    conservar[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return conservar[etiquetas]


def rgb_visual(dn, orden, nodata):
    """Color verdadero (B04, B03, B02) con estiramiento 2-98 % (solo para visualizar)."""
    orden = list(orden)
    idx = [orden.index(b) if b in orden else min(i, dn.shape[0] - 1) for i, b in enumerate(("B04", "B03", "B02"))]
    rgb = np.zeros((dn.shape[1], dn.shape[2], 3), np.uint8)
    validos = ~nodata
    for i, k in enumerate(idx):
        banda = dn[k]
        if validos.any():
            lo, hi = np.percentile(banda[validos], [2, 98])
        else:
            lo, hi = 0.0, 1.0
        rgb[..., i] = (np.clip((banda - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return rgb


def guardar_png(ruta, rgb):
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    cv2.imwrite(ruta, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def guardar_salidas(base, ruta_mascara, escena, prob, pred, opts, rgb=None):
    import cv2
    nodata = escena["nodata"]
    mascara = pred.astype(np.uint8)
    mascara[nodata] = 255
    ext = os.path.splitext(ruta_mascara)[1].lower()
    if ext in (".png", ".jpg", ".jpeg"):
        vis = np.zeros(pred.shape + (3,), np.uint8)
        vis[pred] = (255, 0, 0)
        vis[nodata] = (64, 64, 64)
        guardar_png(ruta_mascara, vis)
    else:
        if ext not in (".tif", ".tiff"):
            ruta_mascara = os.path.splitext(ruta_mascara)[0] + ".tif"
        wc.escribir_raster(ruta_mascara, mascara, escena["meta"], nodata=255)
    salidas = [ruta_mascara]

    if opts["guardar_prob"]:
        p8 = np.clip(np.rint(prob * 254), 0, 254).astype(np.uint8)
        p8[nodata] = 255
        ruta = base + "_prob.tif"
        wc.escribir_raster(ruta, p8, escena["meta"], nodata=255)
        salidas.append(ruta)

    if opts["visualizacion"]:
        if rgb is None:
            rgb = rgb_visual(escena["dn"], escena["orden"], nodata)
        over = rgb.astype(np.float32)
        over[pred] = 0.4 * over[pred] + 0.6 * np.array([255, 0, 0], np.float32)
        over[nodata] = (40, 40, 40)
        guardar_png(base + "_overlay.png", over.astype(np.uint8))
        color = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        color[nodata] = 0
        cv2.imwrite(base + "_prob.png", color)
        salidas += [base + "_overlay.png", base + "_prob.png"]
    return salidas


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------

def predecir_archivo(ruta, cfg, opts=None, nombre_salida=None):
    """Predice una escena. nombre_salida: ruta completa de la máscara (si no, DIR_SALIDA/<nombre>_mascara.tif).
    Devuelve la fila de métricas (o None si no hay referencia)."""
    opts = {**opciones_por_defecto(), **(opts or {})}
    t0 = time.time()
    escena = cargar_entrada(ruta, cfg["escala"], cfg["bandas_lectura"], opts["orden_tiff"],
                            con_mascara=opts["eval"], ruta_mascara=opts["mascara_ref"])
    dn, nodata = escena["dn"], escena["nodata"]
    h, w = nodata.shape
    print(f"  {w}x{h} px | escala '{escena['info'].get('modo_escala')}' | {100 * nodata.mean():.1f}% sin dato")

    rgb = rgb_visual(dn, escena["orden"], nodata) if opts["visualizacion"] else None
    x = wc.construir_entrada(dn, escena["orden"], cfg["bandas_modelo"], cfg["indices"], cfg["norm"], nodata)

    def progreso(hecho, total):
        print(f"\r  parches {hecho}/{total}", end="", flush=True)

    prob = wc.predecir_ventanas(x, cfg["fn_inferencia"], tam=cfg["tile"], traslape=cfg["overlap"],
                                lote=opts["batch_size"], ponderar=True, progreso=progreso)
    print()
    del x
    prob[nodata] = 0.0
    pred = quitar_manchas((prob > cfg["umbral"]) & ~nodata, opts["min_area"])

    if nombre_salida:
        ruta_mascara = nombre_salida
        base = os.path.splitext(nombre_salida)[0]
    else:
        base = os.path.join(opts["dir_salida"], escena["nombre"])
        ruta_mascara = base + "_mascara.tif"
    salidas = guardar_salidas(base, ruta_mascara, escena, prob, pred, opts, rgb)

    validos = int((~nodata).sum())
    quemados = int(pred.sum())
    print(f"  quemado: {quemados:,} px ({100 * quemados / max(validos, 1):.2f}% de lo válido) | {time.time() - t0:.1f}s")
    for s in salidas:
        print(f"  -> {s}")

    if escena["gt"] is not None:
        tp, fp, fn, tn = wc.conteos_confusion(pred, escena["gt"])
        m = wc.metricas(tp, fp, fn, tn)
        print(f"  vs referencia: IoU {wc.fmt(m['iou'])}  F1 {wc.fmt(m['f1'])}  "
              f"P {wc.fmt(m['precision'])}  R {wc.fmt(m['recall'])}")
        return {"escena": escena["nombre"], "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                **{k: wc.fmt(v) for k, v in m.items()}}
    if opts["eval"]:
        print("  (sin capa de referencia: no se calculan métricas)")
    return None


def guardar_metricas(filas, destino):
    tp, fp, fn, tn = (sum(f[k] for f in filas) for k in ("tp", "fp", "fn", "tn"))
    m = wc.metricas(tp, fp, fn, tn)
    filas = filas + [{"escena": "GLOBAL", "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      **{k: wc.fmt(v) for k, v in m.items()}}]
    os.makedirs(destino, exist_ok=True)
    ruta_csv = os.path.join(destino, "metricas.csv")
    with open(ruta_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        wr.writeheader()
        wr.writerows(filas)
    print(f"GLOBAL ({len(filas) - 1} escenas): IoU {wc.fmt(m['iou'])}  F1 {wc.fmt(m['f1'])}  "
          f"P {wc.fmt(m['precision'])}  R {wc.fmt(m['recall'])}  -> {ruta_csv}")


def main():
    entradas = expandir_entradas(ENTRADAS)
    if not entradas:
        sys.exit("ERROR: ENTRADAS no tiene archivos")
    if SALIDA and len(entradas) > 1:
        sys.exit("ERROR: SALIDA es para una sola entrada; con varias usa DIR_SALIDA")
    if MASCARA_REF and len(entradas) > 1:
        sys.exit("ERROR: MASCARA_REF es para una sola entrada")
    modelo = ultimo_best() if MODELO == "auto" else MODELO
    if not modelo or not os.path.isfile(modelo):
        sys.exit(f"ERROR: no existe el modelo {modelo or '(no hay runs/*/best.pth)'}")

    print("=" * 70)
    cfg = preparar_modelo(modelo)
    print("=" * 70)

    filas, errores = [], []
    for i, ruta in enumerate(entradas, 1):
        print(f"[{i}/{len(entradas)}] {ruta}")
        try:
            fila = predecir_archivo(ruta, cfg, nombre_salida=SALIDA)
            if fila:
                filas.append(fila)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            errores.append(ruta)

    if filas:
        print("=" * 70)
        guardar_metricas(filas, os.path.dirname(os.path.abspath(SALIDA)) if SALIDA else DIR_SALIDA)
    if errores:
        print(f"⚠ {len(errores)} entrada(s) con error: {errores}")
    print("=" * 70)
    if errores:
        sys.exit(1)


if __name__ == "__main__":
    main()
