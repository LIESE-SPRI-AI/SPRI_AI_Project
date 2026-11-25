#!/usr/bin/env python3
"""
predict_segmentation.py
=======================
Predicción de área quemada sobre escenas completas de cualquier tamaño.

La entrada puede ser:
  * un TIFF de 4 bandas (R,G,B,NIR) como los que guarda el generador,
  * un zip de EO Browser (se apilan B04,B03,B02,B08 igual que en el generador),
  * una carpeta con esas bandas.

Ejemplos:
  python predict_segmentation.py --model runs/20261015_130000/best.pth --input escena.zip --visualization
  python predict_segmentation.py --model weights/model_best.pth --image test_image.tif --output prediction.tif --gpu auto
  python predict_segmentation.py --model best.pth --input Raw/*.zip --output-dir pred --eval --tta

Con --eval, si el zip/carpeta trae la capa Burned Area Detection (o se da --mask), se calculan
IoU, F1, precisión y recall sobre la escena completa (incluye zonas sin incendio, que es el caso real).

Salidas por escena (en --output-dir, o junto a --output):
  <nombre>_mascara.tif   uint8 georreferenciado: 0 = no quemado, 1 = quemado, 255 = sin dato
  <nombre>_prob.tif      (--save-prob) probabilidad * 254, 255 = sin dato
  <nombre>_overlay.png   (--visualization) color verdadero con lo detectado en rojo
  <nombre>_prob.png      (--visualization) mapa de probabilidad
  metricas.csv           (--eval) métricas por escena y globales

Cambios principales respecto a la versión anterior:
  * Normalización, tamaño de parche y umbral se leen del checkpoint (iguales al entrenamiento).
  * Ventana gaussiana para combinar parches traslapados (sin costuras) y relleno por reflejo
    en los bordes en vez de ceros.
  * Inferencia por lotes, TTA opcional, píxeles sin dato marcados, filtro de manchas pequeñas.
  * Imágenes más chicas que el traslape ya no producen una salida vacía.
"""

import argparse
import csv
import os
import sys
import tempfile
import time

import numpy as np

import wildfire_common as wc


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "si", "sí", "yes", "y")


def parse_args():
    p = argparse.ArgumentParser(description="Predicción de área quemada (MbUNet2D)")
    p.add_argument("--model", required=True, help="best.pth / final.pth (o pesos antiguos .pth)")
    p.add_argument("--input", "--image", dest="inputs", nargs="+", required=True,
                   help="TIFF de 4 bandas, zip de EO Browser o carpeta de escena (uno o varios)")
    p.add_argument("--output", default=None, help="Archivo de máscara de salida (.tif o .png); solo con una entrada")
    p.add_argument("--output-dir", default="predicciones")
    p.add_argument("--gpu", default="auto", help="True/False/auto")
    p.add_argument("--threshold", type=float, default=None, help="Por defecto el umbral guardado en el checkpoint")
    p.add_argument("--tile", type=int, default=None, help="Tamaño de parche (por defecto el del entrenamiento)")
    p.add_argument("--overlap", type=int, default=None, help="Traslape entre parches (por defecto tile/2)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--tta", action="store_true", help="Promediar con volteos (4x más lento, más estable)")
    p.add_argument("--min-area", type=int, default=0, help="Eliminar manchas de menos de N píxeles")
    p.add_argument("--visualization", type=str2bool, nargs="?", const=True, default=False)
    p.add_argument("--save-prob", action="store_true")
    p.add_argument("--eval", action="store_true", help="Calcular métricas contra la capa Burned Area")
    p.add_argument("--mask", default=None, help="Capa Burned Area de referencia (para una entrada TIFF)")
    p.add_argument("--escala", default=None, choices=["auto"] + list(wc.ESCALAS),
                   help="Conversión a DN; por defecto 'auto' (o 'dn' con checkpoints antiguos)")
    return p.parse_args()


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
    from Wildfire_models import MbUNet2D
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
    model = MbUNet2D(in_channels=meta.get("in_channels", 4), out_channels=meta.get("out_channels", 2))
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


# ---------------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------------

def _escena_desde_carpeta(raiz, escala, con_mascara):
    hallados = wc.buscar_archivos_escena(raiz)
    faltan = [b for b in wc.BAND_ORDER if b not in hallados["bandas"]]
    if faltan:
        raise ValueError(f"Faltan bandas {faltan}. Archivos: {hallados['archivos'][:12]}")
    dn, meta, nodata, info = wc.leer_stack_bandas(hallados["bandas"], escala=escala)
    gt = None
    if con_mascara and hallados["mascara"]:
        gt, _ = wc.extraer_mascara_quemado(hallados["mascara"], nodata.shape, meta)
    return dn, meta, nodata, info, gt


def cargar_entrada(ruta, escala, con_mascara=False, ruta_mascara=None):
    """Devuelve dict(nombre, dn (4,H,W) float32, meta, nodata, info, gt o None)."""
    nombre = wc.nombre_seguro(os.path.splitext(os.path.basename(os.path.normpath(ruta)))[0])
    ext = os.path.splitext(ruta)[1].lower()
    if os.path.isdir(ruta):
        dn, meta, nodata, info, gt = _escena_desde_carpeta(ruta, escala, con_mascara)
    elif ext == ".zip":
        with tempfile.TemporaryDirectory(prefix="wf_pred_") as tmp:
            wc.extraer_zip_seguro(ruta, tmp)
            dn, meta, nodata, info, gt = _escena_desde_carpeta(tmp, escala, con_mascara)
    elif ext in (".tif", ".tiff"):
        dn, meta, nodata, info = wc.leer_tiff_4bandas(ruta, escala=escala)
        gt = None
    else:
        raise ValueError(f"Entrada no soportada: {ruta}")
    if ruta_mascara:
        gt, _ = wc.extraer_mascara_quemado(ruta_mascara, nodata.shape, meta)
    if gt is not None:
        gt = gt.copy()
        gt[nodata] = wc.IGNORE_INDEX
    return {"nombre": nombre, "dn": dn, "meta": meta, "nodata": nodata, "info": info, "gt": gt}


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


def rgb_visual(dn, nodata):
    """Color verdadero con estiramiento 2-98 % (solo para visualizar)."""
    rgb = np.zeros((dn.shape[1], dn.shape[2], 3), np.uint8)
    validos = ~nodata
    for i in range(3):
        banda = dn[i]
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


def guardar_salidas(base, ruta_mascara, escena, prob, pred, args, rgb=None):
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

    if args.save_prob:
        p8 = np.clip(np.rint(prob * 254), 0, 254).astype(np.uint8)
        p8[nodata] = 255
        ruta = base + "_prob.tif"
        wc.escribir_raster(ruta, p8, escena["meta"], nodata=255)
        salidas.append(ruta)

    if args.visualization:
        if rgb is None:
            rgb = rgb_visual(escena["dn"], nodata)
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
# Main
# ---------------------------------------------------------------------------

def procesar(ruta, fn_inferencia, cfg, args):
    t0 = time.time()
    escena = cargar_entrada(ruta, cfg["escala"], con_mascara=args.eval, ruta_mascara=args.mask)
    dn, nodata = escena["dn"], escena["nodata"]
    h, w = nodata.shape
    print(f"  {w}x{h} px | escala '{escena['info'].get('modo_escala')}' | "
          f"{100 * nodata.mean():.1f}% sin dato")

    rgb = rgb_visual(dn, nodata) if args.visualization else None
    x = wc.normalizar(dn, cfg["norm"])

    def progreso(hecho, total):
        print(f"\r  parches {hecho}/{total}", end="", flush=True)

    prob = wc.predecir_ventanas(x, fn_inferencia, tam=cfg["tile"], traslape=cfg["overlap"],
                                lote=args.batch_size, ponderar=True, progreso=progreso)
    print()
    del x
    prob[nodata] = 0.0
    pred = quitar_manchas((prob > cfg["umbral"]) & ~nodata, args.min_area)

    if args.output and len(args.inputs) == 1:
        ruta_mascara = args.output
        base = os.path.splitext(args.output)[0]
    else:
        base = os.path.join(args.output_dir, escena["nombre"])
        ruta_mascara = base + "_mascara.tif"
    salidas = guardar_salidas(base, ruta_mascara, escena, prob, pred, args, rgb)

    validos = int((~nodata).sum())
    quemados = int(pred.sum())
    print(f"  quemado: {quemados:,} px ({100 * quemados / max(validos, 1):.2f}% de lo válido) | {time.time() - t0:.1f}s")
    for s in salidas:
        print(f"  -> {s}")

    fila = None
    if escena["gt"] is not None:
        tp, fp, fn, tn = wc.conteos_confusion(pred, escena["gt"])
        m = wc.metricas(tp, fp, fn, tn)
        print(f"  vs referencia: IoU {wc.fmt(m['iou'])}  F1 {wc.fmt(m['f1'])}  "
              f"P {wc.fmt(m['precision'])}  R {wc.fmt(m['recall'])}")
        fila = {"escena": escena["nombre"], "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                **{k: wc.fmt(v) for k, v in m.items()}}
    elif args.eval:
        print("  (sin capa de referencia: no se calculan métricas)")
    return fila


def main():
    args = parse_args()
    if args.output and len(args.inputs) > 1:
        sys.exit("ERROR: --output es para una sola entrada; con varias usa --output-dir")
    if args.mask and len(args.inputs) > 1:
        sys.exit("ERROR: --mask es para una sola entrada")
    if not os.path.isfile(args.model):
        sys.exit(f"ERROR: no existe el modelo {args.model}")

    device = elegir_dispositivo(args.gpu)
    model, meta, formato = cargar_modelo(args.model, device)
    tile = args.tile or int(meta.get("image_size", 128))
    cfg = {
        "norm": meta.get("norm", {"tipo": "div", "valor": 12500.0}),
        "tile": tile,
        "overlap": args.overlap if args.overlap is not None else tile // 2,
        "umbral": args.threshold if args.threshold is not None else float(meta.get("umbral") or 0.5),
        "escala": args.escala or ("auto" if formato == "nuevo" else "dn"),
    }
    if not 0 <= cfg["overlap"] < tile:
        sys.exit("ERROR: --overlap debe estar entre 0 y tile-1")

    print("=" * 70)
    print(f"Modelo: {args.model} ({formato}) | dispositivo {device}")
    print(f"Parche {tile}px, traslape {cfg['overlap']}px | umbral {cfg['umbral']} | norm {cfg['norm']['tipo']}"
          f" | TTA {'sí' if args.tta else 'no'}")
    if formato == "antiguo":
        print("⚠ Checkpoint antiguo (sin metadatos): se asume DN/12500 y umbral 0.5. Si el modelo se entrenó con "
              "el dataset estirado por min/max, las predicciones sobre zips crudos pueden no coincidir.")
    print("=" * 70)

    fn_inferencia = crear_fn_inferencia(model, device, tta=args.tta)
    filas, errores = [], []
    for i, ruta in enumerate(args.inputs, 1):
        print(f"[{i}/{len(args.inputs)}] {ruta}")
        try:
            fila = procesar(ruta, fn_inferencia, cfg, args)
            if fila:
                filas.append(fila)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            errores.append(ruta)

    if filas:
        tp, fp, fn, tn = (sum(f[k] for f in filas) for k in ("tp", "fp", "fn", "tn"))
        m = wc.metricas(tp, fp, fn, tn)
        filas.append({"escena": "GLOBAL", "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      **{k: wc.fmt(v) for k, v in m.items()}})
        destino = os.path.dirname(os.path.abspath(args.output)) if args.output else args.output_dir
        os.makedirs(destino, exist_ok=True)
        ruta_csv = os.path.join(destino, "metricas.csv")
        with open(ruta_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            wr.writeheader()
            wr.writerows(filas)
        print("=" * 70)
        print(f"GLOBAL ({len(filas) - 1} escenas): IoU {wc.fmt(m['iou'])}  F1 {wc.fmt(m['f1'])}  "
              f"P {wc.fmt(m['precision'])}  R {wc.fmt(m['recall'])}  -> {ruta_csv}")
    if errores:
        print(f"⚠ {len(errores)} entrada(s) con error: {errores}")
    print("=" * 70)


if __name__ == "__main__":
    main()
