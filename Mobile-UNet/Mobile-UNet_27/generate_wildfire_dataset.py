#!/usr/bin/env python3
"""
generate_wildfire_dataset.py
============================
Genera el dataset de entrenamiento a partir de los zips exportados de EO Browser
(bandas B02/B03/B04/B08 "Raw" + capa "Burned Area Detection").

Toda la configuración está en el bloque CONFIGURACIÓN (no hay argumentos por terminal).
Las rutas parten de BASE_DIR definido en el archivo .env.

Antes conviene revisar los zips (detecta corruptos, capas faltantes y duplicados):
  python3 revisar_zips.py          # usa los mismos parámetros que este script
  # luego, opcional: LISTA_ZIPS = ruta a reporte_zips/zips_disponibles.txt
  python3 generate_wildfire_dataset.py

Salida en OUTPUT_DIR (= DATA_DIR de trainModelWildfire3D4B.py):
  Images/<escena>_x<col>_y<fila>.tif      bandas BANDAS_DATASET (por defecto R,G,B,NIR) uint16 en DN
                                          (los índices NDVI/BAI... NO se guardan: se calculan al entrenar)
  SegmentationClass/<mismo nombre>.tif    1 banda uint8: 0 = no quemado, 1 = quemado, 255 = sin dato
  train.txt / valid.txt / test.txt        split por SITIO: escenas que se traslapan van al mismo split
  crops.csv      un renglón por recorte (tipo, % quemado, % sin dato, split...)
  escenas.csv    un renglón por zip con su estado y, si se descartó, el motivo y los archivos que tenía
  config.json    parámetros usados (incluye "orden_bandas", que el entrenamiento lee)

Cambios principales respecto a la versión anterior:
  * Búsqueda recursiva y sin distinguir mayúsculas/separadores de la capa Burned Area y de las bandas.
  * Máscara con escala fija según el tipo de dato (antes el min/max por banda hacía que en
    exportaciones float32 el rojo puro quedara < 254 y la escena se descartaba).
  * Sin gdal_translate/gdal_merge.py por subprocess (sus errores se silenciaban).
  * Sin estiramiento min/max por banda: se conservan valores físicos.
  * Recortes negativos (sin quemado) para que el modelo aprenda a decir "no hay incendio".
  * Imágenes más chicas que el recorte se rellenan en vez de descartarse.
  * Split por sitio para evitar fuga de información por el traslape entre recortes.
"""

import csv
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import wildfire_common as wc


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# --- Rutas (BASE_DIR viene del .env) ---
INPUT_DIR = wc.ruta_proyecto("Dataset", "Raw")            # carpeta con los .zip de EO Browser
LISTA_ZIPS = None              # o ruta a un .txt con un zip por línea (p. ej. reporte_zips/zips_disponibles.txt)
OUTPUT_DIR = wc.ruta_proyecto("Mobile-UNet/Mobile-UNet_27", "data")    # el entrenamiento lee de aquí (DATA_DIR)
TMP_DIR = None                 # dónde descomprimir temporalmente (None = /tmp)

# --- Bandas a guardar (en este orden) ---
# R, G, B, NIR. Para agregar SWIR (mejora mucho área quemada, pero la cámara RGB+IR no la tiene):
#   BANDAS_DATASET = ("B04", "B03", "B02", "B08", "B11", "B12")
# Los zips que no traigan alguna de estas bandas se descartan (estado faltan_bandas).
BANDAS_DATASET = ("B04", "B03", "B02", "B08")

# --- Recortes ---
TILE = 128                     # tamaño del recorte (px)
OVERLAP = 64                   # traslape entre recortes positivos (px)
MIN_FIRE = 0.035               # fracción mínima de píxeles quemados para que un recorte sea positivo
INCLUIR_BAJOS = False          # guardar también recortes con 0 < quemado < MIN_FIRE (bordes)
NEG_RATIO = 0.5                # recortes negativos por cada positivo en la misma escena (0 = sin negativos)
NEG_SIN_FUEGO = 4              # negativos a guardar de escenas sin ningún píxel quemado
MAX_NODATA = 0.2               # fracción máxima de píxeles sin dato en un recorte

# --- Split ---
VAL_FRAC = 0.2
TEST_FRAC = 0.1
SPLIT_POR = "sitio"            # "sitio" (agrupa escenas que se traslapan, recomendado) | "escena"

# --- Lectura de bandas y máscara ---
ESCALA = "auto"                # "auto" | "dn" | "reflectancia" | "8bit" | "16bit_completo"
UMBRAL_ROJO = 0.75             # R mínimo (0-1) para considerar quemado
UMBRAL_GB = 0.2                # G y B máximos (0-1) para considerar quemado
PALABRAS_MASCARA = list(wc.DEFAULT_MASK_KEYWORDS)

# --- Ejecución ---
INCLUIR_CARPETAS = False       # procesar también subcarpetas ya descomprimidas dentro de INPUT_DIR
RECURSIVO = False              # buscar zips también en subcarpetas
WORKERS = 4                    # procesos en paralelo
LIMPIAR = False                # borrar Images/ y SegmentationClass/ de OUTPUT_DIR antes de empezar
SEED = 42
# =============================================================================


# ---------------------------------------------------------------------------
# Procesamiento de una escena
# ---------------------------------------------------------------------------

def _seleccionar_negativos(candidatos, k, tam, rng):
    """Elige hasta k ventanas sin traslape entre sí (menos redundancia)."""
    if k <= 0 or not candidatos:
        return []
    orden = list(candidatos)
    rng.shuffle(orden)
    elegidos = []
    for cand in orden:
        y, x = cand[0], cand[1]
        if all(abs(y - e[0]) >= tam or abs(x - e[1]) >= tam for e in elegidos):
            elegidos.append(cand)
            if len(elegidos) >= k:
                break
    return elegidos


def procesar_escena(item, escena_id, cfg):
    """Procesa un zip (o carpeta). Devuelve (registro_escena, lista_registros_crops)."""
    registro = {
        "origen": os.path.basename(item), "escena": escena_id, "estado": "error", "motivo": "",
        "alto": "", "ancho": "", "modo_escala": "", "p99_dn": "", "fraccion_quemada": "",
        "positivos": 0, "negativos": 0, "bajos_descartados": 0, "nodata_descartados": 0,
        "alpha_mascara": "", "mascara_remuestreada": "", "bandas_remuestreadas": "",
        "rellenada": False, "bounds_wgs84": "", "archivo_mascara": "", "avisos": "",
    }
    crops = []
    tmp_ctx = None
    try:
        if os.path.isdir(item):
            raiz = item
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="wf_", dir=cfg["tmp_dir"])
            raiz = tmp_ctx.name
            wc.extraer_zip_seguro(item, raiz)

        hallados = wc.buscar_archivos_escena(raiz, cfg["palabras_mascara"], bandas=cfg["bandas"])
        registro["avisos"] = " | ".join(hallados["avisos"])
        listado = ", ".join(hallados["archivos"][:12]) + (" ..." if len(hallados["archivos"]) > 12 else "")

        if hallados["mascara"] is None:
            registro["estado"] = "sin_mascara"
            registro["motivo"] = f"No hay capa de área quemada. Archivos: {listado}"
            return registro, crops
        faltan = [b for b in cfg["bandas"] if b not in hallados["bandas"]]
        if faltan:
            registro["estado"] = "faltan_bandas"
            registro["motivo"] = f"Faltan {faltan}. Archivos: {listado}"
            return registro, crops
        registro["archivo_mascara"] = os.path.relpath(hallados["mascara"], raiz)

        # --- Bandas y máscara alineadas a la rejilla de la primera banda (B04) ---
        dn, meta, nodata, info_b = wc.leer_stack_bandas(hallados["bandas"], escala=cfg["escala"],
                                                        orden=cfg["bandas"])
        alto, ancho = nodata.shape
        mascara, info_m = wc.extraer_mascara_quemado(
            hallados["mascara"], (alto, ancho), meta,
            umbral_rojo=cfg["umbral_rojo"], umbral_gb=cfg["umbral_gb"])
        mascara[nodata] = wc.IGNORE_INDEX

        registro.update({
            "alto": alto, "ancho": ancho, "modo_escala": info_b["modo_escala"],
            "p99_dn": ";".join(f"{v:.0f}" for v in info_b["p99_dn"]),
            "fraccion_quemada": f"{info_m['fraccion_quemada']:.5f}",
            "alpha_mascara": info_m["alpha"], "mascara_remuestreada": info_m["remuestreada"],
            "bandas_remuestreadas": ";".join(info_b["bandas_remuestreadas"]),
        })
        if info_b.get("aviso"):
            registro["avisos"] = (registro["avisos"] + " | " + info_b["aviso"]).strip(" |")
        b = wc.bounds_wgs84(meta, alto, ancho)
        if b is not None:
            registro["bounds_wgs84"] = ";".join(f"{v:.6f}" for v in b)

        # --- Relleno si la escena es más chica que el recorte ---
        tam = cfg["tile"]
        real = np.ones((alto, ancho), dtype=bool)
        pad_h, pad_w = max(0, tam - alto), max(0, tam - ancho)
        if pad_h or pad_w:
            dn = np.pad(dn, ((0, 0), (0, pad_h), (0, pad_w)))
            mascara = np.pad(mascara, ((0, pad_h), (0, pad_w)), constant_values=wc.IGNORE_INDEX)
            real = np.pad(real, ((0, pad_h), (0, pad_w)), constant_values=False)
            registro["rellenada"] = True

        # --- Clasificar todas las ventanas ---
        positivos, negativos_cand, bajos, n_nodata = wc.clasificar_ventanas(
            mascara, real, tam, cfg["overlap"], cfg["min_fire"], cfg["max_nodata"])
        registro["nodata_descartados"] = n_nodata

        if cfg["incluir_bajos"]:
            positivos += bajos
        else:
            registro["bajos_descartados"] = len(bajos)

        rng = random.Random(f"{cfg['seed']}-{escena_id}")
        k_neg = math.ceil(cfg["neg_ratio"] * len(positivos)) if positivos else cfg["neg_sin_fuego"]
        negativos = _seleccionar_negativos(negativos_cand, k_neg, tam, rng)

        # --- Escribir recortes ---
        for (y, x, quemados, n_validos, frac, frac_nodata, tipo) in positivos + negativos:
            nombre = f"{escena_id}_x{x}_y{y}.tif"
            meta_v = wc.transform_de_ventana(meta, x, y, tam, tam)
            wc.escribir_raster(os.path.join(cfg["dir_img"], nombre),
                               wc.a_uint16(dn[:, y:y + tam, x:x + tam]), meta_v)
            wc.escribir_raster(os.path.join(cfg["dir_mask"], nombre),
                               mascara[y:y + tam, x:x + tam].copy(), meta_v, nodata=wc.IGNORE_INDEX)
            crops.append({
                "archivo": nombre, "escena": escena_id, "x": x, "y": y, "tipo": tipo,
                "pixeles_quemados": quemados, "pixeles_validos": n_validos,
                "porcentaje_quemado": round(100.0 * frac, 4),
                "porcentaje_nodata": round(100.0 * frac_nodata, 4),
            })

        registro["positivos"] = len(positivos)
        registro["negativos"] = len(negativos)
        registro["estado"] = "ok" if crops else "sin_recortes"
        if not crops:
            registro["motivo"] = "Ninguna ventana cumplió los criterios (revisa MAX_NODATA / MIN_FIRE)"
        return registro, crops

    except Exception as e:
        registro["estado"] = "error"
        registro["motivo"] = f"{type(e).__name__}: {e}"
        registro["avisos"] = (registro["avisos"] + " | " + traceback.format_exc(limit=2).replace("\n", " ")).strip(" |")
        return registro, crops
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


# ---------------------------------------------------------------------------
# Agrupación por sitio y split
# ---------------------------------------------------------------------------

def agrupar_por_sitio(escenas):
    """Escenas cuyos footprints (WGS84) se intersectan forman un mismo grupo."""
    cajas = {e["escena"]: (tuple(float(v) for v in e["bounds_wgs84"].split(";")) if e["bounds_wgs84"] else None)
             for e in escenas}
    return wc.agrupar_por_footprint(cajas), sum(c is not None for c in cajas.values())


def asignar_splits(grupo_de, conteo_por_escena, val_frac, test_frac, seed):
    tam_grupo = {}
    for esc, g in grupo_de.items():
        tam_grupo[g] = tam_grupo.get(g, 0) + conteo_por_escena.get(esc, 0)
    grupos = [g for g in tam_grupo if tam_grupo[g] > 0]
    rng = random.Random(seed)
    rng.shuffle(grupos)
    grupos.sort(key=lambda g: tam_grupo[g], reverse=True)   # estable: aleatorio entre empates

    objetivos = {"train": 1.0 - val_frac - test_frac, "valid": val_frac, "test": test_frac}
    objetivos = {k: v for k, v in objetivos.items() if v > 0}
    total = sum(tam_grupo[g] for g in grupos)
    cuenta = {k: 0 for k in objetivos}
    split_de_grupo = {}
    for g in grupos:
        deficit = {k: objetivos[k] * total - cuenta[k] for k in objetivos}
        k = max(deficit, key=deficit.get)
        split_de_grupo[g] = k
        cuenta[k] += tam_grupo[g]

    # Garantizar al menos un grupo en validación si es posible
    if "valid" in objetivos and cuenta.get("valid", 0) == 0:
        de_train = [g for g in grupos if split_de_grupo[g] == "train"]
        if len(de_train) > 1:
            g = min(de_train, key=lambda q: tam_grupo[q])
            split_de_grupo[g] = "valid"
    return split_de_grupo, len(grupos)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def escribir_csv(ruta, filas, campos):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)


def main():
    t0 = time.time()

    if LISTA_ZIPS is None and INPUT_DIR is None:
        sys.exit("ERROR: define INPUT_DIR o LISTA_ZIPS")
    if LISTA_ZIPS is None and not os.path.isdir(INPUT_DIR):
        sys.exit(f"ERROR: no existe INPUT_DIR = {INPUT_DIR}")
    if VAL_FRAC + TEST_FRAC >= 1.0:
        sys.exit("ERROR: VAL_FRAC + TEST_FRAC debe ser < 1")
    if not 0 <= OVERLAP < TILE:
        sys.exit("ERROR: OVERLAP debe estar entre 0 y TILE - 1")
    if BANDAS_DATASET[0] != "B04":
        print("⚠ La primera banda define la rejilla de referencia; se recomienda B04 (10 m).")
    if not wc.HAS_RASTERIO:
        print("⚠ rasterio no está instalado: los recortes NO tendrán georreferencia y no se puede agrupar por sitio.")

    dir_img = os.path.join(OUTPUT_DIR, "Images")
    dir_mask = os.path.join(OUTPUT_DIR, "SegmentationClass")
    if LIMPIAR:
        for d in (dir_img, dir_mask):
            shutil.rmtree(d, ignore_errors=True)
    elif os.path.isdir(dir_img) and os.listdir(dir_img):
        print(f"⚠ {dir_img} ya tiene archivos: los recortes viejos que no se regeneren quedan en la carpeta "
              "(no entran a las listas, pero ocupan espacio). Usa LIMPIAR = True para empezar de cero.")
    os.makedirs(dir_img, exist_ok=True)
    os.makedirs(dir_mask, exist_ok=True)

    # --- Entradas ---
    items = []
    if LISTA_ZIPS:
        with open(LISTA_ZIPS) as f:
            items = [l.split("\t")[0].strip() for l in f if l.strip()]
        faltan = [i for i in items if not os.path.exists(i)]
        if faltan:
            print(f"⚠ {len(faltan)} rutas de la lista no existen (se omiten). Ej: {faltan[:3]}")
        items = [i for i in items if os.path.exists(i)]
    elif RECURSIVO:
        for carpeta, _, archivos in os.walk(INPUT_DIR):
            items += [os.path.join(carpeta, a) for a in archivos if a.lower().endswith(".zip")]
    else:
        items += [os.path.join(INPUT_DIR, a) for a in os.listdir(INPUT_DIR) if a.lower().endswith(".zip")]
    if INCLUIR_CARPETAS and INPUT_DIR and os.path.isdir(INPUT_DIR):
        out_abs = os.path.abspath(OUTPUT_DIR)
        for a in os.listdir(INPUT_DIR):
            ruta = os.path.join(INPUT_DIR, a)
            if os.path.isdir(ruta) and os.path.abspath(ruta) != out_abs:
                items.append(ruta)
    items = sorted(items)
    if not items:
        sys.exit("No se encontraron zips (usa INCLUIR_CARPETAS = True si ya están descomprimidos)")

    usados, tareas = {}, []
    for it in items:
        base = wc.nombre_seguro(os.path.splitext(os.path.basename(it))[0])
        n = usados.get(base, 0)
        usados[base] = n + 1
        tareas.append((it, base if n == 0 else f"{base}_{n}"))

    cfg = {
        "tile": TILE, "overlap": OVERLAP, "min_fire": MIN_FIRE,
        "incluir_bajos": INCLUIR_BAJOS, "neg_ratio": NEG_RATIO,
        "neg_sin_fuego": NEG_SIN_FUEGO, "max_nodata": MAX_NODATA,
        "escala": ESCALA, "umbral_rojo": UMBRAL_ROJO, "umbral_gb": UMBRAL_GB,
        "palabras_mascara": PALABRAS_MASCARA, "tmp_dir": TMP_DIR, "bandas": tuple(BANDAS_DATASET),
        "dir_img": dir_img, "dir_mask": dir_mask, "seed": SEED,
    }

    print("=" * 70)
    print(f"Escenas a procesar: {len(tareas)} | bandas {list(BANDAS_DATASET)} | recorte {TILE}px, "
          f"traslape {OVERLAP}px (paso {TILE - OVERLAP}px) | workers {WORKERS}")
    print("=" * 70)

    escenas, crops = [], []

    def reportar(i, reg, cr):
        estado = reg["estado"]
        extra = f"{reg['positivos']} pos / {reg['negativos']} neg" if estado == "ok" else reg["motivo"][:140]
        print(f"[{i}/{len(tareas)}] {reg['origen']}: {estado} - {extra}", flush=True)

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futuros = {ex.submit(procesar_escena, it, eid, cfg): it for it, eid in tareas}
            for i, fut in enumerate(as_completed(futuros), 1):
                reg, cr = fut.result()
                escenas.append(reg)
                crops += cr
                reportar(i, reg, cr)
    else:
        for i, (it, eid) in enumerate(tareas, 1):
            reg, cr = procesar_escena(it, eid, cfg)
            escenas.append(reg)
            crops += cr
            reportar(i, reg, cr)

    escenas.sort(key=lambda e: e["escena"])
    crops.sort(key=lambda c: c["archivo"])

    # --- Split ---
    con_crops = [e for e in escenas if e["estado"] == "ok"]
    conteo = {}
    for c in crops:
        conteo[c["escena"]] = conteo.get(c["escena"], 0) + 1

    if SPLIT_POR == "sitio":
        grupo_de, n_georef = agrupar_por_sitio(con_crops)
        if n_georef < len(con_crops):
            print(f"⚠ {len(con_crops) - n_georef} escenas sin georreferencia: se agrupan solas (riesgo de fuga si "
                  "hay varias fechas del mismo sitio).")
    else:
        grupo_de = {e["escena"]: e["escena"] for e in con_crops}
    split_de_grupo, n_grupos = asignar_splits(grupo_de, conteo, VAL_FRAC, TEST_FRAC, SEED)
    if n_grupos < 3:
        print(f"⚠ Solo hay {n_grupos} sitio(s) independientes: la validación será poco representativa.")

    for e in escenas:
        g = grupo_de.get(e["escena"])
        e["grupo"] = g or ""
        e["split"] = split_de_grupo.get(g, "") if g else ""
    split_de_escena = {e["escena"]: e["split"] for e in escenas}
    listas = {"train": [], "valid": [], "test": []}
    for c in crops:
        c["split"] = split_de_escena.get(c["escena"], "train")
        listas[c["split"]].append(c["archivo"])

    for nombre, lista in listas.items():
        with open(os.path.join(OUTPUT_DIR, f"{nombre}.txt"), "w") as f:
            f.write("\n".join(lista) + ("\n" if lista else ""))

    escribir_csv(os.path.join(OUTPUT_DIR, "crops.csv"), crops,
                 ["archivo", "escena", "split", "tipo", "x", "y", "pixeles_quemados", "pixeles_validos",
                  "porcentaje_quemado", "porcentaje_nodata"])
    escribir_csv(os.path.join(OUTPUT_DIR, "escenas.csv"), escenas,
                 ["origen", "escena", "estado", "motivo", "split", "grupo", "alto", "ancho", "rellenada",
                  "modo_escala", "p99_dn", "fraccion_quemada", "positivos", "negativos", "bajos_descartados",
                  "nodata_descartados", "alpha_mascara", "mascara_remuestreada", "bandas_remuestreadas",
                  "bounds_wgs84", "archivo_mascara", "avisos"])
    config = {k: v for k, v in globals().items() if k.isupper()}
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump({**config, "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "orden_bandas": list(BANDAS_DATASET), "unidades": "DN = reflectancia*10000",
                   "mascara": {"0": "no quemado", "1": "quemado", "255": "sin dato (ignorar)"}}, f, indent=2, default=str)

    # --- Resumen ---
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    estados = {}
    for e in escenas:
        estados[e["estado"]] = estados.get(e["estado"], 0) + 1
    for k, v in sorted(estados.items()):
        print(f"  {k:15s}: {v}")
    modos = {}
    for e in con_crops:
        modos[e["modo_escala"]] = modos.get(e["modo_escala"], 0) + 1
    if len(modos) > 1:
        print(f"  ⚠ Escenas con escalas distintas: {modos} (revisa 'modo_escala' y 'p99_dn' en escenas.csv)")
    for nombre in ("train", "valid", "test"):
        sel = [c for c in crops if c["split"] == nombre]
        if not sel:
            continue
        pos = sum(c["tipo"] != "negativo" for c in sel)
        q = sum(c["pixeles_quemados"] for c in sel)
        v = sum(c["pixeles_validos"] for c in sel)
        escs = len({c["escena"] for c in sel})
        print(f"  {nombre:5s}: {len(sel):6d} recortes ({pos} pos / {len(sel) - pos} neg) de {escs} escenas, "
              f"{100.0 * q / max(v, 1):.2f}% píxeles quemados")
    malas = [e for e in escenas if e["estado"] != "ok"]
    if malas:
        print("\n  Escenas descartadas (detalle completo en escenas.csv):")
        for e in malas[:20]:
            print(f"    - {e['origen']}: {e['estado']} - {e['motivo'][:160]}")
        if len(malas) > 20:
            print(f"    ... y {len(malas) - 20} más")
    print(f"\nDataset en: {OUTPUT_DIR}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
