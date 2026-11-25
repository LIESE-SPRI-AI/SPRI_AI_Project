#!/usr/bin/env python3
"""
generate_wildfire_dataset.py
============================
Genera el dataset de entrenamiento a partir de los zips exportados de EO Browser
(bandas B02/B03/B04/B08 "Raw" + capa "Burned Area Detection").

Antes conviene revisar los zips (detecta corruptos, capas faltantes y duplicados):
  python revisar_zips.py --input-dir /home/felix/SPRI_AI_Project/Dataset/Raw --profundo
  python generate_wildfire_dataset.py --lista-zips reporte_zips/zips_disponibles.txt --output-dir ...

Ejemplo:
  python generate_wildfire_dataset.py \
      --input-dir  /home/felix/SPRI_AI_Project/Dataset/Raw \
      --output-dir /home/felix/SPRI_AI_Project/Dataset/Dataset_v2 \
      --tile 128 --overlap 64 --min-fire 0.035 --neg-ratio 0.5 \
      --val-frac 0.2 --test-frac 0.1 --workers 4

Salida (lista para trainModelWildfire3D4B.py --data-dir <output-dir>):
  Images/<escena>_x<col>_y<fila>.tif      4 bandas uint16 (R,G,B,NIR) en DN = reflectancia*10000
  SegmentationClass/<mismo nombre>.tif    1 banda uint8: 0 = no quemado, 1 = quemado, 255 = sin dato
  train.txt / valid.txt / test.txt        split por SITIO: escenas que se traslapan van al mismo split
  crops.csv      un renglón por recorte (tipo, % quemado, % sin dato, split...)
  escenas.csv    un renglón por zip con su estado y, si se descartó, el motivo y los archivos que tenía
  config.json    parámetros usados

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

import argparse
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


def parse_args():
    p = argparse.ArgumentParser(description="Generador de dataset de áreas quemadas (Sentinel-2 / EO Browser)")
    p.add_argument("--input-dir", default=None, help="Carpeta con los .zip de EO Browser")
    p.add_argument("--lista-zips", default=None,
                   help="Archivo con una ruta de zip por línea (p. ej. zips_disponibles.txt de revisar_zips.py)")
    p.add_argument("--output-dir", required=True, help="Carpeta de salida del dataset")
    p.add_argument("--tile", type=int, default=128, help="Tamaño del recorte (px)")
    p.add_argument("--overlap", type=int, default=64, help="Traslape entre recortes positivos (px)")
    p.add_argument("--min-fire", type=float, default=0.03,
                   help="Fracción mínima de píxeles quemados para que un recorte sea positivo")
    p.add_argument("--incluir-bajos", action="store_true",
                   help="Guardar también recortes con 0 < fracción quemada < --min-fire (bordes de incendio)")
    p.add_argument("--neg-ratio", type=float, default=0.5,
                   help="Recortes negativos por cada positivo en la misma escena (0 = sin negativos)")
    p.add_argument("--neg-sin-fuego", type=int, default=4,
                   help="Negativos a guardar de escenas sin ningún píxel quemado")
    p.add_argument("--max-nodata", type=float, default=0.2,
                   help="Fracción máxima de píxeles sin dato permitida en un recorte")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--split-por", choices=["sitio", "escena"], default="sitio",
                   help="'sitio' agrupa escenas cuyos footprints se traslapan (recomendado)")
    p.add_argument("--escala", default="auto", choices=["auto"] + list(wc.ESCALAS),
                   help="Cómo convertir las bandas a DN (auto detecta por tipo de dato)")
    p.add_argument("--umbral-rojo", type=float, default=0.75, help="R mínimo (0-1) para considerar quemado")
    p.add_argument("--umbral-gb", type=float, default=0.2, help="G y B máximos (0-1) para considerar quemado")
    p.add_argument("--palabras-mascara", nargs="+", default=list(wc.DEFAULT_MASK_KEYWORDS),
                   help="Palabras para reconocer la capa de área quemada (sin importar mayúsculas)")
    p.add_argument("--incluir-carpetas", action="store_true",
                   help="Procesar también subcarpetas ya descomprimidas dentro de --input-dir")
    p.add_argument("--recursivo", action="store_true", help="Buscar zips también en subcarpetas")
    p.add_argument("--tmp-dir", default=None, help="Dónde descomprimir temporalmente (por defecto /tmp)")
    p.add_argument("--workers", type=int, default=1, help="Procesos en paralelo")
    p.add_argument("--limpiar", action="store_true",
                   help="Borrar Images/ y SegmentationClass/ de --output-dir antes de empezar")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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

        hallados = wc.buscar_archivos_escena(raiz, cfg["palabras_mascara"])
        registro["avisos"] = " | ".join(hallados["avisos"])
        listado = ", ".join(hallados["archivos"][:12]) + (" ..." if len(hallados["archivos"]) > 12 else "")

        if hallados["mascara"] is None:
            registro["estado"] = "sin_mascara"
            registro["motivo"] = f"No hay capa de área quemada. Archivos: {listado}"
            return registro, crops
        faltan = [b for b in wc.BAND_ORDER if b not in hallados["bandas"]]
        if faltan:
            registro["estado"] = "faltan_bandas"
            registro["motivo"] = f"Faltan {faltan}. Archivos: {listado}"
            return registro, crops
        registro["archivo_mascara"] = os.path.relpath(hallados["mascara"], raiz)

        # --- Bandas y máscara alineadas a la rejilla de B04 ---
        dn, meta, nodata, info_b = wc.leer_stack_bandas(hallados["bandas"], escala=cfg["escala"])
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
            registro["motivo"] = "Ninguna ventana cumplió los criterios (revisa --max-nodata / --min-fire)"
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
    args = parse_args()
    t0 = time.time()

    if args.lista_zips is None and args.input_dir is None:
        sys.exit("ERROR: indica --input-dir o --lista-zips")
    if args.input_dir is not None and not os.path.isdir(args.input_dir):
        sys.exit(f"ERROR: no existe {args.input_dir}")
    if args.val_frac + args.test_frac >= 1.0:
        sys.exit("ERROR: --val-frac + --test-frac debe ser < 1")
    if not 0 <= args.overlap < args.tile:
        sys.exit("ERROR: --overlap debe estar entre 0 y --tile - 1")
    if not wc.HAS_RASTERIO:
        print("⚠ rasterio no está instalado: los recortes NO tendrán georreferencia y no se puede agrupar por sitio.")

    dir_img = os.path.join(args.output_dir, "Images")
    dir_mask = os.path.join(args.output_dir, "SegmentationClass")
    if args.limpiar:
        for d in (dir_img, dir_mask):
            shutil.rmtree(d, ignore_errors=True)
    os.makedirs(dir_img, exist_ok=True)
    os.makedirs(dir_mask, exist_ok=True)

    # --- Entradas ---
    items = []
    if args.lista_zips:
        with open(args.lista_zips) as f:
            items = [l.split("\t")[0].strip() for l in f if l.strip()]
        faltan = [i for i in items if not os.path.exists(i)]
        if faltan:
            print(f"⚠ {len(faltan)} rutas de la lista no existen (se omiten). Ej: {faltan[:3]}")
        items = [i for i in items if os.path.exists(i)]
    elif args.recursivo:
        for carpeta, _, archivos in os.walk(args.input_dir):
            items += [os.path.join(carpeta, a) for a in archivos if a.lower().endswith(".zip")]
    else:
        items += [os.path.join(args.input_dir, a) for a in os.listdir(args.input_dir) if a.lower().endswith(".zip")]
    if args.incluir_carpetas and args.input_dir:
        out_abs = os.path.abspath(args.output_dir)
        for a in os.listdir(args.input_dir):
            ruta = os.path.join(args.input_dir, a)
            if os.path.isdir(ruta) and os.path.abspath(ruta) != out_abs:
                items.append(ruta)
    items = sorted(items)
    if not items:
        sys.exit("No se encontraron zips (usa --incluir-carpetas si ya están descomprimidos)")

    usados, tareas = {}, []
    for it in items:
        base = wc.nombre_seguro(os.path.splitext(os.path.basename(it))[0])
        n = usados.get(base, 0)
        usados[base] = n + 1
        tareas.append((it, base if n == 0 else f"{base}_{n}"))

    cfg = {
        "tile": args.tile, "overlap": args.overlap, "min_fire": args.min_fire,
        "incluir_bajos": args.incluir_bajos, "neg_ratio": args.neg_ratio,
        "neg_sin_fuego": args.neg_sin_fuego, "max_nodata": args.max_nodata,
        "escala": args.escala, "umbral_rojo": args.umbral_rojo, "umbral_gb": args.umbral_gb,
        "palabras_mascara": args.palabras_mascara, "tmp_dir": args.tmp_dir,
        "dir_img": dir_img, "dir_mask": dir_mask, "seed": args.seed,
    }

    print("=" * 70)
    print(f"Escenas a procesar: {len(tareas)} | recorte {args.tile}px, traslape {args.overlap}px "
          f"(paso {args.tile - args.overlap}px) | workers {args.workers}")
    print("=" * 70)

    escenas, crops = [], []

    def reportar(i, reg, cr):
        estado = reg["estado"]
        extra = f"{reg['positivos']} pos / {reg['negativos']} neg" if estado == "ok" else reg["motivo"][:140]
        print(f"[{i}/{len(tareas)}] {reg['origen']}: {estado} - {extra}", flush=True)

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
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

    if args.split_por == "sitio":
        grupo_de, n_georef = agrupar_por_sitio(con_crops)
        if n_georef < len(con_crops):
            print(f"⚠ {len(con_crops) - n_georef} escenas sin georreferencia: se agrupan solas (riesgo de fuga si "
                  "hay varias fechas del mismo sitio).")
    else:
        grupo_de = {e["escena"]: e["escena"] for e in con_crops}
    split_de_grupo, n_grupos = asignar_splits(grupo_de, conteo, args.val_frac, args.test_frac, args.seed)
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
        with open(os.path.join(args.output_dir, f"{nombre}.txt"), "w") as f:
            f.write("\n".join(lista) + ("\n" if lista else ""))

    escribir_csv(os.path.join(args.output_dir, "crops.csv"), crops,
                 ["archivo", "escena", "split", "tipo", "x", "y", "pixeles_quemados", "pixeles_validos",
                  "porcentaje_quemado", "porcentaje_nodata"])
    escribir_csv(os.path.join(args.output_dir, "escenas.csv"), escenas,
                 ["origen", "escena", "estado", "motivo", "split", "grupo", "alto", "ancho", "rellenada",
                  "modo_escala", "p99_dn", "fraccion_quemada", "positivos", "negativos", "bajos_descartados",
                  "nodata_descartados", "alpha_mascara", "mascara_remuestreada", "bandas_remuestreadas",
                  "bounds_wgs84", "archivo_mascara", "avisos"])
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "orden_bandas": list(wc.BAND_ORDER), "unidades": "DN = reflectancia*10000",
                   "mascara": {"0": "no quemado", "1": "quemado", "255": "sin dato (ignorar)"}}, f, indent=2)

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
    print(f"\nDataset en: {args.output_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
