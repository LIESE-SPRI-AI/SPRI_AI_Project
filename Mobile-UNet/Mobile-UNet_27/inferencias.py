#!/usr/bin/env python3
"""
inferencias.py
==============
Ejecuta la predicción para cada combinación imagen × modelo.

Cada modelo se carga UNA vez y se aplica a todas las imágenes (antes se lanzaba un proceso
de predict por cada par, recargando el modelo cada vez). La preparación de la entrada
(bandas, índices, normalización, cuantización) sale del checkpoint de cada modelo.

Convenciones de nombres:
  Entrada : DIR_IMAGENES/<nombrebase>_Merged.tif
  Modelos : patrones de MODELOS (weights/model_<nombre>.pth o runs/<corrida>/best.pth)
  Salida  : DIR_SALIDA/<nombrebase>_<nombremodelo>.tif
"""

import glob
import os
import sys
from pathlib import Path

import wildfire_common as wc
import predict_segmentation as ps

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DIR_IMAGENES = Path(wc.ruta_proyecto("Inferencias", "Input"))
PATRON_IMAGENES = "*_Merged.tif"
MODELOS = [                                   # patrones glob de checkpoints
    os.path.join(wc.DIR_CODIGO, "weights", "model_*.pth"),
    os.path.join(wc.DIR_CODIGO, "runs", "*", "best.pth"),
]
DIR_SALIDA = Path(wc.ruta_proyecto("Inferencias", "Output_Inferencias_Mobile-UNet", "Output_MbUN_27"))
ORDEN_BANDAS_TIFF = ("B04", "B03", "B02", "B08")   # orden de bandas en los *_Merged.tif
OMITIR_EXISTENTES = True                      # no recalcular si la salida ya existe
GPU = "auto"
TTA = False
VISUALIZACION = False
GUARDAR_PROB = False
# =============================================================================


def obtener_imagenes():
    archivos = sorted(DIR_IMAGENES.glob(PATRON_IMAGENES))
    if not archivos:
        print(f"[ADVERTENCIA] No se encontraron imágenes '{PATRON_IMAGENES}' en: {DIR_IMAGENES}")
    return archivos


def obtener_modelos():
    archivos = sorted({Path(p) for patron in MODELOS for p in glob.glob(patron)})
    if not archivos:
        print(f"[ADVERTENCIA] No se encontraron modelos con los patrones: {MODELOS}")
    return archivos


def nombre_modelo(modelo: Path) -> str:
    if modelo.stem in ("best", "final", "last"):          # runs/<corrida>/best.pth
        return f"{modelo.parent.name}_{modelo.stem}"
    return modelo.stem.replace("model_", "", 1)


def nombre_salida(imagen: Path, modelo: Path) -> Path:
    nombrebase = imagen.stem.replace("_Merged", "")
    return DIR_SALIDA / f"{nombrebase}_{nombre_modelo(modelo)}.tif"


def main():
    if not DIR_IMAGENES.is_dir():
        sys.exit(f"[ERROR] El directorio de imágenes no existe: {DIR_IMAGENES}")
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    imagenes = obtener_imagenes()
    modelos = obtener_modelos()
    if not imagenes or not modelos:
        sys.exit("[ERROR] No hay imágenes o modelos para procesar.")

    total = len(imagenes) * len(modelos)
    exitos = errores = omitidos = 0
    opts = {"dir_salida": str(DIR_SALIDA), "visualizacion": VISUALIZACION, "guardar_prob": GUARDAR_PROB,
            "eval": False, "mascara_ref": None, "orden_tiff": tuple(ORDEN_BANDAS_TIFF)}

    print(f"\n{'═' * 60}")
    print(f"  Imágenes encontradas : {len(imagenes)}")
    print(f"  Modelos encontrados  : {len(modelos)}")
    print(f"  Inferencias totales  : {total}")
    print(f"  Salida               : {DIR_SALIDA}")
    print(f"{'═' * 60}")

    for modelo in modelos:
        pendientes = [im for im in imagenes if not (OMITIR_EXISTENTES and nombre_salida(im, modelo).exists())]
        omitidos += len(imagenes) - len(pendientes)
        if not pendientes:
            print(f"\n[OMITIDO] {modelo}: todas las salidas ya existen")
            continue
        print(f"\n{'─' * 60}\n  Modelo : {modelo}\n{'─' * 60}")
        try:
            cfg = ps.preparar_modelo(str(modelo), gpu=GPU, tta=TTA)
        except Exception as e:
            print(f"[ERROR] No se pudo cargar {modelo}: {type(e).__name__}: {e}")
            errores += len(pendientes)
            continue
        for imagen in pendientes:
            salida = nombre_salida(imagen, modelo)
            print(f"  Imagen : {imagen.name}  ->  {salida.name}")
            try:
                ps.predecir_archivo(str(imagen), cfg, opts, nombre_salida=str(salida))
                exitos += 1
            except Exception as e:
                print(f"[ERROR] {type(e).__name__}: {e}")
                errores += 1

    print(f"\n{'═' * 60}")
    print("  RESUMEN")
    print(f"  Completadas : {exitos}")
    print(f"  Errores     : {errores}")
    print(f"  Omitidas    : {omitidos}  (salida ya existía)")
    print(f"{'═' * 60}\n")
    if errores:
        sys.exit(1)


if __name__ == "__main__":
    main()
