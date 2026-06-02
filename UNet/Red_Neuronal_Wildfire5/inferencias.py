"""
run_inferences.py
-----------------
Ejecuta predict.py para cada combinación imagen × modelo.

Convenciones de nombres:
  Entrada : <DIR_IMAGENES>/<nombrebase>_Merged.tif
  Modelos : <DIR_MODELOS>/model_<nombremodelo>.pth
  Salida  : <DIR_SALIDA>/<nombrebase>_MbUN_Out_<nombremodelo>.tif
"""

import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIGURACIÓN  ← edita estos tres valores
# ─────────────────────────────────────────────
DIR_IMAGENES = Path("/home/liese2/SPRI_AI_project/Inferencias/Input")       # directorio con los .tif de entrada
DIR_MODELOS  = Path("/home/liese2/SPRI_AI_project/UNet/Red_Neuronal_Wildfire5/weights")        # directorio con los .pth
DIR_SALIDA   = Path("/home/liese2/SPRI_AI_project/Inferencias/Output_Inferencias_UNet/Output_UN_5")         # directorio donde se guardan los resultados
PREDICT_PY   = Path("/home/liese2/SPRI_AI_project/UNet/Red_Neuronal_Wildfire5/predict_segmentation.py")            # ruta al script de inferencia
# ─────────────────────────────────────────────


def obtener_imagenes(directorio: Path) -> list[Path]:
    """Devuelve todos los archivos que terminan en _Merged.tif."""
    archivos = sorted(directorio.glob("*_Merged.tif"))
    if not archivos:
        print(f"[ADVERTENCIA] No se encontraron imágenes '*_Merged.tif' en: {directorio}")
    return archivos


def obtener_modelos(directorio: Path) -> list[Path]:
    """Devuelve todos los archivos que siguen el patrón model_*.pth."""
    archivos = sorted(directorio.glob("model_*.pth"))
    if not archivos:
        print(f"[ADVERTENCIA] No se encontraron modelos 'model_*.pth' en: {directorio}")
    return archivos


def nombre_salida(imagen: Path, modelo: Path) -> Path:
    """Construye el nombre del archivo de salida."""
    # nombrebase = todo lo que está antes de '_Merged'
    nombrebase = imagen.stem.replace("_Merged", "")
    # nombremodelo = lo que sigue después de 'model_' (sin extensión)
    nombremodelo = modelo.stem.replace("model_", "")
    return DIR_SALIDA / f"{nombrebase}_{nombremodelo}.tif"


def ejecutar_inferencia(imagen: Path, modelo: Path, salida: Path) -> bool:
    """Llama a predict.py y devuelve True si terminó sin errores."""
    cmd = [
        sys.executable, str(PREDICT_PY),
        "--image",  str(imagen),
        "--model",  str(modelo),
        "--output", str(salida),
    ]

    print(f"\n{'─'*60}")
    print(f"  Imagen : {imagen.name}")
    print(f"  Modelo : {modelo.name}")
    print(f"  Salida : {salida.name}")
    print(f"  Comando: {' '.join(cmd)}")
    print(f"{'─'*60}")

    resultado = subprocess.run(cmd, capture_output=False, text=True)

    if resultado.returncode != 0:
        print(f"[ERROR] predict.py terminó con código {resultado.returncode}")
        return False

    print(f"[OK] Inferencia completada → {salida}")
    return True


def main():
    # Validaciones previas
    for directorio, nombre in [(DIR_IMAGENES, "DIR_IMAGENES"),
                                (DIR_MODELOS,  "DIR_MODELOS"),
                                (PREDICT_PY.parent if PREDICT_PY.parent != Path(".") else Path("."), "")]:
        pass  # las validaciones detalladas van abajo

    if not DIR_IMAGENES.is_dir():
        sys.exit(f"[ERROR] El directorio de imágenes no existe: {DIR_IMAGENES}")
    if not DIR_MODELOS.is_dir():
        sys.exit(f"[ERROR] El directorio de modelos no existe: {DIR_MODELOS}")
    if not PREDICT_PY.exists():
        sys.exit(f"[ERROR] No se encontró el script: {PREDICT_PY}")

    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    imagenes = obtener_imagenes(DIR_IMAGENES)
    modelos  = obtener_modelos(DIR_MODELOS)

    if not imagenes or not modelos:
        sys.exit("[ERROR] No hay imágenes o modelos para procesar.")

    total    = len(imagenes) * len(modelos)
    exitos   = 0
    errores  = 0
    omitidos = 0

    print(f"\n{'═'*60}")
    print(f"  Imágenes encontradas : {len(imagenes)}")
    print(f"  Modelos encontrados  : {len(modelos)}")
    print(f"  Inferencias totales  : {total}")
    print(f"  Salida               : {DIR_SALIDA}")
    print(f"{'═'*60}")

    for imagen in imagenes:
        for modelo in modelos:
            salida = nombre_salida(imagen, modelo)

            # Omitir si la salida ya existe
            if salida.exists():
                print(f"\n[OMITIDO] Ya existe: {salida.name}")
                omitidos += 1
                continue

            ok = ejecutar_inferencia(imagen, modelo, salida)
            if ok:
                exitos += 1
            else:
                errores += 1

    # Resumen final
    print(f"\n{'═'*60}")
    print(f"  RESUMEN")
    print(f"  Completadas : {exitos}")
    print(f"  Errores     : {errores}")
    print(f"  Omitidas    : {omitidos}  (salida ya existía)")
    print(f"{'═'*60}\n")

    if errores:
        sys.exit(1)


if __name__ == "__main__":
    main()
