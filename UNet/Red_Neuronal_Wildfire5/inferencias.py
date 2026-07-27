"""
run_inferences.py
-----------------
Ejecuta predict.py para cada combinación imagen × modelo.

Convenciones de nombres:
  Entrada : <DIR_IMAGENES>/<nombrebase>_Merged.tif
  Modelos : <DIR_MODELOS>/model_<nombremodelo>.pth
  Salida  : <DIR_SALIDA>/<nombrebase>_MbUN_Out_<nombremodelo>.tif
"""

<<<<<<< HEAD
=======
import csv
import re
>>>>>>> 92a01e3134d8de5c205c83efe0dba0ae3c76c94e
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIGURACIÓN  ← edita estos tres valores
# ─────────────────────────────────────────────
<<<<<<< HEAD
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
=======
DIR_IMAGENES = Path("/home/liese2/SPRI_AI_project/Mobile-UNet/Mobile-UNet_1/data/Images")       # directorio con los .tif de entrada
DIR_MODELOS  = Path("weights")        # directorio con los .pth
DIR_SALIDA   = Path("")         # directorio donde se guardan los resultados
PREDICT_PY   = Path("predict_segmentation.py")            # ruta al script de inferencia
RUTA_VALID = Path("/home/liese2/SPRI_AI_project/Mobile-UNet/Mobile-UNet_1/data/valid.txt")

RUTA_CSV_RESULTADOS = DIR_SALIDA / "prediccion_UN_pixeles.csv"
GUARDAR_IMAGENES = False
# ─────────────────────────────────────────────


def cargar_combinaciones_existentes(ruta_csv: Path) -> set:
    """
    Lee el CSV de resultados (si existe) y regresa un set de (imagen, modelo)
    ya procesados, para poder reanudar sin repetir inferencias.
    """
    existentes = set()
    if not ruta_csv.exists():
        return existentes
 
    with open(ruta_csv, "r", newline="", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            existentes.add((fila["imagen"], fila["modelo"]))
 
    print(f"[INFO] {len(existentes)} combinaciones ya presentes en {ruta_csv.name}, se omitirán.")
    return existentes
 
 
def guardar_fila_csv(ruta_csv: Path, fila: dict):
    """Agrega una fila al CSV de resultados, escribiendo el header si el archivo no existe."""
    existe = ruta_csv.exists()
    with open(ruta_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "imagen", "modelo", "pixeles_incendio", "total_pixeles", "porcentaje_incendio"
        ])
        if not existe:
            writer.writeheader()
        writer.writerow(fila)
 
 
PATRON_CSV_RESULT = re.compile(r"^CSV_RESULT\|(.+)\|(.+)\|(\d+)\|(\d+)\|([\d.]+)\s*$")
 
 
def parsear_resultado(stdout: str):
    """Busca la línea CSV_RESULT en la salida de predict_segmentation.py y la parsea."""
    for linea in stdout.splitlines():
        m = PATRON_CSV_RESULT.match(linea.strip())
        if m:
            imagen, modelo, pixeles, total, porcentaje = m.groups()
            return {
                "imagen": imagen,
                "modelo": modelo,
                "pixeles_incendio": int(pixeles),
                "total_pixeles": int(total),
                "porcentaje_incendio": float(porcentaje)
            }
    return None
 

 
def obtener_imagenes(directorio: Path, ruta_valid_txt: Path) -> list[Path]:
    """
    Lee los nombres de imagen desde ruta_valid_txt (un nombre por línea, p.ej.
    'nombre_imagen.tiff') y devuelve las rutas correspondientes dentro de
    'directorio'. Omite líneas vacías y advierte si algún archivo no existe.
    """
    if not ruta_valid_txt.exists():
        print(f"[ERROR] No se encontró el archivo de nombres: {ruta_valid_txt}")
        return []
 
    with open(ruta_valid_txt, "r", encoding="utf-8") as f:
        nombres = [linea.strip() for linea in f if linea.strip()]
 
    archivos = []
    for nombre in nombres:
        ruta = directorio / nombre
        if ruta.exists():
            archivos.append(ruta)
        else:
            print(f"[ADVERTENCIA] No se encontró la imagen listada en {ruta_valid_txt.name}: {ruta}")
 
    if not archivos:
        print(f"[ADVERTENCIA] No se encontraron imágenes válidas listadas en: {ruta_valid_txt}")
 
    return archivos
 
 
>>>>>>> 92a01e3134d8de5c205c83efe0dba0ae3c76c94e


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


<<<<<<< HEAD
def ejecutar_inferencia(imagen: Path, modelo: Path, salida: Path) -> bool:
=======
def ejecutar_inferencia(imagen: Path, modelo: Path, salida: Path):
>>>>>>> 92a01e3134d8de5c205c83efe0dba0ae3c76c94e
    """Llama a predict.py y devuelve True si terminó sin errores."""
    cmd = [
        sys.executable, str(PREDICT_PY),
        "--image",  str(imagen),
        "--model",  str(modelo),
        "--output", str(salida),
<<<<<<< HEAD
=======
        "--guardar_imagen", "True" if GUARDAR_IMAGENES else "False"
>>>>>>> 92a01e3134d8de5c205c83efe0dba0ae3c76c94e
    ]

    print(f"\n{'─'*60}")
    print(f"  Imagen : {imagen.name}")
    print(f"  Modelo : {modelo.name}")
    print(f"  Salida : {salida.name}")
    print(f"  Comando: {' '.join(cmd)}")
    print(f"{'─'*60}")

<<<<<<< HEAD
    resultado = subprocess.run(cmd, capture_output=False, text=True)

    if resultado.returncode != 0:
        print(f"[ERROR] predict.py terminó con código {resultado.returncode}")
        return False

    print(f"[OK] Inferencia completada → {salida}")
    return True

=======
    resultado = subprocess.run(cmd, capture_output=True, text=True)
    print(resultado.stdout)
    if resultado.stderr:
        print(resultado.stderr, file=sys.stderr)

    if resultado.returncode != 0:
        print(f"[ERROR] predict.py terminó con código {resultado.returncode}")
        return None
    
    fila = parsear_resultado(resultado.stdout)
    if fila is None: 
        print("No hay fila csv result en la salida de predict.py")
        return None

    print(f"[OK] Inferencia completada → {salida}")
    return fila

def ruta_csv_modelo(modelo: Path) -> Path:
    nombremodelo = modelo.stem.replace("model_", "")
    return DIR_SALIDA / f"prediccion_UN_pixeles_{nombremodelo}.csv"
>>>>>>> 92a01e3134d8de5c205c83efe0dba0ae3c76c94e

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

<<<<<<< HEAD
    imagenes = obtener_imagenes(DIR_IMAGENES)
=======
    imagenes = obtener_imagenes(DIR_IMAGENES, RUTA_VALID)
>>>>>>> 92a01e3134d8de5c205c83efe0dba0ae3c76c94e
    modelos  = obtener_modelos(DIR_MODELOS)

    if not imagenes or not modelos:
        sys.exit("[ERROR] No hay imágenes o modelos para procesar.")

<<<<<<< HEAD
=======
    combinaciones_existentes = cargar_combinaciones_existentes(RUTA_CSV_RESULTADOS)

>>>>>>> 92a01e3134d8de5c205c83efe0dba0ae3c76c94e
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

<<<<<<< HEAD
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

=======
    for modelo in modelos:
        ruta_csv = ruta_csv_modelo(modelo)
        combinaciones_existentes = cargar_combinaciones_existentes(ruta_csv)
        
        print(f"\n{'▓'*60}")
        print(f"  MODELO: {modelo.name}  →  CSV: {ruta_csv.name}")
        print(f"{'▓'*60}")
 
        for imagen in imagenes:
            salida = nombre_salida(imagen, modelo)
 
            # Omitir si ya está en el CSV de resultados de este modelo
            if (imagen.name, modelo.name) in combinaciones_existentes:
                print(f"\n[OMITIDO] Ya procesado: {imagen.name} x {modelo.name}")
                omitidos += 1
                continue
 
            fila = ejecutar_inferencia(imagen, modelo, salida)
            if fila is not None:
                fila["imagen"] = imagen.name
                fila["modelo"] = modelo.name
                guardar_fila_csv(ruta_csv, fila)
                exitos += 1
            else:
                errores += 1
 
>>>>>>> 92a01e3134d8de5c205c83efe0dba0ae3c76c94e
    # Resumen final
    print(f"\n{'═'*60}")
    print(f"  RESUMEN")
    print(f"  Completadas : {exitos}")
    print(f"  Errores     : {errores}")
    print(f"  Omitidas    : {omitidos}  (salida ya existía)")
    print(f"{'═'*60}\n")
<<<<<<< HEAD

    if errores:
        sys.exit(1)


if __name__ == "__main__":
    main()
=======
 
    if errores:
        sys.exit(1)
 
 
if __name__ == "__main__":
    main()
 
>>>>>>> 92a01e3134d8de5c205c83efe0dba0ae3c76c94e
