import os
import csv
import shutil
import random
from pathlib import Path

<<<<<<< HEAD
DIRECTORIO_ENTRADA = "/home/liese2/SPRI_AI_project/Dataset/Datasets_generados/Dataset_csv" 
DIRECTORIO_SALIDA_BASE = "/home/liese2/SPRI_AI_project/Mobile-UNet" 
DIRECTORIO_TXTS = "/home/liese2/SPRI_AI_project/Mobile-UNet" 
=======
DIRECTORIO_ENTRADA = "/home/felix/SPRI_AI_Project/Dataset/Dataset_3p5" 
DIRECTORIO_SALIDA_BASE = "/home/felix/SPRI_AI_Project/Dataset/Dataset_3p5_separado" 
DIRECTORIO_TXTS = "/home/felix/SPRI_AI_Project/Dataset/Dataset_3p5_separado" 
>>>>>>> release/1.0.2

# Ruta al CSV generado por el script de creación del dataset (columna 'bloque' y 'pixeles_incendio')
RUTA_CSV_PIXELES = "/home/liese2/SPRI_AI_project/Mobile-UNet/Mobile-UNet_1/data/pixeles_incendio.csv"


def cargar_conteo_pixeles(ruta_csv):
    """
    Lee el CSV de conteo de píxeles de incendio y regresa un diccionario
    { nombre_bloque_sin_extension: pixeles_incendio }
    """
    conteo = {}
    if not os.path.exists(ruta_csv):
        print(f"⚠ No se encontró el CSV de píxeles en: {ruta_csv}")
        print(f"  Los archivos train_incendios.txt/valid_incendios.txt tendrán 0 para todos los bloques.")
        return conteo

    with open(ruta_csv, "r", newline="", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            conteo[fila["bloque"]] = int(fila["pixeles_incendio"])

    print(f"✓ Conteo de píxeles cargado: {len(conteo)} bloques registrados en el CSV")
    return conteo


def procesar_dataset(
    dir_entrada,
    dir_salida_base,
    dir_txts,
    porcentaje_entrenamiento=0.8,
    ruta_csv_pixeles=RUTA_CSV_PIXELES
):
    """
    Procesa el dataset según las especificaciones dadas.
    
    Args:
        dir_entrada: Directorio que contiene las carpetas True y Mask
        dir_salida_base: Directorio base de salida donde se crearán las carpetas
        dir_txts: Directorio donde se guardarán los archivos txt
        porcentaje_entrenamiento: Porcentaje de archivos para entrenamiento (default: 0.8)
        ruta_csv_pixeles: Ruta al CSV con el conteo de píxeles de incendio por bloque
    """
    
    # Cargar el conteo de píxeles de incendio por bloque
    conteo_pixeles = cargar_conteo_pixeles(ruta_csv_pixeles)
    
    # Definir rutas de entrada
    dir_true = Path(dir_entrada) / "True"
    dir_mask = Path(dir_entrada) / "Mask"
    
    # Definir rutas de salida
    dir_imagenes = Path(dir_salida_base) / "Mobile-UNet_5" / "data" / "Images"
    dir_segmentacion = Path(dir_salida_base) / "Mobile-UNet_5" / "data" / "SegmentationClass"
    dir_txts_completo = Path(dir_txts) / "Mobile-UNet_5" / "data" 
    
    # Crear directorios de salida si no existen
    dir_imagenes.mkdir(parents=True, exist_ok=True)
    dir_segmentacion.mkdir(parents=True, exist_ok=True)
    dir_txts_completo.mkdir(parents=True, exist_ok=True)
    
    # Verificar que existen las carpetas True y Mask
    if not dir_true.exists():
        raise FileNotFoundError(f"No se encuentra la carpeta True en: {dir_true}")
    if not dir_mask.exists():
        raise FileNotFoundError(f"No se encuentra la carpeta Mask en: {dir_mask}")
    
    # Obtener listas de archivos
    archivos_true = sorted([f for f in os.listdir(dir_true) if not f.startswith('.')])
    archivos_mask = sorted([f for f in os.listdir(dir_mask) if not f.startswith('.')])
    
    # Verificar que tienen la misma cantidad de archivos
    if len(archivos_true) != len(archivos_mask):
        raise ValueError(
            f"La cantidad de archivos no coincide: "
            f"True tiene {len(archivos_true)} archivos, "
            f"Mask tiene {len(archivos_mask)} archivos"
        )
    
    # Verificar que los nombres coinciden (sin extensión)
    nombres_true = {Path(f).stem for f in archivos_true}
    nombres_mask = {Path(f).stem for f in archivos_mask}
    
    # if nombres_true != nombres_mask:
    #     diferencia_true = nombres_true - nombres_mask
    #     diferencia_mask = nombres_mask - nombres_true
        
    #     error_msg = "Los archivos en True y Mask no coinciden:\n"
    #     if diferencia_true:
    #         error_msg += f"Archivos solo en True: {sorted(diferencia_true)}\n"
    #     if diferencia_mask:
    #         error_msg += f"Archivos solo en Mask: {sorted(diferencia_mask)}"
        
    #     raise ValueError(error_msg)
    
    # print(f"✓ Se encontraron {len(archivos_true)} archivos coincidentes en True y Mask")
    
    # Mezclar aleatoriamente los archivos (manteniendo el emparejamiento)
    pares_archivos = list(zip(archivos_true, archivos_mask))
    random.shuffle(pares_archivos)
    
    # Dividir en conjuntos de entrenamiento y validación
    punto_corte = int(len(pares_archivos) * porcentaje_entrenamiento)
    entrenamiento = pares_archivos[:punto_corte]
    validacion = pares_archivos[punto_corte:]
    
    print(f"✓ {len(entrenamiento)} archivos para entrenamiento ({porcentaje_entrenamiento*100:.0f}%)")
    print(f"✓ {len(validacion)} archivos para validación ({(1-porcentaje_entrenamiento)*100:.0f}%)")
    
    bloques_sin_conteo = []

    def procesar_grupo(grupo):
        """Copia archivos y arma la lista (nombre_sin_ext, pixeles_incendio) para un grupo"""
        registros = []
        for archivo_true, archivo_mask in grupo:
            # Copiar archivo True
            shutil.copy2(
                dir_true / archivo_true,
                dir_imagenes / archivo_true
            )
            
            # Copiar archivo Mask
            shutil.copy2(
                dir_mask / archivo_mask,
                dir_segmentacion / archivo_mask
            )
            
            # Nombre sin extensión (corregido: antes duplicaba ".tiff")
            nombre_sin_ext = Path(archivo_true).stem
            
            pixeles = conteo_pixeles.get(nombre_sin_ext)
            if pixeles is None:
                bloques_sin_conteo.append(nombre_sin_ext)
                pixeles = 0
            
            registros.append((nombre_sin_ext, pixeles))
        
        return registros
    
    # Procesar archivos de entrenamiento y validación
    registros_entrenamiento = procesar_grupo(entrenamiento)
    registros_validacion = procesar_grupo(validacion)
    
    # Ordenar por nombre para que train.txt / train_incendios.txt queden alineados línea a línea
    registros_entrenamiento.sort(key=lambda t: t[0])
    registros_validacion.sort(key=lambda t: t[0])
    
    if bloques_sin_conteo:
        print(f"\n⚠ {len(bloques_sin_conteo)} bloques no se encontraron en el CSV de píxeles "
              f"(se les asignó 0). Ejemplos: {bloques_sin_conteo[:5]}")
    
    # Escribir archivos txt de nombres
    with open(dir_txts_completo / "train.txt", "w") as f:
        for nombre, _ in registros_entrenamiento:
            f.write(f"{nombre}\n")
    
    with open(dir_txts_completo / "valid.txt", "w") as f:
        for nombre, _ in registros_validacion:
            f.write(f"{nombre}\n")
    
    # Escribir archivos txt del conteo de píxeles de incendio: "nombre_bloque pixeles_incendio"
    with open(dir_txts_completo / "train_incendios.txt", "w") as f:
        for nombre, pixeles in registros_entrenamiento:
            f.write(f"{nombre} {pixeles}\n")
    
    with open(dir_txts_completo / "valid_incendios.txt", "w") as f:
        for nombre, pixeles in registros_validacion:
            f.write(f"{nombre} {pixeles}\n")
    
    # Resumen final
    print("\n" + "="*50)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*50)
    print(f"✓ Archivos copiados a:")
    print(f"  - Imágenes: {dir_imagenes}")
    print(f"  - Segmentación: {dir_segmentacion}")
    print(f"\n✓ Archivos de división creados en:")
    print(f"  - Entrenamiento: {dir_txts_completo / 'train.txt'} ({len(registros_entrenamiento)} archivos)")
    print(f"  - Validación: {dir_txts_completo / 'valid.txt'} ({len(registros_validacion)} archivos)")
    print(f"\n✓ Conteo de píxeles de incendio creado en:")
    print(f"  - Entrenamiento: {dir_txts_completo / 'train_incendios.txt'}")
    print(f"  - Validación: {dir_txts_completo / 'valid_incendios.txt'}")
    print("="*50)

if __name__ == "__main__":
    # Configurar semilla para reproducibilidad (opcional)
    random.seed(42)  # Puedes eliminar esta línea si quieres aleatoriedad diferente cada vez
    
    try:
        procesar_dataset(
            dir_entrada=DIRECTORIO_ENTRADA,
            dir_salida_base=DIRECTORIO_SALIDA_BASE,
            dir_txts=DIRECTORIO_TXTS,
            porcentaje_entrenamiento=0.8,
            ruta_csv_pixeles=RUTA_CSV_PIXELES
        )
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")
        print("\nAsegúrate de que:")
        print("1. Las rutas especificadas sean correctas")
        print("2. La carpeta de entrada contenga las subcarpetas 'True' y 'Mask'")
        print("3. Ambos directorios tengan los mismos archivos (mismos nombres)")
        print("4. El CSV de píxeles de incendio exista y tenga las columnas 'bloque' y 'pixeles_incendio'")