import os
import shutil
import random
from pathlib import Path

def procesar_dataset(
    dir_entrada,
    dir_salida_base,
    dir_txts,
    porcentaje_entrenamiento=0.8
):
    """
    Procesa el dataset según las especificaciones dadas.
    
    Args:
        dir_entrada: Directorio que contiene las carpetas True y Mask
        dir_salida_base: Directorio base de salida donde se crearán las carpetas
        dir_txts: Directorio donde se guardarán los archivos txt
        porcentaje_entrenamiento: Porcentaje de archivos para entrenamiento (default: 0.8)
    """
    
    # Definir rutas de entrada
    dir_true = Path(dir_entrada) / "True"
    dir_mask = Path(dir_entrada) / "Mask"
    
    # Definir rutas de salida
    dir_imagenes = Path(dir_salida_base) / "Wildfire7" / "Images"
    dir_segmentacion = Path(dir_salida_base) / "Wildfire7" / "SegmentationClass"
    dir_txts_completo = Path(dir_txts) / "Wildfire7" / "ImageSets" / "Segmentation"
    
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
    
    # Procesar archivos de entrenamiento
    nombres_entrenamiento = []
    for archivo_true, archivo_mask in entrenamiento:
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
        
        # Guardar nombre sin extensión
        nombre_sin_ext = Path(archivo_true).stem
        nombres_entrenamiento.append(nombre_sin_ext)
    
    # Procesar archivos de validación
    nombres_validacion = []
    for archivo_true, archivo_mask in validacion:
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
        
        # Guardar nombre sin extensión
        nombre_sin_ext = Path(archivo_true).stem
        nombres_validacion.append(nombre_sin_ext)
    
    # Escribir archivos txt
    with open(dir_txts_completo / "train.txt", "w") as f:
        for nombre in sorted(nombres_entrenamiento):
            f.write(f"{nombre}\n")
    
    with open(dir_txts_completo / "valid.txt", "w") as f:
        for nombre in sorted(nombres_validacion):
            f.write(f"{nombre}\n")
    
    # Resumen final
    print("\n" + "="*50)
    print("PROCESO COMPLETADO EXITOSAMENTE")
    print("="*50)
    print(f"✓ Archivos copiados a:")
    print(f"  - Imágenes: {dir_imagenes}")
    print(f"  - Segmentación: {dir_segmentacion}")
    print(f"\n✓ Archivos de división creados en:")
    print(f"  - Entrenamiento: {dir_txts_completo / 'train.txt'} ({len(nombres_entrenamiento)} archivos)")
    print(f"  - Validación: {dir_txts_completo / 'valid.txt'} ({len(nombres_validacion)} archivos)")
    print("="*50)

DIRECTORIO_ENTRADA = "/home/liese2/SPRI_AI_project/Dataset/Crops6" 
DIRECTORIO_SALIDA_BASE = "/home/liese2/SPRI_AI_project" 
DIRECTORIO_TXTS = "/home/liese2/SPRI_AI_project" 

if __name__ == "__main__":
    # Configurar semilla para reproducibilidad (opcional)
    random.seed(42)  # Puedes eliminar esta línea si quieres aleatoriedad diferente cada vez
    
    try:
        procesar_dataset(
            dir_entrada=DIRECTORIO_ENTRADA,
            dir_salida_base=DIRECTORIO_SALIDA_BASE,
            dir_txts=DIRECTORIO_TXTS,
            porcentaje_entrenamiento=0.8
        )
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")
        print("\nAsegúrate de que:")
        print("1. Las rutas especificadas sean correctas")
        print("2. La carpeta de entrada contenga las subcarpetas 'True' y 'Mask'")
        print("3. Ambos directorios tengan los mismos archivos (mismos nombres)")