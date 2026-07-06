import os
import csv
from osgeo import gdal
import numpy as np

LISTA_TXT = "/home/liese2/SPRI_AI_project/Mobile-UNet/Mobile-UNet_1/data/valid.txt"          
CARPETA_TIFF = "/home/liese2/SPRI_AI_project/Mobile-UNet/Mobile-UNet_1/data/Images"          
CARPETA_SALIDA_BIN = "/home/liese2/SPRI_AI_project/Dataset/Valid_bin"  

ALTO_ESPERADO = 128
ANCHO_ESPERADO = 128
BANDAS_ESPERADAS = 4

NOMBRES_BANDAS = ["R", "G", "B", "NIR"]

os.makedirs(CARPETA_SALIDA_BIN, exist_ok=True)

def leer_lista(ruta_txt):
    with open(ruta_txt, "r", encoding="utf-8") as f:
        nombres = [linea.strip() for linea in f if linea.strip()]
    return nombres

def main():
    nombres = leer_lista(LISTA_TXT)
    print(f"Total de imágenes listadas: {len(nombres)}")

    filas_indice = []
    convertidas = 0
    faltantes = 0
    tamaño_inesperado = 0

    for i, nombre in enumerate(nombres, 1):
        ruta_tif = os.path.join(CARPETA_TIFF, nombre)

        if not os.path.exists(ruta_tif):
            print(f"  [{i}/{len(nombres)}] ⚠ No encontrado: {nombre}")
            faltantes += 1
            continue

        ds = gdal.Open(ruta_tif)
        arr = ds.ReadAsArray()  # (bandas, alto, ancho)
        ds = None
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        bandas, alto, ancho = arr.shape
        dtype = arr.dtype

        aviso = ""
        if (bandas, alto, ancho) != (BANDAS_ESPERADAS, ALTO_ESPERADO, ANCHO_ESPERADO):
            aviso = "TAMAÑO_INESPERADO"
            tamaño_inesperado += 1
            print(f"  [{i}/{len(nombres)}] ⚠ {nombre}: shape {arr.shape} (esperado {BANDAS_ESPERADAS}x{ALTO_ESPERADO}x{ANCHO_ESPERADO})")

        base = os.path.splitext(nombre)[0]
        carpeta_imagen = os.path.join(CARPETA_SALIDA_BIN, base)
        os.makedirs(carpeta_imagen, exist_ok=True)

        for idx_banda in range(bandas):
            nombre_banda = NOMBRES_BANDAS[idx_banda] if idx_banda < len(NOMBRES_BANDAS) else f"banda{idx_banda+1}"
            ruta_bin_banda = os.path.join(carpeta_imagen, f"{nombre_banda}.bin")
            arr[idx_banda].tofile(ruta_bin_banda)

        # filas_indice.append({
        #     "archivo_original": nombre,
        #     "carpeta_bin": base,
        #     "bandas": bandas,
        #     "alto": alto,
        #     "ancho": ancho,
        #     "dtype": str(dtype),
        #     "aviso": aviso
        # })

        convertidas += 1
        if i % 50 == 0 or i == len(nombres):
            print(f"  [{i}/{len(nombres)}] procesadas...")

    # indice CSV
    # ruta_csv = os.path.join(CARPETA_SALIDA_BIN, "indice_bin.csv")
    # with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
    #     writer = csv.DictWriter(f, fieldnames=[
    #         "archivo_original", "carpeta_bin", "bandas", "alto", "ancho", "dtype", "aviso"
    #     ])
    #     writer.writeheader()
    #     writer.writerows(filas_indice)

    print(f"\nResumen:")
    print(f"  Convertidas:        {convertidas}")
    print(f"  Faltantes:          {faltantes}")
    print(f"  Tamaño inesperado:  {tamaño_inesperado}")
    # print(f"  Índice CSV:         {ruta_csv}")

if __name__ == "__main__":
    main()