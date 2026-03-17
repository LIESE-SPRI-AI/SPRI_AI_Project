import os
import zipfile
import subprocess
import glob
import re
import numpy as np
import cv2
import shutil
import rasterio
from osgeo import gdal, osr
import argparse

# --- CONFIGURACIÓN ---
input_directory = "/home/liese2/SPRI_AI_project/Dataset/Raw" 
output_directory = "/home/liese2/SPRI_AI_project/Dataset/Entrada_firmas"
temp_directory = "/home/liese2/SPRI_AI_project/Dataset/Raw" 


# Configuración del escalado (gdal_translate)
target_min = 0
target_max = (2 ** 16) - 1
new_data_type = "UInt16"  # 'Byte' para 0-255

# Configuración del merge final
final_format = "GTiff"

# Crear directorios necesarios
os.makedirs(output_directory, exist_ok=True)
os.makedirs(temp_directory, exist_ok=True)

def find_file_in_zip(zip_ref, pattern):
    for file_name in zip_ref.namelist():
        if pattern in file_name and file_name.endswith(('.tif', '.tiff')):
            return file_name
    return None

def get_band_min_max(file_path):
    cmd = ["gdalinfo", "-mm", file_path]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        
        # Buscamos la línea "Computed Min/Max=val_min,val_max"
        match = re.search(r"Computed Min/Max=([0-9\.\-]+),([0-9\.\-]+)", output)
        
        if match:
            min_val = match.group(1)
            max_val = match.group(2)
            return min_val, max_val
        else:
            print(f"Advertencia: No se encontró 'Computed Min/Max' en {os.path.basename(file_path)}")
            return None, None
            
    except Exception as e:
        print(f"Error ejecutando gdalinfo: {e}")
        return None, None

def translate_band(input_path, output_path, min_val, max_val):
    cmd = [
        "gdal_translate",
        input_path, output_path,
        "-scale", str(min_val), str(max_val), str(target_min), str(target_max),
        "-ot", new_data_type
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    zip_files = glob.glob(os.path.join(input_directory, "*.zip"))
    
    if not zip_files:
        print("No se encontraron archivos .zip")
        return

    for zip_path in zip_files:
        zip_name = os.path.basename(zip_path)
        base_name = os.path.splitext(zip_name)[0]
        
        print(f"--- Procesando: {zip_name} ---")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 1. Identificación de archivos
                # Nota: Seguimos buscando Burned_Area para validar, aunque no se procese
                file_burned = find_file_in_zip(zip_ref, "Burned_Area_Detection")
                
                bands_map = {
                    'Red': find_file_in_zip(zip_ref, "B04"),
                    'Green': find_file_in_zip(zip_ref, "B03"),
                    'Blue': find_file_in_zip(zip_ref, "B02"),
                    'NIR': find_file_in_zip(zip_ref, "B08")
                }

                # Validación de existencia
                if not file_burned or not all(bands_map.values()):
                    print(f"SALTADO: Faltan archivos en {zip_name}")
                    continue

                abs_zip_path = os.path.abspath(zip_path)
                scaled_files = [] # Lista para guardar las rutas de los archivos temporales
                
                # Procesamos en orden estricto: R, G, B, NIR
                ordered_keys = ['Red', 'Green', 'Blue', 'NIR']
                
                print(f"  > Analizando y Escalando bandas...")
                
                for key in ordered_keys:
                    internal_file = bands_map[key]
                    vsi_path = f"/vsizip/{abs_zip_path}/{internal_file}"
                    
                    # A. OBTENER MIN/MAX
                    src_min, src_max = get_band_min_max(vsi_path)
                    
                    if src_min is None:
                        print(f"    Error obteniendo estadísticas para {key}, usando defecto 0-10000")
                        src_min, src_max = 0, 10000 # Fallback por seguridad
                    
                    # B. GDAL TRANSLATE (ESCALADO)
                    # Creamos un archivo temporal físico
                    temp_output = os.path.join(temp_directory, f"{base_name}_{key}_scaled.tif")
                    translate_band(vsi_path, temp_output, src_min, src_max)
                    
                    scaled_files.append(temp_output)

                # C. GDAL MERGE
                output_file = os.path.join(output_directory, f"{base_name}_Merged.tif")
                
                merge_cmd = [
                    "gdal_merge.py",
                    "-separate",
                    "-ot", new_data_type,
                    "-of", final_format,
                    "-co", "PHOTOMETRIC=RGB"
                ]
                
                # Añadimos los archivos escalados (R, G, B, N)
                merge_cmd.extend(scaled_files)
                
                # Añadimos salida
                merge_cmd.append("-o")
                merge_cmd.append(output_file)
                
                print(f"  > Fusionando en: {os.path.basename(output_file)}")
                result = subprocess.run(merge_cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    print("    Éxito.")
                else:
                    print("    Error en merge:")
                    print(result.stderr)

                # D. LIMPIEZA (Borrar archivos temporales para no llenar el disco)
                for f in scaled_files:
                    if os.path.exists(f):
                        os.remove(f)

        except Exception as e:
            print(f"Error crítico en {zip_name}: {e}")

if __name__ == "__main__":
    main()