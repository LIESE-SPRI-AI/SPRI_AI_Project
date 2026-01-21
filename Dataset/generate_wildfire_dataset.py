import os
import zipfile
import subprocess
import glob
import re
import numpy as np
import shutil
import rasterio
from osgeo import gdal
import time
import gc

# Configuración de rutas
input_directory = "/home/liese2/SPRI_AI_project/Dataset/Raw" 
output_directory = input_directory
temp_directory = input_directory

output_dir_img = "/home/liese2/SPRI_AI_project/Dataset/Crops6/True"
output_dir_mask = "/home/liese2/SPRI_AI_project/Dataset/Crops6/Mask"

# Configuración
target_min = 0
target_max = (2 ** 16) - 1
new_data_type = "UInt16"
final_format = "GTiff"

UMBRAL_PORCENTAJE_ROJO = 0.035
TAMANO_BLOQUE = 128
OVERLAP = 64

os.makedirs(output_directory, exist_ok=True)
os.makedirs(temp_directory, exist_ok=True)
os.makedirs(output_dir_img, exist_ok=True)
os.makedirs(output_dir_mask, exist_ok=True)

def limpiar_archivos_forzado(ruta, max_intentos=3, espera=1):
    """
    Intenta eliminar archivo/directorio con reintentos
    """
    for intento in range(max_intentos):
        try:
            if os.path.isfile(ruta):
                os.remove(ruta)
                return True
            elif os.path.isdir(ruta):
                shutil.rmtree(ruta, ignore_errors=True)
                # Verificar si realmente se eliminó
                if not os.path.exists(ruta):
                    return True
                # Si todavía existe, forzar eliminación
                if os.name == 'nt':  # Windows
                    os.system(f'rmdir /S /Q "{ruta}"')
                else:  # Linux/Mac
                    os.system(f'rm -rf "{ruta}"')
                return not os.path.exists(ruta)
        except PermissionError:
            print(f"    ⚠ Intento {intento+1}/{max_intentos}: Archivo/carpeta en uso, esperando...")
            time.sleep(espera)
            gc.collect()  # Forzar recolección de basura
        except Exception as e:
            print(f"    ⚠ Error limpiando {ruta}: {e}")
            time.sleep(espera)
    
    return False

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
        
        match = re.search(r"Computed Min/Max=([0-9\.\-]+),([0-9\.\-]+)", output)
        
        if match:
            min_val = match.group(1)
            max_val = match.group(2)
            return min_val, max_val
        else:
            print(f"No se encontró Computed Min/Max en {os.path.basename(file_path)}")
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

def crear_mascara_roja_pura(imagen_burned_area):
    """Crea máscara con píxeles rojos puros en UInt16"""
    ds = None
    try:
        with rasterio.open(imagen_burned_area) as src:
            if src.count >= 3:
                banda_r = src.read(1)
                banda_g = src.read(2)
                banda_b = src.read(3)
                
                def normalizar_banda(banda):
                    if banda.dtype != np.uint8:
                        if banda.dtype == np.uint16:
                            return (banda / 256).astype(np.uint8)
                        else:
                            banda_norm = (banda - banda.min()) / (banda.max() - banda.min()) * 255
                            return banda_norm.astype(np.uint8)
                    return banda
                
                banda_r_norm = normalizar_banda(banda_r)
                banda_g_norm = normalizar_banda(banda_g)
                banda_b_norm = normalizar_banda(banda_b)
                
                umbral_rojo = 254
                
                mascara_roja = (
                    (banda_r_norm >= umbral_rojo) &
                    (banda_r_norm > banda_g_norm + 40) &
                    (banda_r_norm > banda_b_norm + 40) &
                    (banda_r_norm > 50)
                )
                
                mascara_verde_bajo = (banda_g_norm < 100)
                mascara_azul_bajo = (banda_b_norm < 100)
                mascara_final = mascara_roja & mascara_verde_bajo & mascara_azul_bajo
                
                rojo_pixels = np.sum(mascara_final)
                total_pixels = mascara_final.size
                porcentaje_rojo = (rojo_pixels / total_pixels) * 100
                
                print(f"    Píxeles rojos detectados: {rojo_pixels} ({porcentaje_rojo:.2f}%)")
                
                if rojo_pixels == 0:
                    print("    ⚠ No se detectaron píxeles rojos puros")
                    return None
                
                mascara_r = np.where(mascara_final, np.uint16(65535), np.uint16(0))
                mascara_g = np.where(mascara_final, np.uint16(0), np.uint16(0))
                mascara_b = np.where(mascara_final, np.uint16(0), np.uint16(0))
                
                return {
                    'r': mascara_r,
                    'g': mascara_g,
                    'b': mascara_b,
                    'profile': src.profile,
                    'transform': src.transform,
                    'crs': src.crs
                }
            else:
                print(f"    Imagen tiene {src.count} bandas, se esperaban 3")
                return None
                
    except Exception as e:
        print(f"    Error creando máscara: {str(e)}")
        return None
    finally:
        # Asegurar cierre de dataset
        if ds is not None:
            ds = None
        gc.collect()

def dividir_y_filtrar_imagen(ruta_imagen_true, ruta_imagen_mask, tamaño_bloque=TAMANO_BLOQUE, 
                            overlap=OVERLAP, zip_name=None):
    """Procesa imagen y genera crops con gestión robusta de recursos"""
    
    src_true = None
    dst_true = None
    dst_mask = None
    
    try:
        print(f"\n    Procesando par de imágenes:")
        print(f"      TRUE: {os.path.basename(ruta_imagen_true)}")
        print(f"      MASK: {os.path.basename(ruta_imagen_mask)}")
        
        print(f"    Creando máscara de píxeles rojos puros...")
        mascara_data = crear_mascara_roja_pura(ruta_imagen_mask)
        
        if mascara_data is None:
            print("    ❌ No se pudo crear máscara válida")
            return 0
        
        print(f"    Leyendo imagen de 4 bandas...")
        with rasterio.open(ruta_imagen_true) as src_true:
            if src_true.count < 4:
                print(f"    ❌ Imagen tiene {src_true.count} bandas, se esperaban 4")
                return 0
            
            # Leer todas las bandas de una vez
            bandas_true = [src_true.read(i) for i in range(1, 5)]
            perfil_base = src_true.profile.copy()
            
        # Cerrar explícitamente src_true
        src_true = None
        gc.collect()
        
        alto, ancho = bandas_true[0].shape
        print(f"    Tamaño: {ancho}x{alto}")
        
        contador = 0
        descartadas = 0
        
        alto_comun = min(alto, mascara_data['r'].shape[0])
        ancho_comun = min(ancho, mascara_data['r'].shape[1])
        
        paso = tamaño_bloque - overlap
        total_pixels_por_bloque = tamaño_bloque * tamaño_bloque
        minimo_pixels_rojos = int(total_pixels_por_bloque * UMBRAL_PORCENTAJE_ROJO)
        
        print(f"    Configuración: {tamaño_bloque}x{tamaño_bloque}, overlap {overlap}px")
        
        max_y = alto_comun - tamaño_bloque
        max_x = ancho_comun - tamaño_bloque
        
        if max_y < 0 or max_x < 0:
            print(f"    ❌ Imagen muy pequeña")
            return 0
        
        y_coords = list(range(0, max_y + 1, paso))
        x_coords = list(range(0, max_x + 1, paso))
        
        if y_coords[-1] != max_y:
            y_coords.append(max_y)
        if x_coords[-1] != max_x:
            x_coords.append(max_x)
        
        print(f"    Grid: {len(y_coords)}x{len(x_coords)} = {len(y_coords)*len(x_coords)} ventanas")
        
        for y in y_coords:
            for x in x_coords:
                bloque_mascara_r = mascara_data['r'][y:y+tamaño_bloque, x:x+tamaño_bloque]
                
                pixeles_rojos = np.sum(bloque_mascara_r > 0)
                porcentaje_rojo = pixeles_rojos / total_pixels_por_bloque
                
                if pixeles_rojos > 0 and porcentaje_rojo >= UMBRAL_PORCENTAJE_ROJO:
                    # Extraer bloques
                    bloque_true = [banda[y:y+tamaño_bloque, x:x+tamaño_bloque] for banda in bandas_true]
                    bloque_mascara_g = mascara_data['g'][y:y+tamaño_bloque, x:x+tamaño_bloque]
                    bloque_mascara_b = mascara_data['b'][y:y+tamaño_bloque, x:x+tamaño_bloque]
                    
                    nombre_base = os.path.splitext(zip_name)[0] if zip_name else "bloque"
                    
                    # GUARDAR TRUE
                    nombre_true = f"{nombre_base}_bloque_{contador}_x{x}_y{y}.tiff"
                    ruta_true = os.path.join(output_dir_img, nombre_true)
                    
                    perfil_true = perfil_base.copy()
                    perfil_true.update({
                        'height': tamaño_bloque,
                        'width': tamaño_bloque,
                        'count': 4,
                        'dtype': 'uint16',
                        'transform': rasterio.Affine(
                            perfil_base['transform'].a,
                            perfil_base['transform'].b,
                            perfil_base['transform'].c + x * perfil_base['transform'].a,
                            perfil_base['transform'].d,
                            perfil_base['transform'].e,
                            perfil_base['transform'].f + y * perfil_base['transform'].e
                        )
                    })
                    
                    try:
                        with rasterio.open(ruta_true, 'w', **perfil_true) as dst_true:
                            for i, bloque_banda in enumerate(bloque_true):
                                dst_true.write(bloque_banda, i+1)
                        dst_true = None
                    except Exception as e:
                        print(f"      ⚠ Error guardando TRUE: {e}")
                        continue
                    
                    # GUARDAR MASK
                    nombre_mask = f"{nombre_base}_bloque_{contador}_x{x}_y{y}.tiff"
                    ruta_mask = os.path.join(output_dir_mask, nombre_mask)
                    
                    perfil_mask = {
                        'driver': 'GTiff',
                        'height': tamaño_bloque,
                        'width': tamaño_bloque,
                        'count': 3,
                        'dtype': 'uint16',
                        'crs': mascara_data.get('crs'),
                        'transform': rasterio.Affine(
                            mascara_data['transform'].a,
                            mascara_data['transform'].b,
                            mascara_data['transform'].c + x * mascara_data['transform'].a,
                            mascara_data['transform'].d,
                            mascara_data['transform'].e,
                            mascara_data['transform'].f + y * mascara_data['transform'].e
                        )
                    }
                    
                    try:
                        with rasterio.open(ruta_mask, 'w', **perfil_mask) as dst_mask:
                            dst_mask.write(bloque_mascara_r, 1)
                            dst_mask.write(bloque_mascara_g, 2)
                            dst_mask.write(bloque_mascara_b, 3)
                        dst_mask = None
                    except Exception as e:
                        print(f"      ⚠ Error guardando MASK: {e}")
                        # Eliminar TRUE si MASK falló
                        if os.path.exists(ruta_true):
                            os.remove(ruta_true)
                        continue
                    
                    if contador < 10:
                        print(f"      ✓ Bloque {contador} en ({x},{y}): {pixeles_rojos} rojos ({porcentaje_rojo*100:.1f}%)")
                    elif contador == 10:
                        print(f"      ... procesando más bloques ...")
                    
                    contador += 1
                else:
                    descartadas += 1
        
        print(f"\n    Resumen: {contador} bloques guardados, {descartadas} descartados")
        return contador
        
    except Exception as e:
        print(f"    Error procesando imágenes: {str(e)}")
        import traceback
        print(f"    Detalles: {traceback.format_exc()}")
        return 0
    finally:
        # LIMPIEZA GARANTIZADA
        src_true = None
        dst_true = None
        dst_mask = None
        gc.collect()

def procesar_zip(zip_path, estadisticas_globales):
    """Procesa ZIP con limpieza robusta garantizada"""
    
    zip_name = os.path.basename(zip_path)
    base_name = os.path.splitext(zip_name)[0]
    
    print(f"\n{'='*60}")
    print(f"PROCESANDO: {zip_name}")
    print('='*60)
    
    extract_dir = os.path.join(temp_directory, base_name)
    imagen_combinada = os.path.join(output_directory, f"{base_name}_merged.tif")
    scaled_files = []
    
    # USAR TRY-FINALLY PARA GARANTIZAR LIMPIEZA
    try:
        stats_zip = {
            'nombre': zip_name,
            'archivos_encontrados': 0,
            'bandas_encontradas': 0,
            'imagen_combinada_creada': False,
            'bloques_generados': 0,
            'error': None,
            'valido': False
        }
        
        print(f"\n1. Extrayendo archivos...")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print(f"\n2. Buscando archivos...")
        
        archivo_burned = None
        archivos_tiff = [f for f in os.listdir(extract_dir) if f.endswith(('.tif', '.tiff'))]
        stats_zip['archivos_encontrados'] = len(archivos_tiff)
        
        for archivo in archivos_tiff:
            if "Burned_Area_Detection" in archivo:
                archivo_burned = os.path.join(extract_dir, archivo)
                print(f"   ✓ Burned Area: {archivo}")
                break
        
        if not archivo_burned:
            print(f"   ❌ No se encontró Burned_Area_Detection")
            stats_zip['error'] = "No Burned_Area_Detection"
            estadisticas_globales['zips_invalidos'].append(stats_zip)
            return 0
        
        bandas = {}
        patrones = {'B02': 'B02', 'B03': 'B03', 'B04': 'B04', 'B08': 'B08'}
        
        for key, patron in patrones.items():
            for archivo in archivos_tiff:
                if patron in archivo and 'Raw' in archivo:
                    bandas[key] = os.path.join(extract_dir, archivo)
                    print(f"   ✓ Banda {key}: {archivo}")
                    break
        
        stats_zip['bandas_encontradas'] = len(bandas)
        if len(bandas) != 4:
            print(f"   ❌ Faltan bandas: {list(bandas.keys())}")
            stats_zip['error'] = f"Faltan bandas: {list(bandas.keys())}"
            estadisticas_globales['zips_invalidos'].append(stats_zip)
            return 0
        
        print(f"\n3. Fusionando bandas...")
        
        orden_bandas = ['B04', 'B03', 'B02', 'B08']
        
        for banda_key in orden_bandas:
            ruta_banda = bandas[banda_key]
            src_min, src_max = get_band_min_max(ruta_banda)
            if src_min is None:
                src_min, src_max = 0, 10000
            
            temp_output = os.path.join(temp_directory, f"{base_name}_{banda_key}_scaled.tif")
            translate_band(ruta_banda, temp_output, src_min, src_max)
            scaled_files.append(temp_output)
        
        merge_cmd = [
            "gdal_merge.py", "-separate", "-ot", "UInt16",
            "-of", "GTiff", "-o", imagen_combinada
        ]
        merge_cmd.extend(scaled_files)
        
        result = subprocess.run(merge_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"   ❌ Error fusionando: {result.stderr}")
            stats_zip['error'] = "Error en gdal_merge"
            estadisticas_globales['zips_invalidos'].append(stats_zip)
            return 0
        
        print(f"   ✓ Imagen combinada creada")
        stats_zip['imagen_combinada_creada'] = True
        
        print(f"\n4. Generando crops...")
        bloques_generados = dividir_y_filtrar_imagen(
            ruta_imagen_true=imagen_combinada,
            ruta_imagen_mask=archivo_burned,
            tamaño_bloque=TAMANO_BLOQUE,
            overlap=OVERLAP,
            zip_name=zip_name
        )
        
        stats_zip['bloques_generados'] = bloques_generados
        
        print(f"\n{'='*60}")
        print(f"COMPLETADO: {zip_name} - {bloques_generados} bloques")
        print('='*60)
        
        if bloques_generados > 0:
            stats_zip['valido'] = True
            estadisticas_globales['zips_validos'].append(stats_zip)
        else:
            stats_zip['valido'] = False
            stats_zip['error'] = "No generó bloques válidos"
            estadisticas_globales['zips_invalidos'].append(stats_zip)
        
        return bloques_generados
        
    except Exception as e:
        print(f"Error procesando {zip_name}: {str(e)}")
        stats_zip['error'] = str(e)
        estadisticas_globales['zips_invalidos'].append(stats_zip)
        return 0
    
    finally:
        # LIMPIEZA GARANTIZADA - SIEMPRE SE EJECUTA
        print(f"\n5. Limpiando archivos temporales...")
        
        # Limpiar archivos escalados
        for scaled_file in scaled_files:
            if os.path.exists(scaled_file):
                if limpiar_archivos_forzado(scaled_file):
                    print(f"   ✓ Eliminado: {os.path.basename(scaled_file)}")
                else:
                    print(f"   ⚠ No se pudo eliminar: {os.path.basename(scaled_file)}")
        
        # Limpiar imagen combinada
        if os.path.exists(imagen_combinada):
            if limpiar_archivos_forzado(imagen_combinada):
                print(f"   ✓ Eliminada imagen combinada")
            else:
                print(f"   ⚠ No se pudo eliminar imagen combinada")
        
        # Limpiar directorio extraído
        if os.path.exists(extract_dir):
            print(f"   Eliminando directorio: {extract_dir}")
            if limpiar_archivos_forzado(extract_dir):
                print(f"   ✓ Directorio eliminado")
            else:
                print(f"   ⚠ Directorio no se pudo eliminar completamente")
                print(f"      Puede requerir eliminación manual")
        
        # Forzar liberación de recursos
        gc.collect()
        time.sleep(0.5)  

def mostrar_estadisticas_detalladas(estadisticas_globales, total_bloques):
    """Muestra estadísticas detalladas"""
    print(f"\n{'='*80}")
    print("ESTADÍSTICAS DETALLADAS")
    print('='*80)
    
    total_zips = estadisticas_globales['total_zips']
    zips_validos = len(estadisticas_globales['zips_validos'])
    zips_invalidos = len(estadisticas_globales['zips_invalidos'])
    
    print(f"\nRESUMEN:")
    print(f"  Total ZIPs: {total_zips}")
    print(f"  Válidos: {zips_validos} ({zips_validos/total_zips*100:.1f}%)")
    print(f"  Inválidos: {zips_invalidos} ({zips_invalidos/total_zips*100:.1f}%)")
    print(f"  Bloques generados: {total_bloques}")
    
    if zips_validos > 0:
        print(f"\nZIPS VÁLIDOS:")
        for i, info in enumerate(estadisticas_globales['zips_validos'], 1):
            print(f"  {i}. {info['nombre']}: {info['bloques_generados']} bloques")
    
    if zips_invalidos > 0:
        print(f"\nZIPS INVÁLIDOS:")
        for i, info in enumerate(estadisticas_globales['zips_invalidos'], 1):
            error = info['error'][:50] if info['error'] else "Desconocido"
            print(f"  {i}. {info['nombre']}: {error}")
    
    print('='*80)

def main():
    zip_files = glob.glob(os.path.join(input_directory, "*.zip"))
    
    if not zip_files:
        print("No se encontraron archivos .zip")
        return
    
    estadisticas_globales = {
        'total_zips': len(zip_files),
        'zips_validos': [],
        'zips_invalidos': [],
        'total_bloques': 0
    }
    
    print(f"\n{'='*60}")
    print("INICIANDO PROCESAMIENTO")
    print('='*60)
    print(f"ZIPs encontrados: {len(zip_files)}")
    print(f"Tamaño bloque: {TAMANO_BLOQUE}x{TAMANO_BLOQUE}")
    print(f"Overlap: {OVERLAP}px ({OVERLAP/TAMANO_BLOQUE*100:.0f}%)")
    print('='*60)
    
    total_bloques = 0
    
    for i, zip_path in enumerate(zip_files, 1):
        print(f"\n[{i}/{len(zip_files)}] Procesando...")
        bloques = procesar_zip(zip_path, estadisticas_globales)
        total_bloques += bloques
        
        # Pausa entre ZIPs para liberar recursos
        time.sleep(1)
        gc.collect()
    
    mostrar_estadisticas_detalladas(estadisticas_globales, total_bloques)
    
    print(f"\n{'='*60}")
    print("PROCESO COMPLETADO")
    print(f"Total bloques: {total_bloques}")
    print('='*60)

if __name__ == "__main__":
    main()