#python predict_wildfire.py --model weights/model_best.pth --image test_image.tif --output prediction.tif --gpu auto
#python predict_wildfire.py --model weights/model_best.pth --image test_image.tif --output prediction.png --gpu False --visualization True

import torch
import numpy as np
from osgeo import gdal, osr
import os
from Wildfire_models import UNet2D as WildfireNet
import cv2
import argparse

def check_gpu_availability():
    """Verificar si CUDA está disponible y funcionando correctamente"""
    if not torch.cuda.is_available():
        return False
    
    try:
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except RuntimeError:
        return False

def predict_image(model_path, image_path, output_path, gpu=False, threshold=0.5, visualization=False):
    """
    Realizar predicción de incendios forestales
    
    Args:
        model_path: Ruta al modelo entrenado (.pth)
        image_path: Ruta a la imagen de entrada (TIFF 5 bandas, UInt16)
        output_path: Ruta para guardar la predicción
        gpu: Usar GPU (True/False/"auto")
        threshold: Umbral de probabilidad para clasificar como incendio (0-1)
        visualization: Generar imagen de visualización con colores
    """
    
    # Determinar uso de GPU
    if isinstance(gpu, str):
        if gpu.lower() == "auto":
            gpu = check_gpu_availability()
        else:
            gpu = gpu.lower() == "true"
    
    print("="*60)
    print("PREDICCIÓN DE INCENDIOS FORESTALES")
    print("="*60)
    print(f"Modelo: {model_path}")
    print(f"Imagen: {image_path}")
    print(f"Salida: {output_path}")
    print(f"Dispositivo: {'GPU' if gpu else 'CPU'}")
    print(f"Umbral: {threshold}")
    print("="*60 + "\n")
    
    # Verificar que existe el modelo
    if not os.path.exists(model_path):
        print(f"ERROR: No se encontró el modelo en {model_path}")
        return
    
    # Verificar que existe la imagen
    if not os.path.exists(image_path):
        print(f"ERROR: No se encontró la imagen en {image_path}")
        return
    
    # Cargar modelo
    print("Cargando modelo...")
    model = WildfireNet(in_channels=5, out_channels=2)
    device = torch.device('cuda' if gpu else 'cpu')
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("✓ Modelo cargado exitosamente\n")
    except Exception as e:
        print(f"ERROR al cargar modelo: {e}")
        return
    
    # Cargar imagen
    print("Cargando imagen...")
    ds = gdal.Open(image_path)
    if ds is None:
        print(f"ERROR: No se pudo abrir la imagen {image_path}")
        return
    
    bands = ds.RasterCount
    height, width = ds.RasterYSize, ds.RasterXSize
    
    print(f"✓ Imagen cargada: {bands} bandas, {width}x{height} píxeles")
    
    if bands != 5:
        print(f"⚠ ADVERTENCIA: La imagen tiene {bands} bandas, se esperaban 5")
        print(f"  Se usarán las primeras {min(bands, 5)} bandas")
    
    # Preparar imagen (normalizar de UInt16 a [0, 1])
    print("\nProcesando imagen...")
    image = np.zeros((min(bands, 5), height, width), dtype=np.float32)
    
    for b in range(min(bands, 5)):
        band_data = ds.GetRasterBand(b+1).ReadAsArray()
        
        # ✅ CORREGIDO: Normalización para UInt16
        # Dividir por 12500 (como en el entrenamiento) en lugar de 65535
        band_data = band_data.astype(np.float32) / 12500.0
        band_data = np.clip(band_data, 0, 1)
        image[b, :, :] = band_data
    
    print(f"✓ Normalización completada (UInt16 → [0,1])")
    
    # Configuración para procesamiento por patches
    patch_size = 128  # Tamaño de los patches (mismo que en entrenamiento)
    overlap = 32      # Superposición entre patches para suavizar bordes
    
    # Arrays para acumular resultados
    output = np.zeros((2, height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    
    print(f"\nProcesando imagen por patches de {patch_size}x{patch_size} con overlap de {overlap}...")
    
    # Calcular número de patches
    step = patch_size - overlap
    n_patches_h = ((height - overlap) + step - 1) // step
    n_patches_w = ((width - overlap) + step - 1) // step
    total_patches = n_patches_h * n_patches_w
    
    print(f"Total de patches a procesar: {total_patches}")
    
    processed = 0
    
    with torch.no_grad():
        for i in range(0, height - overlap, step):
            for j in range(0, width - overlap, step):
                # Asegurar que el patch no exceda los límites de la imagen
                i_end = min(i + patch_size, height)
                j_end = min(j + patch_size, width)
                
                # Ajustar el inicio si es necesario
                i_start = max(0, i_end - patch_size)
                j_start = max(0, j_end - patch_size)
                
                # Obtener patch
                patch = image[:, i_start:i_end, j_start:j_end]
                
                # Si el patch es más pequeño que patch_size, rellenar con ceros
                if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                    temp_patch = np.zeros((min(bands, 5), patch_size, patch_size), dtype=np.float32)
                    temp_patch[:, :patch.shape[1], :patch.shape[2]] = patch
                    patch = temp_patch
                
                # Convertir a tensor
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
                
                if gpu:
                    patch_tensor = patch_tensor.cuda()
                
                # Predicción
                pred_patch = model(patch_tensor)
                pred_patch = torch.softmax(pred_patch, dim=1).cpu().numpy()[0]
                
                # Recortar predicción al tamaño real del patch
                actual_h = i_end - i_start
                actual_w = j_end - j_start
                pred_patch = pred_patch[:, :actual_h, :actual_w]
                
                # Acumular resultados
                output[:, i_start:i_end, j_start:j_end] += pred_patch
                count[i_start:i_end, j_start:j_end] += 1
                
                processed += 1
                if processed % 100 == 0:
                    print(f"  Procesados {processed}/{total_patches} patches ({processed/total_patches*100:.1f}%)")
    
    print(f"✓ Todos los patches procesados\n")
    
    # Promediar superposiciones
    print("Combinando resultados de patches...")
    output = output / np.maximum(count, 1)
    
    # Obtener probabilidades de incendio (clase 1)
    prob_fire = output[1]
    
    # Aplicar umbral
    print(f"Aplicando umbral de {threshold}...")
    pred_mask = (prob_fire > threshold).astype(np.uint8)
    
    # Estadísticas de la predicción
    total_pixels = pred_mask.size
    fire_pixels = np.sum(pred_mask > 0)
    fire_percent = (fire_pixels / total_pixels) * 100
    
    print("\n" + "="*60)
    print("RESULTADOS DE LA PREDICCIÓN")
    print("="*60)
    print(f"Total de píxeles: {total_pixels:,}")
    print(f"Píxeles con incendio: {fire_pixels:,} ({fire_percent:.2f}%)")
    print(f"Píxeles sin incendio: {total_pixels - fire_pixels:,} ({100-fire_percent:.2f}%)")
    print("="*60 + "\n")
    
    # Determinar formato de salida por extensión
    output_ext = os.path.splitext(output_path)[1].lower()
    
    if output_ext in ['.tif', '.tiff']:
        # Guardar como GeoTIFF
        print(f"Guardando predicción como GeoTIFF...")
        
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path, width, height, 1, gdal.GDT_Byte)
        
        # Copiar georeferencia de la imagen original
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        
        # Guardar máscara binaria (0 = no incendio, 1 = incendio)
        out_ds.GetRasterBand(1).WriteArray(pred_mask)
        out_ds.FlushCache()
        out_ds = None
        
        print(f"✓ Predicción guardada en: {output_path}")
        
    elif output_ext in ['.png', '.jpg', '.jpeg']:
        # Guardar como imagen de visualización
        print(f"Guardando predicción como imagen de visualización...")
        
        # Crear imagen RGB para visualización
        visual_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Rojo para incendio, negro para no incendio
        visual_mask[pred_mask == 1] = [255, 0, 0]  # Rojo = incendio
        
        cv2.imwrite(output_path, cv2.cvtColor(visual_mask, cv2.COLOR_RGB2BGR))
        print(f"✓ Visualización guardada en: {output_path}")
        
    else:
        print(f"⚠ Formato no reconocido: {output_ext}")
        print(f"  Se guardará como .tif por defecto")
        output_path = os.path.splitext(output_path)[0] + '.tif'
        
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(output_path, width, height, 1, gdal.GDT_Byte)
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(pred_mask)
        out_ds.FlushCache()
        out_ds = None
        
        print(f"✓ Predicción guardada en: {output_path}")
    
    # Guardar visualización adicional si se solicita
    if visualization:
        visual_path = os.path.splitext(output_path)[0] + '_visualization.png'
        print(f"\nGenerando visualización adicional...")
        
        visual_mask = np.zeros((height, width, 3), dtype=np.uint8)
        visual_mask[pred_mask == 1] = [255, 0, 0]  # Rojo = incendio
        
        cv2.imwrite(visual_path, cv2.cvtColor(visual_mask, cv2.COLOR_RGB2BGR))
        print(f"✓ Visualización guardada en: {visual_path}")
    
    # Guardar también un mapa de probabilidades si se solicita
    if visualization:
        prob_path = os.path.splitext(output_path)[0] + '_probabilities.png'
        print(f"Generando mapa de probabilidades...")
        
        # Convertir probabilidades a escala de colores
        prob_normalized = (prob_fire * 255).astype(np.uint8)
        prob_colored = cv2.applyColorMap(prob_normalized, cv2.COLORMAP_JET)
        
        cv2.imwrite(prob_path, prob_colored)
        print(f"✓ Mapa de probabilidades guardado en: {prob_path}")
    
    # Liberar recursos
    ds = None
    
    print("\n" + "="*60)
    print("PREDICCIÓN COMPLETADA")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicción de Incendios Forestales')
    parser.add_argument('--model', required=True, help='Ruta al modelo entrenado (.pth)')
    parser.add_argument('--image', required=True, help='Ruta a la imagen de entrada (TIFF 5 bandas)')
    parser.add_argument('--output', required=True, help='Ruta para guardar la predicción')
    parser.add_argument('--gpu', type=str, default="auto", 
                       help='Usar GPU: True/False/auto (auto detecta disponibilidad)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Umbral de probabilidad para clasificar como incendio (0-1, default: 0.5)')
    parser.add_argument('--visualization', type=str, default="False",
                       help='Generar visualizaciones adicionales (mapa de probabilidades)')
    
    args = parser.parse_args()
    
    predict_image(
        args.model, 
        args.image, 
        args.output, 
        args.gpu,
        args.threshold,
        args.visualization.lower() == "true"
    )