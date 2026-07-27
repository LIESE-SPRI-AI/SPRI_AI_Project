#!/usr/bin/env python3
"""
Script para visualizar la imagen original y la predicción de incendios
Uso: python3 visualizar_prediccion.py --image imagen.tiff --prediction mascara.txt --output visualizacion.png
"""

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import argparse
import sys
from matplotlib.patches import Patch

def cargar_imagen_rgb(image_path):
    """Cargar imagen y crear visualización RGB"""
    ds = gdal.Open(image_path)
    if ds is None:
        print(f"ERROR: No se pudo abrir {image_path}")
        return None
    
    # Leer bandas
    bands = ds.RasterCount
    
    # Para visualización RGB, usar bandas 4,3,2 (infrarrojo, rojo, verde) o 3,2,1
    if bands >= 4:
        # Usar bandas 4 (NIR), 3 (Red), 2 (Green) para mejor visualización de vegetación
        r = ds.GetRasterBand(4).ReadAsArray().astype(np.float32)
        g = ds.GetRasterBand(3).ReadAsArray().astype(np.float32)
        b = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
    elif bands >= 3:
        r = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        g = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
        b = ds.GetRasterBand(3).ReadAsArray().astype(np.float32)
    else:
        print("ERROR: Se necesitan al menos 3 bandas para visualización RGB")
        return None
    
    # Normalizar para visualización (estirar al 2-98 percentil)
    def normalize(band):
        p2, p98 = np.percentile(band, (2, 98))
        band_norm = (band - p2) / (p98 - p2)
        return np.clip(band_norm, 0, 1)
    
    rgb = np.stack([normalize(r), normalize(g), normalize(b)], axis=2)
    
    return rgb

def cargar_prediccion(mask_txt_path, height, width):
    """Cargar predicción desde archivo de texto"""
    pred_mask = np.zeros((height, width), dtype=np.uint8)
    
    try:
        with open(mask_txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        y, x, valor = int(parts[0]), int(parts[1]), int(parts[2])
                        if 0 <= y < height and 0 <= x < width:
                            pred_mask[y, x] = valor
    except Exception as e:
        print(f"Error cargando predicción: {e}")
    
    return pred_mask

def cargar_mascara_real(mask_path):
    """Cargar máscara real si existe (opcional)"""
    if not mask_path or not os.path.exists(mask_path):
        return None
    
    ds = gdal.Open(mask_path)
    if ds is None:
        return None
    
    mask = ds.GetRasterBand(1).ReadAsArray()
    return mask

def visualizar_prediccion(image_path, prediction_path, output_path, ground_truth_path=None):
    """Visualizar imagen original con predicción superpuesta"""
    
    print("="*60)
    print("VISUALIZACIÓN DE PREDICCIÓN")
    print("="*60)
    print(f"Imagen: {image_path}")
    print(f"Predicción: {prediction_path}")
    print(f"Salida: {output_path}")
    if ground_truth_path:
        print(f"Máscara real: {ground_truth_path}")
    print("="*60 + "\n")
    
    # Cargar imagen
    print("Cargando imagen...")
    rgb = cargar_imagen_rgb(image_path)
    if rgb is None:
        return False
    
    height, width = rgb.shape[:2]
    print(f"✓ Imagen cargada: {width}x{height} píxeles")
    
    # Cargar predicción
    print("Cargando predicción...")
    pred_mask = cargar_prediccion(prediction_path, height, width)
    fire_pixels = np.sum(pred_mask > 0)
    print(f"✓ Predicción cargada: {fire_pixels} píxeles con incendio ({fire_pixels/(height*width)*100:.2f}%)")
    
    # Cargar máscara real si existe
    gt_mask = None
    if ground_truth_path:
        print("Cargando máscara real...")
        gt_mask = cargar_mascara_real(ground_truth_path)
        if gt_mask is not None:
            gt_fire = np.sum(gt_mask > 0)
            print(f"✓ Máscara real cargada: {gt_fire} píxeles con incendio ({gt_fire/(height*width)*100:.2f}%)")
            
            # Calcular métricas si hay ambas máscaras
            if gt_mask is not None and pred_mask is not None:
                intersection = np.logical_and(pred_mask > 0, gt_mask > 0).sum()
                union = np.logical_or(pred_mask > 0, gt_mask > 0).sum()
                if union > 0:
                    iou = intersection / union
                    print(f"\n📊 Métricas (umbral actual):")
                    print(f"  - IoU (Intersection over Union): {iou:.4f}")
                    if (gt_mask > 0).sum() > 0:
                        recall = intersection / (gt_mask > 0).sum()
                        print(f"  - Recall (sensibilidad): {recall:.4f}")
                if (pred_mask > 0).sum() > 0:
                    precision = intersection / (pred_mask > 0).sum()
                    print(f"  - Precision: {precision:.4f}")
    
    # Crear figura
    if gt_mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Imagen original
    axes[0].imshow(rgb)
    axes[0].set_title('Imagen Original (RGB)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Imagen con predicción superpuesta
    axes[1].imshow(rgb)
    # Crear máscara de incendio con transparencia
    fire_overlay = np.zeros((height, width, 4))
    fire_mask = pred_mask > 0
    if np.any(fire_mask):
        fire_overlay[fire_mask] = [1, 0, 0, 0.6]  # Rojo con transparencia 60%
    axes[1].imshow(fire_overlay)
    axes[1].set_title(f'Predicción (umbral actual)\n{fire_pixels} píxeles de incendio', 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Máscara real (si existe)
    if gt_mask is not None:
        axes[2].imshow(rgb)
        gt_overlay = np.zeros((height, width, 4))
        gt_fire = gt_mask > 0
        if np.any(gt_fire):
            gt_overlay[gt_fire] = [0, 1, 0, 0.5]  # Verde con transparencia 50%
        axes[2].imshow(gt_overlay)
        axes[2].set_title(f'Máscara Real (Ground Truth)\n{gt_fire.sum()} píxeles de incendio', 
                          fontsize=14, fontweight='bold')
        axes[2].axis('off')
    
    # Agregar leyenda
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='Predicción (Incendio)'),
        Patch(facecolor='green', alpha=0.5, label='Máscara Real (Incendio)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualización guardada en: {output_path}")
    
    plt.show()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Visualizar predicción de incendios')
    parser.add_argument('--image', required=True, help='Ruta a la imagen original (TIFF)')
    parser.add_argument('--prediction', required=True, help='Ruta al archivo de predicción (.txt)')
    parser.add_argument('--output', default='prediccion_visualizada.png', help='Ruta para guardar visualización')
    parser.add_argument('--ground_truth', help='Ruta a la máscara real (opcional, para comparar)')
    
    args = parser.parse_args()
    
    success = visualizar_prediccion(args.image, args.prediction, args.output, args.ground_truth)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
