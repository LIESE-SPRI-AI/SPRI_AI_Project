#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import sys

# Ruta de la imagen (usa la que funciona)
image_path = "/home/liese2/SPRI_AI_project/Dataset/Merged_4Band/Huamuchil_4band.tiff"
pred_path = "mascara.txt"

print("="*60)
print("VISUALIZACIÓN DE PREDICCIÓN")
print("="*60)
print(f"Imagen: {image_path}")
print(f"Predicción: {pred_path}")
print("="*60)

# Cargar imagen
print("\n1. Cargando imagen...")
ds = gdal.Open(image_path)
if ds is None:
    print(f"ERROR: No se pudo abrir {image_path}")
    sys.exit(1)

print(f"   Bandas: {ds.RasterCount}")
print(f"   Dimensiones: {ds.RasterXSize} x {ds.RasterYSize}")
print(f"   Tipo de dato: {gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)}")

# Leer bandas para RGB (usar bandas 4,3,2 o 3,2,1)
if ds.RasterCount >= 4:
    print("\n2. Usando bandas 4 (NIR), 3 (Red), 2 (Green) para RGB")
    r_band = 4  # NIR
    g_band = 3  # Red  
    b_band = 2  # Green
else:
    print("\n2. Usando bandas 3,2,1 para RGB")
    r_band, g_band, b_band = 1, 2, 3

# Leer datos
r = ds.GetRasterBand(r_band).ReadAsArray().astype(np.float32)
g = ds.GetRasterBand(g_band).ReadAsArray().astype(np.float32)
b = ds.GetRasterBand(b_band).ReadAsArray().astype(np.float32)

print(f"\n3. Estadísticas de los datos originales:")
print(f"   Banda R: min={r.min():.0f}, max={r.max():.0f}, mean={r.mean():.0f}")
print(f"   Banda G: min={g.min():.0f}, max={g.max():.0f}, mean={g.mean():.0f}")
print(f"   Banda B: min={b.min():.0f}, max={b.max():.0f}, mean={b.mean():.0f}")

# Normalización mejorada
def normalize_band(band, method='percentile'):
    """Normalizar banda para visualización"""
    if method == 'percentile':
        # Usar percentiles para evitar valores extremos
        p2 = np.percentile(band, 2)
        p98 = np.percentile(band, 98)
        if p98 - p2 > 0:
            band_norm = (band - p2) / (p98 - p2)
        else:
            band_norm = band - band.min()
            if band_norm.max() > 0:
                band_norm = band_norm / band_norm.max()
    elif method == 'minmax':
        # Normalización min-max simple
        band_min = band.min()
        band_max = band.max()
        if band_max - band_min > 0:
            band_norm = (band - band_min) / (band_max - band_min)
        else:
            band_norm = band - band_min
    elif method == 'sqrt':
        # Transformación sqrt para datos con alta variabilidad
        band_norm = np.sqrt(band)
        band_norm = band_norm / band_norm.max()
    else:
        band_norm = band / band.max()
    
    return np.clip(band_norm, 0, 1)

# Probar diferentes métodos de normalización
print("\n4. Normalizando para visualización...")
rgb_percentile = np.stack([
    normalize_band(r, 'percentile'),
    normalize_band(g, 'percentile'),
    normalize_band(b, 'percentile')
], axis=2)

rgb_minmax = np.stack([
    normalize_band(r, 'minmax'),
    normalize_band(g, 'minmax'),
    normalize_band(b, 'minmax')
], axis=2)

print(f"   Imagen normalizada (percentil): shape={rgb_percentile.shape}, min={rgb_percentile.min():.3f}, max={rgb_percentile.max():.3f}")

# Cargar predicción
print("\n5. Cargando predicción...")
height, width = r.shape
pred_mask = np.zeros((height, width), dtype=np.uint8)

try:
    with open(pred_path, 'r') as f:
        line_count = 0
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        y, x, val = int(parts[0]), int(parts[1]), int(parts[2])
                        if 0 <= y < height and 0 <= x < width:
                            pred_mask[y, x] = val
                            line_count += 1
                    except ValueError:
                        pass
    print(f"   Leídas {line_count} coordenadas de incendio")
except Exception as e:
    print(f"   Error: {e}")

fire_pixels = np.sum(pred_mask == 1)
print(f"   Píxeles con incendio: {fire_pixels} ({fire_pixels/(height*width)*100:.4f}%)")

if fire_pixels == 0:
    print("\n⚠ ADVERTENCIA: No se encontraron píxeles de incendio en la predicción")
    print("   Verifica que el archivo mascara.txt tenga datos")

# Crear visualización
print("\n6. Generando visualización...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Método 1: Normalización por percentiles
axes[0,0].imshow(rgb_percentile)
axes[0,0].set_title('Imagen Original (normalización percentil)', fontsize=12)
axes[0,0].axis('off')

# Método 2: Normalización min-max
axes[0,1].imshow(rgb_minmax)
axes[0,1].set_title('Imagen Original (normalización min-max)', fontsize=12)
axes[0,1].axis('off')

# Predicción sobre percentil
axes[1,0].imshow(rgb_percentile)
if fire_pixels > 0:
    overlay = np.zeros((height, width, 4))
    overlay[pred_mask == 1] = [1, 0, 0, 0.7]  # Rojo intenso
    axes[1,0].imshow(overlay)
axes[1,0].set_title(f'Predicción sobre percentil\n{fire_pixels} píxeles de incendio', fontsize=12)
axes[1,0].axis('off')

# Predicción sobre min-max
axes[1,1].imshow(rgb_minmax)
if fire_pixels > 0:
    axes[1,1].imshow(overlay)
axes[1,1].set_title(f'Predicción sobre min-max\n{fire_pixels} píxeles de incendio', fontsize=12)
axes[1,1].axis('off')

plt.tight_layout()
plt.savefig('visualizacion_fixed.png', dpi=150, bbox_inches='tight')
print("   ✅ Guardado: visualizacion_fixed.png")

# También guardar versión simple con mejor contraste
print("\n7. Guardando versión mejorada...")
fig2, ax = plt.subplots(1, 2, figsize=(14, 6))

# Usar ecualización de histograma para mejor contraste
from skimage import exposure

def enhance_contrast(rgb):
    """Mejorar contraste de la imagen"""
    result = np.zeros_like(rgb)
    for i in range(3):
        result[:,:,i] = exposure.equalize_hist(rgb[:,:,i])
    return result

rgb_enhanced = enhance_contrast(rgb_percentile)

ax[0].imshow(rgb_enhanced)
ax[0].set_title('Imagen con contraste mejorado', fontsize=12)
ax[0].axis('off')

ax[1].imshow(rgb_enhanced)
if fire_pixels > 0:
    ax[1].imshow(overlay)
ax[1].set_title(f'Predicción (contraste mejorado)\n{fire_pixels} píxeles', fontsize=12)
ax[1].axis('off')

plt.tight_layout()
plt.savefig('visualizacion_enhanced.png', dpi=150, bbox_inches='tight')
print("   ✅ Guardado: visualizacion_enhanced.png")

print("\n" + "="*60)
print("VISUALIZACIÓN COMPLETADA")
print("="*60)
print("\nArchivos generados:")
print("  - visualizacion_fixed.png")
print("  - visualizacion_enhanced.png")
print("\nPara verlos:")
print("  eog visualizacion_enhanced.png")
print("  o xdg-open visualizacion_enhanced.png")