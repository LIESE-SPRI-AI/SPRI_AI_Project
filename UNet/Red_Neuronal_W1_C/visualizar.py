import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

# Imagen que funcionó (4 bandas)
ds = gdal.Open('/home/liese2/SPRI_AI_project/Dataset/Merged_4Band/Huamuchil_4band.tiff')

# Crear RGB
r = ds.GetRasterBand(4).ReadAsArray().astype(np.float32)
g = ds.GetRasterBand(3).ReadAsArray().astype(np.float32)
b = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)

def norm(x):
    p2, p98 = np.percentile(x, (2, 98))
    return np.clip((x - p2) / (p98 - p2), 0, 1)

rgb = np.stack([norm(r), norm(g), norm(b)], axis=2)
h, w = rgb.shape[:2]

# Cargar predicción
pred = np.zeros((h, w))
with open('mascara.txt', 'r') as f:
    for line in f:
        if not line.startswith('#') and len(line.split()) >= 3:
            y, x, v = map(int, line.split()[:3])
            if y < h and x < w:
                pred[y, x] = v

fire = (pred == 1).sum()
print(f'Dimensiones: {w}x{h}')
print(f'Píxeles con incendio: {fire} ({fire/(h*w)*100:.4f}%)')

# Graficar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.imshow(rgb)
ax1.set_title('Imagen Original', fontsize=14)
ax1.axis('off')

ax2.imshow(rgb)
if fire > 0:
    overlay = np.zeros((h, w, 4))
    overlay[pred == 1] = [1, 0, 0, 0.6]
    ax2.imshow(overlay)
ax2.set_title(f'Predicción - {fire} píxeles', fontsize=14)
ax2.axis('off')

plt.tight_layout()
plt.savefig('visualizacion.png', dpi=150)
print('\n✅ Visualización guardada: visualizacion.png')
