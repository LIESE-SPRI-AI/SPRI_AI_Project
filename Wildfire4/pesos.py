#!/usr/bin/env python3
import os
import numpy as np
from datasetAugmentationWildFire4B import WildFireDataset

data_root = "/home/liese2/SPRI_AI_project/Wildfire4"
list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
img_dir = os.path.join(data_root, "Images")
mask_dir = os.path.join(data_root, "SegmentationClass")

bands = 4

print("Calculando nuevas estadísticas para Wildfire4...")

# Crear dataset SIN normalización
objects_dataset = WildFireDataset(
    list_file=list_file_path,
    img_dir=img_dir,
    mask_dir=mask_dir,
    size=128,
    normalize=False,  # ¡IMPORTANTE!
    augmentation=False
)

# Calcular mean y std por banda
print(f"Procesando {len(objects_dataset)} imágenes...")

mean = np.zeros(bands, dtype=np.float64)
std = np.zeros(bands, dtype=np.float64)
nb_samples = len(objects_dataset)

for idx in range(nb_samples):
    data = objects_dataset[idx]
    image = data['image'].numpy()  # Shape: [bands, 128, 128]
    
    for b in range(bands):
        mean[b] += image[b, :, :].mean()
        std[b] += image[b, :, :].std()
    
    if (idx + 1) % 100 == 0:
        print(f"  Procesadas {idx + 1}/{nb_samples} imágenes...")

# Promediar
mean /= nb_samples
std /= nb_samples

print("\n" + "="*50)
print("NUEVAS ESTADÍSTICAS PARA Wildfire4:")
print("="*50)
print(f"meanB = {list(mean)}")
print(f"stdB = {list(std)}")

# Mostrar valores formateados
print("\nFormateado para código:")
print(f"meanB = [{', '.join([f'{x:.10f}' for x in mean])}]")
print(f"stdB = [{', '.join([f'{x:.10f}' for x in std])}]")

# Guardar en archivo
output_file = os.path.join(data_root, "dataset_statistics_Wildfire4.txt")
with open(output_file, 'w') as f:
    f.write(f"Dataset: Wildfire4\n")
    f.write(f"Total imágenes: {nb_samples}\n")
    f.write(f"Tamaño: 128x128\n")
    f.write(f"\nmeanB = {list(mean)}\n")
    f.write(f"stdB = {list(std)}\n")
    
    f.write("\nPara copiar al código:\n")
    f.write(f"meanB = [{', '.join([f'{x:.10f}' for x in mean])}]\n")
    f.write(f"stdB = [{', '.join([f'{x:.10f}' for x in std])}]\n")

print(f"\n✓ Estadísticas guardadas en: {output_file}")