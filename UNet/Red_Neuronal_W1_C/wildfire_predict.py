#!/usr/bin/env python3
"""
Script de predicción para incendios forestales
Uso: python3 wildfire_predict.py --weights /ruta/a/pesos.pth --image imagen.tiff --output prediccion.bin --mask prediccion_mask.txt --size 128
"""

import torch
import torch.nn as nn
import numpy as np
from osgeo import gdal
import os
import sys
import argparse
import struct
from pathlib import Path
from collections import OrderedDict

# ============================================================================
# DEFINICIÓN DEL MODELO UNet (versión que coincide con el entrenamiento)
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, features[0])
        self.enc2 = DoubleConv(features[0], features[1])
        self.enc3 = DoubleConv(features[1], features[2])
        self.enc4 = DoubleConv(features[2], features[3])
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3] * 2)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features[3] * 2, features[3])
        
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features[2] * 2, features[2])
        
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features[1] * 2, features[1])
        
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features[0] * 2, features[0])
        
        # Output
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Maxpool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        out = self.out_conv(dec1)
        
        return out

# ============================================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================================

def check_gpu_availability():
    """Verificar si CUDA está disponible"""
    if not torch.cuda.is_available():
        return False
    try:
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except RuntimeError:
        return False

def load_model(weights_path, device):
    """
    Cargar modelo desde archivo o directorio de pesos
    
    Args:
        weights_path: Ruta al archivo .pth o directorio que contiene los pesos
        device: Dispositivo (CPU/GPU)
    """
    print(f"Cargando modelo desde: {weights_path}")
    
    # Determinar si es archivo o directorio
    if os.path.isfile(weights_path):
        weight_file = weights_path
    else:
        # Buscar archivos de pesos en el directorio
        weight_files = []
        for file in os.listdir(weights_path):
            if file.endswith('.pth') or file.endswith('.pt'):
                weight_files.append(file)
        
        if not weight_files:
            print(f"ERROR: No se encontraron archivos .pth en {weights_path}")
            return None
        
        # Ordenar por época si es posible (model_epoch_XXXX.pth)
        def get_epoch_num(filename):
            try:
                if 'epoch' in filename:
                    return int(filename.split('_')[2])
                return 0
            except:
                return 0
        
        weight_files.sort(key=get_epoch_num, reverse=True)
        weight_file = os.path.join(weights_path, weight_files[0])
        print(f"Usando pesos: {weight_file}")
    
    # Crear modelo con la arquitectura correcta
    model = UNet(in_channels=4, out_channels=2)
    
    try:
        # Cargar checkpoint
        checkpoint = torch.load(weight_file, map_location=device)
        
        # Extraer state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filtrar num_batches_tracked
        clean_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'num_batches_tracked' not in k:
                clean_state_dict[k] = v
        
        # Cargar pesos
        missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
        
        if missing_keys:
            print(f"  ℹ Claves faltantes ({len(missing_keys)}): {missing_keys[:3]}..." if len(missing_keys) > 3 else f"  ℹ Claves faltantes: {missing_keys}")
        if unexpected_keys:
            print(f"  ℹ Claves inesperadas ({len(unexpected_keys)}): {unexpected_keys[:3]}..." if len(unexpected_keys) > 3 else f"  ℹ Claves inesperadas: {unexpected_keys}")
        
        # Verificar si se cargaron todas las capas importantes
        if len(missing_keys) < 50:  # Si faltan pocas capas, probablemente está bien
            print("✓ Modelo cargado exitosamente")
        else:
            print("⚠ ADVERTENCIA: Faltan muchas claves, puede haber problemas de arquitectura")
        
        model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        print(f"ERROR al cargar modelo: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_and_save_binary(model, image_path, output_bin_path, mask_txt_path, patch_size, device, threshold=0.5):
    """
    Realizar predicción y guardar en formato binario y máscara de texto
    """
    
    print("="*60)
    print("PREDICCIÓN DE INCENDIOS FORESTALES")
    print("="*60)
    print(f"Imagen: {image_path}")
    print(f"Salida binaria: {output_bin_path}")
    print(f"Máscara texto: {mask_txt_path}")
    print(f"Tamaño patch: {patch_size}")
    print(f"Dispositivo: {'GPU' if device.type == 'cuda' else 'CPU'}")
    print(f"Umbral: {threshold}")
    print("="*60 + "\n")
    
    # Verificar que existe la imagen
    if not os.path.exists(image_path):
        print(f"ERROR: No se encontró la imagen en {image_path}")
        return False
    
    # Cargar imagen
    print("Cargando imagen...")
    ds = gdal.Open(image_path)
    if ds is None:
        print(f"ERROR: No se pudo abrir la imagen {image_path}")
        return False
    
    bands = ds.RasterCount
    height, width = ds.RasterYSize, ds.RasterXSize
    
    print(f"✓ Imagen cargada: {bands} bandas, {width}x{height} píxeles")
    
    if bands < 4:
        print(f"⚠ ADVERTENCIA: La imagen tiene {bands} bandas, se esperaban 4")
    
    # Preparar imagen (normalizar de UInt16 a [0, 1])
    print("\nProcesando imagen...")
    image = np.zeros((min(bands, 4), height, width), dtype=np.float32)
    
    for b in range(min(bands, 4)):
        band_data = ds.GetRasterBand(b+1).ReadAsArray()
        # Normalización para UInt16 (dividir por 12500)
        band_data = band_data.astype(np.float32) / 12500.0
        band_data = np.clip(band_data, 0, 1)
        image[b, :, :] = band_data
    
    print(f"✓ Normalización completada (UInt16 → [0,1])")
    
    # Procesamiento por patches
    overlap = patch_size // 4  # 32 píxeles de overlap para patches de 128
    
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
                # Asegurar que el patch no exceda los límites
                i_end = min(i + patch_size, height)
                j_end = min(j + patch_size, width)
                
                # Ajustar el inicio si es necesario
                i_start = max(0, i_end - patch_size)
                j_start = max(0, j_end - patch_size)
                
                # Obtener patch
                patch = image[:, i_start:i_end, j_start:j_end]
                
                # Si el patch es más pequeño, rellenar con ceros
                if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                    temp_patch = np.zeros((min(bands, 4), patch_size, patch_size), dtype=np.float32)
                    temp_patch[:, :patch.shape[1], :patch.shape[2]] = patch
                    patch = temp_patch
                
                # Convertir a tensor
                patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
                
                if device.type == 'cuda':
                    patch_tensor = patch_tensor.cuda()
                
                # Predicción
                pred_patch = model(patch_tensor)
                pred_patch = torch.softmax(pred_patch, dim=1).cpu().numpy()[0]
                
                # Recortar predicción al tamaño real
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
    
    # Estadísticas
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
    
    # Guardar en formato binario
    print(f"Guardando predicción binaria en: {output_bin_path}")
    with open(output_bin_path, 'wb') as f:
        # Guardar metadatos primero
        f.write(struct.pack('II', height, width))  # Dimensiones
        # Guardar la máscara binaria
        f.write(pred_mask.tobytes())
    
    file_size = os.path.getsize(output_bin_path)
    print(f"✓ Archivo binario guardado ({file_size} bytes)")
    
    # Guardar máscara en formato texto
    print(f"Guardando máscara de texto en: {mask_txt_path}")
    with open(mask_txt_path, 'w') as f:
        f.write(f"# Predicción de incendios forestales\n")
        f.write(f"# =================================\n")
        f.write(f"# Archivo: {os.path.basename(image_path)}\n")
        f.write(f"# Dimensiones: {width} x {height} píxeles\n")
        f.write(f"# Píxeles con incendio: {fire_pixels} ({fire_percent:.2f}%)\n")
        f.write(f"# Umbral usado: {threshold}\n")
        f.write(f"# Tamaño de patch: {patch_size}\n")
        f.write(f"# =================================\n")
        f.write(f"# Formato: fila columna valor (1=incendio, 0=no incendio)\n\n")
        
        # Guardar coordenadas de píxeles con incendio
        fire_coords = np.where(pred_mask == 1)
        for y, x in zip(fire_coords[0], fire_coords[1]):
            f.write(f"{y} {x} 1\n")
        
        # Guardar estadísticas al final
        f.write(f"\n# Estadísticas:\n")
        f.write(f"# Total incendio: {fire_pixels} píxeles\n")
        f.write(f"# Porcentaje: {fire_percent:.4f}%\n")
    
    print(f"✓ Máscara de texto guardada")
    
    # Liberar recursos
    ds = None
    
    print("\n" + "="*60)
    print("PREDICCIÓN COMPLETADA EXITOSAMENTE")
    print("="*60)
    
    return True

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Predicción de Incendios Forestales')
    parser.add_argument('--weights', required=True, help='Ruta al archivo .pth o directorio con los pesos del modelo')
    parser.add_argument('--image', required=True, help='Ruta a la imagen de entrada (TIFF 4 bandas)')
    parser.add_argument('--output', required=True, help='Ruta para guardar predicción binaria (.bin)')
    parser.add_argument('--mask', required=True, help='Ruta para guardar máscara de texto (.txt)')
    parser.add_argument('--size', type=int, default=128, help='Tamaño de patch (default: 128)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Umbral de probabilidad (default: 0.5)')
    parser.add_argument('--gpu', type=str, default='auto', help='Usar GPU: auto/True/False')
    
    args = parser.parse_args()
    
    # Configurar dispositivo
    if args.gpu.lower() == 'auto':
        use_gpu = check_gpu_availability()
    else:
        use_gpu = args.gpu.lower() == 'true'
    
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelo
    model = load_model(args.weights, device)
    
    if model is None:
        print("\n❌ ERROR: No se pudo cargar el modelo")
        sys.exit(1)
    
    # Realizar predicción
    success = predict_and_save_binary(
        model, args.image, args.output, args.mask, 
        args.size, device, args.threshold
    )
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()