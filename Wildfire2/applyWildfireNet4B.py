#!/usr/bin/env python3
# applyWildfireNet4B.py - Salida RGB GeoTIFF (negro con rojo)

from __future__ import print_function
import numpy as np
import torch
import argparse
import os
from osgeo import gdal
from PIL import Image
import sys
import time

# Importar modelo
from wildfire_model import WildFireNet2DV3L_3x3_Residual as WildFireNet

def load_image(path=None, channels=4, normalize=False, mean=None, std=None):
    """Carga imagen TIFF con normalización y georreferenciación"""
    
    ds_img = gdal.Open(path)
    if ds_img is None:
        print(f"Error: No se puede abrir {path}")
        sys.exit(1)
    
    cols = ds_img.RasterXSize
    rows = ds_img.RasterYSize
    bands = ds_img.RasterCount
    
    print(f"Image: {bands} bands, Size: {rows}x{cols}")
    
    if bands != channels:
        print(f"ERROR: Expected {channels} bands, got {bands}")
        sys.exit(1)
    
    # Obtener información de georreferenciación
    geotransform = ds_img.GetGeoTransform()
    projection = ds_img.GetProjection()
    
    print(f"Georeference info:")
    print(f"  Origin: ({geotransform[0]:.6f}, {geotransform[3]:.6f})")
    print(f"  Pixel size: ({geotransform[1]:.6f}, {geotransform[5]:.6f})")

    imx_t = np.empty([channels, rows, cols], dtype=np.float32)
    
    print("Reading bands: ", end="", flush=True)
    for b in range(1, bands + 1):
        print(f"{b} ", end="", flush=True)
        channel = np.array(ds_img.GetRasterBand(b).ReadAsArray())
        
        if channel.dtype == np.uint16:
            np_float = channel.astype(np.float32) / 65535.0
        elif channel.dtype == np.uint8:
            np_float = channel.astype(np.float32) / 255.0
        else:
            np_float = channel.astype(np.float32)
            if np.max(channel) > 1000:
                np_float = np_float / 10000.0
        
        np_float = np.clip(np_float, 0, 1)
        imx_t[b-1, :, :] = np_float
    
    print()
    
    if normalize and mean is not None and std is not None:
        print(f"Applying statistical normalization with mean/std...")
        for b in range(channels):
            imx_t[b, :, :] = (imx_t[b, :, :] - mean[b]) / std[b]
    
    print(f"Image loaded: {imx_t.shape}")

    ds_img = None
    return imx_t, geotransform, projection

def save_rgb_geotiff(output_path, result_mask, geotransform, projection):
    """
    Guarda máscara como GeoTIFF RGB de 3 bandas
    Negro (0,0,0) para background, Rojo (255,0,0) para wildfire
    """
    
    driver = gdal.GetDriverByName('GTiff')
    
    rows, cols = result_mask.shape
    
    # Crear dataset con 3 bandas (RGB)
    ds = driver.Create(
        output_path,
        cols,
        rows,
        3,  
        gdal.GDT_Byte
    )
    
    if ds is None:
        print(f"Error: No se puede crear {output_path}")
        return False
    

    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)

    red_band = np.zeros((rows, cols), dtype=np.uint8)
    green_band = np.zeros((rows, cols), dtype=np.uint8)
    blue_band = np.zeros((rows, cols), dtype=np.uint8)

    red_band[result_mask == 1] = 255

    ds.GetRasterBand(1).WriteArray(red_band)    # Banda R
    ds.GetRasterBand(2).WriteArray(green_band)  # Banda G
    ds.GetRasterBand(3).WriteArray(blue_band)   # Banda B

    ds.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
    ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
    ds.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)

    ds.SetMetadataItem('COMPRESS', 'DEFLATE')
    ds.SetMetadataItem('PREDICTOR', '2')

    ds.FlushCache()
    ds = None
    
    print(f"✓ RGB GeoTIFF saved: {output_path}")
    print(f"  Format: 3 bands RGB (Black=background, Red=wildfire)")
    return True

def save_metadata(metadata_path, input_file, height, width, fire_pixel_count, 
                 total_pixels, geotransform, projection, processing_time):
    """Guarda archivo de metadatos con información del procesamiento"""
    
    with open(metadata_path, 'w') as f:
        f.write("WILDFIRE DETECTION METADATA\n")
        f.write("=" * 50 + "\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Processing date: {time.ctime()}\n")
        f.write(f"Processing time: {processing_time:.2f} seconds\n")
        f.write(f"\nIMAGE INFORMATION:\n")
        f.write(f"  Size: {height}x{width} pixels\n")
        f.write(f"  Fire pixels: {fire_pixel_count:,}\n")
        f.write(f"  Fire percentage: {100*fire_pixel_count/total_pixels:.4f}%\n")
        f.write(f"\nGEOREFERENCE INFORMATION:\n")
        f.write(f"  Origin (X, Y): ({geotransform[0]}, {geotransform[3]})\n")
        f.write(f"  Pixel size (X, Y): ({geotransform[1]}, {geotransform[5]})\n")
        if projection:
            f.write(f"  Projection: {projection[:100]}...\n")
        f.write(f"\nOUTPUT FORMAT:\n")
        f.write(f"  Type: RGB GeoTIFF (3 bands)\n")
        f.write(f"  Color scheme: Black (0,0,0) = Background | Red (255,0,0) = Wildfire\n")
    
    print(f"✓ Metadata saved: {metadata_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict WildFire from satellite image - RGB GeoTIFF output')
    
    parser.add_argument('--input', required=True, help='Input GeoTIFF image')
    parser.add_argument('--output', required=True, help='Output RGB mask (.tif)')
    parser.add_argument('--weights', 
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "Wildfire2_N__128__AUG_1_valid_best_3264.pth"))
    
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--size', default=128, type=int, help='window size')
    parser.add_argument('--dims', type=str, default="(32,64)")
    parser.add_argument('--no_normalize', action='store_true', 
                       help='do not normalize input')
    parser.add_argument('--save_metadata', action='store_true',
                       help='save metadata file')
    
    args = parser.parse_args()

    MEAN = [0.14756309210730303, 0.11773452185142602, 0.11033282306813785, 0.20451259007433334]
    STD = [0.06048959501836055, 0.049134526421286386, 0.0423207987452255, 0.09282711550887447]
    
    normalize = not args.no_normalize
    
    print("=" * 70)
    print("WILDFIRE DETECTION - RGB GeoTIFF OUTPUT")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Channels: {args.channels}")
    print(f"Window size: {args.size}")
    print(f"Normalize: {normalize}")
    print(f"Cuda: {args.cuda}")
    print(f"Output format: RGB GeoTIFF (Black + Red)")
    
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)
    
    dim = eval(args.dims)
    model = WildFireNet(args.channels, 2, dims=dim)
    print(f'Model parameters: {sum(param.numel() for param in model.parameters())}')
    
    if os.path.exists(args.weights):
        print(f"Loading weights from: {args.weights}")
        net_weights = torch.load(args.weights, map_location=device)
        model.load_state_dict(net_weights)
        model.eval()
        model.to(device)
        print("✓ Model loaded successfully")
    else:
        print(f"✗ Error: Model file not found: {args.weights}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("LOADING IMAGE")
    print("=" * 70)

    imgnp, geotransform, projection = load_image(
        args.input, channels=args.channels, 
        normalize=normalize, mean=MEAN, std=STD
    )
    
    k_size = args.size
    
    if k_size <= 32:
        stride = 16
        bord = 8
    elif k_size == 64:
        stride = 32
        bord = 8
    elif k_size == 128:
        stride = 92
        bord = 8
    elif k_size == 256:
        stride = 192
        bord = 8
    else:
        stride = k_size // 2
        bord = 1
    
    _, height, width = imgnp.shape
    
    print(f"\nWindow parameters:")
    print(f"  Window size: {k_size}x{k_size}")
    print(f"  Stride: {stride} (overlap: {100*(1-stride/k_size):.1f}%)")
    print(f"  Border ignored: {bord} pixels")
    print(f"  Image size: {height}x{width}")
    
    y_positions = []
    y = 0
    while y + k_size <= height:
        y_positions.append(y)
        y += stride
    
    if y_positions[-1] + k_size < height:
        y_positions.append(height - k_size)
    
    x_positions = []
    x = 0
    while x + k_size <= width:
        x_positions.append(x)
        x += stride
    
    if x_positions[-1] + k_size < width:
        x_positions.append(width - k_size)
    
    nb_y = len(y_positions)
    nb_x = len(x_positions)
    nb_total = nb_y * nb_x
    
    print(f"\nWindow grid: {nb_y}x{nb_x} = {nb_total} windows")
    print(f"  Y coverage: {y_positions[-1] + k_size}/{height}")
    print(f"  X coverage: {x_positions[-1] + k_size}/{width}")
    
    vote_map = np.zeros((height, width), dtype=np.int32)
    coverage_map = np.zeros((height, width), dtype=np.int32)
    
    imgs_tensor = torch.from_numpy(imgnp).to(device)
    
    print("\n" + "=" * 70)
    print("PREDICTING")
    print("=" * 70)
    
    start_time = time.time()
    window_count = 0
    
    for y in y_positions:
        for x in x_positions:
            crop_img = imgs_tensor[:, y:y+k_size, x:x+k_size]
            crop_img = crop_img.unsqueeze(0)
            
            with torch.no_grad():
                output = model(crop_img)
                _, predict = torch.max(output, dim=1)
                pred = predict.cpu().numpy()[0]
            
            for yy in range(bord, k_size - bord):
                img_y = y + yy
                if img_y >= height:
                    break
                
                for xx in range(bord, k_size - bord):
                    img_x = x + xx
                    if img_x >= width:
                        break
                    
                    coverage_map[img_y, img_x] += 1
                    if pred[yy, xx] == 1:
                        vote_map[img_y, img_x] += 1
            
            window_count += 1
            
            if window_count % 10 == 0 or window_count == nb_total:
                progress = 100 * window_count / nb_total
                elapsed = time.time() - start_time
                print(f"\rProgress: {window_count}/{nb_total} ({progress:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s", end="", flush=True)
    
    total_time = time.time() - start_time
    print()
    
    # Generar máscara final
    print("\nGenerating final prediction...")
    
    result_mask = np.zeros((height, width), dtype=np.uint8)
    fire_pixel_count = 0
    
    for y in range(height):
        for x in range(width):
            if coverage_map[y, x] > 0:
                if vote_map[y, x] > (coverage_map[y, x] / 2):
                    result_mask[y, x] = 1
                    fire_pixel_count += 1
    
    total_pixels = height * width
    fire_percentage = 100 * fire_pixel_count / total_pixels
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Processing time: {total_time:.2f} seconds")
    print(f"Fire pixels: {fire_pixel_count:,} ({fire_percentage:.4f}%)")
    
    print("\n" + "=" * 70)
    print("SAVING RGB GeoTIFF")
    print("=" * 70)
    
    if not args.output.endswith('.tif') and not args.output.endswith('.tiff'):
        output_path = args.output + '.tif'
    else:
        output_path = args.output
    
    save_rgb_geotiff(output_path, result_mask, geotransform, projection)
    
    if args.save_metadata:
        base_name = os.path.splitext(output_path)[0]
        metadata_path = base_name + '_metadata.txt'
        save_metadata(metadata_path, args.input, height, width, 
                     fire_pixel_count, total_pixels, geotransform, 
                     projection, total_time)
    
    print("\n" + "=" * 70)
    print("✓ PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"Output saved: {output_path}")
    print(f"Format: RGB GeoTIFF (3 bands)")
    print(f"  - Black (0,0,0): Background")
    print(f"  - Red (255,0,0): Wildfire detected")
    
    if fire_pixel_count == 0:
        print("\n   NOTE: No fire pixels detected.")
        print("   Consider checking the input image or model parameters.")
    
    sys.exit(0)
