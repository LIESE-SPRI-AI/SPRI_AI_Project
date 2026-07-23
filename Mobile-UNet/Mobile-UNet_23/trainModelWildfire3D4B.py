#python trainModelWildfire3D4B.py --epochs 100 --lr 0.001 -b 4 --gpu auto --image-size 128

import argparse
import os
import tempfile
import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
from osgeo import gdal
from Wildfire_models import UNet2D as WildfireNet
import cv2
from PIL import Image
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
if not BASE_DIR:
    print("Warning: BASE_DIR is not set in the environment.")
    exit(0)

parser = argparse.ArgumentParser(description='Wildfire Segmentation Training')
DATA_PATH = os.path.join(BASE_DIR, "Mobile-UNet/Mobile-UNet_19/data")
print(f"Ruta de datos: {DATA_PATH}")

parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=4, type=int, help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--gpu', default="auto", type=str, help='use GPU: True/False/auto (auto detecta disponibilidad)')
parser.add_argument('--image-size', default=128, type=int, help='target image size for resizing')
parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
parser.add_argument('--resume', default='weights/checkpoint.pth', type=str, help='ruta a checkpoint por si falla training')
parser.add_argument('--checkpoint-freq', default=5, type=int, help='guardar checkpoint')
best_iou = 0

def check_gpu_availability():
    print(f"PyTorch version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("CUDA no está disponible en este sistema")
        return False
    
    try:
        print(f"CUDA disponible: Sí")
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        
        device_count = torch.cuda.device_count()
        print(f"Número de GPUs disponibles: {device_count}")
        
        if device_count > 0:
            # Intentar acceso seguro
            try:
                device_name = torch.cuda.get_device_name(0)
                print(f"GPU 0: {device_name}")
                
                # Test pequeño para verificar funcionamiento
                test_tensor = torch.tensor([1.0], device='cuda:0')
                del test_tensor
                torch.cuda.empty_cache()
                print("✓ GPU verificada y funcionando correctamente")
                return True
            except RuntimeError as e:
                print(f"⚠ Error al acceder a GPU: {e}")
                print("⚠ Posible problema de compatibilidad CUDA/drivers")
                return False
        else:
            print("No se detectaron GPUs")
            return False
            
    except Exception as e:
        print(f"Error inesperado: {e}")
        return False

def custom_collate_fn(batch):
    return batch

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_list, target_size=128, augment = False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.target_size = target_size
        self.augment = augment
        self.augmentation = Augmentation() if augment else None

        with open(image_list, 'r') as f:
            self.images = [line.strip() for line in f if line.strip()]
            
        print(f"Cargadas {len(self.images)} imágenes desde {image_list}")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        base_name = os.path.splitext(img_name)[0]
        mask_name = None
        for ext in ['.tif', '.tiff', '.png', '.jpg']:
            potential_mask = base_name + ext
            potential_path = os.path.join(self.masks_dir, potential_mask)
            if os.path.exists(potential_path):
                mask_name = potential_mask
                mask_path = potential_path
                break
        
        if mask_name is None:
            print(f"ERROR: No se encontró máscara para {img_name}")
            return torch.zeros((4, self.target_size, self.target_size)), \
                   torch.zeros((self.target_size, self.target_size), dtype=torch.long)
        
        try:
            img = self.load_and_resize_image(img_path, self.target_size)
            mask = self.load_and_resize_mask(mask_path, self.target_size)
            if self.augment:
                img, mask = self.augmentation(img, mask)
            return img, mask
        except Exception as e:
            print(f"Error cargando {img_name}: {e}")
            return torch.zeros((4, self.target_size, self.target_size)), \
                   torch.zeros((self.target_size, self.target_size), dtype=torch.long)
        
    def load_and_resize_image(self, path, target_size):
        """Cargar imagen TIFF multibanda con GDAL"""
        ds = gdal.Open(path)
        if ds is None:
            raise ValueError(f"No se pudo abrir la imagen: {path}")
            
        bands = ds.RasterCount
        if bands != 4:
            print(f"ADVERTENCIA: La imagen {path} tiene {bands} bandas, se esperaban 4")
        
        height, width = ds.RasterYSize, ds.RasterXSize
        
        if height == target_size and width == target_size:
            image = np.zeros((min(bands, 4), target_size, target_size), dtype=np.float32)
            for b in range(min(bands, 4)):
                band_data = ds.GetRasterBand(b+1).ReadAsArray()
                band_data = band_data.astype(np.float32) / 12500.0
                band_data = np.clip(band_data, 0, 1)
                image[b, :, :] = band_data
            ds = None
            return torch.tensor(image, dtype=torch.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            gdal.Warp(temp_path, ds, width=target_size, height=target_size, 
                     resampleAlg=gdal.GRIORA_Bilinear, format='GTiff')
            
            ds_resized = gdal.Open(temp_path)
            image = np.zeros((min(bands, 4), target_size, target_size), dtype=np.float32)
            
            for b in range(min(bands, 4)):
                band_data = ds_resized.GetRasterBand(b+1).ReadAsArray()
                band_data = band_data.astype(np.float32) / 12500.0
                band_data = np.clip(band_data, 0, 1)
                image[b, :, :] = band_data
            
            ds_resized = None
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        ds = None
        return torch.tensor(image, dtype=torch.float32)
    
    def load_and_resize_mask(self, path, target_size):
        """Cargar máscara y convertir rojo=incendio, negro=no incendio"""
        try:
            mask = cv2.imread(path)
            
            if mask is None:
                mask_img = Image.open(path)
                mask = np.array(mask_img)
            
            if len(mask.shape) == 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            if mask.shape[2] == 3:
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            else:
                mask_rgb = mask
            
            if mask_rgb.shape[0] != target_size or mask_rgb.shape[1] != target_size:
                mask_rgb = cv2.resize(mask_rgb, (target_size, target_size), 
                                     interpolation=cv2.INTER_NEAREST)
            
            red_channel = mask_rgb[:, :, 0]
            mask_binary = np.where(red_channel > 127, 1, 0).astype(np.uint8)
            mask_tensor = torch.tensor(mask_binary, dtype=torch.long)
            
            return mask_tensor
            
        except Exception as e:
            print(f"Error cargando máscara {path}: {e}")
            return torch.zeros((target_size, target_size), dtype=torch.long)

def calculate_iou(pred, target):
    """Calculate Intersection over Union para segmentación binaria"""
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    
    if pred.shape != target.shape:
        min_height = min(pred.shape[1], target.shape[1])
        min_width = min(pred.shape[2], target.shape[2])
        pred = pred[:, :min_height, :min_width]
        target = target[:, :min_height, :min_width]
    
    pred_fire = pred == 1
    target_fire = target == 1
    
    intersection = (pred_fire & target_fire).float().sum()
    union = (pred_fire | target_fire).float().sum()
    
    if union == 0:
        return 0.0
    else:
        return (intersection / union).item()

def check_class_balance(dataset, name="Dataset"):
    """Verificar el balance entre clases"""
    total_pixels = 0
    fire_pixels = 0
    
    num_samples = min(10, len(dataset))
    print(f"\n{name} - Verificando balance de clases en {num_samples} muestras...")
    
    for i in range(num_samples):
        _, mask = dataset[i]
        total_pixels += mask.numel()
        fire_pixels += (mask == 1).sum().item()
    
    if total_pixels > 0:
        fire_percent = (fire_pixels / total_pixels) * 100
        no_fire_percent = 100 - fire_percent
        print(f"{name} - Píxeles de incendio: {fire_percent:.2f}%")
        print(f"{name} - Píxeles sin incendio: {no_fire_percent:.2f}%")
    else:
        print(f"{name} - No se pudieron contar píxeles")

def process_batch_train(model, batch, criterion, device, optimizer):
    """Procesar un batch para entrenamiento"""
    batch_loss = 0
    batch_iou = 0
    n_samples = 0
    
    for i, (input, target) in enumerate(batch):
        input = input.to(device)
        target = target.to(device)
        
        if input.dim() == 3:
            input = input.unsqueeze(0)
        
        if target.dim() == 2:
            target = target.unsqueeze(0)
        
        output = model(input)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss += loss.item()
        
        with torch.no_grad():
            iou = calculate_iou(output, target)
            if not np.isnan(iou):
                batch_iou += iou
        
        n_samples += 1
    
    return batch_loss / n_samples, batch_iou / n_samples if n_samples > 0 else 0

def process_batch_validate(model, batch, device):
    """Procesar un batch para validación"""
    batch_iou = 0
    n_samples = 0
    
    for i, (input, target) in enumerate(batch):
        input = input.to(device)
        target = target.to(device)
        
        if input.dim() == 3:
            input = input.unsqueeze(0)
        
        if target.dim() == 2:
            target = target.unsqueeze(0)
        
        with torch.no_grad():
            output = model(input)
            iou = calculate_iou(output, target)
            if not np.isnan(iou):
                batch_iou += iou
        
        n_samples += 1
    
    return batch_iou / n_samples if n_samples > 0 else 0

def train(train_loader, model, criterion, optimizer, epoch, device, args):
    model.train()
    losses = AverageMeter()
    ious = AverageMeter()
    
    for i, batch in enumerate(train_loader):
        batch_loss, batch_iou = process_batch_train(model, batch, criterion, device, optimizer)
        losses.update(batch_loss, len(batch))
        ious.update(batch_iou, len(batch))
        
        print(f'\r Batch [{i+1}/{len(train_loader)}] '
              f'Loss: {losses.avg:.4f} IOU: {ious.avg:.4f}', end='', flush=True)
    
    return losses.avg, ious.avg

def validate(val_loader, model, epoch, device, args):
    model.eval()
    ious = AverageMeter()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch_iou = process_batch_validate(model, batch, device)
            ious.update(batch_iou, len(batch))
            
    return ious.avg

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

PUSHBULLET_API_TOKEN = os.getenv("PUSHBULLET_API_TOKEN")
if not PUSHBULLET_API_TOKEN:
    print("Warning: PUSHBULLET_API_TOKEN is not set in the environment.")

def enviar_notificacion(mensaje):
    try:
        requests.post(
            "https://api.pushbullet.com/v2/pushes",
            headers = {
                'Access-Token': PUSHBULLET_API_TOKEN,
                'Content-Type': 'application/json'
            },
            json = {
                "type": "note",
                "title": "Alerta de entrenamiento SPRI",
                "body": mensaje
            },
            timeout=10
        )
        print(f"Notificación enviada amush: {mensaje}")
        
    except Exception as e:
        print(f"No me la hagas jochis: {e}")

class Augmentation:
    def __init__(self, p_flip = 0.5, p_rotate = 0.5, p_bright = 0.3, p_noise = 0.2, bright_range = 0.15, noise_std = 0.02):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_bright = p_bright
        self.p_noise = p_noise
        self.bright_range = bright_range
        self.noise_std = noise_std

    def __call__(self, image, mask):
        if torch.rand(1).item() < self.p_flip:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if torch.rand(1).item() < self.p_flip:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])
        if torch.rand(1).item() < self.p_rotate:
            k = torch.randint(1, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[0, 1])
        if torch.rand(1).item() < self.p_bright:
            factor = 1.0 + (torch.rand(image.shape[0], 1, 1) * 2 - 1) * self.bright_range
            image = torch.clamp(image * factor, 0.0, 1.0)
        if torch.rand(1).item() < self.p_noise:
            noise = torch.randn_like(image) * self.noise_std
            image = torch.clamp(image + noise, 0.0, 1.0)
        return image.contiguous(), mask.contiguous()

def main():
    global best_iou
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*60)
    print("VERIFICANDO SISTEMA")
    print("="*60)
    
    # Determinar uso de GPU
    if args.gpu.lower() == "auto" or args.gpu.lower() == "true":
        gpu_available = check_gpu_availability()
        if gpu_available:
            try:
                device = torch.device("cuda:0")
                print(f"\n✓ Usando GPU: {torch.cuda.get_device_name(0)}")
            except:
                device = torch.device("cpu")
                print("\n⚠ No se pudo inicializar GPU. Usando CPU.")
                gpu_available = False
        else:
            device = torch.device("cpu")
            print("\nℕ Usando CPU (GPU no disponible)")
    else:
        device = torch.device("cpu")
        gpu_available = False
        print("\nℕ Modo CPU seleccionado manualmente")
    
    print("\n" + "="*60)
    print("CONFIGURACIÓN DE ENTRENAMIENTO - DETECCIÓN DE INCENDIOS")
    print("="*60)
    print(f"Dispositivo: {device}")
    print(f"Tamaño de imagen: {args.image_size}x{args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Épocas: {args.epochs}")
    print("="*60)
    
    # Crear directorio de pesos
    weights_dir = os.path.join(script_dir, 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    images_dir = os.path.join(DATA_PATH, 'Images')
    masks_dir = os.path.join(DATA_PATH, 'SegmentationClass')
    train_list = os.path.join(DATA_PATH, 'train.txt')
    val_list = os.path.join(DATA_PATH, 'valid.txt')

    print("\nVerificando directorios y archivos...")
    for path, name in [(images_dir, 'Imágenes'),
                       (masks_dir, 'Máscaras'),
                       (train_list, 'Lista train.txt'),
                       (val_list, 'Lista valid.txt')]:
        if not os.path.exists(path):
            print(f"ERROR: No existe {name}: {path}")
            return
        else:
            print(f"✓ {name}: {path}")
    
    train_dataset = SegmentationDataset(images_dir, masks_dir, train_list, args.image_size, augment=True)
    val_dataset = SegmentationDataset(images_dir, masks_dir, val_list, args.image_size, augment=False)
    
    print(f"\nImágenes de entrenamiento: {len(train_dataset)}")
    print(f"Imágenes de validación: {len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: No se cargaron imágenes. Verifica los archivos .txt y las rutas.")
        return
    
    check_class_balance(train_dataset, "Entrenamiento")
    check_class_balance(val_dataset, "Validación")
    
    # Ajustar batch size para CPU si es necesario
    if device.type == 'cpu' and args.batch_size > 2:
        print(f"\n⚠ Reduciendo batch size de {args.batch_size} a 2 para CPU")
        args.batch_size = 2
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=(device.type == 'cuda'))
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
        pin_memory=(device.type == 'cuda'))
    
    # Crear modelo
    model = WildfireNet(in_channels=4, out_channels=2)
    model.to(device)
    print(f"\n✓ Modelo cargado en {device}")
    
    # Loss con pesos de clase
    class_weights = torch.tensor([1.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n Cargando checkpoint de: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_iou = checkpoint['best_iou']
            print(f"Reanudando training desde epoch {start_epoch}, best_iou: {best_iou:.4f}")
        else: 
            print(f"No se encontro checkpoint, carita triste.")
    
    # Archivos de log
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = open(os.path.join(script_dir, f'training_log_{timestamp}.txt'), 'w')
    iou_file = open(os.path.join(script_dir, f'iou_metrics_{timestamp}.txt'), 'w')
    log_file.write("Epoch,Loss,IOU_Train,IOU_Val,Time\n")
    iou_file.write("Epoch,IOU_Train,IOU_Val\n")
    
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO")
    print("="*60 + "\n")

    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):

        start_time = time.time()
        
        train_loss, train_iou = train(train_loader, model, criterion, optimizer, epoch, device, args)
        val_iou = validate(val_loader, model, epoch, device, args)
        
        epoch_time = time.time() - start_time
        
        log_file.write(f"{epoch},{train_loss:.6f},{train_iou:.4f},{val_iou:.4f},{epoch_time:.2f}\n")
        iou_file.write(f"{epoch},{train_iou:.4f},{val_iou:.4f}\n")
        log_file.flush()
        iou_file.flush()
        
        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)

        print(f'\rEpoch [{epoch+1}/{args.epochs}]  '
            f'Loss: {train_loss:.4f}  '
            f'IOU Train: {train_iou:.4f}  '
            f'IOU Val: {val_iou:.4f}  '
            f'Tiempo: {epoch_time:.1f}s'
            + (' <- mejor\n' if is_best else ''))
        
        if is_best:
            torch.save(model.state_dict(), os.path.join(weights_dir, 'model_best.pth'))
            print(f"Modelo mas mejor guardado - IOU Val: {best_iou:.4f}\n")
        
        if (epoch + 1) % args.checkpoint_freq == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_iou': best_iou,
                        }, os.path.join(weights_dir, 'checkpoint.pth'))
    
    log_file.close()
    iou_file.close()
    torch.save(model.state_dict(), os.path.join(weights_dir, 'model_final.pth'))
    
    total_time = time.time() - training_start_time
    horas = int(total_time // 3600)
    minutos = int((total_time % 3600) // 60)
    segundos = int(total_time % 60)

    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"Mejor IOU alcanzado: {best_iou:.4f}")
    print(f"Modelos guardados en: {weights_dir}/")
    print(f"Tiempo total: {horas:02d}h {minutos:02d}m {segundos:02d}s")
    print("="*60)

    enviar_notificacion(f"Entrenamiento Completado! Mejor IOU: {best_iou:.4f}, Tiempo: {horas:02d}h{minutos:02d}m")

if __name__ == '__main__':
    main()
