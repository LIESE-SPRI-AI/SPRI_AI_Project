"""Pascal VOC Dataset Segmentation Dataloader"""
import random

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
 
from PIL import Image
from torch.utils.data import DataLoader
from osgeo import gdal
import cv2

gdal.UseExceptions()

WILDFIRE_CLASSES = ('background',  # always index 0
               'wildfire' )

NUM_CLASSES = len(WILDFIRE_CLASSES) 

palettedata = [0,0,0,255,0,0, 0,255,0, 0,0,255, 255,255,0, 255,0,255, 0,255,255, 127,0,0, 0,127,0,  0,0,127, 127,127,0, 127,0,127, 0,127,127]
   
meanB = [0.15265982818134405, 0.12155882904556131, 0.11179684095377862, 0.2149179913310541]
stdB = [0.06175473860122212, 0.04970513459616255, 0.04281088892387709, 0.09208002357648823]

class WildFireDataset(Dataset):
    """Pascal VOC 2007 Dataset"""
    def __init__(self, list_file, img_dir, mask_dir, size=224, augmentation=False,transform=None,normalize=False,bands=4,niv=1):
        # Pquoi size = 224?
        print("Init WildFireDataset size",size)
        self.images = open(list_file, "rt").read().split("\n")[:-1] #revisar
        #self.images = "bloque_0_x1408_y0.tiff"
        self.transform = transform

        #self.img_extension = ".png" 
        self.img_extension = ".tiff" 
        self.mask_extension = ".tiff"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir
        
        self.size=size
        print("Dataset size",size)
    
        self.bands=bands
        print("Dataset bands",bands)
        
        self.normalize=normalize
        print("Dataset Normalize",normalize)
        
        self.augmentation=augmentation
        print("Dataset augmentation",augmentation)
        
        self.niv_aug=niv
        # self.counts = self.__compute_class_probability()
        # print("counts" ,self.counts)
        # print("Prob ",self.get_class_probability())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)
        
        
        if self.augmentation :

            if random.uniform(0, 1) > 0.25 :    #3/4
                ty=random.randint(0,3)  # 0 1 ou 2
                if ty == 0:

                    image[:,:,:]=np.flip(image,1)
                    gt_mask[:,:]=np.flip(gt_mask,0)
                    
                elif ty == 1 :

                    image[:,:,:]=np.flip(image,2)
                    gt_mask[:,:]=np.flip(gt_mask,1)
                elif ty == 2 : # rot 90
                    image[:,:,:]=np.rot90(image,1,(1,2))
                    gt_mask[:,:]=np.rot90(gt_mask,1)  #default (0,1)
                elif ty == 3 :# rot - 90
                    image[:,:,:]=np.rot90(image,3,(1,2))
                    gt_mask[:,:]=np.rot90(gt_mask,3)
                    
                else:    
                    pass
      
            if self.niv_aug > 1:

                if self.niv_aug == 2:
                    max_b = 1
                    ecart = 0.05
                elif self.niv_aug == 3:
                    max_b = 3
                    ecart = 0.1
                elif self.niv_aug == 4:
                    ecart = 0.25

                    max_b = 4

                if random.uniform(0, 1) > 0.5:

                    if max_b == 1:
                        b = random.randint(0, self.bands-1)
                        delta = random.uniform(-ecart, ecart)
                        # print('AP',image.shape,delta)
                        image[b, :, :] += delta
                        image[b, :, :] = np.clip(image[b, :, :], 0, 1)
                    else:
                      # max_b bandes  modifiées au max
                      nb = random.randint(1, max_b)
                      for _ in range(nb):

                        b = random.randint(0, self.bands-1)
                        delta = random.uniform(-ecart, ecart)
                        image[b, :, :] += delta
                        image[b, :, :] = np.clip(image[b, :, :], 0, 1)
 
        if self.normalize :       
            for b in range(4) :
             image  [b,::]=(image[b,::]  -meanB [b] )/stdB [b] 
             
        data = {
                    'image': torch.FloatTensor(image),
                    'mask' : torch.LongTensor(gt_mask)
                    }
        

        return data


    def compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for name in self.images:
            mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)
            print(mask_path)
            
            ds_mask = gdal.Open(mask_path)
            if ds_mask is None:
                print(f"Error: No se pudo abrir {mask_path}")
                continue
                
            mask_array = ds_mask.GetRasterBand(1).ReadAsArray()
            
            if mask_array.shape != (self.size, self.size):
                mask_array = cv2.resize(mask_array, (self.size, self.size), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # CORRECCIÓN: Binarizar correctamente según tipo de dato
            if mask_array.dtype == np.uint16:
                binary_mask = np.where(mask_array > 30000, 1, 0)
            else:
                binary_mask = np.where(mask_array > 20, 1, 0)
            
            # Contar píxeles por clase
            counts[0] += np.sum(binary_mask == 0)  # Background
            counts[1] += np.sum(binary_mask == 1)  # Wildfire
        
        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values/np.sum(values)

        return torch.Tensor(p_values)


    def load_image(self, path=None):
        
        
        #print(path)
        
        ds_img = gdal.Open(path )
        
        #print("NB bands ",ds_img.RasterCount)
        
        if ds_img.RasterCount !=  self.bands :
            print("ERROR   BAND  NUMBER    ",self.bands,ds_img.RasterCount)
            print(path)
            quit()
            
        
        imx_t=np.empty([ds_img.RasterCount,self.size, self.size], dtype=np.float32)
        
        
        for b in range(1,ds_img.RasterCount +1) :
            channel = np.array(ds_img.GetRasterBand(b).ReadAsArray())
            
            if channel.dtype == np.uint16:
                np_float = np.array(channel, dtype=np.float32) / 65535.0  # ✓ Dividir por 65535
            elif channel.dtype == np.uint8:
                np_float = np.array(channel, dtype=np.float32) / 255.0
            else:
                np_float = (channel - channel.min()) / (channel.max() - channel.min())
        
            np_float = np.clip(np_float, 0.0, 1.0)  # Asegurar rango [0,1]
            imx_t[b-1, :, :] = np_float
            

        return imx_t

    def load_mask(self, path=None):
        ds_mask = gdal.Open(path)
        if ds_mask is None:
            raise ValueError(f"No se pudo abrir la máscara: {path}")
    
        # Leer banda R (la que contiene la información)
        mask_array = ds_mask.GetRasterBand(1).ReadAsArray()
    
        # Redimensionar si es necesario
        if mask_array.shape != (self.size, self.size):
            mask_array = cv2.resize(mask_array, (self.size, self.size), 
                               interpolation=cv2.INTER_NEAREST)
    
        # CORRECCIÓN: Umbral apropiado para UInt16
        if mask_array.dtype == np.uint16:
            # Valores esperados: 0 o 65535
            imx_t = np.where(mask_array > 30000, 1, 0)  # ✓ Umbral para UInt16
        else:
            # Fallback para UInt8
            imx_t = np.where(mask_array > 20, 1, 0)
    
        return imx_t

    def get_name(self, index):
       return self.images[index]
   


if __name__ == "__main__":
   
    data_root = "/home/liese2/SPRI_AI_project/Wildfire3" 
    list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
    img_dir = os.path.join(data_root, "Images")
    mask_dir = os.path.join(data_root, "SegmentationClass")

    bands = 4

    objects_dataset = WildFireDataset(list_file=list_file_path,
                                       img_dir=img_dir,
                                       mask_dir=mask_dir,
                                       size=128)
    
    # Calcular pesos de clase
    print("\n" + "="*50)
    print("CALCULANDO PESOS DE CLASE...")
    print("="*50)
    objects_dataset.counts = objects_dataset.compute_class_probability()
    class_weights = objects_dataset.get_class_probability()
    print(f"Distribución de clases: {class_weights}")
    print(f"  Background: {class_weights[0]:.2%}")
    print(f"  Wildfire: {class_weights[1]:.2%}")
    
    # Calcular mean y std por banda
    print("\n" + "="*50)
    print("CALCULANDO MEAN Y STD POR BANDA...")
    print("="*50)
    
    mean = np.zeros(bands, dtype=np.float64)
    std = np.zeros(bands, dtype=np.float64)
    nb_samples = len(objects_dataset)
    
    print(f"Procesando {nb_samples} imágenes...")
    
    for idx, data in enumerate(objects_dataset):
        image = data['image'].cpu().numpy()  # Shape: [bands, 128, 128]
        
        # Calcular mean y std por banda
        for b in range(bands):
            mean[b] += image[b, :, :].mean()
            std[b] += image[b, :, :].std()
        
        # Mostrar progreso cada 100 imágenes
        if (idx + 1) % 100 == 0:
            print(f"  Procesadas {idx + 1}/{nb_samples} imágenes...")
    
    # Promediar sobre todas las imágenes
    mean /= nb_samples
    std /= nb_samples
    
    print("\n" + "="*50)
    print("RESULTADOS FINALES")
    print("="*50)
    print(f"Mean por banda: {mean}")
    print(f"  Banda R (B04): {mean[0]:.6f}")
    print(f"  Banda G (B03): {mean[1]:.6f}")
    print(f"  Banda B (B02): {mean[2]:.6f}")
    print(f"  Banda NIR (B08): {mean[3]:.6f}")
    
    print(f"\nStd por banda: {std}")
    print(f"  Banda R (B04): {std[0]:.6f}")
    print(f"  Banda G (B03): {std[1]:.6f}")
    print(f"  Banda B (B02): {std[2]:.6f}")
    print(f"  Banda NIR (B08): {std[3]:.6f}")
    
    # Guardar resultados en un archivo
    output_stats = os.path.join(data_root, "dataset_statistics.txt")
    with open(output_stats, 'w') as f:
        f.write("="*50 + "\n")
        f.write("ESTADÍSTICAS DEL DATASET\n")
        f.write("="*50 + "\n")
        f.write(f"Total de imágenes: {nb_samples}\n")
        f.write(f"Tamaño de imagen: 128x128\n")
        f.write(f"Número de bandas: {bands}\n\n")
        
        f.write("PESOS DE CLASE:\n")
        f.write(f"  Background: {class_weights[0]:.6f} ({class_weights[0]:.2%})\n")
        f.write(f"  Wildfire: {class_weights[1]:.6f} ({class_weights[1]:.2%})\n\n")
        
        f.write("MEAN POR BANDA:\n")
        f.write(f"meanB = {list(mean)}\n\n")
        
        f.write("STD POR BANDA:\n")
        f.write(f"stdB = {list(std)}\n\n")
        
        f.write("CÓDIGO PARA USAR EN ENTRENAMIENTO:\n")
        f.write(f"meanB = {list(mean)}\n")
        f.write(f"stdB = {list(std)}\n")
    
    print(f"\n✓ Estadísticas guardadas en: {output_stats}")
    
    # Validar un sample
    print("\n" + "="*50)
    print("VALIDACIÓN DE UN SAMPLE")
    print("="*50)
    
    sample = objects_dataset[0]
    image, mask = sample['image'], sample['mask']
    
    print(f"Shape imagen: {image.shape}")  # [4, 128, 128]
    print(f"Shape máscara: {mask.shape}")  # [128, 128]
    print(f"Rango imagen: [{image.min():.4f}, {image.max():.4f}]")
    print(f"Valores únicos máscara: {torch.unique(mask)}")
    print(f"Píxeles de fuego: {(mask == 1).sum().item()} ({(mask == 1).sum().item() / mask.numel() * 100:.2f}%)")
    print(f"Píxeles de fondo: {(mask == 0).sum().item()} ({(mask == 0).sum().item() / mask.numel() * 100:.2f}%)")
    
    print("\n" + "="*50)
    print("✓ PROCESO COMPLETADO")
    print("="*50)
