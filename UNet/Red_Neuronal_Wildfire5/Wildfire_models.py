"""Models for agricultural segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv2D, self).__init__()
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

class UNet2D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        super(UNet2D, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = DoubleConv2D(in_channels, 64)
        self.enc2 = DoubleConv2D(64, 128)
        self.enc3 = DoubleConv2D(128, 256)
        self.enc4 = DoubleConv2D(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv2D(512, 1024)
        
        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv2D(1024, 512)  # 512 + 512 = 1024
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv2D(512, 256)   # 256 + 256 = 512
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv2D(256, 128)   # 128 + 128 = 256
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv2D(128, 64)    # 64 + 64 = 128
        
        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 512, H/8, W/8]
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(e4))  # [B, 1024, H/16, W/16]
        
        # Decoder with skip connections
        d4 = self.upconv4(bottleneck)  # [B, 512, H/8, W/8]
        # Asegurar que e4 y d4 tengan el mismo tamaño
        if e4.size()[2:] != d4.size()[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([e4, d4], dim=1)  # [B, 1024, H/8, W/8]
        d4 = self.dec4(d4)               # [B, 512, H/8, W/8]
        
        d3 = self.upconv3(d4)            # [B, 256, H/4, W/4]
        if e3.size()[2:] != d3.size()[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([e3, d3], dim=1)  # [B, 512, H/4, W/4]
        d3 = self.dec3(d3)               # [B, 256, H/4, W/4]
        
        d2 = self.upconv2(d3)            # [B, 128, H/2, W/2]
        if e2.size()[2:] != d2.size()[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([e2, d2], dim=1)  # [B, 256, H/2, W/2]
        d2 = self.dec2(d2)               # [B, 128, H/2, W/2]
        
        d1 = self.upconv1(d2)            # [B, 64, H, W]
        if e1.size()[2:] != d1.size()[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([e1, d1], dim=1)  # [B, 128, H, W]
        d1 = self.dec1(d1)               # [B, 64, H, W]
        
        # Output - asegurar que tenga el mismo tamaño espacial que la entrada
        out = self.out_conv(d1)          # [B, 2, H, W]
        
        return out

# Modelo principal a usar
WildfireNet = UNet2D

def test_operaciones_paso_a_paso():
    """Prueba cada operación de la arquitectura paso a paso"""
    
    print("="*50)
    print("PRUEBA DE OPERACIONES INDIVIDUALES")
    print("="*50)
    
    # Configuración
    batch_size = 2
    in_channels = 4
    out_channels = 2
    height, width = 128, 128
    
    # Crear entrada de prueba
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"\n✅ Entrada creada: shape = {x.shape}")
    
    # Instanciar modelo
    model = UNet2D(in_channels=in_channels, out_channels=out_channels)
    model.eval()  # Modo evaluación
    
    # ============ 1. PRUEBA ENCODER ============
    print("\n" + "="*50)
    print("1. ENCODER (Downsampling)")
    print("="*50)
    
    # Encoder 1
    print(f"\n📌 enc1 (DoubleConv2D {in_channels}→64):")
    e1 = model.enc1(x)
    print(f"   Input: {x.shape} → Output: {e1.shape}")
    
    # MaxPool + Encoder 2
    print(f"\n📌 pool + enc2 (64→128):")
    pooled = model.pool(e1)
    print(f"   After pool: {pooled.shape}")
    e2 = model.enc2(pooled)
    print(f"   After enc2: {e2.shape}")
    
    # MaxPool + Encoder 3
    print(f"\n📌 pool + enc3 (128→256):")
    pooled = model.pool(e2)
    e3 = model.enc3(pooled)
    print(f"   After pool: {pooled.shape} → After enc3: {e3.shape}")
    
    # MaxPool + Encoder 4
    print(f"\n📌 pool + enc4 (256→512):")
    pooled = model.pool(e3)
    e4 = model.enc4(pooled)
    print(f"   After pool: {pooled.shape} → After enc4: {e4.shape}")
    
    # ============ 2. PRUEBA BOTTLENECK ============
    print("\n" + "="*50)
    print("2. BOTTLENECK")
    print("="*50)
    
    print(f"\n📌 pool + bottleneck (512→1024):")
    pooled = model.pool(e4)
    print(f"   After pool: {pooled.shape}")
    bottleneck = model.bottleneck(pooled)
    print(f"   After bottleneck: {bottleneck.shape}")
    
    # ============ 3. PRUEBA DECODER ============
    print("\n" + "="*50)
    print("3. DECODER (Upsampling + Skip Connections)")
    print("="*50)
    
    # Decoder 4
    print(f"\n📌 upconv4 (1024→512):")
    d4_up = model.upconv4(bottleneck)
    print(f"   Input: {bottleneck.shape} → Output: {d4_up.shape}")
    print(f"   Skip connection e4 shape: {e4.shape}")
    
    print(f"\n📌 Ajuste de tamaño (si necesario):")
    if e4.size()[2:] != d4_up.size()[2:]:
        print(f"   ⚠️  Dimensiones diferentes. Aplicando interpolación...")
        d4_up = F.interpolate(d4_up, size=e4.shape[2:], mode='bilinear', align_corners=True)
        print(f"   Después de interpolación: {d4_up.shape}")
    else:
        print(f"   ✅ Dimensiones coinciden")
    
    print(f"\n📌 Concatenación e4 + d4_up:")
    d4_cat = torch.cat([e4, d4_up], dim=1)
    print(f"   e4: {e4.shape}, d4_up: {d4_up.shape} → concatenado: {d4_cat.shape}")
    
    print(f"\n📌 dec4 (1024→512):")
    d4 = model.dec4(d4_cat)
    print(f"   Output: {d4.shape}")
    
    # Decoder 3
    print(f"\n📌 upconv3 (512→256):")
    d3_up = model.upconv3(d4)
    print(f"   Output: {d3_up.shape}")
    print(f"   Skip connection e3 shape: {e3.shape}")
    
    if e3.size()[2:] != d3_up.size()[2:]:
        d3_up = F.interpolate(d3_up, size=e3.shape[2:], mode='bilinear', align_corners=True)
    
    d3_cat = torch.cat([e3, d3_up], dim=1)
    print(f"   Después concatenación: {d3_cat.shape}")
    d3 = model.dec3(d3_cat)
    print(f"   Después dec3: {d3.shape}")
    
    # Decoder 2
    print(f"\n📌 upconv2 (256→128):")
    d2_up = model.upconv2(d3)
    if e2.size()[2:] != d2_up.size()[2:]:
        d2_up = F.interpolate(d2_up, size=e2.shape[2:], mode='bilinear', align_corners=True)
    d2_cat = torch.cat([e2, d2_up], dim=1)
    print(f"   Después dec2: {model.dec2(d2_cat).shape}")
    
    # Decoder 1
    print(f"\n📌 upconv1 (128→64):")
    d1_up = model.upconv1(model.dec2(d2_cat))
    if e1.size()[2:] != d1_up.size()[2:]:
        d1_up = F.interpolate(d1_up, size=e1.shape[2:], mode='bilinear', align_corners=True)
    d1_cat = torch.cat([e1, d1_up], dim=1)
    print(f"   Después dec1: {model.dec1(d1_cat).shape}")
    
    # ============ 4. PRUEBA COMPLETA ============
    print("\n" + "="*50)
    print("4. FORWARD COMPLETO")
    print("="*50)
    
    with torch.no_grad():
        output = model(x)
        print(f"\n✅ Forward pass completo:")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        
        # Verificar dimensiones
        if output.shape[2:] == x.shape[2:]:
            print(f"   ✅ Las dimensiones espaciales se conservan: {output.shape[2:]}")
        else:
            print(f"   ⚠️  Dimensiones cambiaron: input {x.shape[2:]} → output {output.shape[2:]}")
    
    return output


def test_capa_doubleconv():
    """Prueba específica para DoubleConv2D"""
    print("\n" + "="*50)
    print("TEST DOUBLE CONV LAYER")
    print("="*50)
    
    # Probar diferentes configuraciones
    configs = [
        (4, 64, (8, 8)),    # (in, out, size)
        (64, 128, (16, 16)),
        (128, 256, (32, 32)),
        (256, 512, (64, 64))
    ]
    
    for in_ch, out_ch, size in configs:
        x = torch.randn(1, in_ch, size[0], size[1])
        layer = DoubleConv2D(in_ch, out_ch)
        layer.eval()
        
        with torch.no_grad():
            y = layer(x)
            
        print(f"\n📌 DoubleConv2D({in_ch}→{out_ch}):")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {y.shape}")
        
        # Verificar parámetros
        conv1 = layer.double_conv[0]
        conv2 = layer.double_conv[3]
        print(f"   Conv1: {conv1.weight.shape}, bias: {conv1.bias is not None}")
        print(f"   Conv2: {conv2.weight.shape}, bias: {conv2.bias is not None}")


def test_skip_connections():
    """Prueba las conexiones de salto con diferentes tamaños"""
    print("\n" + "="*50)
    print("TEST SKIP CONNECTIONS")
    print("="*50)
    
    # Simular diferentes tamaños espaciales
    sizes = [(128, 128), (129, 129), (127, 127)]
    
    for h, w in sizes:
        print(f"\n📌 Probando con tamaño: ({h}, {w})")
        
        # Asegurar que sea divisible por 16 (requerido para UNet)
        if h % 16 != 0 or w % 16 != 0:
            print(f"   ⚠️  Advertencia: {h}x{w} no es divisible por 16")
            print(f"   UNet requiere tamaños múltiplos de 16")
            continue
            
        x = torch.randn(1, 4, h, w)
        model = UNet2D()
        model.eval()
        
        with torch.no_grad():
            try:
                y = model(x)
                print(f"   ✅ Exitoso: {x.shape} → {y.shape}")
            except Exception as e:
                print(f"   ❌ Error: {e}")


if __name__ == "_main_":
    # Ejecutar todas las pruebas
    test_capa_doubleconv()
    test_operaciones_paso_a_paso()
    test_skip_connections()
    
    print("\n" + "="*50)
    print("✅ TODAS LAS PRUEBAS COMPLETADAS")
    print("="*50)
    
    # Información adicional
    print("\n📊 Resumen de la arquitectura:")
    model = UNet2D()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parámetros totales: {total_params:,}")
    print(f"   Parámetros entrenables: {trainable_params:,}")
    
    # Verificar que la salida tiene el formato correcto
    x_test = torch.randn(1, 4, 128, 128)
    with torch.no_grad():
        y_test = model(x_test)
    print(f"   Shape de salida esperado: [1, 2, 128, 128]")
    print(f"   Shape de salida obtenido: {list(y_test.shape)}")