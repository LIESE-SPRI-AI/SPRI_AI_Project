"""Models for agricultural segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super().__init__()

        expanded_channels = in_channels * expand_ratio
        self.use_residual = (in_channels == out_channels)

        self.block = nn.Sequential(
            # la pointwise
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            # la depthwise
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            # la otra pointwise
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super(DoubleConv, self).__init__()

        self.block1 = InvertedResidual(in_channels, out_channels, expand_ratio)

    def forward(self, x):
        x = self.block1(x)
        return x
    

class UNet2D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        super(UNet2D, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 512 + 512 = 1024
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)   # 256 + 256 = 512
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)   # 128 + 128 = 256
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)    # 64 + 64 = 128
        
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
