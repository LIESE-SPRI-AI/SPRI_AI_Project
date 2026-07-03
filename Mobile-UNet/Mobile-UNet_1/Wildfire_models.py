"""Models for agricultural segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size = 3, 
            stride = stride,
            padding=1,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class MobileDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseConv(in_channels, out_channels),
            DepthwiseConv(out_channels, out_channels),
        )

    def forward(self, x):
        return self.conv(x)
    
class MobileUp(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), # con convtransposed
            nn.BatchNorm2d(out_channels), #quitar
            nn.ReLU6(inplace=True) # quitar
        )
        self.double_conv = MobileDoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True) #upsample
        x = self.up_conv(x) #reducir canales
        x = torch.cat([skip, x], dim=1) #concatener con skip connection
        return self.double_conv(x)  #conv
    

class MobileUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, width_mult=1.0):
        super().__init__()
        def c(n): return max(1, int(n * width_mult))

        self.enc1 = MobileDoubleConv(in_channels, c(16))
        self.enc2 = MobileDoubleConv(c(16), c(32))
        self.enc3 = MobileDoubleConv(c(32), c(64))
        self.enc4 = MobileDoubleConv(c(64), c(128))

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = MobileDoubleConv(c(128), c(256))

        self.up4 = MobileUp(c(256), c(128), c(128)) 
        self.up3 = MobileUp(c(128), c(64), c(64)) 
        self.up2 = MobileUp(c(64), c(32), c(32)) 
        self.up1 = MobileUp(c(32), c(16), c(16)) 

        self.out_conv = nn.Conv2d(c(16), out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        return self.out_conv(d1)


# Modelo principal a usar
UNet2D = MobileUNet
WildfireNet = MobileUNet