"""Models for agricultural segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, expand_ratio = 6):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride ==  1 and in_channels == out_channels)
        hidden = int(in_channels * expand_ratio)

        layers = []

        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        
        layers += [ #depthwise
                nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride, padding=1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        
        layers += [ #pointwise?
                nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)
    
class MobileNetV2Encoder(nn.Module):
    # magia negra de claude
    block_cfg = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    def __init__(self, in_channels = 4, width_mult = 1.0):
        super().__init__()

        def c(n):
            return max(8, int(n * width_mult + 4) // 8 * 8)
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, c(32), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU6(inplace=True), 
        )

        in_ch = c(32)

        self.layer1 = self._make_layer(in_ch, c(16), 1, 1, t=1);  in_ch = c(16)
        self.layer2 = self._make_layer(in_ch, c(24), 2, 2, t=6);  in_ch = c(24)
        self.layer3 = self._make_layer(in_ch, c(32), 3, 2, t=6);  in_ch = c(32)
        self.layer4 = self._make_layer(in_ch, c(64), 4, 2, t=6);  in_ch = c(64)
        self.layer5 = self._make_layer(in_ch, c(96), 3, 1, t=6);  in_ch = c(96)
        self.layer6 = self._make_layer(in_ch, c(160), 3, 2, t=6);  in_ch = c(160)
        self.layer7 = self._make_layer(in_ch, c(320), 1, 1, t=6);  in_ch = c(320)

        self.out_channels = {
            "s0": c(32),
            "s1": c(16),
            "s2": c(24),
            "s3": c(32),
            "s4": c(96),
            "s5": c(320),
        }
    
    @staticmethod
    def _make_layer(in_ch, out_ch, n, stride, t):
        layers = [InvertedResidual(in_ch, out_ch, stride=stride, expand_ratio=t)]
        for _ in range(1, n):
            layers.append(InvertedResidual(out_ch, out_ch, stride=1, expand_ratio=t))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        s0 = self.first_conv(x)
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer5(self.layer4(s3))
        s5 = self.layer7(self.layer6(s4))

        return s0, s1, s2, s3, s4, s5

class ConvBnRelu6(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )
    def forward(self, x):
        return self.block(x)
    
class MobileUp(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, expand_ratio = 6):
        super().__init__()
        self.reduce = ConvBnRelu6(in_ch, out_ch, kernel_size=1)
        self.fuse = InvertedResidual(out_ch + skip_ch, out_ch, stride=1, expand_ratio=expand_ratio)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1) #concatener con skip connection
        return self.fuse(x)  

class MobileUNet(nn.Module):
    def __init__(self, in_channels = 4, out_channels = 2, width_mult = 1.0):
        super().__init__()

        self.encoder = MobileNetV2Encoder(in_channels=in_channels, width_mult=width_mult)
        ch = self.encoder.out_channels

        self.up4 = MobileUp(ch["s5"], ch["s4"], ch["s4"]) 
        self.up3 = MobileUp(ch["s4"], ch["s3"], ch["s3"]) 
        self.up2 = MobileUp(ch["s3"], ch["s2"], ch["s2"]) 
        self.up1 = MobileUp(ch["s2"], ch["s1"], ch["s1"]) 
        self.up0 = MobileUp(ch["s1"], ch["s0"], ch["s0"]) 

        self.seg_head = nn.Sequential(
            ConvBnRelu6(ch["s0"], ch["s0"], kernel_size=3),
            nn.Conv2d(ch["s0"], out_channels, kernel_size=1),
        )

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

        s0, s1, s2, s3, s4, s5 = self.encoder(x)

        d4 = self.up4(s5, s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)

        return self.seg_head(d0)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Modelo principal a usar
UNet2D = MobileUNet
WildfireNet = MobileUNet