"""Models for agricultural segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1):
        super().__init__()

        expanded_channels = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.block = nn.Sequential(
            # la pointwise
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            # la depthwise
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False),
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

def make_stage(in_channels, out_channels, expand_ratio, n, stride):
    layers = [InvertedResidual(in_channels, out_channels, expand_ratio, stride)]
    for _ in range(n - 1):
        layers.append(InvertedResidual(out_channels, out_channels, expand_ratio, stride=1))
    return nn.Sequential(*layers)

class UNet2D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        super(UNet2D, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        #enconder de chava
        self.d1 = make_stage(32, 16, expand_ratio=1, n=1, stride=1)
        self.d2 = make_stage(16, 24, expand_ratio=6, n=2, stride=2)
        self.d3 = make_stage(24, 32, expand_ratio=6, n=3, stride=2)
        self.d4a = make_stage(32, 64, expand_ratio=6, n=4, stride=2)
        self.d4b = make_stage(64, 96, expand_ratio=6, n=3, stride=1)
        self.d5a = make_stage(96, 160, expand_ratio=6, n=3, stride=2)
        self.d5b = make_stage(160, 320, expand_ratio=6, n=1, stride=1)
        #decoder fresquito
        self.upconv1 = nn.ConvTranspose2d(320, 96, kernel_size=4, stride=2, padding=1)
        self.ir1 = InvertedResidual(96 + 96, 96, expand_ratio=6, stride=1)
        self.upconv2 = nn.ConvTranspose2d(96, 32, kernel_size=4, stride=2, padding=1)
        self.ir2 = InvertedResidual(32 + 32, 32, expand_ratio=6, stride=1)
        self.upconv3 = nn.ConvTranspose2d(32, 24, kernel_size=4, stride=2, padding=1)
        self.ir3 = InvertedResidual(24 + 24, 24, expand_ratio=6, stride=1)
        self.upconv4 = nn.ConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1)
        self.ir4 = InvertedResidual(16 + 16, 16, expand_ratio=6, stride=1)

        self.upconv5 = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.stem(x)
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4b(self.d4a(x3))
        x5 = self.d5b(self.d5a(x4))

        l1 = self._match_and_cat(self.upconv1(x5), x4)
        l2 = self.ir1(l1)
        l3 = self._match_and_cat(self.upconv2(l2), x3)
        l4 = self.ir2(l3)
        l5 = self._match_and_cat(self.upconv3(l4), x2)
        l6 = self.ir3(l5)
        l7 = self._match_and_cat(self.upconv4(l6), x1)
        l8 = self.ir4(l7)

        out = self.upconv5(l8)

        return out
    
    @staticmethod # magia negra de claude
    def _match_and_cat(up, skip):
        if up.size()[2:] != skip.size()[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([up, skip], dim=1)
    
# Modelo principal a usar
WildfireNet = UNet2D
