import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FakeQuantSTE(torch.autograd.Function): # strainght-through estimator
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, qmin, qmax)
        x_q = torch.round(x_clamped)
        mask = (x_scaled >= qmin) & (x_scaled <= qmax)
        ctx.save_for_backward(mask)
        return x_q * scale

    @staticmethod 
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        grad_input = grad_output * mask.to(grad_output.dtype)
        return grad_input, None, None, None
# detach() pytorch calcula el grafo automaticamente pero hace mas operaciones
# si se usa detach() el grad siempre es 1 
# con autograd.Function se controla mejor que se guarda en ctx ???    

def fake_quantize(x, int_bits=0, frac_bits=7, signed=True, total_bits=8):
    scale = 2.0 ** (-frac_bits)
    if signed:
        qmin, qmax = -(2 ** (total_bits -1)), 2 ** (total_bits -1) - 1
    else:
        qmin, qmax = 0, 2 ** total_bits - 1
    return FakeQuantSTE.apply(x, scale, qmin, qmax)

class FakeQuantAct(nn.Module): # cuantiza activaciones
    def __init__(self, int_bits=0, frac_bits=7, signed=True):
        super().__init__()
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.signed = signed

    def forward(self, x):
        return fake_quantize(x, int_bits=self.int_bits, frac_bits=self.frac_bits, signed=self.signed)

class QATConv2d(nn.Conv2d):
    def __init__(self, *args, weight_frac_bits=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_frac_bits = weight_frac_bits

    def forward(self, x):
        w_q = fake_quantize(self.weight, frac_bits=self.weight_frac_bits, signed=True)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QATConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, weight_frac_bits=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_frac_bits = weight_frac_bits

    def forward(self, x):
        w_q = fake_quantize(self.weight, frac_bits=self.weight_frac_bits, signed=True)
        return F.conv_transpose2d(x, w_q, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
    


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1):
        super().__init__()

        expanded_channels = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.block = nn.Sequential(
            # la pointwise
            QATConv2d(in_channels, expanded_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            FakeQuantAct(int_bits=3, frac_bits=4, signed=False),
            # la depthwise
            QATConv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            FakeQuantAct(int_bits=3, frac_bits=4, signed=False),
            # la otra pointwise
            QATConv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            FakeQuantAct(int_bits=2, frac_bits=5, signed=True)
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

        num = 16 # original dr adan 64, min 64 max 1024
        a1, a2, a3, a4, a5 = num, num*2, num*4, num*8, num*16

        # Encoder (Downsampling)
        self.enc1 = InvertedResidual(in_channels, a1)
        self.enc2 = InvertedResidual(a1, a2)
        self.enc3 = InvertedResidual(a2, a3)
        self.enc4 = InvertedResidual(a3, a4)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = InvertedResidual(a4, a5)
        
        # Decoder (Upsampling)
        self.upconv4 = QATConvTranspose2d(a5, a4, kernel_size=2, stride=2)
        self.dec4 = InvertedResidual(a5, a4)  # 512 + 512 = 1024
        
        self.upconv3 = QATConvTranspose2d(a4, a3, kernel_size=2, stride=2)
        self.dec3 = InvertedResidual(a4, a3)   # 256 + 256 = 512
        
        self.upconv2 = QATConvTranspose2d(a3, a2, kernel_size=2, stride=2)
        self.dec2 = InvertedResidual(a3, a2)   # 128 + 128 = 256
        
        self.upconv1 = QATConvTranspose2d(a2, a1, kernel_size=2, stride=2)
        self.dec1 = InvertedResidual(a2, a1)    # 64 + 64 = 128
        
        # Output layer
        self.out_conv = QATConv2d(a1, out_channels, kernel_size=1)

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
