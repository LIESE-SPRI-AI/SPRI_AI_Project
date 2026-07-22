import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def calc_frac(max_abs_value, signed=True, total_bits=8):
    avail_bits = (total_bits -1) if signed else total_bits
    int_bits = max(0, math.ceil(math.log2(max_abs_value + 1e-8))) # ?
    frac_bits = max(0, avail_bits - int_bits)
    return frac_bits

class FakeQuantSTE(torch.autograd.Function): # strainght-through estimator
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, qmin, qmax)
        x_q = torch.round(x_clamped)
        mask = (x_scaled >= qmin) & (x_scaled <= qmax)
        ctx.save_for_backward(mask)
        return x_q * scale

    @staticmethod # magia negra de claude ???
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        grad_input = grad_output * mask.to(grad_output.dtype)
        return grad_input, None, None, None
    
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
        
        self.input_quant = FakeQuantAct(int_bits=0, frac_bits=8, signed=False)
        self.stem = nn.Sequential(
            QATConv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            FakeQuantAct(int_bits=3, frac_bits=4, signed=False)
        ) # 32 64 64

        #enconder de chava
        self.d1 = make_stage(32, 16, expand_ratio=1, n=1, stride=1) # 16 64 64
        self.d2 = make_stage(16, 24, expand_ratio=6, n=2, stride=2) # 24 32 32
        self.d3 = make_stage(24, 32, expand_ratio=6, n=3, stride=2) # 32 16 16
        self.d4a = make_stage(32, 64, expand_ratio=6, n=4, stride=2) # 64 8 8
        self.d4b = make_stage(64, 96, expand_ratio=6, n=3, stride=1) # 96 8 8
        self.d5a = make_stage(96, 160, expand_ratio=6, n=3, stride=2) # 160 4 4
        self.d5b = make_stage(160, 320, expand_ratio=6, n=1, stride=1) # 320 4 4

        #decoder de mujer
        self.upconv1 = QATConvTranspose2d(320, 96, kernel_size=4, stride=2, padding=1)
        self.upconv1_quant = FakeQuantAct(int_bits=4, frac_bits=3, signed=True)
        self.ir1 = InvertedResidual(96 + 96, 96, expand_ratio=6, stride=1)

        self.upconv2 = QATConvTranspose2d(96, 32, kernel_size=4, stride=2, padding=1)
        self.upconv2_quant = FakeQuantAct(int_bits=4, frac_bits=3, signed=True)
        self.ir2 = InvertedResidual(32 + 32, 32, expand_ratio=6, stride=1)

        self.upconv3 = QATConvTranspose2d(32, 24, kernel_size=4, stride=2, padding=1)
        self.upconv3_quant = FakeQuantAct(int_bits=4, frac_bits=3, signed=True)
        self.ir3 = InvertedResidual(24 + 24, 24, expand_ratio=6, stride=1)

        self.upconv4 = QATConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1)
        self.upconv4_quant = FakeQuantAct(int_bits=4, frac_bits=3, signed=True)
        self.ir4 = InvertedResidual(16 + 16, 16, expand_ratio=6, stride=1)

        self.upconv5 = QATConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # encoder
        x = self.stem(x) # 32 64 64
        x1 = self.d1(x) # 16 64 64
        x2 = self.d2(x1) # 24 32 32
        x3 = self.d3(x2) # 32 16 16
        x4 = self.d4b(self.d4a(x3)) # 96 8 8
        x5 = self.d5b(self.d5a(x4)) # 320 4 4

        # decoder con skip connections
        l1 = self._match_and_cat(self.upconv1_quant(self.upconv1(x5)), x4)
        l2 = self.ir1(l1) # 16x16x96
        l3 = self._match_and_cat(self.upconv2_quant(self.upconv2(l2)), x3)
        l4 = self.ir2(l3) # 32x32x32
        l5 = self._match_and_cat(self.upconv3_quant(self.upconv3(l4)), x2)
        l6 = self.ir3(l5) # 64x64x24
        l7 = self._match_and_cat(self.upconv4_quant(self.upconv4(l6)), x1)
        l8 = self.ir4(l7) # 128x128x16

        out = self.upconv5(l8) # 256x256x2 

        return out
    
    @staticmethod # magia negra de claude
    def _match_and_cat(up, skip):
        if up.size()[2:] != skip.size()[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([up, skip], dim=1)
    
# Modelo principal a usar
WildfireNet = UNet2D

#funciones equis
