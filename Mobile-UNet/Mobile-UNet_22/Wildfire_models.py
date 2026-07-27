<<<<<<< HEAD
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
=======
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

    @staticmethod # magia negra de claude
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
        ) #32 64 64
        #enconder de chava
        self.d1 = make_stage(32, 16, expand_ratio=1, n=1, stride=1) # 16 64 64
        self.d2 = make_stage(16, 24, expand_ratio=6, n=2, stride=2) # 24 32 32
        self.d3 = make_stage(24, 32, expand_ratio=6, n=3, stride=2) # 32 16 16
        self.d4a = make_stage(32, 64, expand_ratio=6, n=4, stride=2) # 64 8 8
        self.d4b = make_stage(64, 96, expand_ratio=6, n=3, stride=1) # 96 8 8
        self.d5a = make_stage(96, 160, expand_ratio=6, n=3, stride=2) # 160 4 4
        self.d5b = make_stage(160, 320, expand_ratio=6, n=1, stride=1) # 320 4 4
        self.c1 = nn.Sequential(
            QATConv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            FakeQuantAct(int_bits=4, frac_bits=3, signed=True)
        ) # 1280 4 4

        #decoder de mujer
        self.upconv1 = QATConvTranspose2d(1280, 96, kernel_size=4, stride=2, padding=1) # 96 8 8
        self.upconv1_quant = FakeQuantAct(int_bits=4, frac_bits=3, signed=True) # 96 8 8
        self.ir1 = InvertedResidual(96 + 96, 96, expand_ratio=6, stride=1) # 96 8 8

        self.upconv2 = QATConvTranspose2d(96, 32, kernel_size=4, stride=2, padding=1) # 32 16 16
        self.upconv2_quant = FakeQuantAct(int_bits=4, frac_bits=3, signed=True) # 32 16 16
        self.ir2 = InvertedResidual(32 + 32, 32, expand_ratio=6, stride=1) # 32 16 16

        self.upconv3 = QATConvTranspose2d(32, 24, kernel_size=4, stride=2, padding=1) # 24 32 32
        self.upconv3_quant = FakeQuantAct(int_bits=4, frac_bits=3, signed=True) # 24 32 32
        self.ir3 = InvertedResidual(24 + 24, 24, expand_ratio=6, stride=1) # 24 32 32

        self.upconv4 = QATConvTranspose2d(24, 16, kernel_size=4, stride=2, padding=1) # 16 64 64
        self.upconv4_quant = FakeQuantAct(int_bits=4, frac_bits=3, signed=True) # 16 64 64
        self.ir4 = InvertedResidual(16 + 16, 16, expand_ratio=6, stride=1) # 16 64 64

        self.upconv5 = QATConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1) # 2 128 128

    def forward(self, x):
        # encoder
        x = self.stem(x) # 32 64 64 
        x1 = self.d1(x) # 16 64 64
        x2 = self.d2(x1) # 24 32 32
        x3 = self.d3(x2) # 32 16 16
        x4 = self.d4b(self.d4a(x3)) # 96 8 8
        x5 = self.c1(self.d5b(self.d5a(x4))) # 1280 4 4

        # decoder con skip connections
        l1 = self._match_and_cat(self.upconv1_quant(self.upconv1(x5)), x4) 
        l2 = self.ir1(l1) # 96 8 8
        l3 = self._match_and_cat(self.upconv2_quant(self.upconv2(l2)), x3) 
        l4 = self.ir2(l3) # 32 16 16
        l5 = self._match_and_cat(self.upconv3_quant(self.upconv3(l4)), x2) 
        l6 = self.ir3(l5) # 24 32 32
        l7 = self._match_and_cat(self.upconv4_quant(self.upconv4(l6)), x1)
        l8 = self.ir4(l7) # 16 64 64

        out = self.upconv5(l8) # 2 128 128

        return out
    
    @staticmethod # magia negra de claude
    def _match_and_cat(up, skip):
        if up.size()[2:] != skip.size()[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([up, skip], dim=1)
    
WildfireNet = UNet2D

#funciones equis
#def calc_frac(max_abs_value, signed=True, total_bits=8):
    #avail_bits = (total_bits -1) if signed else total_bits
    #int_bits = max(0, math.ceil(math.log2(max_abs_value + 1e-8))) # ?
    #frac_bits = max(0, avail_bits - int_bits)
#    return frac_bits

>>>>>>> 92a01e3134d8de5c205c83efe0dba0ae3c76c94e
