import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# QAT_VERIFY=1 activa comprobaciones de rango en los quantizadores marcados como
# clamp_free. Cuesta un min()/max() por llamada, asi que solo para depurar.
_QAT_VERIFY = os.getenv("QAT_VERIFY", "").lower() not in ("", "0", "false", "no")


# --------------------------------------------------------------------------- #
# Cuantizacion
# --------------------------------------------------------------------------- #
class FakeQuantSTE(torch.autograd.Function):
    """
    Straight-through estimator con cuantizacion simulada.

    La version original encadenaba  x/scale -> clamp -> round -> *scale,
    creando CUATRO tensores float del tamano completo por llamada. En dec1
    (288 canales expandidos a 256x256) eso son 4 x 72 MB por muestra y por
    quantizador; con 2 quantizadores por bloque y batch 16, 9 GB de transito
    solo en ese bloque.

    Aqui `x_scaled` es un tensor nuevo del que somos duenos, asi que el resto
    de las operaciones van in-place y el pico baja a UN tensor float.
    Numericamente es identico: scale es una potencia exacta de 2, asi que
    multiplicar por 1/scale da el mismo resultado bit a bit que dividir.
    """

    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        x_scaled = x.mul(1.0 / scale)
        mask = x_scaled >= qmin
        mask &= x_scaled <= qmax          # el &= evita un tercer tensor bool
        x_scaled.clamp_(qmin, qmax).round_().mul_(scale)
        ctx.save_for_backward(mask)
        return x_scaled

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask.to(grad_output.dtype), None, None, None


class FakeQuantPassthroughSTE(torch.autograd.Function):
    """
    Variante para los casos en que se puede demostrar que el clamp NUNCA actua.

    Caso concreto: los FakeQuantAct que van justo despues de un ReLU6, con
    signed=False y frac_bits=4. La salida de ReLU6 esta en [0, 6]; con
    scale = 2^-4, x/scale queda en [0, 96], y el rango sin signo de 8 bits es
    [0, 255]. Nunca se satura, asi que la mascara es siempre True y el gradiente
    pasa entero.

    Esto ahorra el tensor bool guardado para el backward, que es 1 byte por
    elemento: en dec1 son 2 x 18 MB por muestra, 600 MB a batch 16.

    Si alguien cambia ReLU6 por ReLU la premisa deja de valer. Corre con
    QAT_VERIFY=1 para que se compruebe en tiempo de ejecucion.
    """

    @staticmethod
    def forward(ctx, x, scale):
        return x.mul(1.0 / scale).round_().mul_(scale)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def q_range(frac_bits=7, signed=True, total_bits=8):
    """Limites enteros del formato."""
    if signed:
        return -(2 ** (total_bits - 1)), 2 ** (total_bits - 1) - 1
    return 0, 2 ** total_bits - 1


def fake_quantize(x, int_bits=0, frac_bits=7, signed=True, total_bits=8):
    """
    NOTA: `int_bits` se acepta por compatibilidad con la firma original pero no
    interviene en el calculo. El rango sale de `frac_bits` y `total_bits`.
    Usa `qformat()` para imprimir el formato Qm.n real.
    """
    scale = 2.0 ** (-frac_bits)
    qmin, qmax = q_range(frac_bits, signed, total_bits)
    return FakeQuantSTE.apply(x, scale, qmin, qmax)


def qformat(frac_bits, signed=True, total_bits=8):
    """(m, n, rango) reales, para documentar el formato de hardware."""
    n = frac_bits
    m = total_bits - frac_bits - (1 if signed else 0)
    qmin, qmax = q_range(frac_bits, signed, total_bits)
    scale = 2.0 ** (-frac_bits)
    return m, n, (qmin * scale, qmax * scale)


class FakeQuantAct(nn.Module):
    def __init__(self, int_bits=0, frac_bits=7, signed=True, total_bits=8,
                 clamp_free=False):
        super().__init__()
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.signed = signed
        self.total_bits = total_bits
        self.clamp_free = clamp_free
        self.scale = 2.0 ** (-frac_bits)
        self.qmin, self.qmax = q_range(frac_bits, signed, total_bits)

    def forward(self, x):
        if self.clamp_free:
            if _QAT_VERIFY:
                lo = float(x.min())
                hi = float(x.max())
                rlo, rhi = self.qmin * self.scale, self.qmax * self.scale
                if lo < rlo - 1e-6 or hi > rhi + 1e-6:
                    raise RuntimeError(
                        f"FakeQuantAct(clamp_free=True) recibio valores en "
                        f"[{lo:.4f}, {hi:.4f}], fuera del rango representable "
                        f"[{rlo:.4f}, {rhi:.4f}]. La premisa de no saturacion ya "
                        f"no se cumple; pon clamp_free=False.")
            return FakeQuantPassthroughSTE.apply(x, self.scale)
        return FakeQuantSTE.apply(x, self.scale, self.qmin, self.qmax)

    def extra_repr(self):
        m, n, rng = qformat(self.frac_bits, self.signed, self.total_bits)
        return (f"declarado=Q{self.int_bits}.{self.frac_bits}, "
                f"real=Q{m}.{n}{'s' if self.signed else 'u'}, "
                f"rango=[{rng[0]:.4f}, {rng[1]:.4f}]"
                + (", clamp_free" if self.clamp_free else ""))


class QATConv2d(nn.Conv2d):
    def __init__(self, *args, weight_frac_bits=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_frac_bits = weight_frac_bits

    def forward(self, x):
        w_q = fake_quantize(self.weight, frac_bits=self.weight_frac_bits, signed=True)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class QATConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, weight_frac_bits=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_frac_bits = weight_frac_bits

    def forward(self, x):
        w_q = fake_quantize(self.weight, frac_bits=self.weight_frac_bits, signed=True)
        return F.conv_transpose2d(x, w_q, self.bias, self.stride, self.padding,
                                  self.output_padding, self.groups, self.dilation)


# --------------------------------------------------------------------------- #
# Bloques
# --------------------------------------------------------------------------- #
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, stride=1):
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.block = nn.Sequential(
            QATConv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            # clamp_free: ReLU6 acota a [0,6] y el rango sin signo llega a 15.94
            FakeQuantAct(int_bits=3, frac_bits=4, signed=False, clamp_free=True),

            QATConv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride,
                      padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            FakeQuantAct(int_bits=3, frac_bits=4, signed=False, clamp_free=True),

            QATConv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # Aqui NO: la salida de BatchNorm no esta acotada y el clamp si actua.
            FakeQuantAct(int_bits=2, frac_bits=5, signed=True),
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class Stem(nn.Module):
    """
    Convolucion inicial 3 -> base_channels.

    Con entrada RGB el primer InvertedResidual expandiria 3 * 6 = 18 canales
    antes de la depthwise, un cuello de botella en la capa con mas informacion
    espacial. MobileNetV2 mete un stem por la misma razon. Tras el stem, enc1
    pasa a ser InvertedResidual(24, 24), lo que ademas activa su residual
    interna (in == out, stride 1).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            QATConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            FakeQuantAct(int_bits=3, frac_bits=4, signed=False, clamp_free=True),
        )

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------------------------------- #
# UNet
# --------------------------------------------------------------------------- #
class UNet2D(nn.Module):
    """
    UNet cuantizado (QAT) para segmentacion de embarcaciones aereas.

    in_channels = 3   (RGB normalizado a [0,1] dividiendo entre 255)
    out_channels = 3  (0=fondo, 1=boat, 2=ship)

    Con entrada de 256x256 el bottleneck queda en 16x16.

    use_checkpoint recomputa el forward de los cuatro bloques mas pesados
    durante el backward en vez de guardar sus activaciones. Cuesta ~30% de
    tiempo y permite subir el batch de 8 a 16-24 en una GPU de 16 GB.
    Los bloques elegidos son los de mayor resolucion, que concentran la memoria:

        dec1  288 canales @ 256^2  ->  72 MB por muestra y tensor
        enc1  144 canales @ 256^2  ->  36 MB
        dec2  576 canales @ 128^2  ->  36 MB
        dec3 1152 canales @  64^2  ->  18 MB
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=24,
                 use_stem=True, use_checkpoint=False):
        super().__init__()

        num = base_channels
        a1, a2, a3, a4, a5 = num, num * 2, num * 4, num * 8, num * 16
        self.use_checkpoint = use_checkpoint

        self.use_stem = use_stem
        if use_stem:
            self.stem = Stem(in_channels, a1)
            enc1_in = a1
        else:
            self.stem = nn.Identity()
            enc1_in = in_channels

        self.enc1 = InvertedResidual(enc1_in, a1)
        self.enc2 = InvertedResidual(a1, a2)
        self.enc3 = InvertedResidual(a2, a3)
        self.enc4 = InvertedResidual(a3, a4)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = InvertedResidual(a4, a5)

        self.upconv4 = QATConvTranspose2d(a5, a4, kernel_size=2, stride=2)
        self.dec4 = InvertedResidual(a5, a4)

        self.upconv3 = QATConvTranspose2d(a4, a3, kernel_size=2, stride=2)
        self.dec3 = InvertedResidual(a4, a3)

        self.upconv2 = QATConvTranspose2d(a3, a2, kernel_size=2, stride=2)
        self.dec2 = InvertedResidual(a3, a2)

        self.upconv1 = QATConvTranspose2d(a2, a1, kernel_size=2, stride=2)
        self.dec1 = InvertedResidual(a2, a1)

        self.out_conv = QATConv2d(a1, out_channels, kernel_size=1)

    def _maybe_ckpt(self, module, x):
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x):
        s = self.stem(x)

        e1 = self._maybe_ckpt(self.enc1, s)          # [B, a1, H,    W   ]
        e2 = self.enc2(self.pool(e1))                # [B, a2, H/2,  W/2 ]
        e3 = self.enc3(self.pool(e2))                # [B, a3, H/4,  W/4 ]
        e4 = self.enc4(self.pool(e3))                # [B, a4, H/8,  W/8 ]

        b = self.bottleneck(self.pool(e4))           # [B, a5, H/16, W/16]

        d4 = self.upconv4(b)
        if e4.shape[2:] != d4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self.upconv3(d4)
        if e3.shape[2:] != d3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self._maybe_ckpt(self.dec3, torch.cat([e3, d3], dim=1))

        d2 = self.upconv2(d3)
        if e2.shape[2:] != d2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self._maybe_ckpt(self.dec2, torch.cat([e2, d2], dim=1))

        d1 = self.upconv1(d2)
        if e1.shape[2:] != d1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self._maybe_ckpt(self.dec1, torch.cat([e1, d1], dim=1))

        return self.out_conv(d1)                     # [B, out_channels, H, W]


VesselNet = UNet2D


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--checkpoint", action="store_true")
    ap.add_argument("--bench", action="store_true",
                    help="Mide la memoria real en GPU para varios batch sizes")
    a = ap.parse_args()

    net = UNet2D(in_channels=3, out_channels=3, use_checkpoint=a.checkpoint)
    x = torch.randn(a.batch, 3, a.size, a.size)
    y = net(x)
    print(f"entrada   : {tuple(x.shape)}")
    print(f"salida    : {tuple(y.shape)}")
    print(f"parametros: {sum(p.numel() for p in net.parameters()):,}")
    print(f"checkpoint: {a.checkpoint}")

    print("\nFormatos de cuantizacion reales:")
    for tag, fb, sg in [("act ReLU6 (unsigned)", 4, False),
                        ("act salida bloque (signed)", 5, True),
                        ("pesos", 7, True)]:
        m, n, rng = qformat(fb, sg)
        print(f"  {tag:<28s} Q{m}.{n}  rango [{rng[0]:.4f}, {rng[1]:.4f}]")
    print("  ReLU6 satura en 6.0 y el formato unsigned llega a 15.94: no hay recorte,")
    print("  por eso esos quantizadores usan clamp_free y no guardan mascara.")

    if a.bench and torch.cuda.is_available():
        print("\nMemoria pico en GPU (forward + backward):")
        net = net.cuda().train()
        crit = torch.nn.CrossEntropyLoss()
        for bs in (4, 8, 16, 24, 32):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                xb = torch.randn(bs, 3, a.size, a.size, device="cuda")
                tb = torch.randint(0, 3, (bs, a.size, a.size), device="cuda")
                crit(net(xb), tb).backward()
                net.zero_grad(set_to_none=True)
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3
                print(f"  batch {bs:3d}: {peak:6.2f} GB")
                del xb, tb
            except torch.OutOfMemoryError:
                print(f"  batch {bs:3d}: OOM")
                torch.cuda.empty_cache()
                break