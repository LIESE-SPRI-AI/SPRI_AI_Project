"""
Wildfire_models.py
==================
MbUNet2D: UNet con bloques invertidos (MobileNetV2) y entrenamiento consciente de la
cuantización (QAT) en punto fijo Qm.n.

Formato de punto fijo
---------------------
  total_bits = signo (1 si signed) + int_bits + frac_bits
  int_bits se DERIVA de total_bits y frac_bits (antes el parámetro int_bits existía pero se
  ignoraba y todo quedaba en 8 bits).
  escala = 2^-frac_bits ;  rango signed = [-2^(int_bits), 2^(int_bits) - escala]

Puntos de cuantización (cada uno se puede apagar poniendo None en la configuración):
  entrada   -> antes de la primera convolución
  pesos     -> pesos de Conv2d / ConvTranspose2d
  bias      -> bias de las capas que lo tienen (upconv y capa de salida)
  relu      -> salida de cada ReLU6 (sin signo)
  lineal    -> salida de la proyección 1x1 + BN de cada bloque
  residual  -> después de la suma residual
  upconv    -> salida de cada ConvTranspose2d
  salida    -> logits

Nota FPGA: las BatchNorm NO están plegadas en los pesos. Al exportar hay que plegar BN en
la convolución y volver a cuantizar; los pesos plegados no son exactamente los entrenados.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
IN_CHANNELS = 6          # valor por defecto; el entrenamiento lo calcula: bandas + índices
OUT_CHANNELS = 2         # 0 = no quemado, 1 = quemado
BASE_CHANNELS = 16       # canales del primer nivel; luego x2, x4, x8, x16 (bottleneck)
EXPAND_RATIO = 6         # expansión de los bloques invertidos

# --- Cuantización 16 bits ---
QUANT_BITS = 8
QUANT_REDONDEO = "half_up"   # "half_up": floor(x + 0.5), como suele hacerse en HDL
                             # "nearest_even": torch.round (redondeo bancario, el de la versión anterior)
Q_ENTRADA_FRAC = 12      # Q3.12  con signo   rango [-8, 8)    bandas normalizadas e índices
Q_PESOS_FRAC = 13        # Q2.13  con signo   rango [-4, 4)
Q_RELU_FRAC = 13         # UQ3.13 sin signo   rango [0, 8)     ReLU6 llega a 6
Q_LINEAL_FRAC = 11       # Q4.11  con signo   rango [-16, 16)  proyección, residual, upconv, logits
Q_BIAS_FRAC = 11         # bias en el mismo formato que la salida a la que se suma


def formato(frac_bits, signed=True, bits=QUANT_BITS):
    return {"bits": int(bits), "frac_bits": int(frac_bits), "signed": bool(signed)}


QUANT_16BIT = {
    "redondeo": QUANT_REDONDEO,
    "entrada": formato(Q_ENTRADA_FRAC, signed=True),
    "pesos": formato(Q_PESOS_FRAC, signed=True),
    "bias": formato(Q_BIAS_FRAC, signed=True),
    "relu": formato(Q_RELU_FRAC, signed=False),
    "lineal": formato(Q_LINEAL_FRAC, signed=True),
    "residual": formato(Q_LINEAL_FRAC, signed=True),
    "upconv": formato(Q_LINEAL_FRAC, signed=True),
    "salida": formato(Q_LINEAL_FRAC, signed=True),
}

# Reproduce EXACTAMENTE la versión anterior (8 bits) para cargar checkpoints viejos.
QUANT_8BIT_LEGACY = {
    "redondeo": "nearest_even",
    "entrada": None,
    "pesos": formato(7, signed=True, bits=8),      # Q0.7  [-1, 1)
    "bias": None,
    "relu": formato(4, signed=False, bits=8),      # UQ4.4 [0, 16)  (el código decía int_bits=3)
    "lineal": formato(5, signed=True, bits=8),     # Q2.5  [-4, 4)
    "residual": None,
    "upconv": None,
    "salida": None,
}

# Sin cuantización (punto flotante), útil para comparar.
QUANT_DESACTIVADA = {k: (None if k != "redondeo" else "nearest_even") for k in QUANT_16BIT}

QUANT_CONFIG = QUANT_8BIT_LEGACY       # la que usa un modelo nuevo si no se indica otra


# =============================================================================
# Cuantización
# =============================================================================

def int_bits(fmt):
    return fmt["bits"] - fmt["frac_bits"] - (1 if fmt["signed"] else 0)


def nombre_formato(fmt):
    if fmt is None:
        return "float"
    return f"{'' if fmt['signed'] else 'U'}Q{int_bits(fmt)}.{fmt['frac_bits']}"


def limites(fmt):
    """(qmin, qmax) enteros y escala del formato."""
    b = fmt["bits"]
    if fmt["signed"]:
        qmin, qmax = -(2 ** (b - 1)), 2 ** (b - 1) - 1
    else:
        qmin, qmax = 0, 2 ** b - 1
    return qmin, qmax, 2.0 ** (-fmt["frac_bits"])


def validar_quant(quant):
    for clave, fmt in quant.items():
        if clave == "redondeo" or fmt is None:
            continue
        if int_bits(fmt) < 0:
            raise ValueError(f"Formato inválido en '{clave}': {fmt} (frac_bits mayor que los bits disponibles)")
    if quant.get("redondeo") not in ("half_up", "nearest_even"):
        raise ValueError(f"Redondeo desconocido: {quant.get('redondeo')}")


class FakeQuantSTE(torch.autograd.Function):
    """Cuantiza en el forward; en el backward pasa el gradiente solo dentro del rango (STE)."""

    @staticmethod
    def forward(ctx, x, scale, qmin, qmax, half_up):
        x_scaled = x / scale
        mask = (x_scaled >= qmin) & (x_scaled <= qmax)
        x_clamped = torch.clamp(x_scaled, qmin, qmax)
        x_q = torch.floor(x_clamped + 0.5) if half_up else torch.round(x_clamped)
        x_q = torch.clamp(x_q, qmin, qmax)       # floor(qmax + 0.5) = qmax, por si acaso
        ctx.save_for_backward(mask)
        return x_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask.to(grad_output.dtype), None, None, None, None


def fake_quantize(x, fmt, redondeo="half_up"):
    if fmt is None:
        return x
    qmin, qmax, scale = limites(fmt)
    return FakeQuantSTE.apply(x, scale, qmin, qmax, redondeo == "half_up")


class FakeQuantAct(nn.Module):
    """Cuantiza activaciones. fmt=None -> identidad (sin parámetros: no cambia el state_dict)."""

    def __init__(self, fmt, redondeo="half_up"):
        super().__init__()
        self.fmt = copy.deepcopy(fmt)
        self.redondeo = redondeo

    @property
    def frac_bits(self):
        return None if self.fmt is None else self.fmt["frac_bits"]

    def forward(self, x):
        return fake_quantize(x, self.fmt, self.redondeo)

    def extra_repr(self):
        return nombre_formato(self.fmt)


class QATConv2d(nn.Conv2d):
    def __init__(self, *args, w_fmt=None, b_fmt=None, redondeo="half_up", **kwargs):
        super().__init__(*args, **kwargs)
        self.w_fmt, self.b_fmt, self.redondeo = copy.deepcopy(w_fmt), copy.deepcopy(b_fmt), redondeo

    def forward(self, x):
        w_q = fake_quantize(self.weight, self.w_fmt, self.redondeo)
        b_q = fake_quantize(self.bias, self.b_fmt, self.redondeo) if self.bias is not None else None
        return F.conv2d(x, w_q, b_q, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return super().extra_repr() + f", w={nombre_formato(self.w_fmt)}, b={nombre_formato(self.b_fmt)}"


class QATConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, w_fmt=None, b_fmt=None, redondeo="half_up", **kwargs):
        super().__init__(*args, **kwargs)
        self.w_fmt, self.b_fmt, self.redondeo = copy.deepcopy(w_fmt), copy.deepcopy(b_fmt), redondeo

    def forward(self, x):
        w_q = fake_quantize(self.weight, self.w_fmt, self.redondeo)
        b_q = fake_quantize(self.bias, self.b_fmt, self.redondeo) if self.bias is not None else None
        return F.conv_transpose2d(x, w_q, b_q, self.stride, self.padding, self.output_padding,
                                  self.groups, self.dilation)

    def extra_repr(self):
        return super().extra_repr() + f", w={nombre_formato(self.w_fmt)}, b={nombre_formato(self.b_fmt)}"


# =============================================================================
# Arquitectura
# =============================================================================

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=EXPAND_RATIO, stride=1, quant=None):
        super().__init__()
        q = quant or QUANT_CONFIG
        r = q["redondeo"]
        expanded = in_channels * expand_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # Los índices del Sequential son los mismos que en la versión anterior (compatibilidad de pesos)
        self.block = nn.Sequential(
            # pointwise de expansión
            QATConv2d(in_channels, expanded, kernel_size=1, bias=False, w_fmt=q["pesos"], redondeo=r),
            nn.BatchNorm2d(expanded),
            nn.ReLU6(inplace=True),
            FakeQuantAct(q["relu"], r),
            # depthwise
            QATConv2d(expanded, expanded, kernel_size=3, stride=stride, padding=1, groups=expanded,
                      bias=False, w_fmt=q["pesos"], redondeo=r),
            nn.BatchNorm2d(expanded),
            nn.ReLU6(inplace=True),
            FakeQuantAct(q["relu"], r),
            # pointwise de proyección (lineal)
            QATConv2d(expanded, out_channels, kernel_size=1, bias=False, w_fmt=q["pesos"], redondeo=r),
            nn.BatchNorm2d(out_channels),
            FakeQuantAct(q["lineal"], r),
        )
        self.q_residual = FakeQuantAct(q["residual"], r)

    def forward(self, x):
        if self.use_residual:
            return self.q_residual(x + self.block(x))
        return self.block(x)


def make_stage(in_channels, out_channels, expand_ratio, n, stride, quant=None):
    layers = [InvertedResidual(in_channels, out_channels, expand_ratio, stride, quant)]
    for _ in range(n - 1):
        layers.append(InvertedResidual(out_channels, out_channels, expand_ratio, 1, quant))
    return nn.Sequential(*layers)


class MbUNet2D(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, base_channels=BASE_CHANNELS,
                 expand_ratio=EXPAND_RATIO, quant=None):
        super().__init__()
        quant = copy.deepcopy(quant or QUANT_CONFIG)
        validar_quant(quant)
        self.config = {"in_channels": in_channels, "out_channels": out_channels,
                       "base_channels": base_channels, "expand_ratio": expand_ratio, "quant": quant}
        q, r = quant, quant["redondeo"]
        a1, a2, a3, a4, a5 = (base_channels * m for m in (1, 2, 4, 8, 16))

        def bloque(cin, cout):
            return InvertedResidual(cin, cout, expand_ratio, 1, quant)

        def upconv(cin, cout):
            return QATConvTranspose2d(cin, cout, kernel_size=2, stride=2, w_fmt=q["pesos"], b_fmt=q["bias"],
                                      redondeo=r)

        self.q_entrada = FakeQuantAct(q["entrada"], r)

        # Encoder
        self.enc1 = bloque(in_channels, a1)
        self.enc2 = bloque(a1, a2)
        self.enc3 = bloque(a2, a3)
        self.enc4 = bloque(a3, a4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = bloque(a4, a5)

        # Decoder
        self.upconv4 = upconv(a5, a4)
        self.q_up4 = FakeQuantAct(q["upconv"], r)
        self.dec4 = bloque(a5, a4)
        self.upconv3 = upconv(a4, a3)
        self.q_up3 = FakeQuantAct(q["upconv"], r)
        self.dec3 = bloque(a4, a3)
        self.upconv2 = upconv(a3, a2)
        self.q_up2 = FakeQuantAct(q["upconv"], r)
        self.dec2 = bloque(a3, a2)
        self.upconv1 = upconv(a2, a1)
        self.q_up1 = FakeQuantAct(q["upconv"], r)
        self.dec1 = bloque(a2, a1)

        # Salida
        self.out_conv = QATConv2d(a1, out_channels, kernel_size=1, w_fmt=q["pesos"], b_fmt=q["bias"], redondeo=r)
        self.q_salida = FakeQuantAct(q["salida"], r)

    @staticmethod
    def _unir(skip, up):
        if skip.shape[2:] != up.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return torch.cat([skip, up], dim=1)

    def forward(self, x):
        x = self.q_entrada(x)
        e1 = self.enc1(x)                             # [B, a1, H, W]
        e2 = self.enc2(self.pool(e1))                 # [B, a2, H/2, W/2]
        e3 = self.enc3(self.pool(e2))                 # [B, a3, H/4, W/4]
        e4 = self.enc4(self.pool(e3))                 # [B, a4, H/8, W/8]
        b = self.bottleneck(self.pool(e4))            # [B, a5, H/16, W/16]

        d4 = self.dec4(self._unir(e4, self.q_up4(self.upconv4(b))))
        d3 = self.dec3(self._unir(e3, self.q_up3(self.upconv3(d4))))
        d2 = self.dec2(self._unir(e2, self.q_up2(self.upconv2(d3))))
        d1 = self.dec1(self._unir(e1, self.q_up1(self.upconv1(d2))))
        return self.q_salida(self.out_conv(d1))      # [B, out_channels, H, W]


def config_desde_meta(meta):
    """Argumentos de MbUNet2D a partir de los metadatos de un checkpoint.
    Sin metadatos (pesos viejos) -> 4 canales, 64 base, expansión 6, cuantización 8 bits original."""
    meta = meta or {}
    arq = meta.get("arquitectura") or {}
    return {
        "in_channels": int(meta.get("in_channels", 4)),
        "out_channels": int(meta.get("out_channels", 2)),
        "base_channels": int(arq.get("base_channels", 64)),
        "expand_ratio": int(arq.get("expand_ratio", 6)),
        "quant": meta.get("quant") or QUANT_8BIT_LEGACY,
    }


def modelo_desde_meta(meta):
    return MbUNet2D(**config_desde_meta(meta))


def resumen_quant(quant):
    return ", ".join(f"{k}={nombre_formato(v)}" for k, v in quant.items() if k != "redondeo") \
        + f", redondeo={quant['redondeo']}"


# Modelo principal a usar
WildfireNet = MbUNet2D
