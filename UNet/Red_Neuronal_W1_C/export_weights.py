"""
export_weights.py

Exporta todos los pesos de WildfireNet (UNet2D) a archivos binarios
float32 que el código C puede leer con fread().

También genera:
  - test_input.bin   : tensor de entrada [1,4,128,128] con valores fijos
  - test_output_py.bin : salida del modelo PyTorch para esa entrada

Uso:
  python export_weights.py --model weights/model_best.pth
  python export_weights.py --random   (pesos aleatorios, para probar sin .pth)

Genera la carpeta  weights_bin/  con un archivo por tensor de pesos.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn


# ── Definición del modelo (copia de Wildfire_models.py) ──────────────────────
class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        super().__init__()
        self.enc1 = DoubleConv2D(in_channels, 64)
        self.enc2 = DoubleConv2D(64,  128)
        self.enc3 = DoubleConv2D(128, 256)
        self.enc4 = DoubleConv2D(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv2D(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4    = DoubleConv2D(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3    = DoubleConv2D(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2    = DoubleConv2D(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128,  64, kernel_size=2, stride=2)
        self.dec1    = DoubleConv2D(128,  64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        import torch.nn.functional as F
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        d4 = self.upconv4(bn)
        if e4.size()[2:] != d4.size()[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([e4, d4], dim=1))
        d3 = self.upconv3(d4)
        if e3.size()[2:] != d3.size()[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([e3, d3], dim=1))
        d2 = self.upconv2(d3)
        if e2.size()[2:] != d2.size()[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))
        d1 = self.upconv1(d2)
        if e1.size()[2:] != d1.size()[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))
        return self.out_conv(d1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_bin(arr: np.ndarray, path: str):
    """Guarda arreglo float32 en binario puro (sin cabecera)."""
    arr.astype(np.float32).flatten().tofile(path)
    print(f"  guardado {path}  shape={arr.shape}  bytes={arr.nbytes}")

def export_double_conv(dc: DoubleConv2D, prefix: str, out_dir: str):
    """
    Exporta los 12 tensores de un bloque DoubleConv:
    conv1.weight, conv1.bias,
    bn1.weight(gamma), bn1.bias(beta), bn1.running_mean, bn1.running_var,
    conv2.weight, conv2.bias,
    bn2.weight(gamma), bn2.bias(beta), bn2.running_mean, bn2.running_var
    """
    seq = dc.double_conv
    # índices en Sequential: 0=Conv, 1=BN, 2=ReLU, 3=Conv, 4=BN, 5=ReLU
    conv1, bn1 = seq[0], seq[1]
    conv2, bn2 = seq[3], seq[4]

    save_bin(conv1.weight.detach().numpy(), f"{out_dir}/{prefix}_conv1_w.bin")
    save_bin(conv1.bias.detach().numpy(),   f"{out_dir}/{prefix}_conv1_b.bin")
    save_bin(bn1.weight.detach().numpy(),   f"{out_dir}/{prefix}_bn1_gamma.bin")
    save_bin(bn1.bias.detach().numpy(),     f"{out_dir}/{prefix}_bn1_beta.bin")
    save_bin(bn1.running_mean.numpy(),      f"{out_dir}/{prefix}_bn1_mean.bin")
    save_bin(bn1.running_var.numpy(),       f"{out_dir}/{prefix}_bn1_var.bin")

    save_bin(conv2.weight.detach().numpy(), f"{out_dir}/{prefix}_conv2_w.bin")
    save_bin(conv2.bias.detach().numpy(),   f"{out_dir}/{prefix}_conv2_b.bin")
    save_bin(bn2.weight.detach().numpy(),   f"{out_dir}/{prefix}_bn2_gamma.bin")
    save_bin(bn2.bias.detach().numpy(),     f"{out_dir}/{prefix}_bn2_beta.bin")
    save_bin(bn2.running_mean.numpy(),      f"{out_dir}/{prefix}_bn2_mean.bin")
    save_bin(bn2.running_var.numpy(),       f"{out_dir}/{prefix}_bn2_var.bin")

def export_conv_transpose(ct: nn.ConvTranspose2d, prefix: str, out_dir: str):
    save_bin(ct.weight.detach().numpy(), f"{out_dir}/{prefix}_w.bin")
    if ct.bias is not None:
        save_bin(ct.bias.detach().numpy(), f"{out_dir}/{prefix}_b.bin")
    else:
        # bias cero si no existe
        np.zeros(ct.out_channels, dtype=np.float32).tofile(
            f"{out_dir}/{prefix}_b.bin")

def export_conv1x1(conv: nn.Conv2d, prefix: str, out_dir: str):
    save_bin(conv.weight.detach().numpy(), f"{out_dir}/{prefix}_w.bin")
    if conv.bias is not None:
        save_bin(conv.bias.detach().numpy(), f"{out_dir}/{prefix}_b.bin")
    else:
        np.zeros(conv.out_channels, dtype=np.float32).tofile(
            f"{out_dir}/{prefix}_b.bin")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default=None,
                        help='Ruta al .pth entrenado')
    parser.add_argument('--random', action='store_true',
                        help='Usar pesos aleatorios (no necesita .pth)')
    parser.add_argument('--outdir', default='weights_bin',
                        help='Carpeta de salida (default: weights_bin)')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Tamaño del patch de prueba (default: 128)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Cargar modelo
    model = UNet2D(in_channels=4, out_channels=2)

    if args.model:
        print(f"\nCargando pesos desde {args.model} ...")
        state = torch.load(args.model, map_location='cpu')
        model.load_state_dict(state)
        print("✓ Pesos cargados")
    elif args.random:
        print("\nUsando pesos aleatorios (torch.manual_seed(42))")
        torch.manual_seed(42)
        # pesos ya inicializados aleatoriamente por defecto
    else:
        print("ERROR: indica --model <ruta.pth>  o  --random")
        return

    model.eval()

    # ── Exportar todos los pesos ──────────────────────────────────────────────
    print(f"\nExportando pesos a {args.outdir}/")

    export_double_conv(model.enc1,       "enc1",       args.outdir)
    export_double_conv(model.enc2,       "enc2",       args.outdir)
    export_double_conv(model.enc3,       "enc3",       args.outdir)
    export_double_conv(model.enc4,       "enc4",       args.outdir)
    export_double_conv(model.bottleneck, "bottleneck", args.outdir)

    export_conv_transpose(model.upconv4, "upconv4", args.outdir)
    export_double_conv(model.dec4,       "dec4",    args.outdir)

    export_conv_transpose(model.upconv3, "upconv3", args.outdir)
    export_double_conv(model.dec3,       "dec3",    args.outdir)

    export_conv_transpose(model.upconv2, "upconv2", args.outdir)
    export_double_conv(model.dec2,       "dec2",    args.outdir)

    export_conv_transpose(model.upconv1, "upconv1", args.outdir)
    export_double_conv(model.dec1,       "dec1",    args.outdir)

    export_conv1x1(model.out_conv, "out_conv", args.outdir)

    # ── Generar entrada de prueba y referencia ────────────────────────────────
    S = args.image_size
    print(f"\nGenerando entrada de prueba [{1},{4},{S},{S}] con semilla 0 ...")

    rng = np.random.default_rng(0)
    test_input_np = rng.random((1, 4, S, S)).astype(np.float32)

    save_bin(test_input_np, f"{args.outdir}/test_input.bin")

    # Guardar también shape para que C lo lea fácilmente
    with open(f"{args.outdir}/test_input_shape.txt", "w") as f:
        f.write(f"1 4 {S} {S}\n")

    # Forward pass en PyTorch para obtener referencia numérica
    print("Corriendo forward pass PyTorch para obtener referencia ...")
    with torch.no_grad():
        x = torch.tensor(test_input_np)
        y = model(x)
        # softmax para obtener probabilidades (igual que predict_wildfire.py)
        prob = torch.softmax(y, dim=1)

    ref_logits = y.numpy()       # [1, 2, S, S]
    ref_probs  = prob.numpy()    # [1, 2, S, S]

    save_bin(ref_logits, f"{args.outdir}/test_output_logits.bin")
    save_bin(ref_probs,  f"{args.outdir}/test_output_probs.bin")

    # Guardar también un resumen legible
    fire_prob = ref_probs[0, 1]   # canal 1 = incendio
    print(f"\nResumen de salida PyTorch (canal incendio):")
    print(f"  min={fire_prob.min():.6f}  max={fire_prob.max():.6f}"
          f"  mean={fire_prob.mean():.6f}")
    print(f"  píxeles > 0.5: {(fire_prob > 0.5).sum()}")

    np.savetxt(f"{args.outdir}/test_output_summary.txt",
               [fire_prob.min(), fire_prob.max(), fire_prob.mean(),
                float((fire_prob > 0.5).sum())],
               header="min max mean n_fire_pixels")

    print(f"\n✓ Exportación completada en '{args.outdir}/'")
    print(f"  Archivos generados: {len(os.listdir(args.outdir))}")


if __name__ == '__main__':
    main()
