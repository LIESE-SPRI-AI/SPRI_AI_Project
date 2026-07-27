import torch
import os
import sys

sys.path.append("/home/liese2/SPRI_AI_project/UNet/UWildfire5b1")
from Wildfire_models import UNet2D

PTH_PATH = "/home/liese2/SPRI_AI_project/UNet/UWildfire5b1/weights/model_final.pth"

# Cargar modelo
model = UNet2D(in_channels=5, out_channels=2)
state = torch.load(PTH_PATH, map_location="cpu")

if isinstance(state, dict):
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    if "model_state_dict" in state:
        state = state["model_state_dict"]

model.load_state_dict(state)
model.eval()

# ── Hooks para capturar input/output de cada operación ────────────────────────
operations = []

def make_hook(name):
    def hook(module, input, output):
        in_shape  = list(input[0].shape)  if isinstance(input,  tuple) else list(input.shape)
        out_shape = list(output.shape)
        params    = sum(p.numel() for p in module.parameters(recurse=False))
        dtype     = str(output.dtype).replace("torch.", "")

        # MACs estimados solo para Conv2d y ConvTranspose2d
        macs = 0
        if isinstance(module, torch.nn.Conv2d):
            macs = (params * out_shape[-2] * out_shape[-1]) // module.groups
        elif isinstance(module, torch.nn.ConvTranspose2d):
            macs = params * in_shape[-2] * in_shape[-1]

        operations.append({
            "name":      name,
            "type":      type(module).__name__,
            "in_shape":  in_shape,
            "out_shape": out_shape,
            "params":    params,
            "dtype":     dtype,
            "macs":      macs,
        })
    return hook

# Registrar hooks en capas hoja (sin hijos)
hooks = []
for name, module in model.named_modules():
    if len(list(module.children())) == 0:  # solo capas hoja
        hooks.append(module.register_forward_hook(make_hook(name)))

# Forward pass
x = torch.randn(1, 5, 256, 256)
with torch.no_grad():
    y = model(x)

# Eliminar hooks
for h in hooks:
    h.remove()

# ── Imprimir reporte ───────────────────────────────────────────────────────────
SECTIONS = {
    "enc1":       "ENCODER — bloque 1",
    "enc2":       "ENCODER — bloque 2",
    "enc3":       "ENCODER — bloque 3",
    "enc4":       "ENCODER — bloque 4",
    "pool":       "DOWNSAMPLING",
    "bottleneck": "BOTTLENECK",
    "upconv4":    "DECODER — bloque 4",
    "dec4":       "DECODER — bloque 4",
    "upconv3":    "DECODER — bloque 3",
    "dec3":       "DECODER — bloque 3",
    "upconv2":    "DECODER — bloque 2",
    "dec2":       "DECODER — bloque 2",
    "upconv1":    "DECODER — bloque 1",
    "dec1":       "DECODER — bloque 1",
    "out_conv":   "OUTPUT",
}

current_section = None
total_params = 0
total_macs   = 0

print("=" * 100)
print("DESCRIPCIÓN DE OPERACIONES — UNet2D")
print("=" * 100)
print(f"{'Operación':<40} {'Tipo':<18} {'Input shape':<22} {'Output shape':<22} {'Params':>8} {'MACs':>12} {'Dtype'}")
print("-" * 100)

for op in operations:
    # Detectar sección
    prefix = op["name"].split(".")[0]
    section = SECTIONS.get(prefix)
    if section and section != current_section:
        current_section = section
        print(f"\n  ── {section} {'─' * (60 - len(section))}")

    macs_str   = f"{op['macs']:,}"   if op["macs"] > 0 else "—"
    params_str = f"{op['params']:,}" if op["params"] > 0 else "—"

    print(f"  {op['name']:<38} {op['type']:<18} {str(op['in_shape']):<22} {str(op['out_shape']):<22} {params_str:>8} {macs_str:>12} {op['dtype']}")

    total_params += op["params"]
    total_macs   += op["macs"]

print("\n" + "=" * 100)
print(f"  Total parámetros en operaciones : {total_params:>15,}")
print(f"  Total MACs estimados            : {total_macs:>15,}  ({total_macs / 1e9:.2f} GMACs)")
print(f"  Tipo de dato                    : float32")
print(f"  Input  → [1, 5, 256, 256]")
print(f"  Output → [1, 2, 256, 256]  (2 clases de segmentación)")
print("=" * 100)

from thop import profile
macs, params = profile(model, inputs=(x,))
print(f"MACs exactos: {macs / 1e9:.2f} GMACs — Params: {params / 1e6:.2f} M")