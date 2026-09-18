import torch
import os
import sys
sys.path.append("/home/liese2/SPRI_AI_project/Mobile-UNet")
from Wildfire_models import UNet2D
from torchinfo import summary

PTH_PATH = "weights/model_final.pth"

# Redirigir stdout a archivo en el mismo directorio del script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "model_report.txt")

# Guardar stdout original y redirigir
original_stdout = sys.stdout
sys.stdout = open(output_path, "w")

try:
    # Cargar modelo
    model = UNet2D(in_channels=4, out_channels=2)
    state = torch.load(PTH_PATH, map_location="cpu")
    if isinstance(state, dict):
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        if "model_state_dict" in state:
            state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    # ── 1. Estructura básica ───────────────────────────────────────────────────
    print(model)

    # ── 2. Resumen completo con torchinfo ─────────────────────────────────────
    summary(
        model,
        input_size=(1, 4, 128, 128),
        col_names=["input_size", "output_size", "num_params", "params_percent", "trainable"],
        col_width=20,
        depth=4,
        row_settings=["var_names"],
        verbose=1,
    )
finally:
    sys.stdout.close()
    sys.stdout = original_stdout

print(f"Reporte guardado en: {output_path}")