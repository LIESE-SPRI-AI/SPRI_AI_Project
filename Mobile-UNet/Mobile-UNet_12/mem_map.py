import torch
import os
import sys

sys.path.append("/home/liese2/SPRI_AI_project/Mobile-UNet")
from Wildfire_models import UNet2D
from torchinfo import summary

PTH_PATH = "/home/liese2/SPRI_AI_project/Mobile-UNet/Mobile-UNet_12/weights/model_final.pth"

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

# ── 1. Estructura básica ───────────────────────────────────────────────────────
print(model)

# ── 2. Resumen completo con torchinfo ─────────────────────────────────────────
summary(
    model,
    input_size=(1, 4, 256, 256),
    col_names=["input_size", "output_size", "num_params", "params_percent", "trainable"],
    col_width=20,
    depth=4,
    row_settings=["var_names"],
    verbose=1,
)
