import torch
import os
import sys

# Apunta al directorio de tu proyecto
sys.path.append("/home/liese2/SPRI_AI_project/Mobile-UNet/Mobile-UNet_1")
from Wildfire_models import UNet2D

PTH_PATH   = "/home/liese2/SPRI_AI_project/Mobile-UNet/Mobile-UNet_1/weights/model_final.pth"
OUTPUT_DIR = "/home/liese2/SPRI_AI_project/Mobile-UNet/Mobile-UNet_1/graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
print("✓ model_final.pth cargado")

# Grafo torchviz
from torchviz import make_dot
x   = torch.randn(1, 4, 256, 256)
y   = model(x)
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render(os.path.join(OUTPUT_DIR, "model_final"), format="png", cleanup=True)
print(f"✓ Grafo guardado en {OUTPUT_DIR}/model_final.png")

# ONNX para Netron (más detallado e interactivo)
torch.onnx.export(
    model,
    x,
    os.path.join(OUTPUT_DIR, "model_final.onnx"),
    opset_version=11,
    input_names=["input"],
    output_names=["segmentation_mask"],
)