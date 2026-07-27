import torch
import os
import sys
import socket
import time
import logging

logging.getLogger("torch.onnx").setLevel(logging.WARNING)

sys.path.append("/home/liese2/SPRI_AI_project/Mobile-UNet/Mobile-UNet_5")
from Wildfire_models import UNet2D

PTH_PATH   = "/home/liese2/SPRI_AI_project/Mobile-UNet/Mobile-UNet_5/weights/model_final.pth"
OUTPUT_DIR = "/home/liese2/SPRI_AI_project/Mobile-UNet/Mobile-UNet_5/graphs"
ONNX_PATH  = os.path.join(OUTPUT_DIR, "model_final.onnx")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Cargar modelo ──────────────────────────────────────────────
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

# ── Grafo torchviz ─────────────────────────────────────────────
from torchviz import make_dot
x   = torch.randn(1, 4, 256, 256)
y   = model(x)
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render(os.path.join(OUTPUT_DIR, "model_final"), format="png", cleanup=True)
print(f"✓ Grafo torchviz guardado en {OUTPUT_DIR}/model_final.png")

# ── Exportar ONNX ──────────────────────────────────────────────
torch.onnx.export(
    model, x, ONNX_PATH,
    opset_version=18,
    input_names=["input"],
    output_names=["segmentation_mask"],
    export_params=True,
    do_constant_folding=True,
)
print(f"✓ ONNX guardado en {ONNX_PATH}")

# ── Netron + Selenium ──────────────────────────────────────────
import netron
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Detectar qué argumentos acepta netron.start en esta versión
import inspect
sig = inspect.signature(netron.start)
print(f"netron.start signature: {sig}")

PORT = 8080

def start_netron():
    params = inspect.signature(netron.start).parameters
    if "address" in params:
        netron.start(ONNX_PATH, browse=False, address=PORT)
    elif "port" in params:
        netron.start(ONNX_PATH, browse=False, port=PORT)
    else:
        # Versión antigua: solo acepta (file, browse)
        netron.start(ONNX_PATH, browse=False)

t = threading.Thread(target=start_netron, daemon=True)
t.start()

# Esperar que el servidor esté realmente listo
def wait_for_port(port, timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False

# Si no usa puerto fijo, buscar en puertos comunes de netron
NETRON_PORTS = [8080, 8081, 8082, 5000]
active_port = None

for p in NETRON_PORTS:
    if wait_for_port(p, timeout=5):
        active_port = p
        break

if active_port is None:
    print("✗ No se encontró netron en ningún puerto. Abortando.")
    sys.exit(1)

print(f"✓ Netron activo en puerto {active_port}")

# Capturar screenshot
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=3840,8000")
options.add_argument("--force-device-scale-factor=2")

driver = webdriver.Chrome(options=options)
driver.get(f"http://localhost:{active_port}")
time.sleep(5)

# Fondo blanco vía JS
driver.execute_script("""
    document.body.style.background = 'white';
    document.querySelectorAll('canvas, svg, .graph, .view').forEach(el => {
        el.style.background = 'white';
    });
""")
time.sleep(2)

screenshot_path = os.path.join(OUTPUT_DIR, "model_final_netron.png")
driver.save_screenshot(screenshot_path)
driver.quit()
print(f"✓ Screenshot guardado en {screenshot_path}")