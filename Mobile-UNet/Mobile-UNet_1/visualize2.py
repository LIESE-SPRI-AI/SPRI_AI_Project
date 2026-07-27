import netron
import threading
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import inspect
print(inspect.signature(netron.start))

PORT = 8080

# Iniciar netron en background
def start_netron():
    try:
        netron.start("graphs/model_final.onnx", browse=False, address=PORT)
    except TypeError:
        netron.start("graphs/model_final.onnx", browse=False)

t = threading.Thread(target=start_netron, daemon=True)
t.start()

import socket

def wait_port(port, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False

if not wait_port(PORT):
    print("Netron no levanto el puerto")
    exit(1)

print(f"Netron activo en el puerto {PORT}")

# Capturar con selenium
options = Options()
options.add_argument("--headless")
# options.add_argument("--no-sandbox")
# options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,4000")  # altura grande para el modelo completo

driver = webdriver.Chrome(options=options)
driver.get(f"http://localhost:{PORT}")
time.sleep(5)  # esperar renderizado

driver.save_screenshot("graphs/model_final_netron.png")
driver.quit()
print("Guardado en graphs/model_final_netron.png")

