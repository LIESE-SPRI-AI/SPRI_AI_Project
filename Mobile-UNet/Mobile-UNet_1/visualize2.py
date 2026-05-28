import netron
import threading
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Iniciar netron en background
def start_netron():
    netron.start("graphs/model_final.onnx", browse=False, port=8080)

t = threading.Thread(target=start_netron, daemon=True)
t.start()
time.sleep(3)  # esperar que cargue

# Capturar con selenium
options = Options()
options.add_argument("--headless")
options.add_argument("--window-size=1920,4000")  # altura grande para el modelo completo

driver = webdriver.Chrome(options=options)
driver.get("http://localhost:8080")
time.sleep(4)  # esperar renderizado

driver.save_screenshot("graphs/model_final_netron.png")
driver.quit()
print("✓ Guardado en graphs/model_final_netron.png")