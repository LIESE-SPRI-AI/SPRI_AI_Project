#!/usr/bin/env python3
"""
trainModelWildfire3D4B.py
=========================
Entrenamiento de MbUNet2D (Wildfire_models.py) para segmentar área quemada.

Toda la configuración está en el bloque CONFIGURACIÓN de abajo (no hay argumentos por
terminal). Las rutas parten de BASE_DIR definido en el archivo .env.

  python3 trainModelWildfire3D4B.py          (o ./run_training.sh para reintentar si se cae)

Cada corrida se guarda en OUT_DIR/<fecha_hora>/:
  best.pth   mejor modelo en validación + metadatos (bandas, índices, normalización,
             cuantización, umbral, tamaño, métricas). predict_segmentation.py lee todo de aquí.
  last.pth   estado completo para reanudar (modelo, optimizador, scheduler, época)
  final.pth  modelo de la última época (su existencia marca la corrida como terminada)
  log.csv    métricas por época      config.json   parámetros

Novedades de esta versión:
  * Entrada = bandas (BANDAS_MODELO) + índices espectrales calculados al vuelo (INDICES: NDVI, BAI...).
  * Aumentaciones sobre DN (valores físicos) ANTES de calcular los índices: así el NDVI de una
    imagen aumentada es coherente con sus bandas. Antes el brillo/ruido se aplicaba ya normalizado
    (y con z-score eso movía la media de forma incorrecta).
  * Aumentaciones nuevas: escala aleatoria, bruma (offset atmosférico por banda), desenfoque.
  * Cuantización 16 bits (formatos Qm.n declarados abajo) guardada en el checkpoint.
  * REANUDAR = "auto": si la corrida anterior se cayó (no tiene final.pth) se reanuda desde su
    last.pth. Antes run_training.sh reintentaba desde cero en cada caída.
"""

import csv
import glob
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import wildfire_common as wc
import Wildfire_models as wm

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# --- Rutas (BASE_DIR viene del .env) ---
DATA_DIR = wc.ruta_proyecto("Mobile-UNet/Mobile-UNet_27", "data")   # debe ser el OUTPUT_DIR del generador
IMAGES_SUBDIR = "Images"
MASKS_SUBDIR = "SegmentationClass"
TRAIN_LIST = "train.txt"
VAL_LIST = "valid.txt"
OUT_DIR = os.path.join(wc.DIR_CODIGO, "runs")
RUN_NAME = None              # None = fecha_hora
REANUDAR = "auto"            # None | "auto" (última corrida sin final.pth en OUT_DIR) | ruta a un last.pth
PESOS_INICIALES = None       # ruta a .pth para afinar (sin reanudar optimizador); requiere mismos canales

# --- Entrada y salida del modelo ---
BANDAS_MODELO = ("B04", "B03", "B02", "B08")   # R, G, B, NIR. Con SWIR en el dataset: + ("B11", "B12")
INDICES = ("NDVI", "BAI")                      # canales extra; opciones en wc.INDICES_DISPONIBLES
                                               # (NBR, NBR2, NDMI requieren B11/B12 -> no los da la cámara RGB+IR)
IN_CHANNELS = len(BANDAS_MODELO) + len(INDICES)
OUT_CHANNELS = 2                               # 0 = no quemado, 1 = quemado
IMAGE_SIZE = 128                               # recorte de entrada (no se reescala)

# --- Arquitectura ---
BASE_CHANNELS = 64           # canales del primer nivel (x2, x4, x8, x16 en los siguientes)
EXPAND_RATIO = 6

# --- Cuantización (punto fijo Qm.n; int_bits = BITS - FRAC - signo) ---
QUANT_ACTIVA = True
QUANT_BITS = 16
QUANT_REDONDEO = "half_up"   # "half_up" (floor(x+0.5), típico en HDL) | "nearest_even"
Q_ENTRADA_FRAC = 12          # Q3.12  con signo   [-8, 8)
Q_PESOS_FRAC = 13            # Q2.13  con signo   [-4, 4)
Q_RELU_FRAC = 13             # UQ3.13 sin signo   [0, 8)
Q_LINEAL_FRAC = 11           # Q4.11  con signo   [-16, 16)  proyección, residual, upconv, logits
Q_BIAS_FRAC = 11             # Q4.11  con signo

# --- Optimización ---
EPOCHS = 1000
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
GRAD_CLIP = 5.0              # 0 = sin recorte
PATIENCE = 100               # épocas sin mejora antes de parar (0 = desactivado)
SEED = 42
GPU = "auto"                 # "auto" | "true" | "false"
WORKERS = 4
CACHE = False                # cargar todo el dataset en RAM
PRINT_FREQ = 10              # cada cuántos batches actualizar la línea de progreso

# --- Pérdida y normalización ---
LOSS = "ce_dice"             # "ce" | "ce_dice"
DICE_WEIGHT = 1.0
POS_WEIGHT = "auto"          # "auto" o número (peso de la clase quemado en CE)
NORM = "div"                 # "div": DN / NORM_DIV recortado a [0,1] (cómodo en punto fijo) | "zscore"
NORM_DIV = 12500.0

# --- Aumentaciones (solo entrenamiento; se aplican en DN antes de calcular índices) ---
AUGMENT = False
AUG_P_FLIP = 0.5             # volteo horizontal y vertical (cada uno)
AUG_P_ROT90 = 0.5            # rotación 90/180/270
AUG_P_ESCALA = 0.3           # reescalado aleatorio (simula otra altura / resolución)
AUG_ESCALA_RANGO = (0.8, 1.25)
AUG_P_BRILLO = 0.5           # ganancia global + variación por banda
AUG_BRILLO_RANGO = 0.15      # ±15 %
AUG_JITTER_BANDA = 0.05      # ±5 % por banda (conserva casi intactas las razones entre bandas)
AUG_P_BRUMA = 0.2            # offset aditivo tipo bruma/atmósfera, mayor en azul
AUG_BRUMA_MAX_DN = 300.0     # offset máximo en B02 (azul); las demás bandas según AUG_BRUMA_FACTOR
AUG_BRUMA_FACTOR = {"B02": 1.0, "B03": 0.7, "B04": 0.5, "B08": 0.2, "B8A": 0.15, "B11": 0.05, "B12": 0.03}
AUG_P_RUIDO = 0.3            # ruido gaussiano del sensor
AUG_RUIDO_STD_DN = 60.0
AUG_P_DESENFOQUE = 0.15      # desenfoque gaussiano (óptica / movimiento)
AUG_DESENFOQUE_SIGMA = (0.5, 1.2)

UMBRALES = [round(0.1 * i, 1) for i in range(1, 10)]   # 0.1 ... 0.9 (búsqueda en validación)
BANDAS_DATASET_VIEJO = ("B04", "B03", "B02", "B08")     # si el dataset no tiene config.json
# =============================================================================


CODIGO_ERROR_CONFIG = 2   # run_training.sh no reintenta con este código (no tiene caso reintentar)


def abortar(mensaje):
    print(mensaje, file=sys.stderr)
    sys.exit(CODIGO_ERROR_CONFIG)


def config_quant():
    if not QUANT_ACTIVA:
        return dict(wm.QUANT_DESACTIVADA)
    f = lambda frac, signed: wm.formato(frac, signed, QUANT_BITS)
    return {
        "redondeo": QUANT_REDONDEO,
        "entrada": f(Q_ENTRADA_FRAC, True),
        "pesos": f(Q_PESOS_FRAC, True),
        "bias": f(Q_BIAS_FRAC, True),
        "relu": f(Q_RELU_FRAC, False),
        "lineal": f(Q_LINEAL_FRAC, True),
        "residual": f(Q_LINEAL_FRAC, True),
        "upconv": f(Q_LINEAL_FRAC, True),
        "salida": f(Q_LINEAL_FRAC, True),
    }


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def orden_bandas_dataset(data_dir):
    ruta = os.path.join(data_dir, "config.json")
    if os.path.isfile(ruta):
        with open(ruta, encoding="utf-8") as f:
            orden = json.load(f).get("orden_bandas")
        if orden:
            return tuple(orden)
    return BANDAS_DATASET_VIEJO


def buscar_mascara(dir_masks, nombre_img):
    base = os.path.splitext(nombre_img)[0]
    for ext in (".tif", ".tiff", ".png", ".jpg"):
        ruta = os.path.join(dir_masks, base + ext)
        if os.path.exists(ruta):
            return ruta
    return None


def _u(a, b):
    return a + (b - a) * torch.rand(1).item()


def _p(p):
    return p > 0 and torch.rand(1).item() < p


class Aumentacion:
    """
    Aumentaciones sobre DN (C,H,W float32) + máscara (H,W uint8).
    Usa el RNG de torch (cada worker del DataLoader tiene su propia semilla).
    Los píxeles sin dato (todas las bandas en 0) siguen en 0 para no inventar datos.
    """

    def __init__(self, orden_bandas):
        self.bruma = np.array([AUG_BRUMA_FACTOR.get(b, 0.0) for b in orden_bandas], np.float32)[:, None, None]

    @staticmethod
    def escala(img, msk):
        if not _p(AUG_P_ESCALA):
            return img, msk
        s = _u(*AUG_ESCALA_RANGO)
        _, h, w = img.shape
        nh, nw = max(8, int(round(h * s))), max(8, int(round(w * s)))
        hwc = np.ascontiguousarray(img.transpose(1, 2, 0))
        if hwc.shape[2] <= 4:
            r = cv2.resize(hwc, (nw, nh), interpolation=cv2.INTER_LINEAR)
            r = r[..., None] if r.ndim == 2 else r
        else:  # cv2.resize con muchos canales: por banda
            r = np.stack([cv2.resize(hwc[..., c], (nw, nh), interpolation=cv2.INTER_LINEAR)
                          for c in range(hwc.shape[2])], axis=-1)
        img = np.ascontiguousarray(r.transpose(2, 0, 1))
        msk = cv2.resize(msk, (nw, nh), interpolation=cv2.INTER_NEAREST)
        return img, msk

    def __call__(self, img, msk):
        # --- Geométricas (grupo diedral completo: 8 orientaciones) ---
        if _p(AUG_P_FLIP):
            img, msk = img[:, :, ::-1], msk[:, ::-1]
        if _p(AUG_P_FLIP):
            img, msk = img[:, ::-1, :], msk[::-1, :]
        if _p(AUG_P_ROT90):
            k = int(torch.randint(1, 4, (1,)).item())
            img, msk = np.rot90(img, k, axes=(1, 2)), np.rot90(msk, k, axes=(0, 1))
        img = np.ascontiguousarray(img, dtype=np.float32)
        msk = np.ascontiguousarray(msk)

        # --- Fotométricas en DN ---
        validos = (img != 0).any(axis=0)
        if _p(AUG_P_BRILLO):
            g = 1.0 + _u(-AUG_BRILLO_RANGO, AUG_BRILLO_RANGO)
            b = 1.0 + (torch.rand(img.shape[0], 1, 1).numpy() * 2 - 1) * AUG_JITTER_BANDA
            img = img * (g * b).astype(np.float32)
        if _p(AUG_P_BRUMA):
            img = img + _u(0.0, AUG_BRUMA_MAX_DN) * self.bruma
        if _p(AUG_P_DESENFOQUE):
            sigma = _u(*AUG_DESENFOQUE_SIGMA)
            img = np.stack([cv2.GaussianBlur(c, (0, 0), sigma) for c in img], axis=0)
        if _p(AUG_P_RUIDO):
            img = img + torch.randn(img.shape).numpy() * AUG_RUIDO_STD_DN
        img = np.maximum(img, 0.0).astype(np.float32)
        img[:, ~validos] = 0.0
        # un píxel válido no debe quedar en 0 en todas las bandas (se confundiría con "sin dato")
        img[0][validos & (img == 0).all(axis=0)] = 1.0
        return img, msk


class WildfireCropDataset(Dataset):
    def __init__(self, data_dir, lista, orden_dataset, norm, entrenamiento):
        self.dir_img = os.path.join(data_dir, IMAGES_SUBDIR)
        self.dir_mask = os.path.join(data_dir, MASKS_SUBDIR)
        self.size = IMAGE_SIZE
        self.norm = norm
        self.entrenamiento = entrenamiento
        # bandas a leer (las del modelo + las que piden los índices), en ese orden
        self.orden = wc.bandas_requeridas(BANDAS_MODELO, INDICES)
        faltan = [b for b in self.orden if b not in orden_dataset]
        if faltan:
            abortar(f"ERROR: el dataset ({list(orden_dataset)}) no tiene las bandas {faltan} que piden "
                     f"BANDAS_MODELO/INDICES. Regenera el dataset con esas bandas o quítalas.")
        self.n_dataset = len(orden_dataset)
        self.sel = [list(orden_dataset).index(b) for b in self.orden]
        self.augment = Aumentacion(self.orden) if (entrenamiento and AUGMENT) else None

        with open(os.path.join(data_dir, lista)) as f:
            nombres = [l.strip() for l in f if l.strip()]
        self.pares, faltan = [], []
        for n in nombres:
            img = os.path.join(self.dir_img, n)
            msk = buscar_mascara(self.dir_mask, n)
            if os.path.exists(img) and msk is not None:
                self.pares.append((img, msk))
            else:
                faltan.append(n)
        if faltan:
            print(f"⚠ {lista}: {len(faltan)} imágenes sin archivo o sin máscara (se omiten). Ej: {faltan[:3]}")
        print(f"{lista}: {len(self.pares)} pares imagen/máscara")

        self.cache = None
        if CACHE:
            t0 = time.time()
            self.cache = [self._leer(i) for i in range(len(self.pares))]
            mb = sum(a.nbytes + m.nbytes for a, m in self.cache) / 1e6
            print(f"  en RAM: {mb:.0f} MB ({time.time() - t0:.0f}s)")

    def __len__(self):
        return len(self.pares)

    def _leer(self, i):
        ruta_img, ruta_msk = self.pares[i]
        arr, _ = wc.leer_raster(ruta_img)
        if arr.shape[0] < self.n_dataset:
            raise ValueError(f"{ruta_img}: {arr.shape[0]} bandas, el dataset declara {self.n_dataset}")
        img = arr[self.sel]
        if img.dtype.kind == "f":
            img = img.astype(np.float32)
        msk = wc.cargar_mascara_entrenamiento(ruta_msk)
        if msk.shape != img.shape[1:]:
            raise ValueError(f"Tamaños distintos imagen {img.shape[1:]} / máscara {msk.shape}: {ruta_img}")
        return img, msk

    def leer_mascara(self, i):
        return self.cache[i][1] if self.cache is not None else wc.cargar_mascara_entrenamiento(self.pares[i][1])

    def leer_imagen_dn(self, i):
        img = self.cache[i][0] if self.cache is not None else self._leer(i)[0]
        return img.astype(np.float32)

    def _ajustar(self, img, msk):
        """Recorta (aleatorio en train, centrado en val) o rellena hasta self.size."""
        s = self.size
        _, h, w = img.shape
        if h > s or w > s:
            if self.entrenamiento:
                y = int(torch.randint(0, max(h - s, 0) + 1, (1,)).item())
                x = int(torch.randint(0, max(w - s, 0) + 1, (1,)).item())
            else:
                y, x = max(h - s, 0) // 2, max(w - s, 0) // 2
            img, msk = img[:, y:y + s, x:x + s], msk[y:y + s, x:x + s]
            _, h, w = img.shape
        if h < s or w < s:
            img = np.pad(img, ((0, 0), (0, s - h), (0, s - w)))
            msk = np.pad(msk, ((0, s - h), (0, s - w)), constant_values=wc.IGNORE_INDEX)
        return img, msk

    def __getitem__(self, i):
        img, msk = self.cache[i] if self.cache is not None else self._leer(i)
        img = img.astype(np.float32)          # copia: nunca modificar el caché
        if self.augment is not None:
            img, msk = self.augment.escala(img, msk)
        img, msk = self._ajustar(img, msk)
        if self.augment is not None:
            img, msk = self.augment(img, msk)
        nodata = (img == 0).all(axis=0)
        x = wc.construir_entrada(img, self.orden, BANDAS_MODELO, INDICES, self.norm, nodata)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(msk.astype(np.int64))


def estadisticas_bandas(ds, max_imgs=1000, seed=0):
    """Media y desviación por banda del modelo (en DN) sobre píxeles válidos de una muestra del train."""
    n_b = len(BANDAS_MODELO)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    suma = np.zeros(n_b, np.float64)
    suma2 = np.zeros(n_b, np.float64)
    n = 0
    for i in idx[:max_imgs]:
        img = ds.leer_imagen_dn(i)[:n_b].reshape(n_b, -1)   # las bandas del modelo van primero
        validos = (img != 0).any(axis=0)
        v = img[:, validos].astype(np.float64)
        suma += v.sum(axis=1)
        suma2 += (v ** 2).sum(axis=1)
        n += v.shape[1]
    media = suma / max(n, 1)
    desv = np.sqrt(np.maximum(suma2 / max(n, 1) - media ** 2, 1e-6))
    return [float(m) for m in media], [float(d) for d in desv]


def balance_clases(ds):
    fuego = nofuego = ignorados = 0
    for i in range(len(ds)):
        m = ds.leer_mascara(i)
        fuego += int((m == 1).sum())
        ignorados += int((m == wc.IGNORE_INDEX).sum())
        nofuego += int((m == 0).sum())
    return fuego, nofuego, ignorados


# ---------------------------------------------------------------------------
# Pérdida y métricas
# ---------------------------------------------------------------------------

class SegLoss(nn.Module):
    def __init__(self, pos_weight, usar_dice=True, dice_weight=1.0):
        super().__init__()
        self.register_buffer("pesos", torch.tensor([1.0, float(pos_weight)]))
        self.usar_dice = usar_dice
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        validos = target != wc.IGNORE_INDEX
        if not validos.any():
            return logits.sum() * 0.0
        loss = F.cross_entropy(logits, target, weight=self.pesos, ignore_index=wc.IGNORE_INDEX)
        if self.usar_dice:
            p = torch.softmax(logits, dim=1)[:, 1] * validos
            t = (target == 1).float() * validos
            inter = (p * t).sum()
            dice = 1.0 - (2.0 * inter + 1.0) / (p.sum() + t.sum() + 1.0)
            loss = loss + self.dice_weight * dice
        return loss


class Confusion:
    """Acumula TP/FP/FN/TN por umbral sobre todo el conjunto."""

    def __init__(self, umbrales):
        self.umbrales = torch.tensor(umbrales).view(-1, 1, 1, 1)
        self.c = torch.zeros(len(umbrales), 4, dtype=torch.int64)

    @torch.no_grad()
    def update(self, prob, target):
        prob, target = prob.detach().float().cpu(), target.detach().cpu()
        validos = target != wc.IGNORE_INDEX
        t = (target == 1) & validos
        f = (target == 0) & validos
        pred = prob.unsqueeze(0) > self.umbrales
        self.c[:, 0] += (pred & t).flatten(1).sum(1)
        self.c[:, 1] += (pred & f).flatten(1).sum(1)
        self.c[:, 2] += (~pred & t).flatten(1).sum(1)
        self.c[:, 3] += (~pred & f).flatten(1).sum(1)

    def metricas(self, i):
        tp, fp, fn, tn = (int(v) for v in self.c[i])
        return wc.metricas(tp, fp, fn, tn)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def elegir_dispositivo(opcion):
    opcion = str(opcion).lower()
    if opcion in ("false", "cpu", "0", "no"):
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda:0")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda} | PyTorch {torch.__version__}")
            return torch.device("cuda:0")
        except RuntimeError as e:
            print(f"⚠ CUDA disponible pero no utilizable ({e}); se usa CPU")
    elif opcion == "true":
        print("⚠ Se pidió GPU pero CUDA no está disponible; se usa CPU")
    print(f"Usando CPU | PyTorch {torch.__version__}")
    return torch.device("cpu")


def entrenar_epoca(model, loader, criterio, optimizer, device):
    model.train()
    conf = Confusion([0.5])
    suma_loss, n = 0.0, 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterio(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Pérdida no finita en el batch {i}: revisa LR / normalización de entrada")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        suma_loss += loss.item() * x.size(0)
        n += x.size(0)
        conf.update(torch.softmax(logits.detach(), 1)[:, 1], y)
        if (i + 1) % PRINT_FREQ == 0 or i + 1 == len(loader):
            m = conf.metricas(0)
            print(f"\r  batch {i + 1}/{len(loader)}  loss {suma_loss / n:.4f}  IoU {wc.fmt(m['iou'])}  "
                  f"({time.time() - t0:.0f}s)", end="", flush=True)
    print("\r" + " " * 90 + "\r", end="")
    return suma_loss / max(n, 1), conf.metricas(0)


@torch.no_grad()
def validar(model, loader, criterio, device):
    model.eval()
    conf = Confusion(UMBRALES)
    suma_loss, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        suma_loss += criterio(logits, y).item() * x.size(0)
        n += x.size(0)
        conf.update(torch.softmax(logits, 1)[:, 1], y)
    i05 = UMBRALES.index(0.5)
    m05 = conf.metricas(i05)
    ious = [conf.metricas(i)["iou"] for i in range(len(UMBRALES))]
    ious_ok = [v if not math.isnan(v) else -1 for v in ious]
    i_best = int(np.argmax(ious_ok))
    return {"loss": suma_loss / max(n, 1), **m05,
            "mejor_umbral": UMBRALES[i_best], "iou_mejor_umbral": ious[i_best]}


def guardar_modelo(ruta, model, meta):
    torch.save({"model_state_dict": model.state_dict(), "meta": meta}, ruta)


def cargar_checkpoint(ruta, device):
    try:
        return torch.load(ruta, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 1.13
        return torch.load(ruta, map_location=device)


def enviar_notificacion(mensaje):
    token = os.getenv("PUSHBULLET_API_TOKEN")
    if not token:
        return
    try:
        import requests
        requests.post("https://api.pushbullet.com/v2/pushes",
                      headers={"Access-Token": token, "Content-Type": "application/json"},
                      json={"type": "note", "title": "Alerta de entrenamiento SPRI", "body": mensaje},
                      timeout=10)
        print(f"Notificación enviada: {mensaje}")
    except Exception as e:
        print(f"No se pudo enviar la notificación: {e}")


def a_python(v):
    """Convierte valores numpy/torch/tuplas a tipos de Python (para weights_only=True y JSON)."""
    if isinstance(v, dict):
        return {k: a_python(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [a_python(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def firma_config(meta):
    """Lo que debe coincidir para poder reanudar una corrida."""
    return (list(meta.get("bandas_modelo") or []), list(meta.get("indices") or []), meta.get("quant"),
            meta.get("arquitectura"), meta.get("image_size"))


def buscar_reanudacion(meta_actual):
    """Ruta a last.pth de la corrida más reciente en OUT_DIR que no terminó (sin final.pth)."""
    if REANUDAR is None:
        return None
    if REANUDAR != "auto":
        if not os.path.isfile(REANUDAR):
            abortar(f"ERROR: no existe el checkpoint REANUDAR = {REANUDAR}")
        return REANUDAR
    candidatos = [p for p in glob.glob(os.path.join(OUT_DIR, "*", "last.pth"))
                  if not os.path.exists(os.path.join(os.path.dirname(p), "final.pth"))]
    if not candidatos:
        return None
    ruta = max(candidatos, key=os.path.getmtime)
    try:
        meta = cargar_checkpoint(ruta, "cpu").get("meta", {})
    except Exception as e:
        print(f"⚠ No se pudo leer {ruta} ({e}); se empieza una corrida nueva")
        return None
    if firma_config(a_python(meta)) != firma_config(a_python(meta_actual)):
        print(f"⚠ {ruta} es de otra configuración (bandas/índices/cuantización); se empieza una corrida nueva")
        return None
    return ruta


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    device = elegir_dispositivo(GPU)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    for ruta, nombre in [(DATA_DIR, "DATA_DIR"),
                         (os.path.join(DATA_DIR, TRAIN_LIST), "lista train"),
                         (os.path.join(DATA_DIR, VAL_LIST), "lista validación")]:
        if not os.path.exists(ruta):
            abortar(f"ERROR: no existe {nombre}: {ruta}")

    quant = config_quant()
    meta_base = {
        "modelo": "MbUNet2D", "in_channels": IN_CHANNELS, "out_channels": OUT_CHANNELS, "image_size": IMAGE_SIZE,
        "bandas_modelo": list(BANDAS_MODELO), "indices": list(INDICES),
        "canales": wc.nombres_canales(BANDAS_MODELO, INDICES),
        "orden_bandas": list(BANDAS_MODELO), "unidades_entrada": "DN = reflectancia*10000",
        "arquitectura": {"base_channels": BASE_CHANNELS, "expand_ratio": EXPAND_RATIO},
        "quant": quant,
    }
    if IN_CHANNELS != len(meta_base["canales"]):
        abortar("ERROR: IN_CHANNELS no coincide con BANDAS_MODELO + INDICES")

    ruta_resume = buscar_reanudacion(meta_base)
    ckpt = None
    if ruta_resume:
        ckpt = cargar_checkpoint(ruta_resume, device)
        if "optimizer_state_dict" not in ckpt or "meta" not in ckpt:
            abortar(f"ERROR: {ruta_resume} no es un last.pth de esta versión (no se puede reanudar)")

    # --- Normalización ---
    if ckpt is not None:
        norm = ckpt["meta"]["norm"]
    elif NORM == "div":
        norm = {"tipo": "div", "valor": NORM_DIV}
    else:
        norm = None  # se calcula con el train

    orden_ds = orden_bandas_dataset(DATA_DIR)
    print(f"Dataset: bandas {list(orden_ds)} | entrada del modelo: {meta_base['canales']} ({IN_CHANNELS} canales)")
    norm_tmp = norm or {"tipo": "div", "valor": NORM_DIV}
    train_ds = WildfireCropDataset(DATA_DIR, TRAIN_LIST, orden_ds, norm_tmp, entrenamiento=True)
    val_ds = WildfireCropDataset(DATA_DIR, VAL_LIST, orden_ds, norm_tmp, entrenamiento=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        abortar("ERROR: train o validación vacíos")
    if norm is None:
        media, desv = estadisticas_bandas(train_ds, seed=SEED)
        norm = {"tipo": "zscore", "media": media, "desv": desv}
        train_ds.norm = val_ds.norm = norm
        print(f"Normalización z-score: media {[round(m) for m in media]}, desv {[round(d) for d in desv]}")
    else:
        print(f"Normalización: {norm}")
    print(f"Aumentaciones: {'sí' if AUGMENT else 'no'}")

    # --- Balance y peso de clase ---
    print("Contando píxeles por clase...")
    fuego, nofuego, ign = balance_clases(train_ds)
    total = max(fuego + nofuego, 1)
    print(f"Train: {100 * fuego / total:.2f}% píxeles quemados, {100 * ign / max(total + ign, 1):.2f}% sin dato")
    fv, nv, _ = balance_clases(val_ds)
    print(f"Valid: {100 * fv / max(fv + nv, 1):.2f}% píxeles quemados")
    if ckpt is not None:
        pos_weight = ckpt["meta"]["pos_weight"]
    elif POS_WEIGHT == "auto":
        pos_weight = float(np.clip(math.sqrt(nofuego / max(fuego, 1)), 1.0, 20.0))
    else:
        pos_weight = float(POS_WEIGHT)
    print(f"Peso clase incendio (CE): {pos_weight:.2f} | pérdida: {LOSS}")

    pin = device.type == "cuda"
    persist = WORKERS > 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=len(train_ds) > BATCH_SIZE,
                              num_workers=WORKERS, pin_memory=pin, persistent_workers=persist)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=WORKERS, pin_memory=pin, persistent_workers=persist)

    # --- Modelo ---
    model = wm.MbUNet2D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, base_channels=BASE_CHANNELS,
                        expand_ratio=EXPAND_RATIO, quant=quant).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Modelo MbUNet2D: {n_param / 1e6:.2f} M parámetros "
          f"({n_param * QUANT_BITS / 8 / 1e6:.1f} MB de pesos a {QUANT_BITS} bits)")
    print(f"Cuantización: {wm.resumen_quant(quant)}")
    criterio = SegLoss(pos_weight, usar_dice=(LOSS == "ce_dice"), dice_weight=DICE_WEIGHT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    warm = max(0, min(WARMUP_EPOCHS, EPOCHS - 1))
    coseno = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - warm), eta_min=LR * 0.01)
    if warm > 0:
        calentamiento = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warm)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [calentamiento, coseno], milestones=[warm])
    else:
        scheduler = coseno

    start_epoch, best_iou, sin_mejora = 0, -1.0, 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt["best_iou"]
        sin_mejora = ckpt.get("sin_mejora", 0)
        run_dir = os.path.dirname(os.path.abspath(ruta_resume))
        print(f"Reanudando {run_dir} en la época {start_epoch + 1}, mejor IoU val {best_iou:.4f}")
    else:
        if PESOS_INICIALES:
            ini = cargar_checkpoint(PESOS_INICIALES, device)
            sd = ini["model_state_dict"] if isinstance(ini, dict) and "model_state_dict" in ini else ini
            try:
                model.load_state_dict(sd)
            except RuntimeError as e:
                abortar(f"ERROR: PESOS_INICIALES no son compatibles (¿otro número de canales?): {e}")
            print(f"Pesos iniciales cargados de {PESOS_INICIALES}")
        nombre = RUN_NAME or time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(OUT_DIR, nombre)
        os.makedirs(run_dir, exist_ok=True)

    meta_base.update({"norm": norm, "pos_weight": pos_weight})
    meta_base = a_python(meta_base)
    config_run = {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(a_python({**config_run, **meta_base, "n_train": len(train_ds), "n_val": len(val_ds),
                            "porcentaje_quemado_train": 100 * fuego / total}), f, indent=2, default=str)

    ruta_log = os.path.join(run_dir, "log.csv")
    nuevo_log = not os.path.exists(ruta_log)
    log = open(ruta_log, "a", newline="")
    escritor = csv.writer(log)
    if nuevo_log:
        escritor.writerow(["epoch", "lr", "train_loss", "train_iou", "val_loss", "val_iou", "val_f1",
                           "val_precision", "val_recall", "mejor_umbral", "iou_mejor_umbral", "segundos"])

    print("=" * 70)
    print(f"Entrenando: {len(train_ds)} train / {len(val_ds)} val | batch {BATCH_SIZE} | "
          f"{EPOCHS} épocas | salida {run_dir}")
    print("=" * 70)

    t_inicio = time.time()
    epoch = start_epoch - 1
    interrumpido = False
    try:
        for epoch in range(start_epoch, EPOCHS):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            tr_loss, tr_m = entrenar_epoca(model, train_loader, criterio, optimizer, device)
            val = validar(model, val_loader, criterio, device)
            scheduler.step()
            dt = time.time() - t0

            val_iou = val["iou"] if not math.isnan(val["iou"]) else 0.0
            es_mejor = val_iou > best_iou
            if es_mejor:
                best_iou, sin_mejora = val_iou, 0
                meta = a_python({**meta_base, "epoch": epoch, "umbral": val["mejor_umbral"], "val": val})
                guardar_modelo(os.path.join(run_dir, "best.pth"), model, meta)
            else:
                sin_mejora += 1

            escritor.writerow([epoch + 1, f"{lr:.6g}", f"{tr_loss:.5f}", wc.fmt(tr_m["iou"]), f"{val['loss']:.5f}",
                               wc.fmt(val["iou"]), wc.fmt(val["f1"]), wc.fmt(val["precision"]),
                               wc.fmt(val["recall"]), val["mejor_umbral"], wc.fmt(val["iou_mejor_umbral"]),
                               f"{dt:.1f}"])
            log.flush()
            print(f"Época {epoch + 1:3d}/{EPOCHS}  lr {lr:.2e}  loss {tr_loss:.4f}  IoU train {wc.fmt(tr_m['iou'])}"
                  f" | val loss {val['loss']:.4f}  IoU {wc.fmt(val['iou'])}  F1 {wc.fmt(val['f1'])}"
                  f"  P {wc.fmt(val['precision'], 3)}  R {wc.fmt(val['recall'], 3)}"
                  f"  (umbral {val['mejor_umbral']}: {wc.fmt(val['iou_mejor_umbral'])})  {dt:.0f}s"
                  + ("  <- mejor" if es_mejor else ""))

            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch, "best_iou": best_iou, "sin_mejora": sin_mejora,
                        "meta": meta_base}, os.path.join(run_dir, "last.pth"))

            if PATIENCE > 0 and sin_mejora >= PATIENCE:
                print(f"Early stopping: {PATIENCE} épocas sin mejorar IoU de validación")
                break
    except KeyboardInterrupt:
        interrumpido = True
        print("\nInterrumpido por el usuario. last.pth/best.pth quedan guardados; con REANUDAR = \"auto\" "
              "la próxima ejecución continúa esta corrida (no se escribe final.pth).")
    finally:
        log.close()

    if interrumpido:
        return
    guardar_modelo(os.path.join(run_dir, "final.pth"), model, a_python({**meta_base, "epoch": epoch}))
    total_s = time.time() - t_inicio
    h, m = int(total_s // 3600), int(total_s % 3600 // 60)
    print("=" * 70)
    print(f"Mejor IoU val: {best_iou:.4f} | tiempo {h:02d}h{m:02d}m | modelos en {run_dir}")
    print("=" * 70)
    enviar_notificacion(f"Entrenamiento terminado. Mejor IoU: {best_iou:.4f}, tiempo {h:02d}h{m:02d}m")


if __name__ == "__main__":
    main()
