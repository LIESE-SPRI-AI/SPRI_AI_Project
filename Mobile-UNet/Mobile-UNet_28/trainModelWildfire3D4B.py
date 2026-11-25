#!/usr/bin/env python3
"""
trainModelWildfire3D4B.py
=========================
Entrenamiento de MbUNet2D (Wildfire_models.py, misma arquitectura que Wildfire_models_base.py)
para segmentar área quemada.

Todos los parámetros se pasan por terminal; los valores por defecto están en el bloque
VALORES POR DEFECTO de abajo. Las rutas parten de BASE_DIR definido en el archivo .env.

  python3 trainModelWildfire3D4B.py -h                      # lista de argumentos
  python3 trainModelWildfire3D4B.py                         # todo por defecto
  python3 trainModelWildfire3D4B.py --optimizador sgd --lr 0.02 --scheduler plateau
  python3 trainModelWildfire3D4B.py --quant base8 --pos-weight 1 --umbral-min-precision 0.6
  ./run_training.sh --optimizador adamw --epochs 300        # reintenta si se cae (pasa los argumentos)

Cada corrida se guarda en OUT_DIR/<fecha_hora>_<optimizador>_<scheduler>/:
  best.pth   mejor modelo en validación + metadatos (bandas, índices, normalización,
             cuantización, umbral, tamaño, métricas). predict_segmentation.py lee todo de aquí.
  last.pth   estado completo para reanudar (modelo, optimizador, scheduler, época)
  final.pth  modelo de la última época (su existencia marca la corrida como terminada)
  log.csv    métricas por época      config.json   parámetros (incluye la línea de comandos)

Cambios de esta versión:
  * Argumentos por terminal (argparse) con valores por defecto declarados arriba.
  * Modelo por defecto igual a Wildfire_models_base.py: 32 canales base y los mismos puntos de
    cuantización (pesos, ReLU, proyección lineal). --quant base8 lo reproduce bit a bit.
  * Optimizador configurable (adamw, adam, sgd, rmsprop, radam, nadam).
  * Regularización: weight decay sin bias/BatchNorm, L1, label smoothing y Dropout2d antes de la
    capa de salida (hook; no cambia el modelo ni el state_dict).
  * Bajada de learning rate: step, multistep, plateau, coseno o constante, después del calentamiento.
    Cada bajada reinicia la paciencia del early stopping.
  * El mejor modelo y su umbral se eligen por IoU al MEJOR umbral (el que usa la predicción), con
    opción de exigir una precisión mínima (--umbral-min-precision).
  * POS_WEIGHT = 1 por defecto (antes "auto" hasta 20: empujaba a marcar de más -> baja precisión).
  * Aumentaciones activas por defecto (--no-augment para quitarlas).
"""

import argparse
import copy
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
# VALORES POR DEFECTO (todos se pueden cambiar por terminal: -h para ver la lista)
# =============================================================================

# --- Rutas (BASE_DIR viene del .env) ---
DATA_DIR = wc.ruta_proyecto("Mobile-UNet/Mobile-UNet_28", "data")   # OUTPUT_DIR del generador
IMAGES_SUBDIR = "Images"
MASKS_SUBDIR = "SegmentationClass"
TRAIN_LIST = "train.txt"
VAL_LIST = "valid.txt"
OUT_DIR = os.path.join(wc.DIR_CODIGO, "runs")
RUN_NAME = None              # None = <fecha_hora>_<optimizador>_<scheduler>
REANUDAR = "auto"            # "no" | "auto" (última corrida sin final.pth en OUT_DIR) | ruta a un last.pth
PESOS_INICIALES = None       # ruta a .pth para afinar (sin reanudar optimizador); requiere mismos canales

# --- Entrada y salida del modelo ---
BANDAS_MODELO = ("B04", "B03", "B02", "B08")   # R, G, B, NIR
INDICES = "" #("NDVI", "BAI")                      # canales extra (wc.INDICES_DISPONIBLES); "" = ninguno
OUT_CHANNELS = 2                               # 0 = no quemado, 1 = quemado
IMAGE_SIZE = 128                               # recorte de entrada (no se reescala)

# --- Arquitectura (igual que Wildfire_models_base.py) ---
BASE_CHANNELS = 16           # num = 32 en la base (antes 64 aquí)
EXPAND_RATIO = 6

# --- Cuantización ---
QUANT = "base16"             # base8 (idéntica a la base) | base16 (mismos puntos, 16 bits) | completa16 | ninguna
REDONDEO = None              # None = el del preset | "half_up" | "nearest_even"

# --- Entrenamiento ---
EPOCHS = 100
BATCH_SIZE = 8
PATIENCE = 100                # épocas sin mejora antes de parar (0 = desactivado)
REINICIAR_PACIENCIA_AL_BAJAR_LR = True
SEED = 42
GPU = "auto"                 # "auto" | "true" | "false"
WORKERS = 4
CACHE = False                # cargar todo el dataset en RAM
PRINT_FREQ = 10

# --- Optimizador ---
OPTIMIZADOR = "adam"        # adamw | adam | sgd | rmsprop | radam | nadam
LR = 1e-3                    # sugerido: familia adam 1e-3 | sgd 1e-2 a 5e-2 | rmsprop 1e-4 a 5e-4
MOMENTUM = 0.9               # sgd y rmsprop
NESTEROV = True              # solo sgd
BETAS = (0.9, 0.999)         # familia adam

# --- Regularización ---
WEIGHT_DECAY = 1e-2          # adamw: decaimiento desacoplado | resto: L2 en el gradiente (usa 1e-4 a 5e-4)
DECAY_EN_BIAS_NORM = False   # False = sin weight decay en bias ni BatchNorm
L1_LAMBDA = 0.0              # penalización L1 de pesos conv (prueba 1e-6 a 1e-5)
LABEL_SMOOTHING = 0.0        # suavizado de etiquetas en la CE (prueba 0.05 a 0.1)
DROPOUT_SALIDA = 0.1         # Dropout2d a la entrada de out_conv (0 = desactivado)
GRAD_CLIP = 5.0              # 0 = sin recorte

# --- Learning rate ---
WARMUP_EPOCHS = 3            # subida lineal de 0.1*LR a LR
SCHEDULER = "multistep"      # step | multistep | plateau | coseno | constante
LR_GAMMA = 0.2               # factor de cada bajada (step, multistep, plateau)
LR_STEP_EPOCAS = 50          # step: baja cada N épocas (contadas después del calentamiento)
LR_HITOS = (60, 120, 160)    # multistep: épocas (absolutas) en las que baja
LR_PLATEAU_PACIENCIA = 10    # plateau: épocas sin mejorar antes de bajar (debe ser < PATIENCE)
LR_MIN = 1e-6                # piso del LR (plateau y coseno)

# --- Pérdida, normalización y umbral ---
LOSS = "ce_dice"             # ce | ce_dice
DICE_WEIGHT = 0.5
POS_WEIGHT = "1.0"           # número o "auto" (sqrt(no quemado / quemado), entre 1 y 20)
NORM = "div"                 # div: DN / NORM_DIV recortado a [0,1] | zscore
NORM_DIV = 12500.0
UMBRAL_MIN_PRECISION = 0.0   # >0: el umbral guardado es el de mejor IoU con al menos esta precisión en validación

# --- Aumentaciones (solo entrenamiento; en DN antes de calcular índices) ---
AUGMENT = True
AUG_P_FLIP = 0.5
AUG_P_ROT90 = 0.5
AUG_P_ESCALA = 0.3
AUG_ESCALA_RANGO = (0.8, 1.25)
AUG_P_BRILLO = 0.5
AUG_BRILLO_RANGO = 0.15
AUG_JITTER_BANDA = 0.05
AUG_P_BRUMA = 0.2
AUG_BRUMA_MAX_DN = 300.0
AUG_BRUMA_FACTOR = {"B02": 1.0, "B03": 0.7, "B04": 0.5, "B08": 0.2, "B8A": 0.15, "B11": 0.05, "B12": 0.03}
AUG_P_RUIDO = 0.3
AUG_RUIDO_STD_DN = 60.0
AUG_P_DESENFOQUE = 0.15
AUG_DESENFOQUE_SIGMA = (0.5, 1.2)

UMBRALES = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 ... 0.95 (búsqueda en validación)
BANDAS_DATASET_VIEJO = ("B04", "B03", "B02", "B08")      # si el dataset no tiene config.json
# =============================================================================


CODIGO_ERROR_CONFIG = 2   # run_training.sh no reintenta con este código (argparse también sale con 2)
CODIGO_INTERRUMPIDO = 130


def abortar(mensaje):
    print(mensaje, file=sys.stderr)
    sys.exit(CODIGO_ERROR_CONFIG)


# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------

def _lista(texto):
    """'B04,B03 B02' -> ('B04', 'B03', 'B02'); '' -> ()."""
    return tuple(p for p in texto.replace(",", " ").split() if p)


def leer_argumentos(argv=None):
    p = argparse.ArgumentParser(
        description="Entrenamiento de MbUNet2D para área quemada",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    BOOL = argparse.BooleanOptionalAction

    g = p.add_argument_group("rutas")
    g.add_argument("--data-dir", default=DATA_DIR, help="carpeta del dataset (OUTPUT_DIR del generador)")
    g.add_argument("--out-dir", default=OUT_DIR, help="carpeta de corridas")
    g.add_argument("--run-name", default=RUN_NAME, help="nombre de la corrida (por defecto fecha_optimizador_scheduler)")
    g.add_argument("--reanudar", default=REANUDAR, help='"no", "auto" o ruta a un last.pth')
    g.add_argument("--pesos-iniciales", default=PESOS_INICIALES, help=".pth para afinar")

    g = p.add_argument_group("entrada y modelo")
    g.add_argument("--bandas", type=_lista, default=BANDAS_MODELO, help='bandas, p. ej. "B04,B03,B02,B08"')
    g.add_argument("--indices", type=_lista, default=INDICES, help='índices, p. ej. "NDVI,BAI" ("" = ninguno)')
    g.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    g.add_argument("--base-channels", type=int, default=BASE_CHANNELS)
    g.add_argument("--expand-ratio", type=int, default=EXPAND_RATIO)
    g.add_argument("--quant", choices=sorted(wm.PRESETS_QUANT), default=QUANT)
    g.add_argument("--redondeo", choices=["half_up", "nearest_even"], default=REDONDEO)

    g = p.add_argument_group("entrenamiento")
    g.add_argument("--epochs", type=int, default=EPOCHS)
    g.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    g.add_argument("--patience", type=int, default=PATIENCE, help="0 = sin early stopping")
    g.add_argument("--reiniciar-paciencia", action=BOOL, default=REINICIAR_PACIENCIA_AL_BAJAR_LR,
                   help="reiniciar la paciencia cuando baja el LR")
    g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--gpu", default=GPU, choices=["auto", "true", "false"])
    g.add_argument("--workers", type=int, default=WORKERS)
    g.add_argument("--cache", action=BOOL, default=CACHE)
    g.add_argument("--print-freq", type=int, default=PRINT_FREQ)

    g = p.add_argument_group("optimizador")
    g.add_argument("--optimizador", choices=["adamw", "adam", "sgd", "rmsprop", "radam", "nadam"], default=OPTIMIZADOR)
    g.add_argument("--lr", type=float, default=LR)
    g.add_argument("--momentum", type=float, default=MOMENTUM)
    g.add_argument("--nesterov", action=BOOL, default=NESTEROV)

    g = p.add_argument_group("regularización")
    g.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    g.add_argument("--decay-en-bias-norm", action=BOOL, default=DECAY_EN_BIAS_NORM)
    g.add_argument("--l1", type=float, default=L1_LAMBDA)
    g.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    g.add_argument("--dropout", type=float, default=DROPOUT_SALIDA, help="Dropout2d antes de out_conv")
    g.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    g.add_argument("--augment", action=BOOL, default=AUGMENT)

    g = p.add_argument_group("learning rate")
    g.add_argument("--warmup", type=int, default=WARMUP_EPOCHS)
    g.add_argument("--scheduler", choices=["step", "multistep", "plateau", "coseno", "constante"], default=SCHEDULER)
    g.add_argument("--lr-gamma", type=float, default=LR_GAMMA)
    g.add_argument("--lr-step", type=int, default=LR_STEP_EPOCAS)
    g.add_argument("--lr-hitos", type=int, nargs="+", default=list(LR_HITOS))
    g.add_argument("--plateau-paciencia", type=int, default=LR_PLATEAU_PACIENCIA)
    g.add_argument("--lr-min", type=float, default=LR_MIN)

    g = p.add_argument_group("pérdida, normalización y umbral")
    g.add_argument("--loss", choices=["ce", "ce_dice"], default=LOSS)
    g.add_argument("--dice-weight", type=float, default=DICE_WEIGHT)
    g.add_argument("--pos-weight", default=POS_WEIGHT, help='número o "auto"')
    g.add_argument("--norm", choices=["div", "zscore"], default=NORM)
    g.add_argument("--norm-div", type=float, default=NORM_DIV)
    g.add_argument("--umbral-min-precision", type=float, default=UMBRAL_MIN_PRECISION)

    a = p.parse_args(argv)
    if a.pos_weight != "auto":
        try:
            float(a.pos_weight)
        except ValueError:
            p.error('--pos-weight debe ser un número o "auto"')
    return a


def aplicar_argumentos(a):
    """Copia los argumentos a las variables globales que usa el resto del script."""
    global DATA_DIR, OUT_DIR, RUN_NAME, REANUDAR, PESOS_INICIALES, BANDAS_MODELO, INDICES, IMAGE_SIZE
    global BASE_CHANNELS, EXPAND_RATIO, QUANT, REDONDEO, EPOCHS, BATCH_SIZE, PATIENCE
    global REINICIAR_PACIENCIA_AL_BAJAR_LR, SEED, GPU, WORKERS, CACHE, PRINT_FREQ, OPTIMIZADOR, LR, MOMENTUM
    global NESTEROV, WEIGHT_DECAY, DECAY_EN_BIAS_NORM, L1_LAMBDA, LABEL_SMOOTHING, DROPOUT_SALIDA, GRAD_CLIP
    global AUGMENT, WARMUP_EPOCHS, SCHEDULER, LR_GAMMA, LR_STEP_EPOCAS, LR_HITOS, LR_PLATEAU_PACIENCIA, LR_MIN
    global LOSS, DICE_WEIGHT, POS_WEIGHT, NORM, NORM_DIV, UMBRAL_MIN_PRECISION
    DATA_DIR, OUT_DIR, RUN_NAME = a.data_dir, a.out_dir, a.run_name
    REANUDAR = None if str(a.reanudar).lower() in ("no", "none", "") else a.reanudar
    PESOS_INICIALES = a.pesos_iniciales
    BANDAS_MODELO, INDICES, IMAGE_SIZE = tuple(a.bandas), tuple(a.indices), a.image_size
    BASE_CHANNELS, EXPAND_RATIO, QUANT, REDONDEO = a.base_channels, a.expand_ratio, a.quant, a.redondeo
    EPOCHS, BATCH_SIZE, PATIENCE = a.epochs, a.batch_size, a.patience
    REINICIAR_PACIENCIA_AL_BAJAR_LR, SEED, GPU = a.reiniciar_paciencia, a.seed, a.gpu
    WORKERS, CACHE, PRINT_FREQ = a.workers, a.cache, a.print_freq
    OPTIMIZADOR, LR, MOMENTUM, NESTEROV = a.optimizador, a.lr, a.momentum, a.nesterov
    WEIGHT_DECAY, DECAY_EN_BIAS_NORM, L1_LAMBDA = a.weight_decay, a.decay_en_bias_norm, a.l1
    LABEL_SMOOTHING, DROPOUT_SALIDA, GRAD_CLIP, AUGMENT = a.label_smoothing, a.dropout, a.grad_clip, a.augment
    WARMUP_EPOCHS, SCHEDULER, LR_GAMMA, LR_STEP_EPOCAS = a.warmup, a.scheduler, a.lr_gamma, a.lr_step
    LR_HITOS, LR_PLATEAU_PACIENCIA, LR_MIN = tuple(a.lr_hitos), a.plateau_paciencia, a.lr_min
    LOSS, DICE_WEIGHT, POS_WEIGHT = a.loss, a.dice_weight, a.pos_weight
    NORM, NORM_DIV, UMBRAL_MIN_PRECISION = a.norm, a.norm_div, a.umbral_min_precision


def config_quant():
    quant = copy.deepcopy(wm.PRESETS_QUANT[QUANT])
    if REDONDEO:
        quant["redondeo"] = REDONDEO
    wm.validar_quant(quant)
    return quant


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
        # Se guardan en la instancia (no se leen de las globales en los workers del DataLoader)
        self.bandas_modelo = tuple(BANDAS_MODELO)
        self.indices = tuple(INDICES)
        # bandas a leer (las del modelo + las que piden los índices), en ese orden
        self.orden = wc.bandas_requeridas(self.bandas_modelo, self.indices)
        faltan = [b for b in self.orden if b not in orden_dataset]
        if faltan:
            abortar(f"ERROR: el dataset ({list(orden_dataset)}) no tiene las bandas {faltan} que piden "
                    f"--bandas/--indices. Regenera el dataset con esas bandas o quítalas.")
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
        x = wc.construir_entrada(img, self.orden, self.bandas_modelo, self.indices, self.norm, nodata)
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
    def __init__(self, pos_weight, usar_dice=True, dice_weight=1.0, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("pesos", torch.tensor([1.0, float(pos_weight)]))
        self.usar_dice = usar_dice
        self.dice_weight = dice_weight
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, target):
        validos = target != wc.IGNORE_INDEX
        if not validos.any():
            return logits.sum() * 0.0
        loss = F.cross_entropy(logits, target, weight=self.pesos, ignore_index=wc.IGNORE_INDEX,
                               label_smoothing=self.label_smoothing)
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


def _sin_nan(v, reemplazo=-1.0):
    return reemplazo if v is None or (isinstance(v, float) and math.isnan(v)) else v


def elegir_umbral(conf):
    """Índice del umbral con mejor IoU; si UMBRAL_MIN_PRECISION > 0, solo entre los que la cumplen
    (si ninguno la cumple, el de mayor precisión)."""
    ms = [conf.metricas(i) for i in range(len(UMBRALES))]
    cands = list(range(len(UMBRALES)))
    if UMBRAL_MIN_PRECISION > 0:
        ok = [i for i in cands if _sin_nan(ms[i]["precision"]) >= UMBRAL_MIN_PRECISION]
        if not ok:
            return max(cands, key=lambda i: _sin_nan(ms[i]["precision"])), ms
        cands = ok
    return max(cands, key=lambda i: _sin_nan(ms[i]["iou"])), ms


# ---------------------------------------------------------------------------
# Optimizador, scheduler y regularización
# ---------------------------------------------------------------------------

def separar_parametros(model):
    """(con_decay, sin_decay). Bias y parámetros de 1 dimensión (BatchNorm) van sin decay."""
    con, sin = [], []
    for nombre, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not DECAY_EN_BIAS_NORM and (p.ndim <= 1 or nombre.endswith(".bias")):
            sin.append(p)
        else:
            con.append(p)
    return con, sin


def crear_optimizador(model):
    con, sin = separar_parametros(model)
    grupos = [g for g in ({"params": con, "weight_decay": WEIGHT_DECAY},
                          {"params": sin, "weight_decay": 0.0}) if g["params"]]
    o = OPTIMIZADOR
    if o == "adamw":
        opt = torch.optim.AdamW(grupos, lr=LR, betas=BETAS)
    elif o == "adam":
        opt = torch.optim.Adam(grupos, lr=LR, betas=BETAS)
    elif o == "radam":
        opt = torch.optim.RAdam(grupos, lr=LR, betas=BETAS)
    elif o == "nadam":
        opt = torch.optim.NAdam(grupos, lr=LR, betas=BETAS)
    elif o == "sgd":
        opt = torch.optim.SGD(grupos, lr=LR, momentum=MOMENTUM, nesterov=NESTEROV and MOMENTUM > 0)
    elif o == "rmsprop":
        opt = torch.optim.RMSprop(grupos, lr=LR, momentum=MOMENTUM, alpha=0.99, eps=1e-8)
    else:
        abortar(f"ERROR: optimizador desconocido: {o}")
    print(f"Optimizador: {o} | LR {LR:g} | weight decay {WEIGHT_DECAY:g} en {len(con)} tensores "
          f"({len(sin)} sin decay) | L1 {L1_LAMBDA:g} | label smoothing {LABEL_SMOOTHING:g}")
    return opt, [p for p in con if p.ndim > 1]


def crear_scheduler(optimizer, warm):
    """El scheduler empieza a contar al terminar el calentamiento."""
    s = SCHEDULER
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_EPOCAS, gamma=LR_GAMMA)
    if s == "multistep":
        hitos = sorted(h - warm for h in LR_HITOS if h > warm)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=hitos, gamma=LR_GAMMA)
    if s == "plateau":
        if 0 < PATIENCE <= LR_PLATEAU_PACIENCIA:
            print("⚠ --plateau-paciencia >= --patience: el early stopping cortará antes de bajar el LR")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=LR_GAMMA,
                                                          patience=LR_PLATEAU_PACIENCIA, min_lr=LR_MIN)
    if s == "coseno":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - warm), eta_min=LR_MIN)
    return None   # constante


def fijar_lr_calentamiento(optimizer, epoch, warm):
    """Durante el calentamiento el LR sube lineal de 0.1*LR a LR; en la época 'warm' queda en LR."""
    if warm <= 0 or epoch > warm:
        return
    factor = 1.0 if epoch == warm else 0.1 + 0.9 * epoch / warm
    for g in optimizer.param_groups:
        g["lr"] = LR * factor


def instalar_dropout_salida(model, p):
    """Dropout2d a la entrada de out_conv (la última Conv2d con OUT_CHANNELS salidas).
    Forward pre-hook: solo actúa en model.train() y no cambia el state_dict."""
    if p <= 0:
        return None
    cabezas = [(n, m) for n, m in model.named_modules()
               if isinstance(m, nn.Conv2d) and m.out_channels == OUT_CHANNELS]
    if not cabezas:
        print("⚠ No se encontró la capa de salida: --dropout no se aplica")
        return None
    nombre, capa = cabezas[-1]

    def hook(mod, entradas):
        if not mod.training:
            return None
        return (F.dropout2d(entradas[0], p=p, training=True),) + tuple(entradas[1:])

    print(f"Dropout2d p={p} a la entrada de '{nombre}' ({capa.in_channels} -> {capa.out_channels} canales)")
    return capa.register_forward_pre_hook(hook)


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


def entrenar_epoca(model, loader, criterio, optimizer, device, pesos_l1=None):
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
        total = loss
        if L1_LAMBDA > 0 and pesos_l1:
            total = loss + L1_LAMBDA * sum(p.abs().sum() for p in pesos_l1)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
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
    m05 = conf.metricas(UMBRALES.index(0.5))
    i_best, ms = elegir_umbral(conf)
    mb = ms[i_best]
    return {"loss": suma_loss / max(n, 1), **m05,
            "mejor_umbral": UMBRALES[i_best], "iou_mejor_umbral": mb["iou"],
            "precision_mejor_umbral": mb["precision"], "recall_mejor_umbral": mb["recall"]}


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
            meta.get("arquitectura"), meta.get("image_size"), meta.get("entrenamiento"))


def buscar_reanudacion(meta_actual):
    """Ruta a last.pth de la corrida más reciente en OUT_DIR que no terminó (sin final.pth)."""
    if REANUDAR is None:
        return None
    if REANUDAR != "auto":
        if not os.path.isfile(REANUDAR):
            abortar(f"ERROR: no existe el checkpoint --reanudar {REANUDAR}")
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
        print(f"⚠ {ruta} es de otra configuración; se empieza una corrida nueva")
        return None
    return ruta


def main(argv=None):
    aplicar_argumentos(leer_argumentos(argv))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 70)
    print("Comando: " + " ".join(sys.argv))
    device = elegir_dispositivo(GPU)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    for ruta, nombre in [(DATA_DIR, "--data-dir"),
                         (os.path.join(DATA_DIR, TRAIN_LIST), "lista train"),
                         (os.path.join(DATA_DIR, VAL_LIST), "lista validación")]:
        if not os.path.exists(ruta):
            abortar(f"ERROR: no existe {nombre}: {ruta}")

    in_channels = len(BANDAS_MODELO) + len(INDICES)
    quant = config_quant()
    meta_base = {
        "modelo": "MbUNet2D", "in_channels": in_channels, "out_channels": OUT_CHANNELS, "image_size": IMAGE_SIZE,
        "bandas_modelo": list(BANDAS_MODELO), "indices": list(INDICES),
        "canales": wc.nombres_canales(BANDAS_MODELO, INDICES),
        "orden_bandas": list(BANDAS_MODELO), "unidades_entrada": "DN = reflectancia*10000",
        "arquitectura": {"base_channels": BASE_CHANNELS, "expand_ratio": EXPAND_RATIO},
        "quant": quant, "quant_preset": QUANT,
        # si cambia algo de esto, --reanudar auto empieza una corrida nueva en vez de mezclar pruebas
        "entrenamiento": {"optimizador": OPTIMIZADOR, "lr": LR, "weight_decay": WEIGHT_DECAY,
                          "scheduler": SCHEDULER, "lr_gamma": LR_GAMMA, "l1": L1_LAMBDA,
                          "label_smoothing": LABEL_SMOOTHING, "dropout_salida": DROPOUT_SALIDA,
                          "pos_weight": POS_WEIGHT, "loss": LOSS, "dice_weight": DICE_WEIGHT,
                          "norm": NORM, "augment": AUGMENT, "data_dir": DATA_DIR},
    }

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
    print(f"Dataset: {DATA_DIR} | bandas {list(orden_ds)} | entrada: {meta_base['canales']} ({in_channels} canales)")
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
    pct_val = 100 * fv / max(fv + nv, 1)
    print(f"Valid: {pct_val:.2f}% píxeles quemados")
    if pct_val > 20:
        print("⚠ Validación con más de 20% de píxeles quemados: probablemente NO es exhaustiva y su precisión "
              "saldrá inflada respecto a escenas completas. Regenera el dataset con EVAL_EXHAUSTIVO = True.")
    if ckpt is not None:
        pos_weight = ckpt["meta"]["pos_weight"]
    elif POS_WEIGHT == "auto":
        pos_weight = float(np.clip(math.sqrt(nofuego / max(fuego, 1)), 1.0, 20.0))
    else:
        pos_weight = float(POS_WEIGHT)
    print(f"Peso clase quemado (CE): {pos_weight:.2f} | pérdida: {LOSS} (dice {DICE_WEIGHT})")

    pin = device.type == "cuda"
    persist = WORKERS > 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=len(train_ds) > BATCH_SIZE,
                              num_workers=WORKERS, pin_memory=pin, persistent_workers=persist)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=WORKERS, pin_memory=pin, persistent_workers=persist)

    # --- Modelo ---
    model = wm.MbUNet2D(in_channels=in_channels, out_channels=OUT_CHANNELS, base_channels=BASE_CHANNELS,
                        expand_ratio=EXPAND_RATIO, quant=quant).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    bits = (quant.get("pesos") or {}).get("bits", 32)
    print(f"Modelo MbUNet2D: base {BASE_CHANNELS}, expansión {EXPAND_RATIO}, {n_param / 1e6:.2f} M parámetros "
          f"({n_param * bits / 8 / 1e6:.1f} MB de pesos a {bits} bits)")
    print(f"Cuantización ({QUANT}): {wm.resumen_quant(quant)}")
    criterio = SegLoss(pos_weight, usar_dice=(LOSS == "ce_dice"), dice_weight=DICE_WEIGHT,
                       label_smoothing=LABEL_SMOOTHING).to(device)
    instalar_dropout_salida(model, DROPOUT_SALIDA)
    optimizer, pesos_l1 = crear_optimizador(model)

    warm = max(0, min(WARMUP_EPOCHS, EPOCHS - 1))
    scheduler = crear_scheduler(optimizer, warm)
    es_plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    bajadas_discretas = SCHEDULER in ("step", "multistep", "plateau")
    detalle = {"step": f"cada {LR_STEP_EPOCAS} épocas", "multistep": f"en {list(LR_HITOS)}",
               "plateau": f"tras {LR_PLATEAU_PACIENCIA} épocas sin mejora"}.get(SCHEDULER, "")
    print(f"Scheduler: {SCHEDULER} {detalle} | calentamiento {warm} épocas"
          + (f" | gamma {LR_GAMMA}" if bajadas_discretas else ""))
    if UMBRAL_MIN_PRECISION > 0:
        print(f"Umbral: mejor IoU con precisión >= {UMBRAL_MIN_PRECISION}")

    start_epoch, best_iou, sin_mejora = 0, -1.0, 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
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
                abortar(f"ERROR: --pesos-iniciales no son compatibles (¿otro número de canales?): {e}")
            print(f"Pesos iniciales cargados de {PESOS_INICIALES}")
        nombre = RUN_NAME or f"{time.strftime('%Y%m%d_%H%M%S')}_{OPTIMIZADOR}_{SCHEDULER}"
        run_dir = os.path.join(OUT_DIR, nombre)
        os.makedirs(run_dir, exist_ok=True)

    meta_base.update({"norm": norm, "pos_weight": pos_weight})
    meta_base = a_python(meta_base)
    config_run = {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(a_python({**config_run, **meta_base, "comando": sys.argv, "n_train": len(train_ds),
                            "n_val": len(val_ds), "porcentaje_quemado_train": 100 * fuego / total,
                            "porcentaje_quemado_val": pct_val}), f, indent=2, default=str)

    ruta_log = os.path.join(run_dir, "log.csv")
    nuevo_log = not os.path.exists(ruta_log)
    log = open(ruta_log, "a", newline="")
    escritor = csv.writer(log)
    if nuevo_log:
        escritor.writerow(["epoch", "lr", "train_loss", "train_iou", "val_loss", "val_iou", "val_f1",
                           "val_precision", "val_recall", "mejor_umbral", "iou_mejor_umbral",
                           "precision_mejor_umbral", "recall_mejor_umbral", "segundos"])

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
            fijar_lr_calentamiento(optimizer, epoch, warm)
            lr = optimizer.param_groups[0]["lr"]
            tr_loss, tr_m = entrenar_epoca(model, train_loader, criterio, optimizer, device, pesos_l1)
            val = validar(model, val_loader, criterio, device)
            dt = time.time() - t0

            # Se elige por IoU al mejor umbral: es el umbral que usará predict_segmentation.py
            val_iou = _sin_nan(val["iou_mejor_umbral"], 0.0)
            es_mejor = val_iou > best_iou
            if es_mejor:
                best_iou, sin_mejora = val_iou, 0
                meta = a_python({**meta_base, "epoch": epoch, "umbral": val["mejor_umbral"], "val": val})
                guardar_modelo(os.path.join(run_dir, "best.pth"), model, meta)
            else:
                sin_mejora += 1

            # --- Bajada de LR (el scheduler cuenta desde que termina el calentamiento) ---
            if scheduler is not None and epoch >= warm:
                if es_plateau:
                    scheduler.step(val_iou)
                else:
                    scheduler.step()
            lr_nuevo = optimizer.param_groups[0]["lr"]
            bajo_lr = bajadas_discretas and epoch >= warm and lr_nuevo < lr * 0.999
            if bajo_lr and REINICIAR_PACIENCIA_AL_BAJAR_LR:
                sin_mejora = 0

            escritor.writerow([epoch + 1, f"{lr:.6g}", f"{tr_loss:.5f}", wc.fmt(tr_m["iou"]), f"{val['loss']:.5f}",
                               wc.fmt(val["iou"]), wc.fmt(val["f1"]), wc.fmt(val["precision"]),
                               wc.fmt(val["recall"]), val["mejor_umbral"], wc.fmt(val["iou_mejor_umbral"]),
                               wc.fmt(val["precision_mejor_umbral"]), wc.fmt(val["recall_mejor_umbral"]),
                               f"{dt:.1f}"])
            log.flush()
            print(f"Época {epoch + 1:3d}/{EPOCHS}  lr {lr:.2e}  loss {tr_loss:.4f}  IoU train {wc.fmt(tr_m['iou'])}"
                  f" | val loss {val['loss']:.4f}  @0.5: IoU {wc.fmt(val['iou'])} P {wc.fmt(val['precision'], 3)}"
                  f" R {wc.fmt(val['recall'], 3)} | @{val['mejor_umbral']}: IoU {wc.fmt(val['iou_mejor_umbral'])}"
                  f" P {wc.fmt(val['precision_mejor_umbral'], 3)} R {wc.fmt(val['recall_mejor_umbral'], 3)}"
                  f"  {dt:.0f}s" + ("  <- mejor" if es_mejor else ""))
            if bajo_lr:
                print(f"  ↓ LR {lr:.2e} -> {lr_nuevo:.2e}")

            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                        "epoch": epoch, "best_iou": best_iou, "sin_mejora": sin_mejora,
                        "meta": meta_base}, os.path.join(run_dir, "last.pth"))

            if PATIENCE > 0 and sin_mejora >= PATIENCE:
                print(f"Early stopping: {PATIENCE} épocas sin mejorar IoU de validación")
                break
    except KeyboardInterrupt:
        interrumpido = True
        print("\nInterrumpido por el usuario. last.pth/best.pth quedan guardados; con --reanudar auto "
              "la próxima ejecución continúa esta corrida (no se escribe final.pth).")
    finally:
        log.close()

    if interrumpido:
        sys.exit(CODIGO_INTERRUMPIDO)    # antes salía con 0 y run_training.sh decía "completado"
    guardar_modelo(os.path.join(run_dir, "final.pth"), model, a_python({**meta_base, "epoch": epoch}))
    total_s = time.time() - t_inicio
    h, m = int(total_s // 3600), int(total_s % 3600 // 60)
    print("=" * 70)
    print(f"Mejor IoU val: {best_iou:.4f} | tiempo {h:02d}h{m:02d}m | modelos en {run_dir}")
    print("=" * 70)
    enviar_notificacion(f"Entrenamiento terminado. Mejor IoU: {best_iou:.4f}, tiempo {h:02d}h{m:02d}m")


if __name__ == "__main__":
    main()
