#!/usr/bin/env python3
"""
trainModelWildfire3D4B.py
=========================
Entrenamiento de MbUNet2D (Wildfire_models.py, sin cambios) para segmentar área quemada.

Ejemplos:
  python trainModelWildfire3D4B.py --data-dir /ruta/Dataset_v2 --epochs 100 --lr 1e-3 -b 16 --gpu auto
  python trainModelWildfire3D4B.py --data-dir /ruta/Dataset_v2 --cache --workers 8      # dataset en RAM
  python trainModelWildfire3D4B.py --data-dir /ruta/Dataset_v2 --resume runs/20261015_130000/last.pth
  python trainModelWildfire3D4B.py --data-dir /ruta/Dataset_v2 --pesos-iniciales weights/model_best.pth --lr 2e-4

Si no se da --data-dir se usa $BASE_DIR/Mobile-UNet/Mobile-UNet_1/data (como antes, vía .env).

Cada corrida se guarda en runs/<fecha_hora>/:
  best.pth   mejor modelo en validación + metadatos (normalización, umbral, tamaño, métricas)
  last.pth   estado completo para reanudar (modelo, optimizador, scheduler, época)
  final.pth  modelo de la última época
  log.csv    métricas por época      config.json   parámetros

Cambios principales respecto a la versión anterior:
  * Batches reales: antes el optimizador daba un paso por imagen (batch efectivo = 1).
  * IoU/F1/precisión/recall acumulados sobre TODO el conjunto (antes promedio por imagen,
    y una imagen sin incendio bien predicha contaba como IoU = 0).
  * Pérdida CE + Dice con píxeles "sin dato" ignorados; peso de clase calculado del dataset.
  * Warmup + cosine LR, recorte de gradiente, early stopping, semilla.
  * Umbral óptimo buscado en validación y guardado en el checkpoint.
  * --resume explícito (antes reanudaba en silencio cualquier checkpoint viejo).
  * Recortes de otro tamaño se recortan/rellenan en vez de reescalarse (no cambia la resolución).
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import wildfire_common as wc
from Wildfire_models import MbUNet2D as WildfireNet

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

UMBRALES = [round(0.1 * i, 1) for i in range(1, 10)]   # 0.1 ... 0.9


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento - segmentación de área quemada")
    base = os.getenv("BASE_DIR")
    defecto = os.path.join(base, "Mobile-UNet/Mobile-UNet_26/data") if base else None
    p.add_argument("--data-dir", default=defecto, help="Carpeta con Images/, SegmentationClass/, train.txt, valid.txt")
    p.add_argument("--images-subdir", default="Images")
    p.add_argument("--masks-subdir", default="SegmentationClass")
    p.add_argument("--train-list", default="train.txt")
    p.add_argument("--val-list", default="valid.txt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("-b", "--batch-size", type=int, default=8)
    p.add_argument("--lr", "--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=5.0, help="0 = sin recorte de gradiente")
    p.add_argument("--gpu", default="auto", help="True/False/auto")
    p.add_argument("--image-size", type=int, default=128, help="Tamaño de entrada (recorta o rellena, no reescala)")
    p.add_argument("--workers", type=int, default=4, help="Procesos de carga de datos")
    p.add_argument("--cache", action="store_true", help="Cargar todo el dataset en RAM")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--loss", choices=["ce", "ce_dice"], default="ce_dice")
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--pos-weight", default="auto", help="Peso de la clase incendio en CE: 'auto' o número")
    p.add_argument("--norm", choices=["div", "zscore"], default="div",
                   help="div: DN/--norm-div recortado a [0,1] (cómodo para punto fijo); zscore: media/desv por banda")
    p.add_argument("--norm-div", type=float, default=12500.0)
    p.add_argument("--patience", type=int, default=25, help="Épocas sin mejora antes de parar (0 = desactivado)")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="Ruta a last.pth para reanudar")
    p.add_argument("--pesos-iniciales", default=None,
                   help="Iniciar desde unos pesos (.pth nuevo o antiguo) sin reanudar optimizador; útil para afinar")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-freq", type=int, default=10, help="Cada cuántos batches actualizar la línea de progreso")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def buscar_mascara(dir_masks, nombre_img):
    base = os.path.splitext(nombre_img)[0]
    for ext in (".tif", ".tiff", ".png", ".jpg"):
        ruta = os.path.join(dir_masks, base + ext)
        if os.path.exists(ruta):
            return ruta
    return None


class Augmentation:
    """Geométricas (flips, rot90) + fotométricas suaves. Usa el RNG de torch (seguro con varios workers)."""

    def __init__(self, recortar_a_unidad, p_flip=0.5, p_rotate=0.5, p_bright=0.3, p_noise=0.2,
                 bright_range=0.15, band_jitter=0.05, noise_std=0.02):
        self.recortar = recortar_a_unidad
        self.p_flip, self.p_rotate, self.p_bright, self.p_noise = p_flip, p_rotate, p_bright, p_noise
        self.bright_range, self.band_jitter, self.noise_std = bright_range, band_jitter, noise_std

    def __call__(self, image, mask):
        if torch.rand(1).item() < self.p_flip:
            image, mask = torch.flip(image, dims=[2]), torch.flip(mask, dims=[1])
        if torch.rand(1).item() < self.p_flip:
            image, mask = torch.flip(image, dims=[1]), torch.flip(mask, dims=[0])
        if torch.rand(1).item() < self.p_rotate:
            k = int(torch.randint(1, 4, (1,)).item())
            image, mask = torch.rot90(image, k, dims=[1, 2]), torch.rot90(mask, k, dims=[0, 1])
        if torch.rand(1).item() < self.p_bright:
            # factor global + pequeña variación por banda (no destruye las razones entre bandas)
            g = 1.0 + (torch.rand(1) * 2 - 1) * self.bright_range
            b = 1.0 + (torch.rand(image.shape[0], 1, 1) * 2 - 1) * self.band_jitter
            image = image * g * b
        if torch.rand(1).item() < self.p_noise:
            image = image + torch.randn_like(image) * self.noise_std
        if self.recortar:
            image = image.clamp(0.0, 1.0)
        return image.contiguous(), mask.contiguous()


class WildfireCropDataset(Dataset):
    def __init__(self, data_dir, lista, args, norm, augment):
        self.dir_img = os.path.join(data_dir, args.images_subdir)
        self.dir_mask = os.path.join(data_dir, args.masks_subdir)
        self.size = args.image_size
        self.norm = norm
        self.augment = Augmentation(recortar_a_unidad=(norm["tipo"] == "div")) if augment else None
        self.entrenamiento = augment

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
        if args.cache:
            t0 = time.time()
            self.cache = [self._leer(i) for i in range(len(self.pares))]
            mb = sum(a.nbytes + m.nbytes for a, m in self.cache) / 1e6
            print(f"  en RAM: {mb:.0f} MB ({time.time() - t0:.0f}s)")

    def __len__(self):
        return len(self.pares)

    def _leer(self, i):
        ruta_img, ruta_msk = self.pares[i]
        arr, _ = wc.leer_raster(ruta_img)
        if arr.shape[0] < 4:
            raise ValueError(f"{ruta_img}: {arr.shape[0]} bandas, se esperaban 4")
        img = arr[:4]
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
        img, msk = self._ajustar(img.astype(np.float32), msk)
        x = torch.from_numpy(wc.normalizar(img, self.norm))
        y = torch.from_numpy(msk.astype(np.int64))
        if self.augment is not None:
            x, y = self.augment(x, y)
        return x, y


def estadisticas_bandas(ds, max_imgs=1000, seed=0):
    """Media y desviación por banda (en DN) sobre píxeles válidos de una muestra del train."""
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    suma = np.zeros(4, np.float64)
    suma2 = np.zeros(4, np.float64)
    n = 0
    for i in idx[:max_imgs]:
        img = ds.leer_imagen_dn(i).reshape(4, -1)
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


def entrenar_epoca(model, loader, criterio, optimizer, device, args):
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
            raise RuntimeError(f"Pérdida no finita en el batch {i}: revisa lr / normalización de entrada")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        suma_loss += loss.item() * x.size(0)
        n += x.size(0)
        conf.update(torch.softmax(logits.detach(), 1)[:, 1], y)
        if (i + 1) % args.print_freq == 0 or i + 1 == len(loader):
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


def a_python(d):
    """Convierte valores numpy/torch a tipos de Python (para weights_only=True y JSON)."""
    salida = {}
    for k, v in d.items():
        if isinstance(v, dict):
            salida[k] = a_python(v)
        elif isinstance(v, (np.floating, np.integer)):
            salida[k] = v.item()
        elif isinstance(v, float) and math.isnan(v):
            salida[k] = None
        else:
            salida[k] = v
    return salida


def main():
    args = parse_args()
    if not args.data_dir:
        sys.exit("ERROR: indica --data-dir o define BASE_DIR en .env")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    device = elegir_dispositivo(args.gpu)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ckpt = None
    if args.resume:
        if not os.path.isfile(args.resume):
            sys.exit(f"ERROR: no existe el checkpoint {args.resume}")
        ckpt = cargar_checkpoint(args.resume, device)
        if "optimizer_state_dict" not in ckpt or "meta" not in ckpt:
            sys.exit("ERROR: ese checkpoint no es un last.pth de esta versión (no se puede reanudar)")

    # --- Normalización ---
    if ckpt is not None:
        norm = ckpt["meta"]["norm"]
    elif args.norm == "div":
        norm = {"tipo": "div", "valor": args.norm_div}
    else:
        norm = None  # se calcula con el train

    for ruta, nombre in [(args.data_dir, "data-dir"),
                         (os.path.join(args.data_dir, args.train_list), "lista train"),
                         (os.path.join(args.data_dir, args.val_list), "lista validación")]:
        if not os.path.exists(ruta):
            sys.exit(f"ERROR: no existe {nombre}: {ruta}")

    norm_tmp = norm or {"tipo": "div", "valor": args.norm_div}
    train_ds = WildfireCropDataset(args.data_dir, args.train_list, args, norm_tmp, augment=not args.no_augment)
    val_ds = WildfireCropDataset(args.data_dir, args.val_list, args, norm_tmp, augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        sys.exit("ERROR: train o validación vacíos")
    if norm is None:
        media, desv = estadisticas_bandas(train_ds, seed=args.seed)
        norm = {"tipo": "zscore", "media": media, "desv": desv}
        train_ds.norm = val_ds.norm = norm
        print(f"Normalización z-score: media {[round(m) for m in media]}, desv {[round(d) for d in desv]}")
    else:
        print(f"Normalización: {norm}")

    # --- Balance y peso de clase ---
    print("Contando píxeles por clase...")
    fuego, nofuego, ign = balance_clases(train_ds)
    total = max(fuego + nofuego, 1)
    print(f"Train: {100 * fuego / total:.2f}% píxeles quemados, {100 * ign / max(total + ign, 1):.2f}% sin dato")
    fv, nv, _ = balance_clases(val_ds)
    print(f"Valid: {100 * fv / max(fv + nv, 1):.2f}% píxeles quemados")
    if ckpt is not None:
        pos_weight = ckpt["meta"]["pos_weight"]
    elif args.pos_weight == "auto":
        pos_weight = float(np.clip(math.sqrt(nofuego / max(fuego, 1)), 1.0, 20.0))
    else:
        pos_weight = float(args.pos_weight)
    print(f"Peso clase incendio (CE): {pos_weight:.2f} | pérdida: {args.loss}")

    pin = device.type == "cuda"
    persist = args.workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=len(train_ds) > args.batch_size,
                              num_workers=args.workers, pin_memory=pin, persistent_workers=persist)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin, persistent_workers=persist)

    # --- Modelo ---
    model = WildfireNet(in_channels=4, out_channels=2).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Modelo MbUNet2D: {n_param / 1e6:.2f} M parámetros")
    criterio = SegLoss(pos_weight, usar_dice=(args.loss == "ce_dice"), dice_weight=args.dice_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warm = max(0, min(args.warmup_epochs, args.epochs - 1))
    coseno = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warm), eta_min=args.lr * 0.01)
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
        run_dir = os.path.dirname(os.path.abspath(args.resume))
        print(f"Reanudando en la época {start_epoch + 1}, mejor IoU val {best_iou:.4f}")
    else:
        if args.pesos_iniciales:
            ini = cargar_checkpoint(args.pesos_iniciales, device)
            sd = ini["model_state_dict"] if isinstance(ini, dict) and "model_state_dict" in ini else ini
            model.load_state_dict(sd)
            print(f"Pesos iniciales cargados de {args.pesos_iniciales}")
        nombre = args.run_name or time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.out_dir, nombre)
        os.makedirs(run_dir, exist_ok=True)

    meta_base = {
        "modelo": "MbUNet2D", "in_channels": 4, "out_channels": 2, "image_size": args.image_size,
        "orden_bandas": list(wc.BAND_ORDER), "unidades_entrada": "DN = reflectancia*10000",
        "norm": norm, "pos_weight": pos_weight,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(a_python({**vars(args), **meta_base, "n_train": len(train_ds), "n_val": len(val_ds),
                            "porcentaje_quemado_train": 100 * fuego / total}), f, indent=2)

    ruta_log = os.path.join(run_dir, "log.csv")
    nuevo_log = not os.path.exists(ruta_log)
    log = open(ruta_log, "a", newline="")
    escritor = csv.writer(log)
    if nuevo_log:
        escritor.writerow(["epoch", "lr", "train_loss", "train_iou", "val_loss", "val_iou", "val_f1",
                           "val_precision", "val_recall", "mejor_umbral", "iou_mejor_umbral", "segundos"])

    print("=" * 70)
    print(f"Entrenando: {len(train_ds)} train / {len(val_ds)} val | batch {args.batch_size} | "
          f"{args.epochs} épocas | salida {run_dir}")
    print("=" * 70)

    t_inicio = time.time()
    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            tr_loss, tr_m = entrenar_epoca(model, train_loader, criterio, optimizer, device, args)
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
            print(f"Época {epoch + 1:3d}/{args.epochs}  lr {lr:.2e}  loss {tr_loss:.4f}  IoU train {wc.fmt(tr_m['iou'])}"
                  f" | val loss {val['loss']:.4f}  IoU {wc.fmt(val['iou'])}  F1 {wc.fmt(val['f1'])}"
                  f"  P {wc.fmt(val['precision'], 3)}  R {wc.fmt(val['recall'], 3)}"
                  f"  (umbral {val['mejor_umbral']}: {wc.fmt(val['iou_mejor_umbral'])})  {dt:.0f}s"
                  + ("  <- mejor" if es_mejor else ""))

            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch, "best_iou": best_iou, "sin_mejora": sin_mejora,
                        "meta": a_python(meta_base)}, os.path.join(run_dir, "last.pth"))

            if args.patience > 0 and sin_mejora >= args.patience:
                print(f"Early stopping: {args.patience} épocas sin mejorar IoU de validación")
                break
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario; se guarda lo que hay (reanuda con --resume .../last.pth)")
    finally:
        log.close()

    guardar_modelo(os.path.join(run_dir, "final.pth"), model, a_python({**meta_base, "epoch": epoch}))
    total_s = time.time() - t_inicio
    h, m = int(total_s // 3600), int(total_s % 3600 // 60)
    print("=" * 70)
    print(f"Mejor IoU val: {best_iou:.4f} | tiempo {h:02d}h{m:02d}m | modelos en {run_dir}")
    print("=" * 70)
    enviar_notificacion(f"Entrenamiento terminado. Mejor IoU: {best_iou:.4f}, tiempo {h:02d}h{m:02d}m")


if __name__ == "__main__":
    main()
