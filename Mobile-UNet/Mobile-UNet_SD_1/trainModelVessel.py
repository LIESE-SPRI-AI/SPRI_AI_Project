#!/usr/bin/env python3
"""
trainModelVessel.py
-------------------
Entrenamiento QAT de UNet para segmentacion de embarcaciones aereas.

Entrada: los tiles de 256x256 generados por build_tiles.py

    $BASE_DIR/Dataset/Vessels/tiles/
        images/<split>/*.jpg     masks/<split>/*.png
        <split>.txt              manifest_<split>.csv

Clases: 0=fondo, 1=boat, 2=ship

Cambios de fondo respecto al pipeline de incendios:
  - Sin GDAL. RGB uint8 -> /255.0 (antes 4 bandas UInt16 -> /12500.0).
  - Batching real. El codigo anterior devolvia una lista sin apilar y hacia un
    optimizer.step() POR MUESTRA, o sea batch efectivo de 1 y N pasos por batch.
  - Metricas acumuladas a nivel de epoca via matriz de confusion, no promediadas
    por imagen. El IoU anterior devolvia 0.0 cuando la union era cero, asi que un
    tile de puro mar predicho correctamente puntuaba 0. Con tiles de mar abierto
    eso hunde la metrica sin que el modelo este mal.
  - Pesos de clase calculados del manifiesto, no fijados a mano.
  - Loss combinada CE + Dice, mas estable con desbalance fuerte.

Sin AMP a proposito: el redondeo de fp16 interactua mal con el fake-quantize del
QAT y ensucia el straight-through estimator.

Uso:
    export BASE_DIR=/home/liese2/SPRI_AI_project
    python trainModelVessel.py --epochs 60 -b 16 --lr 1e-3 --gpu auto
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import requests
except ImportError:
    requests = None

from Vessel_models import UNet2D as VesselNet


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
def resolve(base: Path, relative: str):
    """Resuelve `relative` bajo `base` tolerando diferencias de mayusculas."""
    cur = base
    for part in Path(relative).parts:
        if not cur.is_dir():
            return None
        exact = cur / part
        if exact.exists():
            cur = exact
            continue
        low = part.lower()
        match = next((c for c in cur.iterdir() if c.name.lower() == low), None)
        if match is None:
            return None
        cur = match
    return cur


def get_data_root(cli_root):
    if cli_root is not None:
        root = Path(cli_root).expanduser().resolve()
        if not root.is_dir():
            print(f"ERROR: --data-root no existe: {root}")
            sys.exit(1)
        return root
    base = os.getenv("BASE_DIR")
    if not base:
        print("ERROR: BASE_DIR no esta definida en el entorno.")
        print("  export BASE_DIR=/home/liese2/SPRI_AI_project")
        print("  ...o pasa --data-root /ruta/a/Dataset/Vessels")
        sys.exit(1)
    root = resolve(Path(base).expanduser().resolve(), "Dataset/Vessels")
    if root is None:
        print(f"ERROR: no encuentro 'Dataset/Vessels' dentro de BASE_DIR={base}")
        sys.exit(1)
    return root


def safe_load(path, device):
    """torch.load compatible con versiones con y sin weights_only."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
def check_gpu_availability():
    print(f"PyTorch version: {torch.__version__}")
    if not torch.cuda.is_available():
        print("CUDA no esta disponible en este sistema")
        return False
    try:
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        n = torch.cuda.device_count()
        print(f"GPUs disponibles: {n}")
        if n == 0:
            return False
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        t = torch.tensor([1.0], device="cuda:0")
        del t
        torch.cuda.empty_cache()
        print("GPU verificada y funcionando")
        return True
    except RuntimeError as e:
        print(f"Error al acceder a GPU: {e}")
        return False


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class VesselTileDataset(Dataset):
    """
    Lee tiles ya recortados a resolucion nativa. No reescala nada: el tamano del
    tile lo fijo build_tiles.py y cambiarlo aqui reintroduciria el colapso de
    resolucion que el tiling existe para evitar.
    """

    MAX_WARN = 10

    def __init__(self, tiles_root: Path, split: str, num_classes=3, augment=False):
        self.img_dir = tiles_root / "images" / split
        self.msk_dir = tiles_root / "masks" / split
        self.num_classes = num_classes
        self.augment = Augmentation() if augment else None
        self.n_errors = 0

        list_path = tiles_root / f"{split}.txt"
        if not list_path.is_file():
            raise FileNotFoundError(
                f"No existe {list_path}. Corre build_tiles.py --splits {split} primero.")
        with open(list_path, encoding="utf-8") as fh:
            self.names = [ln.strip() for ln in fh if ln.strip()]

        # El manifiesto trae el conteo de pixeles por clase de cada tile: sirve
        # para pesos de clase y muestreo balanceado sin releer el disco.
        self.counts = {}
        man = tiles_root / f"manifest_{split}.csv"
        if man.is_file():
            with open(man, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    self.counts[row["tile"]] = [
                        int(row[f"px_c{c}"]) for c in range(num_classes)
                    ]

        print(f"  {split}: {len(self.names):,} tiles"
              + (f", manifiesto con {len(self.counts):,} entradas" if self.counts else
                 ", SIN manifiesto"))

    def __len__(self):
        return len(self.names)

    def class_pixels(self):
        """Suma de pixeles por clase en todo el split (desde el manifiesto)."""
        if not self.counts:
            return None
        tot = np.zeros(self.num_classes, dtype=np.int64)
        for name in self.names:
            c = self.counts.get(name)
            if c:
                tot += np.asarray(c, dtype=np.int64)
        return tot

    def positive_flags(self):
        """Vector booleano: True si el tile contiene alguna embarcacion."""
        if not self.counts:
            return None
        return np.array(
            [sum(self.counts.get(n, [0] * self.num_classes)[1:]) > 0 for n in self.names],
            dtype=bool)

    def _blank(self, size=256):
        return (torch.zeros((3, size, size), dtype=torch.float32),
                torch.zeros((size, size), dtype=torch.long))

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = self.img_dir / f"{name}.jpg"
        msk_path = self.msk_dir / f"{name}.png"

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(msk_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or msk is None:
            self.n_errors += 1
            if self.n_errors <= self.MAX_WARN:
                print(f"\n  [warn] no se pudo leer el tile '{name}'; se usa uno vacio")
            return self._blank()

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0)

        if msk.ndim == 3:
            msk = msk[:, :, 0]
        # Clamp defensivo: un valor fuera de rango en la mascara provoca un
        # device-side assert en CrossEntropyLoss que es dificil de depurar.
        msk = np.clip(msk, 0, self.num_classes - 1)
        mask = torch.from_numpy(np.ascontiguousarray(msk).astype(np.int64))

        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return img, mask


class Augmentation:
    """
    Flips, rot90, brillo y ruido. Todas validas para vista cenital: no hay
    orientacion privilegiada en imagenes de dron mirando hacia abajo.
    No incluye reescalado a proposito, para no deshacer el tiling nativo.
    """

    def __init__(self, p_flip=0.5, p_rotate=0.5, p_bright=0.3, p_noise=0.2,
                 bright_range=0.15, noise_std=0.02):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_bright = p_bright
        self.p_noise = p_noise
        self.bright_range = bright_range
        self.noise_std = noise_std

    def __call__(self, image, mask):
        if torch.rand(1).item() < self.p_flip:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if torch.rand(1).item() < self.p_flip:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0])
        if torch.rand(1).item() < self.p_rotate:
            k = int(torch.randint(1, 4, (1,)).item())
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[0, 1])
        if torch.rand(1).item() < self.p_bright:
            factor = 1.0 + (torch.rand(image.shape[0], 1, 1) * 2 - 1) * self.bright_range
            image = torch.clamp(image * factor, 0.0, 1.0)
        if torch.rand(1).item() < self.p_noise:
            image = torch.clamp(image + torch.randn_like(image) * self.noise_std, 0.0, 1.0)
        return image.contiguous(), mask.contiguous()


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #
class ConfusionMatrix:
    """
    Acumula la matriz de confusion de toda la epoca en GPU.

    De aqui salen IoU, precision, recall y F1 por clase de una sola pasada.
    Es lo que reemplaza al IoU promediado por imagen del pipeline anterior:
    con tiles mayoritariamente vacios, ese promedio castigaba los aciertos
    triviales (union cero -> 0.0) y no reflejaba la calidad real.
    """

    def __init__(self, num_classes, device):
        self.nc = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    def reset(self):
        self.mat.zero_()

    @torch.no_grad()
    def update(self, target, pred):
        t = target.reshape(-1)
        p = pred.reshape(-1)
        k = (t >= 0) & (t < self.nc)
        idx = self.nc * t[k] + p[k]
        self.mat += torch.bincount(idx, minlength=self.nc ** 2).reshape(self.nc, self.nc)

    def compute(self):
        m = self.mat.double().cpu()
        tp = m.diag()
        support = m.sum(1)                 # pixeles reales de cada clase
        predicted = m.sum(0)               # pixeles predichos como cada clase
        fp = predicted - tp
        fn = support - tp

        denom_iou = tp + fp + fn
        iou = torch.where(denom_iou > 0, tp / denom_iou.clamp(min=1),
                          torch.full_like(tp, float("nan")))
        prec = torch.where(predicted > 0, tp / predicted.clamp(min=1),
                           torch.full_like(tp, float("nan")))
        rec = torch.where(support > 0, tp / support.clamp(min=1),
                          torch.full_like(tp, float("nan")))
        f1 = 2 * prec * rec / (prec + rec)

        # Las clases ausentes quedan NaN y se excluyen del promedio, en vez de
        # entrar como 0 y arrastrar la media hacia abajo.
        fg = iou[1:]
        miou_fg = float(np.nanmean(fg.numpy())) if fg.numel() else float("nan")
        return {"iou": iou.numpy(), "precision": prec.numpy(), "recall": rec.numpy(),
                "f1": f1.numpy(), "support": support.numpy(), "miou_fg": miou_fg}


def format_metrics(met, class_names):
    lines = [f"    {'clase':<10s} {'IoU':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'px reales':>14s}"]
    for c, name in enumerate(class_names):
        lines.append(f"    {name:<10s} {met['iou'][c]:8.4f} {met['precision'][c]:8.4f} "
                     f"{met['recall'][c]:8.4f} {met['f1'][c]:8.4f} {int(met['support'][c]):14,d}")
    lines.append(f"    mIoU (solo embarcaciones): {met['miou_fg']:.4f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
class CombinedLoss(nn.Module):
    """CrossEntropy ponderada + Dice sobre las clases de primer plano."""

    def __init__(self, class_weights, num_classes, dice_weight=0.5, eps=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.nc = num_classes
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits, target):
        loss = self.ce(logits, target)
        if self.dice_weight <= 0:
            return loss
        prob = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target, self.nc).permute(0, 3, 1, 2).to(prob.dtype)
        dims = (0, 2, 3)
        inter = (prob * onehot).sum(dims)
        card = prob.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.eps) / (card + self.eps)
        return loss + self.dice_weight * (1.0 - dice[1:].mean())


def compute_class_weights(class_px, cap=50.0):
    """
    Inversa de la raiz de la frecuencia, normalizada al fondo y con tope.
    La inversa pura (1/freq) da pesos de miles para clases raras y desestabiliza
    la loss en las primeras epocas. Misma formula que imprime build_tiles.py.
    """
    total = float(class_px.sum())
    freq = np.maximum(class_px.astype(np.float64) / max(total, 1.0), 1e-12)
    w = 1.0 / np.sqrt(freq)
    return np.minimum(w / w[0], cap)


# --------------------------------------------------------------------------- #
# Bucles
# --------------------------------------------------------------------------- #
def train_one_epoch(loader, model, criterion, optimizer, device, conf, epoch, args):
    model.train()
    conf.reset()
    running, seen = 0.0, 0
    t0 = time.time()

    for i, (images, targets) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()          # UN paso por batch, no uno por muestra

        bs = images.size(0)
        running += loss.item() * bs
        seen += bs
        conf.update(targets, logits.detach().argmax(1))

        if i % args.print_freq == 0 or i == len(loader):
            el = time.time() - t0
            ips = seen / el if el > 0 else 0.0
            print(f"\r  [{epoch + 1}] batch {i}/{len(loader)}  "
                  f"loss {running / max(seen, 1):.4f}  {ips:6.1f} tiles/s",
                  end="", flush=True)

    return running / max(seen, 1), conf.compute()


@torch.no_grad()
def validate(loader, model, criterion, device, conf):
    model.eval()
    conf.reset()
    running, seen = 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        running += criterion(logits, targets).item() * images.size(0)
        seen += images.size(0)
        conf.update(targets, logits.argmax(1))
    return running / max(seen, 1), conf.compute()


# --------------------------------------------------------------------------- #
def enviar_notificacion(mensaje):
    token = os.getenv("PUSHBULLET_API_TOKEN")
    if not token or requests is None:
        return
    try:
        requests.post("https://api.pushbullet.com/v2/pushes",
                      headers={"Access-Token": token, "Content-Type": "application/json"},
                      json={"type": "note", "title": "Alerta de entrenamiento SPRI",
                            "body": mensaje}, timeout=10)
        print(f"Notificacion enviada: {mensaje}")
    except Exception as e:
        print(f"No se pudo enviar la notificacion: {e}")


def parse_args():
    p = argparse.ArgumentParser(description="Vessel Segmentation Training (QAT UNet)")
    p.add_argument("--data-root", default=None, help="Por defecto: $BASE_DIR/Dataset/Vessels")
    p.add_argument("--tiles-subdir", default="tiles")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="valid")

    p.add_argument("--epochs", default=30, type=int)
    p.add_argument("-b", "--batch-size", default=8, type=int,
                   help="8 cabe en 16 GB sin checkpointing; con --grad-checkpoint "
                        "puedes subir a 16-24")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Recomputa los 4 bloques de mayor resolucion en el backward "
                        "en vez de guardar sus activaciones. ~30%% mas lento, "
                        "pero permite batch 2-3x mayor.")
    p.add_argument("--lr", "--learning-rate", default=1e-3, type=float)
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument("--grad-clip", default=1.0, type=float, help="0 para desactivar")
    p.add_argument("--gpu", default="auto", type=str, help="True/False/auto")
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--print-freq", default=20, type=int)

    p.add_argument("--num-classes", default=3, type=int)
    p.add_argument("--class-names", default="fondo,boat,ship")
    p.add_argument("--base-channels", default=24, type=int)
    p.add_argument("--no-stem", action="store_true",
                   help="Desactiva el stem RGB (para medir cuanto aporta)")

    p.add_argument("--dice-weight", default=0.5, type=float, help="0 para solo CE")
    p.add_argument("--class-weights", default=None,
                   help="Coma-separados. Por defecto se calculan del manifiesto.")
    p.add_argument("--weight-cap", default=50.0, type=float)
    p.add_argument("--pos-ratio", default=0.0, type=float,
                   help="Fraccion objetivo de tiles con embarcacion por epoca "
                        "(0 = sin muestreo balanceado; prueba 0.5 si el modelo "
                        "colapsa a predecir solo fondo)")

    p.add_argument("--resume", default="weights/checkpoint.pth", type=str)
    p.add_argument("--checkpoint-freq", default=5, type=int)
    p.add_argument("--subset", default=0, type=int, help="Usar solo N tiles (smoke test)")
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    data_root = get_data_root(args.data_root)
    tiles_root = data_root / args.tiles_subdir
    class_names = [n.strip() for n in args.class_names.split(",")][: args.num_classes]

    print("=" * 70)
    print("VERIFICANDO SISTEMA")
    print("=" * 70)
    if args.gpu.lower() in ("auto", "true") and check_gpu_availability():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("Usando CPU")

    if not tiles_root.is_dir():
        print(f"\nERROR: no existe {tiles_root}")
        print("  Corre primero:  python build_masks.py --splits all")
        print("                  python build_tiles.py --splits all")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("DATOS")
    print("=" * 70)
    print(f"DATA_PATH : {data_root}")
    print(f"tiles     : {tiles_root}")
    train_ds = VesselTileDataset(tiles_root, args.train_split, args.num_classes, augment=True)
    val_ds = VesselTileDataset(tiles_root, args.val_split, args.num_classes, augment=False)

    if args.subset:
        train_ds.names = train_ds.names[: args.subset]
        val_ds.names = val_ds.names[: max(1, args.subset // 4)]
        print(f"  SMOKE TEST: recortado a {len(train_ds)} / {len(val_ds)} tiles")

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("ERROR: algun split esta vacio.")
        sys.exit(1)

    # ---- pesos de clase ----
    if args.class_weights:
        w = np.array([float(v) for v in args.class_weights.split(",")], dtype=np.float64)
        origen = "linea de comandos"
    else:
        px = train_ds.class_pixels()
        if px is None:
            print("\nAVISO: sin manifiesto, no puedo calcular pesos. Usando 1.0 para todo.")
            w = np.ones(args.num_classes)
            origen = "por defecto"
        else:
            w = compute_class_weights(px, args.weight_cap)
            origen = "manifiesto"
            total = px.sum()
            print("\n  Distribucion de pixeles en train:")
            for c, name in enumerate(class_names):
                print(f"    {name:<10s} {px[c]:14,d} px  ({px[c] / total * 100:8.4f}%)")
    print(f"  Pesos de clase ({origen}): [" + ", ".join(f"{v:.3f}" for v in w) + "]")
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)

    # ---- sampler balanceado opcional ----
    sampler, shuffle = None, True
    if args.pos_ratio > 0:
        flags = train_ds.positive_flags()
        if flags is None:
            print("  AVISO: --pos-ratio necesita manifiesto; se ignora.")
        else:
            n_pos, n_neg = int(flags.sum()), int((~flags).sum())
            if n_pos and n_neg:
                wp = args.pos_ratio / n_pos
                wn = (1.0 - args.pos_ratio) / n_neg
                weights = np.where(flags, wp, wn)
                sampler = WeightedRandomSampler(
                    torch.as_tensor(weights, dtype=torch.double),
                    num_samples=len(train_ds), replacement=True)
                shuffle = False
                print(f"  Muestreo balanceado: {args.pos_ratio:.0%} positivos por epoca "
                      f"({n_pos:,} pos / {n_neg:,} neg en disco)")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle,
                              sampler=sampler, num_workers=args.num_workers,
                              pin_memory=pin, drop_last=True,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin,
                            persistent_workers=args.num_workers > 0)

    # ---- modelo ----
    model = VesselNet(in_channels=3, out_channels=args.num_classes,
                      base_channels=args.base_channels, use_stem=not args.no_stem,
                      use_checkpoint=args.grad_checkpoint).to(device)
    n_par = sum(p.numel() for p in model.parameters())

    # Las entradas son siempre 256x256, asi que dejar que cuDNN elija el mejor
    # algoritmo una vez al principio compensa.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("\n" + "=" * 70)
    print("CONFIGURACION - SEGMENTACION DE EMBARCACIONES")
    print("=" * 70)
    print(f"Dispositivo   : {device}")
    print(f"Parametros    : {n_par:,}")
    print(f"Batch size    : {args.batch_size}   workers: {args.num_workers}"
          + ("   grad-checkpoint: ON" if args.grad_checkpoint else ""))
    print(f"Learning rate : {args.lr}   weight decay: {args.weight_decay}")
    print(f"Epocas        : {args.epochs}")
    print(f"Loss          : CE ponderada" +
          (f" + {args.dice_weight} * Dice" if args.dice_weight > 0 else ""))
    print(f"Clases        : {', '.join(class_names)}")
    print("=" * 70)

    criterion = CombinedLoss(class_weights, args.num_classes, args.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    weights_dir = script_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    best_miou = 0.0
    start_epoch = 0
    resume_path = Path(args.resume)
    if not resume_path.is_absolute():
        resume_path = script_dir / resume_path
    if args.resume and resume_path.is_file():
        print(f"\nCargando checkpoint: {resume_path}")
        ck = safe_load(str(resume_path), device)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_miou = ck.get("best_miou", ck.get("best_iou", 0.0))
        print(f"Reanudando en epoca {start_epoch}, mejor mIoU: {best_miou:.4f}")
    elif args.resume:
        print(f"\nSin checkpoint previo en {resume_path}; empezando de cero.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = script_dir / f"training_log_{timestamp}.csv"
    log = open(log_path, "w", newline="", encoding="utf-8")
    logw = csv.writer(log)
    header = ["epoch", "lr", "train_loss", "val_loss", "train_miou", "val_miou"]
    for n in class_names[1:]:
        header += [f"val_iou_{n}", f"val_prec_{n}", f"val_rec_{n}", f"val_f1_{n}"]
    header.append("seconds")
    logw.writerow(header)

    conf_train = ConfusionMatrix(args.num_classes, device)
    conf_val = ConfusionMatrix(args.num_classes, device)

    print("\n" + "=" * 70)
    print("INICIANDO ENTRENAMIENTO")
    print("=" * 70)
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        try:
            tr_loss, tr_met = train_one_epoch(train_loader, model, criterion, optimizer,
                                              device, conf_train, epoch, args)
            va_loss, va_met = validate(val_loader, model, criterion, device, conf_val)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("\n\n" + "=" * 70)
            print("SIN MEMORIA EN GPU")
            print("=" * 70)
            print(f"Configuracion actual: batch {args.batch_size}, "
                  f"grad-checkpoint {'ON' if args.grad_checkpoint else 'OFF'}")
            print("\nOpciones, de menos a mas costosa en tiempo:")
            if not args.grad_checkpoint:
                print(f"  1. --grad-checkpoint            (mismo batch, ~30% mas lento)")
                print(f"  2. -b {max(1, args.batch_size // 2)}                        "
                      f"(la mitad del batch)")
            else:
                print(f"  1. -b {max(1, args.batch_size // 2)}")
                print(f"  2. --base-channels 16           (modelo mas estrecho)")
            print("  3. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (si hay "
                  "fragmentacion)")
            print("\nPara ver el consumo real por batch size:")
            print("  python3 Vessel_models.py --bench"
                  + (" --checkpoint" if args.grad_checkpoint else ""))
            print("=" * 70)
            log.close()
            sys.exit(1)

        scheduler.step(va_met["miou_fg"])
        dt = time.time() - t0

        is_best = va_met["miou_fg"] > best_miou
        if is_best:
            best_miou = va_met["miou_fg"]

        print(f"\r  epoca {epoch + 1}/{args.epochs}  "
              f"loss {tr_loss:.4f}/{va_loss:.4f}  "
              f"mIoU {tr_met['miou_fg']:.4f}/{va_met['miou_fg']:.4f}  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}  {dt:.0f}s"
              + ("   <- mejor" if is_best else "") + "        ")
        print("  Validacion:")
        print(format_metrics(va_met, class_names))

        row = [epoch, optimizer.param_groups[0]["lr"], tr_loss, va_loss,
               tr_met["miou_fg"], va_met["miou_fg"]]
        for c in range(1, args.num_classes):
            row += [va_met["iou"][c], va_met["precision"][c],
                    va_met["recall"][c], va_met["f1"][c]]
        row.append(dt)
        logw.writerow(row)
        log.flush()

        if is_best:
            torch.save(model.state_dict(), weights_dir / "model_best.pth")
            print(f"  Guardado model_best.pth (mIoU {best_miou:.4f})")

        if (epoch + 1) % args.checkpoint_freq == 0:
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_miou": best_miou,
                        "args": vars(args)}, weights_dir / "checkpoint.pth")

    log.close()
    torch.save(model.state_dict(), weights_dir / "model_final.pth")

    total = time.time() - t_start
    h, m, s = int(total // 3600), int((total % 3600) // 60), int(total % 60)
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print(f"Mejor mIoU (embarcaciones): {best_miou:.4f}")
    print(f"Pesos en : {weights_dir}/")
    print(f"Log en   : {log_path}")
    print(f"Tiempo   : {h:02d}h {m:02d}m {s:02d}s")
    print("=" * 70)

    enviar_notificacion(f"Entrenamiento completado. mIoU {best_miou:.4f}, {h:02d}h{m:02d}m")


if __name__ == "__main__":
    main()