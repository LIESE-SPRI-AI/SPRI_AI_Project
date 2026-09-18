#!/usr/bin/env python3
"""
build_tiles.py
--------------
Corta los frames del Aerial Vessels Detection Dataset y sus mascaras en tiles de
256x256 A RESOLUCION NATIVA (sin reescalar), para que las embarcaciones pequenas
no desaparezcan.

Por que no reescalar: un frame de 1920x1080 llevado a 128x128 encoge 15x en X y
8.4x en Y. Un barco de 40 px pasa a ~3x5 px y tras los 4 max-pool del encoder se
pierde antes de llegar al bottleneck. Ademas se destruye el aspect ratio.

Las rutas se derivan de BASE_DIR, igual que build_masks.py:

    $BASE_DIR/Dataset/Vessels/
        train/images/   train/masks/     <- ENTRADA
        valid/images/   valid/masks/
        test/images/    test/masks/
        tiles/                           <- SALIDA

Salida:
    tiles/images/<split>/  *.jpg    (RGB, calidad configurable)
    tiles/masks/<split>/   *.png    (1 canal, 0=fondo, 1=boat, 2=ship)
    tiles/<split>.txt              lista de nombres
    tiles/manifest_<split>.csv     conteo de pixeles por clase y tile

El manifiesto permite calcular pesos de clase y hacer muestreo balanceado en el
entrenamiento sin volver a recorrer el disco.

Uso:
    export BASE_DIR=/home/liese2/SPRI_AI_project
    python build_tiles.py --splits all
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ALL_SPLITS = ["train", "valid", "test"]


# --------------------------------------------------------------------------- #
def resolve(base: Path, relative: str) -> Path | None:
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


def get_data_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        root = cli_root.expanduser().resolve()
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


def tile_starts(total: int, tile: int, stride: int) -> list[int]:
    """
    Posiciones de inicio a lo largo de un eje. El ultimo tile se pega al borde
    para no perder la franja final ni meter relleno innecesario.
    """
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def pad_to(arr: np.ndarray, tile: int) -> np.ndarray:
    """Rellena con ceros si el frame es menor al tile (caso raro con >=720p)."""
    h, w = arr.shape[:2]
    if h >= tile and w >= tile:
        return arr
    ph, pw = max(0, tile - h), max(0, tile - w)
    return cv2.copyMakeBorder(arr, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)


# --------------------------------------------------------------------------- #
def run_split(split: str, data_root: Path, out_root: Path, neg_keep: float, args) -> bool:
    images_dir = resolve(data_root, f"{split}/{args.images_subdir}")
    if images_dir is None:
        images_dir = resolve(data_root, split)
    masks_dir = resolve(data_root, f"{split}/{args.masks_subdir}")

    if images_dir is None:
        print(f"[{split}] SALTADO: no encuentro la carpeta de imagenes")
        return False
    if masks_dir is None:
        print(f"[{split}] SALTADO: no encuentro '{split}/{args.masks_subdir}'. "
              f"Corre build_masks.py primero.")
        return False

    img_out = out_root / "images" / split
    msk_out = out_root / "masks" / split
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"[{split}] SALTADO: no hay imagenes en {images_dir}")
        return False

    print("-" * 70)
    print(f"[{split}]  tile={args.tile}  stride={args.stride}  neg-keep={neg_keep:.2f}")
    print(f"  imagenes : {images_dir}   ({len(images)} frames)")
    print(f"  mascaras : {masks_dir}")
    print(f"  salida   : {img_out}  |  {msk_out}")

    rng = random.Random(args.seed)
    manifest_path = out_root / f"manifest_{split}.csv"
    list_path = out_root / f"{split}.txt"

    nc = args.num_classes
    class_px = np.zeros(nc, dtype=np.int64)
    tiles_with_class = np.zeros(nc, dtype=np.int64)
    n_pos = n_neg_kept = n_neg_drop = n_missing = 0

    t0 = time.time()
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf, \
         open(list_path, "w", encoding="utf-8") as lf:

        writer = csv.writer(mf)
        writer.writerow(["tile", "src", "x", "y"] + [f"px_c{c}" for c in range(nc)])

        for idx, img_path in enumerate(images, 1):
            mask_path = masks_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                n_missing += 1
                continue

            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            msk = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if bgr is None or msk is None:
                n_missing += 1
                continue
            if msk.ndim == 3:
                msk = msk[:, :, 0]
            if bgr.shape[:2] != msk.shape[:2]:
                print(f"\n  [warn] tamanos distintos en {img_path.name}: "
                      f"{bgr.shape[:2]} vs {msk.shape[:2]}; se omite")
                n_missing += 1
                continue

            bgr = pad_to(bgr, args.tile)
            msk = pad_to(msk, args.tile)
            H, W = bgr.shape[:2]

            for y in tile_starts(H, args.tile, args.stride):
                for x in tile_starts(W, args.tile, args.stride):
                    m_tile = msk[y:y + args.tile, x:x + args.tile]
                    counts = np.bincount(m_tile.ravel(), minlength=nc)[:nc]
                    obj_px = int(counts[1:].sum())

                    positive = obj_px >= args.min_obj_px
                    if not positive:
                        if rng.random() >= neg_keep:
                            n_neg_drop += 1
                            continue
                        n_neg_kept += 1
                    else:
                        n_pos += 1

                    name = f"{img_path.stem}_{x:05d}_{y:05d}"
                    cv2.imwrite(str(img_out / f"{name}.jpg"),
                                bgr[y:y + args.tile, x:x + args.tile],
                                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                    cv2.imwrite(str(msk_out / f"{name}.png"), m_tile)

                    lf.write(name + "\n")
                    writer.writerow([name, img_path.name, x, y] + counts.tolist())

                    class_px += counts
                    tiles_with_class += (counts > 0).astype(np.int64)

            if idx % 20 == 0 or idx == len(images):
                el = time.time() - t0
                rate = idx / el if el > 0 else 0.0
                eta = (len(images) - idx) / rate if rate > 0 else 0.0
                print(f"\r  {idx}/{len(images)} frames ({idx / len(images) * 100:5.1f}%)  "
                      f"{rate:4.1f} f/s  ETA {eta / 60:5.1f} min  "
                      f"[+{n_pos} pos / +{n_neg_kept} neg]", end="", flush=True)

    total_px = int(class_px.sum())
    print(f"\n  Frames sin mascara/ilegibles : {n_missing:,}")
    print(f"  Tiles positivos              : {n_pos:,}")
    print(f"  Tiles negativos guardados    : {n_neg_kept:,}")
    print(f"  Tiles negativos descartados  : {n_neg_drop:,}")
    print(f"  TOTAL de tiles               : {n_pos + n_neg_kept:,}")

    if total_px == 0:
        print("  No se contaron pixeles: revisa las mascaras.")
        return False

    print("  Distribucion de pixeles:")
    for c in range(nc):
        print(f"    clase {c}: {class_px[c]:14,d} px  "
              f"({class_px[c] / total_px * 100:8.4f}%)  "
              f"en {tiles_with_class[c]:7,d} tiles")

    if split == "train":
        # Inversa de la raiz de la frecuencia, normalizada al fondo y con tope.
        # La inversa pura (1/freq) explota con clases raras y desestabiliza la loss.
        freq = np.maximum(class_px.astype(np.float64) / total_px, 1e-12)
        w = 1.0 / np.sqrt(freq)
        w = np.minimum(w / w[0], 50.0)
        print("\n  Pesos sugeridos para CrossEntropyLoss (inv-sqrt, tope 50):")
        print("    class_weights = torch.tensor([" +
              ", ".join(f"{v:.3f}" for v in w) + "])")

    print(f"\n  Manifiesto: {manifest_path}")
    print(f"  Lista     : {list_path}")
    print(f"  Tiempo    : {(time.time() - t0) / 60:.1f} min")
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Tiling a resolucion nativa")
    p.add_argument("--data-root", type=Path, default=None,
                   help="Por defecto: $BASE_DIR/Dataset/Vessels")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Por defecto: <data-root>/tiles")
    p.add_argument("--splits", nargs="+", default=["train"],
                   help="train valid test, o 'all'")
    p.add_argument("--images-subdir", default="images")
    p.add_argument("--masks-subdir", default="masks")

    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--stride", type=int, default=256,
                   help="256 = sin solape. Usa 192 o 128 para mas muestras.")
    p.add_argument("--num-classes", type=int, default=3,
                   help="Incluyendo el fondo. 3 para bg/boat/ship.")

    p.add_argument("--neg-keep", type=float, default=0.08,
                   help="Fraccion de tiles sin objetos que se conserva en train")
    p.add_argument("--neg-keep-eval", type=float, default=0.15,
                   help="Idem para valid/test: mas alto, para que la validacion se "
                        "parezca a un vuelo real donde casi todo es mar")
    p.add_argument("--min-obj-px", type=int, default=24,
                   help="Un tile con menos pixeles de objeto que esto cuenta como "
                        "negativo (recortes de borde con 3 px de barco son ruido)")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    splits = ALL_SPLITS if "all" in [s.lower() for s in args.splits] else args.splits

    data_root = get_data_root(args.data_root)
    out_root = args.out_dir or (data_root / "tiles")
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TILING A RESOLUCION NATIVA")
    print("=" * 70)
    print(f"BASE_DIR  : {os.getenv('BASE_DIR', '(no usada, --data-root explicito)')}")
    print(f"DATA_PATH : {data_root}")
    print(f"SALIDA    : {out_root}")
    print(f"splits    : {', '.join(splits)}")
    print(f"clases    : {args.num_classes} (0=fondo, 1=boat, 2=ship)")
    print("=" * 70)

    done = 0
    for split in splits:
        neg = args.neg_keep if split == "train" else args.neg_keep_eval
        if run_split(split, data_root, out_root, neg, args):
            done += 1

    print("-" * 70)
    print(f"Splits procesados: {done}/{len(splits)}")
    if done == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()