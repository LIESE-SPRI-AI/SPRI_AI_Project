#!/usr/bin/env python3
"""
build_masks.py
--------------
Convierte las cajas YOLO del Aerial Vessels Detection Dataset (Zenodo 7076145)
en mascaras de segmentacion semantica.

Las rutas se derivan de la variable de entorno BASE_DIR:

    $BASE_DIR/Dataset/Vessels/
        labels.txt
        train/images/            valid/images/            test/images/
        train/masks/   <- SALIDA de este script
        Annotations/Yolo/train/  Annotations/Yolo/valid/  Annotations/Yolo/test/

La resolucion es tolerante a mayusculas: 'Annotations/Yolo', 'annotations/yolo'
y 'ANNOTATIONS/YOLO' funcionan igual.

Dos estrategias:
  --strategy rect  (DEFECTO, opcion A)  Rectangulo relleno por caja. Sin GPU y
                                        sin decodificar el JPEG: solo lee la
                                        cabecera para sacar W/H.
  --strategy sam   (opcion B)           Pseudo-mascaras con Segment Anything en
                                        dos pasadas (imagen completa en batch +
                                        refinamiento por recorte), con QC y
                                        fallback al rectangulo.

labels.txt del dataset es:  person=0, boat=1, ship=2
Con --classes boat,ship se descarta 'person':

    yolo 1 (boat) -> 1      yolo 2 (ship) -> 2      fondo y person -> 0

Uso:
    export BASE_DIR=/home/liese2/SPRI_AI_project
    python build_masks.py --splits all --shrink-px 2

Requiere: opencv-python numpy pillow   (+ torch y segment-anything solo con SAM)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ALL_SPLITS = ["train", "valid", "test"]


# --------------------------------------------------------------------------- #
# Resolucion de rutas
# --------------------------------------------------------------------------- #
def resolve(base: Path, relative: str) -> Path | None:
    """
    Resuelve `relative` bajo `base` tolerando diferencias de mayusculas en cada
    componente. Devuelve None si algun tramo no existe.
    """
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


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def load_label_names(labels_txt: Path) -> list[str]:
    names = [ln.strip().lower() for ln in open(labels_txt, encoding="utf-8") if ln.strip()]
    if not names:
        raise ValueError(f"labels.txt vacio: {labels_txt}")
    return names


def build_class_map(names: list[str], keep_csv: str) -> dict[int, int]:
    """{yolo_id: valor_en_mascara}. Lo no listado se descarta. Valores desde 1."""
    lookup = {n: i for i, n in enumerate(names)}
    cmap: dict[int, int] = {}
    keep = [k.strip().lower() for k in keep_csv.split(",") if k.strip()]
    if not keep:
        raise ValueError("--classes no puede estar vacio")
    for new_id, name in enumerate(keep, start=1):
        if name not in lookup:
            raise ValueError(f"Clase '{name}' no esta en labels.txt ({names})")
        cmap[lookup[name]] = new_id
    return cmap


def image_size(path: Path) -> tuple[int, int] | None:
    """(W, H) leyendo solo la cabecera: evita decodificar 10k JPEG de 1080p."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return None if arr is None else (arr.shape[1], arr.shape[0])


def read_yolo_boxes(label_path: Path, W: int, H: int,
                    cmap: dict[int, int], stats: dict) -> list[tuple[int, np.ndarray]]:
    """Lee 'cls cx cy w h' normalizado -> [(valor_mascara, [x1,y1,x2,y2]), ...]."""
    out: list[tuple[int, np.ndarray]] = []
    if not label_path.exists():
        stats["no_label_file"] += 1
        return out

    for line in open(label_path, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
        except ValueError:
            stats["bad_lines"] += 1
            continue

        stats["boxes_seen"][cls] = stats["boxes_seen"].get(cls, 0) + 1
        if cls not in cmap:
            stats["boxes_dropped"] += 1
            continue

        x1 = int(np.clip(round((cx - bw / 2) * W), 0, W - 1))
        y1 = int(np.clip(round((cy - bh / 2) * H), 0, H - 1))
        x2 = int(np.clip(round((cx + bw / 2) * W), 1, W))
        y2 = int(np.clip(round((cy + bh / 2) * H), 1, H))

        if x2 - x1 < 2 or y2 - y1 < 2:
            stats["degenerate"] += 1
            continue
        out.append((cmap[cls], np.array([x1, y1, x2, y2], dtype=np.int32)))
    return out


def shrink_box(box: np.ndarray, px: int) -> np.ndarray:
    """
    Encoge la caja `px` pixeles por lado sin dejarla vacia.

    Motivo: en las marinas del dataset hay decenas de barcos amarrados en
    paralelo y sus cajas alineadas a ejes se tocan. Rasterizadas tal cual se
    funden en un solo blob y las componentes conexas del post-proceso cuentan 1
    donde hay 12 (verificado). Encoger separa las instancias sin sesgar el IoU,
    porque el ground truth ES el rectangulo rasterizado.
    """
    if px <= 0:
        return box
    x1, y1, x2, y2 = box
    mx = min(px, max(0, (x2 - x1 - 1) // 2))
    my = min(px, max(0, (y2 - y1 - 1) // 2))
    return np.array([x1 + mx, y1 + my, x2 - mx, y2 - my], dtype=np.int32)


# --------------------------------------------------------------------------- #
# QC y SAM (solo --strategy sam)
# --------------------------------------------------------------------------- #
def qc_mask(mask, box, min_fill, max_fill, min_inside):
    """Valida una mascara de SAM y devuelve su componente conexa mayor, o None."""
    x1, y1, x2, y2 = (int(v) for v in box)
    box_area = float((x2 - x1) * (y2 - y1))
    if box_area <= 0:
        return None
    total = float(mask.sum())
    if total < 1:
        return None
    inside = float(mask[y1:y2, x1:x2].sum())
    if inside / total < min_inside:      # SAM se fue al muelle o a la estela
        return None
    fill = inside / box_area
    if fill < min_fill or fill > max_fill:
        return None

    clipped = np.zeros_like(mask, dtype=np.uint8)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2].astype(np.uint8)
    n_cc, labels, cc, _ = cv2.connectedComponentsWithStats(clipped, connectivity=8)
    if n_cc <= 1:
        return None
    largest = 1 + int(np.argmax(cc[1:, cv2.CC_STAT_AREA]))
    out = labels == largest
    return out if out.sum() / box_area >= min_fill else None


class SamEngine:
    def __init__(self, checkpoint, model_type, device, backend="segment_anything"):
        import torch
        if backend == "mobile_sam":
            from mobile_sam import SamPredictor, sam_model_registry
        else:
            from segment_anything import SamPredictor, sam_model_registry
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        self.device = device
        self.torch = torch

    def masks_full_image(self, rgb, boxes):
        torch = self.torch
        with torch.no_grad():
            self.predictor.set_image(rgb)
            b = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
            b = self.predictor.transform.apply_boxes_torch(b, rgb.shape[:2])
            masks, _, _ = self.predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=b, multimask_output=False)
        return masks[:, 0].cpu().numpy().astype(bool)

    def mask_from_crop(self, rgb, box, pad_ratio, target_side):
        torch = self.torch
        H, W = rgb.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(x2 - x1, y2 - y1) * pad_ratio / 2
        cx1, cy1 = int(max(0, cx - half)), int(max(0, cy - half))
        cx2, cy2 = int(min(W, cx + half)), int(min(H, cy + half))

        crop = rgb[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return np.zeros((H, W), dtype=bool)
        scale = max(1.0, target_side / max(crop.shape[0], crop.shape[1]))
        crop_up = (cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                              interpolation=cv2.INTER_CUBIC) if scale > 1 else crop)

        with torch.no_grad():
            self.predictor.set_image(crop_up)
            local = np.array([(x1 - cx1) * scale, (y1 - cy1) * scale,
                              (x2 - cx1) * scale, (y2 - cy1) * scale], dtype=np.float32)
            masks, scores, _ = self.predictor.predict(box=local[None, :], multimask_output=True)
        best = masks[int(np.argmax(scores))]
        back = cv2.resize(best.astype(np.uint8), (crop.shape[1], crop.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        full = np.zeros((H, W), dtype=bool)
        full[cy1:cy2, cx1:cx2] = back
        return full


# --------------------------------------------------------------------------- #
# Procesamiento por imagen
# --------------------------------------------------------------------------- #
def process_rect(img_path, label_path, out_path, cmap, args, stats):
    size = image_size(img_path)
    if size is None:
        stats["unreadable"] += 1
        return
    W, H = size

    boxes = read_yolo_boxes(label_path, W, H, cmap, stats)
    mask = np.zeros((H, W), dtype=np.uint8)
    if not boxes:
        cv2.imwrite(str(out_path), mask)
        stats["empty_images"] += 1
        return

    # De mayor a menor area: las embarcaciones chicas quedan encima de las grandes.
    boxes.sort(key=lambda cb: -((cb[1][2] - cb[1][0]) * (cb[1][3] - cb[1][1])))
    for val, box in boxes:
        b = shrink_box(box, args.shrink_px)
        mask[b[1]:b[3], b[0]:b[2]] = val
        stats["painted"][val] = stats["painted"].get(val, 0) + 1

    cv2.imwrite(str(out_path), mask)


def process_sam(img_path, label_path, out_path, cmap, engine, args, stats):
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        stats["unreadable"] += 1
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    boxes = read_yolo_boxes(label_path, W, H, cmap, stats)
    mask = np.zeros((H, W), dtype=np.uint8)
    if not boxes:
        cv2.imwrite(str(out_path), mask)
        stats["empty_images"] += 1
        return

    boxes.sort(key=lambda cb: -((cb[1][2] - cb[1][0]) * (cb[1][3] - cb[1][1])))
    arr = np.stack([b for _, b in boxes]).astype(np.float32)

    coarse = None
    if not args.crop_only:
        try:
            coarse = engine.masks_full_image(rgb, arr)
        except RuntimeError as exc:
            print(f"\n  [warn] SAM imagen completa fallo: {exc}")

    for i, (val, box) in enumerate(boxes):
        side = max(box[2] - box[0], box[3] - box[1])
        final = None
        if coarse is not None and side >= args.small_side:
            final = qc_mask(coarse[i], box, args.min_fill, args.max_fill, args.min_inside)
            if final is not None:
                stats["from_full"] += 1
        if final is None:
            try:
                cand = engine.mask_from_crop(rgb, box, args.pad_ratio, args.crop_target)
                final = qc_mask(cand, box, args.min_fill, args.max_fill, args.min_inside)
                if final is not None:
                    stats["from_crop"] += 1
            except RuntimeError as exc:
                print(f"\n  [warn] SAM crop fallo: {exc}")
        if final is None:
            b = shrink_box(box, args.shrink_px)
            final = np.zeros((H, W), dtype=bool)
            final[b[1]:b[3], b[0]:b[2]] = True
            stats["from_rect"] += 1

        mask[final] = val
        stats["painted"][val] = stats["painted"].get(val, 0) + 1

    cv2.imwrite(str(out_path), mask)


# --------------------------------------------------------------------------- #
def new_stats() -> dict:
    return {"boxes_seen": {}, "boxes_dropped": 0, "degenerate": 0, "bad_lines": 0,
            "no_label_file": 0, "painted": {}, "empty_images": 0, "unreadable": 0,
            "skipped": 0, "from_full": 0, "from_crop": 0, "from_rect": 0}


def run_split(split: str, data_root: Path, names: list[str], cmap: dict[int, int],
              engine, args) -> dict | None:
    images_dir = resolve(data_root, f"{split}/{args.images_subdir}")
    if images_dir is None:
        images_dir = resolve(data_root, split)     # imagenes sueltas en el split
    labels_dir = resolve(data_root, f"{args.annot_subdir}/{split}")

    if images_dir is None:
        print(f"[{split}] SALTADO: no encuentro la carpeta de imagenes")
        return None
    if labels_dir is None:
        print(f"[{split}] SALTADO: no encuentro '{args.annot_subdir}/{split}'")
        return None

    out_dir = (args.out_dir if args.out_dir else data_root / split / args.masks_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"[{split}] SALTADO: no hay imagenes en {images_dir}")
        return None

    sample = images[: min(200, len(images))]
    matched = sum(1 for p in sample if (labels_dir / (p.stem + ".txt")).exists())

    print("-" * 70)
    print(f"[{split}]")
    print(f"  imagenes : {images_dir}   ({len(images)} archivos)")
    print(f"  etiquetas: {labels_dir}")
    print(f"  salida   : {out_dir}")
    print(f"  emparejamiento en muestra: {matched}/{len(sample)}")
    if matched == 0:
        print("  ERROR: ningun .txt coincide con el nombre de las imagenes. Saltado.")
        ejemplo = sample[0].stem if sample else "?"
        print(f"         se buscaba: {labels_dir / (ejemplo + '.txt')}")
        return None
    if matched < len(sample):
        print("  AVISO: hay imagenes sin .txt; se trataran como frames sin objetos.")

    stats = new_stats()
    t0 = time.time()
    for idx, img_path in enumerate(images, 1):
        out_path = out_dir / (img_path.stem + ".png")
        if out_path.exists() and not args.overwrite:
            stats["skipped"] += 1
            continue
        label_path = labels_dir / (img_path.stem + ".txt")

        if args.strategy == "rect":
            process_rect(img_path, label_path, out_path, cmap, args, stats)
        else:
            process_sam(img_path, label_path, out_path, cmap, engine, args, stats)

        if idx % 50 == 0 or idx == len(images):
            el = time.time() - t0
            rate = idx / el if el > 0 else 0.0
            eta = (len(images) - idx) / rate if rate > 0 else 0.0
            print(f"\r  {idx}/{len(images)} ({idx / len(images) * 100:5.1f}%)  "
                  f"{rate:6.1f} img/s  ETA {eta / 60:5.1f} min", end="", flush=True)

    print()
    print("  Cajas leidas por clase YOLO:")
    for yid in sorted(stats["boxes_seen"]):
        nm = names[yid] if yid < len(names) else f"id{yid}"
        flag = "" if yid in cmap else "    <- DESCARTADA"
        print(f"    {yid} '{nm}': {stats['boxes_seen'][yid]:,}{flag}")
    print(f"  Instancias pintadas : "
          f"{ {k: v for k, v in sorted(stats['painted'].items())} }")
    print(f"  Degeneradas / sin .txt / sin objetos / ilegibles / ya existentes: "
          f"{stats['degenerate']:,} / {stats['no_label_file']:,} / "
          f"{stats['empty_images']:,} / {stats['unreadable']:,} / {stats['skipped']:,}")

    if args.strategy == "sam":
        tot = stats["from_full"] + stats["from_crop"] + stats["from_rect"]
        if tot:
            print(f"  Origen: SAM-full {stats['from_full']/tot*100:.1f}% | "
                  f"SAM-crop {stats['from_crop']/tot*100:.1f}% | "
                  f"rect {stats['from_rect']/tot*100:.1f}%")

    print(f"  Tiempo: {(time.time() - t0) / 60:.1f} min")

    report = out_dir.parent / f"masks_report_{split}.json"
    with open(report, "w", encoding="utf-8") as fh:
        json.dump({"split": split, "classes": names,
                   "class_map": {str(k): v for k, v in cmap.items()},
                   "strategy": args.strategy, "shrink_px": args.shrink_px,
                   "images_dir": str(images_dir), "labels_dir": str(labels_dir),
                   "out_dir": str(out_dir), "stats": stats}, fh, indent=2)
    return stats


def parse_args():
    p = argparse.ArgumentParser(description="Mascaras de segmentacion desde cajas YOLO")
    p.add_argument("--data-root", type=Path, default=None,
                   help="Por defecto: $BASE_DIR/Dataset/Vessels")
    p.add_argument("--splits", nargs="+", default=["train"],
                   help="train valid test, o 'all'")
    p.add_argument("--images-subdir", default="images")
    p.add_argument("--annot-subdir", default="Annotations/Yolo")
    p.add_argument("--masks-subdir", default="masks")
    p.add_argument("--labels-txt", type=Path, default=None,
                   help="Por defecto: <data-root>/labels.txt")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Override de la salida. Solo valido con un unico split.")

    p.add_argument("--strategy", choices=["rect", "sam"], default="rect")
    p.add_argument("--classes", default="boat,ship",
                   help="Clases a conservar, separadas por coma y en orden. "
                        "El resto se descarta (queda como fondo).")
    p.add_argument("--shrink-px", type=int, default=2,
                   help="Encoger cada rectangulo N px por lado. Evita que barcos "
                        "amarrados en paralelo se fundan en un solo blob.")

    p.add_argument("--sam-checkpoint", default=None)
    p.add_argument("--sam-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--backend", default="segment_anything",
                   choices=["segment_anything", "mobile_sam"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--small-side", type=int, default=64)
    p.add_argument("--pad-ratio", type=float, default=2.5)
    p.add_argument("--crop-target", type=int, default=512)
    p.add_argument("--crop-only", action="store_true")
    p.add_argument("--min-fill", type=float, default=0.10)
    p.add_argument("--max-fill", type=float, default=0.97)
    p.add_argument("--min-inside", type=float, default=0.70)

    p.add_argument("--limit", type=int, default=0, help="Procesar solo N imagenes (debug)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    splits = ALL_SPLITS if "all" in [s.lower() for s in args.splits] else args.splits
    if args.out_dir is not None and len(splits) > 1:
        print("ERROR: --out-dir solo puede usarse con un unico split.")
        sys.exit(1)

    data_root = get_data_root(args.data_root)

    labels_txt = args.labels_txt or resolve(data_root, "labels.txt")
    if labels_txt is None or not Path(labels_txt).is_file():
        print(f"ERROR: no encuentro labels.txt en {data_root}")
        print("  Pasalo explicitamente con --labels-txt")
        sys.exit(1)
    labels_txt = Path(labels_txt)

    names = load_label_names(labels_txt)
    cmap = build_class_map(names, args.classes)

    print("=" * 70)
    print(f"GENERACION DE MASCARAS  (estrategia: {args.strategy})")
    print("=" * 70)
    print(f"BASE_DIR   : {os.getenv('BASE_DIR', '(no usada, --data-root explicito)')}")
    print(f"DATA_PATH  : {data_root}")
    print(f"labels.txt : {labels_txt}")
    print(f"splits     : {', '.join(splits)}")
    print("\nMapeo de clases:")
    for yid, name in enumerate(names):
        tgt = cmap.get(yid)
        etiqueta = f"  yolo {yid} '{name}'"
        print(etiqueta.ljust(26) + ("-> valor " + str(tgt) if tgt else "-> IGNORADA"))
    print("  fondo".ljust(26) + "-> valor 0")
    print(f"\nCanales de salida del modelo: {len(cmap) + 1}")
    print(f"Shrink de rectangulos       : {args.shrink_px} px por lado")
    print("=" * 70)

    engine = None
    if args.strategy == "sam":
        if not args.sam_checkpoint:
            print("ERROR: --strategy sam requiere --sam-checkpoint")
            sys.exit(1)
        import torch
        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("AVISO: CUDA no disponible -> CPU. Esto tardara dias, no horas.")
            device = "cpu"
        print(f"\nCargando SAM ({args.sam_type}) en {device}...")
        engine = SamEngine(args.sam_checkpoint, args.sam_type, device, args.backend)
        print("SAM listo.")

    done = 0
    for split in splits:
        if run_split(split, data_root, names, cmap, engine, args) is not None:
            done += 1

    print("-" * 70)
    print(f"Splits procesados: {done}/{len(splits)}")
    if done == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()