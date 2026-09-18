#!/usr/bin/env python3
"""
predict_vessels.py
------------------
Inferencia de segmentacion de embarcaciones sobre frames de dron, carpetas de
frames o video.

Cambios respecto al predictor de incendios:
  - Sin GDAL. No hay georreferencia que copiar: son fotogramas de dron, no
    escenas satelitales. Se lee con OpenCV y se normaliza /255.0.
  - 3 canales de entrada, 3 clases de salida (0=fondo, 1=boat, 2=ship).
  - Ventana deslizante de 256 con mezcla ponderada (ventana coseno) en vez de
    promedio uniforme: elimina las costuras visibles entre patches.
  - Post-proceso a cajas: componentes conexas -> bounding boxes -> CSV/JSON.
    Es lo que convierte la salida de segmentacion en deteccion utilizable.

Sobre --expand-px: si generaste las mascaras con build_masks.py --shrink-px 2,
el modelo aprendio blobs 2 px mas chicos que la caja real por lado. Este flag
compensa esa diferencia al recuperar las cajas. Manten los dos valores iguales.

Uso:
    python predict_vessels.py --model weights/model_best.pth --input frame.jpg \
        --output-dir out --visualization

    python predict_vessels.py --model weights/model_best.pth --input carpeta/ \
        --output-dir out --gpu auto

    python predict_vessels.py --model weights/model_best.pth --input vuelo.mp4 \
        --output-dir out --frame-stride 5
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from Vessel_models import UNet2D as VesselNet

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")

# BGR, para dibujar con OpenCV
CLASS_COLORS = {1: (0, 200, 255), 2: (0, 80, 255)}   # boat = ambar, ship = naranja


# --------------------------------------------------------------------------- #
def check_gpu_availability():
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.zeros(1).cuda()
        del t
        torch.cuda.empty_cache()
        return True
    except RuntimeError:
        return False


def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def tile_starts(total, tile, stride):
    """Mismo esquema que build_tiles.py: el ultimo tile se pega al borde."""
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def cosine_window(size, floor=0.02):
    """
    Ventana 2D separable con caida coseno en los bordes.

    Promediar patches con peso uniforme deja costuras: los pixeles del borde de
    un patch se predicen con poco contexto y valen lo mismo que los del centro.
    Ponderar por esta ventana hace que cada pixel lo dominen los patches que lo
    ven cerca de su centro.

    El piso se aplica DESPUES del producto exterior, no antes. Si se aplica a la
    ventana 1D, el minimo se eleva al cuadrado: con un piso de 1e-3 la esquina
    de la ventana acaba en 1e-6. El pixel (0,0) del frame lo cubre un unico
    patch, justo en esa esquina, asi que su peso acumulado seria 1e-6 y la
    division final perderia precision en fp32.
    """
    w = np.hanning(size + 2)[1:-1].astype(np.float32)
    return np.maximum(np.outer(w, w), floor).astype(np.float32)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_frame(model, bgr, device, num_classes, tile=256, overlap=64, batch=8):
    """
    Devuelve el mapa de probabilidades (num_classes, H, W) de un frame completo.
    """
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    pad_b = max(0, tile - H)
    pad_r = max(0, tile - W)
    if pad_b or pad_r:
        rgb = cv2.copyMakeBorder(rgb, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT_101)
    Hp, Wp = rgb.shape[:2]

    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    stride = max(1, tile - overlap)
    win = cosine_window(tile)

    acc = np.zeros((num_classes, Hp, Wp), dtype=np.float32)
    wsum = np.zeros((Hp, Wp), dtype=np.float32)

    coords = [(y, x) for y in tile_starts(Hp, tile, stride)
              for x in tile_starts(Wp, tile, stride)]

    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        patches = np.stack([chw[:, y:y + tile, x:x + tile] for y, x in chunk])
        t = torch.from_numpy(patches).to(device)
        probs = torch.softmax(model(t), dim=1).cpu().numpy()

        for (y, x), p in zip(chunk, probs):
            acc[:, y:y + tile, x:x + tile] += p * win
            wsum[y:y + tile, x:x + tile] += win

    acc /= np.maximum(wsum, 1e-6)
    return acc[:, :H, :W]


def probs_to_labels(probs, threshold=None):
    """
    argmax por defecto. Con --threshold se exige que P(primer plano) supere el
    umbral antes de asignar clase, lo que da un punto de operacion ajustable
    sin reentrenar.
    """
    if threshold is None:
        return probs.argmax(0).astype(np.uint8)
    fg = 1.0 - probs[0]
    lab = probs[1:].argmax(0).astype(np.uint8) + 1
    return np.where(fg > threshold, lab, 0).astype(np.uint8)


def labels_to_boxes(labels, num_classes, min_area=32, expand_px=0, shape=None):
    """
    Componentes conexas por clase -> cajas. Es el paso que devuelve la salida al
    dominio del dataset original (deteccion con bounding boxes).
    """
    H, W = shape if shape else labels.shape
    out = []
    for c in range(1, num_classes):
        binary = (labels == c).astype(np.uint8)
        if not binary.any():
            continue
        n, _, stats, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[k, cv2.CC_STAT_LEFT]) - expand_px
            y = int(stats[k, cv2.CC_STAT_TOP]) - expand_px
            w = int(stats[k, cv2.CC_STAT_WIDTH]) + 2 * expand_px
            h = int(stats[k, cv2.CC_STAT_HEIGHT]) + 2 * expand_px
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append({"class_id": c, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "area_px": area,
                        "cx": round(float(cent[k][0]), 2),
                        "cy": round(float(cent[k][1]), 2)})
    return out


def draw_overlay(bgr, labels, boxes, class_names, alpha=0.45):
    vis = bgr.copy()
    tint = np.zeros_like(bgr)
    for c, color in CLASS_COLORS.items():
        tint[labels == c] = color
    mask_any = labels > 0
    vis[mask_any] = cv2.addWeighted(bgr, 1 - alpha, tint, alpha, 0)[mask_any]

    for b in boxes:
        color = CLASS_COLORS.get(b["class_id"], (255, 255, 255))
        cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
        name = class_names[b["class_id"]] if b["class_id"] < len(class_names) else "?"
        cv2.putText(vis, name, (b["x1"], max(12, b["y1"] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


def colorize_mask(labels):
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for c, color in CLASS_COLORS.items():
        out[labels == c] = color
    return out


# --------------------------------------------------------------------------- #
def process_one(model, bgr, device, args, class_names):
    probs = predict_frame(model, bgr, device, args.num_classes,
                          args.tile, args.overlap, args.batch_size)
    labels = probs_to_labels(probs, args.threshold)
    boxes = labels_to_boxes(labels, args.num_classes, args.min_area,
                            args.expand_px, bgr.shape[:2])
    return probs, labels, boxes


def save_outputs(stem, bgr, probs, labels, boxes, out_dir, args, class_names):
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), labels)

    if args.visualization:
        cv2.imwrite(str(out_dir / f"{stem}_overlay.jpg"),
                    draw_overlay(bgr, labels, boxes, class_names),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(out_dir / f"{stem}_mask_color.png"), colorize_mask(labels))
        fg = ((1.0 - probs[0]) * 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{stem}_prob.png"),
                    cv2.applyColorMap(fg, cv2.COLORMAP_JET))

    with open(out_dir / f"{stem}_boxes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class_name", "x1", "y1", "x2", "y2", "cx", "cy", "area_px"])
        for b in boxes:
            name = class_names[b["class_id"]] if b["class_id"] < len(class_names) else "?"
            w.writerow([b["class_id"], name, b["x1"], b["y1"], b["x2"], b["y2"],
                        b["cx"], b["cy"], b["area_px"]])


def summarize(labels, boxes, class_names, num_classes):
    total = labels.size
    parts = []
    for c in range(1, num_classes):
        px = int((labels == c).sum())
        n = sum(1 for b in boxes if b["class_id"] == c)
        parts.append(f"{class_names[c]}: {n} obj, {px:,} px ({px / total * 100:.3f}%)")
    return "  |  ".join(parts)


# --------------------------------------------------------------------------- #
def run_images(model, paths, device, args, class_names, out_dir):
    all_rows = {}
    for i, path in enumerate(paths, 1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  [warn] no se pudo leer {path.name}")
            continue
        t0 = time.time()
        probs, labels, boxes = process_one(model, bgr, device, args, class_names)
        save_outputs(path.stem, bgr, probs, labels, boxes, out_dir, args, class_names)
        all_rows[path.name] = boxes
        print(f"  [{i}/{len(paths)}] {path.name}  {bgr.shape[1]}x{bgr.shape[0]}  "
              f"{time.time() - t0:.2f}s  ->  {summarize(labels, boxes, class_names, args.num_classes)}")

    with open(out_dir / "detections.json", "w", encoding="utf-8") as fh:
        json.dump({"class_names": class_names, "detections": all_rows}, fh, indent=2)
    print(f"\nDetecciones agregadas: {out_dir / 'detections.json'}")


def run_video(model, path, device, args, class_names, out_dir):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"ERROR: no se pudo abrir el video {path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {W}x{H} @ {fps:.1f} fps, {n_frames} frames, "
          f"procesando 1 de cada {args.frame_stride}")

    writer = cv2.VideoWriter(str(out_dir / f"{path.stem}_annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.frame_stride, (W, H))
    rows = []
    idx = kept = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.frame_stride != 0:
            idx += 1
            continue
        _, labels, boxes = process_one(model, frame, device, args, class_names)
        writer.write(draw_overlay(frame, labels, boxes, class_names))
        for b in boxes:
            rows.append({"frame": idx, **b})
        kept += 1
        if kept % 10 == 0:
            el = time.time() - t0
            print(f"\r  frame {idx}/{n_frames}  {kept / el:.2f} fps procesados",
                  end="", flush=True)
        idx += 1

    cap.release()
    writer.release()
    print(f"\n  Video anotado: {out_dir / (path.stem + '_annotated.mp4')}")

    with open(out_dir / f"{path.stem}_tracks.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "class_id", "class_name", "x1", "y1", "x2", "y2",
                    "cx", "cy", "area_px"])
        for r in rows:
            name = class_names[r["class_id"]] if r["class_id"] < len(class_names) else "?"
            w.writerow([r["frame"], r["class_id"], name, r["x1"], r["y1"],
                        r["x2"], r["y2"], r["cx"], r["cy"], r["area_px"]])
    print(f"  Detecciones por frame: {out_dir / (path.stem + '_tracks.csv')}")
    print("  Nota: son detecciones independientes por frame, sin tracking. "
          "Para IDs persistentes hace falta asociar entre frames (p. ej. IoU + Hungarian).")


def parse_args():
    p = argparse.ArgumentParser(description="Inferencia de embarcaciones aereas")
    p.add_argument("--model", required=True, help="Ruta al .pth entrenado")
    p.add_argument("--input", required=True, help="Imagen, carpeta o video")
    p.add_argument("--output-dir", default="predictions")

    p.add_argument("--gpu", default="auto", help="True/False/auto")
    p.add_argument("--num-classes", default=3, type=int)
    p.add_argument("--class-names", default="fondo,boat,ship")
    p.add_argument("--base-channels", default=24, type=int)
    p.add_argument("--no-stem", action="store_true")

    p.add_argument("--tile", default=256, type=int,
                   help="Debe coincidir con el usado en build_tiles.py")
    p.add_argument("--overlap", default=64, type=int)
    p.add_argument("-b", "--batch-size", default=8, type=int,
                   help="Patches por lote en la ventana deslizante")

    p.add_argument("--threshold", default=None, type=float,
                   help="Umbral sobre P(primer plano). Sin esto se usa argmax.")
    p.add_argument("--min-area", default=32, type=int,
                   help="Descarta componentes menores a N pixeles")
    p.add_argument("--expand-px", default=2, type=int,
                   help="Compensa el --shrink-px usado al generar las mascaras")
    p.add_argument("--visualization", action="store_true",
                   help="Genera overlay, mascara en color y mapa de probabilidad")
    p.add_argument("--frame-stride", default=1, type=int, help="Solo para video")
    return p.parse_args()


def main():
    args = parse_args()
    class_names = [n.strip() for n in args.class_names.split(",")][: args.num_classes]

    gpu = check_gpu_availability() if args.gpu.lower() == "auto" else args.gpu.lower() == "true"
    device = torch.device("cuda" if gpu else "cpu")

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"ERROR: no existe el modelo {model_path}")
        sys.exit(1)

    inp = Path(args.input)
    if not inp.exists():
        print(f"ERROR: no existe la entrada {inp}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("INFERENCIA - SEGMENTACION DE EMBARCACIONES")
    print("=" * 68)
    print(f"Modelo      : {model_path}")
    print(f"Entrada     : {inp}")
    print(f"Salida      : {out_dir}")
    print(f"Dispositivo : {'GPU' if gpu else 'CPU'}")
    print(f"Ventana     : {args.tile} px, solape {args.overlap}")
    print(f"Umbral      : {'argmax' if args.threshold is None else args.threshold}")
    print(f"Clases      : {', '.join(class_names)}")
    print("=" * 68)

    model = VesselNet(in_channels=3, out_channels=args.num_classes,
                      base_channels=args.base_channels, use_stem=not args.no_stem)
    try:
        state = safe_load(str(model_path), device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    except Exception as e:
        print(f"ERROR al cargar el modelo: {e}")
        print("  Si entrenaste con --no-stem o --base-channels distinto, "
              "pasa los mismos valores aqui.")
        sys.exit(1)
    model.to(device).eval()
    print("Modelo cargado.\n")

    if inp.is_dir():
        paths = [p for p in sorted(inp.iterdir()) if p.suffix.lower() in IMG_EXTS]
        if not paths:
            print(f"ERROR: no hay imagenes en {inp}")
            sys.exit(1)
        print(f"Procesando {len(paths)} imagenes...")
        run_images(model, paths, device, args, class_names, out_dir)
    elif inp.suffix.lower() in VIDEO_EXTS:
        print("Procesando video...")
        run_video(model, inp, device, args, class_names, out_dir)
    elif inp.suffix.lower() in IMG_EXTS:
        run_images(model, [inp], device, args, class_names, out_dir)
    else:
        print(f"ERROR: extension no reconocida: {inp.suffix}")
        sys.exit(1)

    print("\n" + "=" * 68)
    print("PREDICCION COMPLETADA")
    print("=" * 68)


if __name__ == "__main__":
    main()