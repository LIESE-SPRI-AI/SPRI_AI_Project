#!/usr/bin/env python3
"""
comparador_mbun.py
==================
Compara las inferencias de dos modelos MbUN contra la capa etiquetada de ESA (Burned Area).

Archivos esperados (los que genera inferencias.py):
  DIR_BURNED        : <nombrebase>_Burned.tiff          RGB(A), quemado en ROJO, alfa 0 = sin dato
  DIR_INFERENCIAS_A : <nombrebase>_<nombremodelo>.tif   1 banda uint8: 0 = no quemado, 1 = quemado,
  DIR_INFERENCIAS_B : <nombrebase>_<nombremodelo>.tif                  255 = sin dato

  nombremodelo es lo que pone inferencias.py: "<corrida>_best" para runs/<corrida>/best.pth,
  o "<nombre>" para weights/model_<nombre>.pth.

Las métricas se calculan solo en los píxeles válidos en los TRES rasters (referencia y ambos
modelos), así los dos modelos se evalúan exactamente sobre los mismos píxeles.

Salida en DIR_SALIDA:
  <nombrebase>_comparacion.png   figura por escena
  comparacion_global.png         barras con las métricas acumuladas de todas las escenas
  metricas.csv                   métricas por escena y GLOBAL (acumulando VP/FP/FN/VN)

Toda la configuración está en el bloque CONFIGURACIÓN (no hay argumentos por terminal).
"""

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import wildfire_common as wc

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
# --- Rutas (BASE_DIR viene del .env) ---
DIR_BURNED = Path(wc.ruta_proyecto("Inferencias", "Burned"))
SUFIJO_BURNED = "_Burned"                 # <nombrebase>_Burned.tif/.tiff (no distingue mayúsculas)
DIR_INFERENCIAS_A = Path(wc.ruta_proyecto("Inferencias", "Output_Inferencias_Mobile-UNet", "Output_MbUN_27"))
DIR_INFERENCIAS_B = Path(wc.ruta_proyecto("Inferencias", "Output_Inferencias_Mobile-UNet", "Output_MbUN_27"))
DIR_SALIDA = Path(wc.ruta_proyecto("Inferencias", "Resultados_MbUN_23_vs_27"))

# --- Qué archivo de cada carpeta usar ---
# nombremodelo del archivo <nombrebase>_<MODELO>.tif, p. ej. "20261109_153000_best".
# None = el único .tif de la carpeta que empiece con <nombrebase>_ (si hay varios, se avisa y se omite).
MODELO_A = None
MODELO_B = None

# --- Nombres para las figuras ---
NOMBRE_MODELO_A = "MbUN 23 (8 bits, RGB+NIR)"
NOMBRE_MODELO_B = "MbUN 27 (16 bits, +NDVI+BAI)"

# --- Referencia ESA: mismos umbrales que generate_wildfire_dataset.py (escala 0-1) ---
UMBRAL_ROJO = 0.75                        # R >= 0.75 (antes 255 con ">" en 0-255: nunca se cumplía)
UMBRAL_GB = 0.2                           # G y B <= 0.2

# --- Inferencias ---
VALOR_QUEMADO = 1
VALOR_SIN_DATO = 255                      # wc.IGNORE_INDEX; None si las máscaras no tienen "sin dato"

# --- Figuras ---
DPI = 200
FIGURA_GLOBAL = True
EXTENSIONES = (".tif", ".tiff")
EXCLUIR_SUFIJOS = ("_prob",)              # no confundir con mapas de probabilidad de predict
# =============================================================================

COLORES = {"A": "#2980b9", "B": "#c0392b"}
GRIS_SIN_DATO = (190, 190, 190)
METRICAS = [("Precisión", "precision"), ("Recall", "recall"), ("F1-Score", "f1"), ("IoU", "iou"),
            ("Exactitud", "exactitud"), ("Especificidad", "especificidad")]


# ───────────────────────────────────────────────────────────────────
# BÚSQUEDA DE ARCHIVOS
# ───────────────────────────────────────────────────────────────────

def buscar_burned(directorio: Path):
    """[(nombrebase, ruta)] de todos los <nombrebase>_Burned.tif/.tiff (sin distinguir mayúsculas)."""
    suf = SUFIJO_BURNED.lower()
    salida = []
    for f in sorted(directorio.iterdir()):
        if f.is_file() and f.suffix.lower() in EXTENSIONES and f.stem.lower().endswith(suf):
            salida.append((f.stem[: -len(suf)], f))
    return salida


def buscar_inferencia(directorio: Path, nombrebase: str, modelo):
    """Devuelve (ruta o None, mensaje)."""
    base = nombrebase.lower()
    candidatos = [f for f in sorted(directorio.iterdir())
                  if f.is_file() and f.suffix.lower() in EXTENSIONES
                  and f.stem.lower().startswith(base + "_")
                  and not any(f.stem.lower().endswith(s) for s in EXCLUIR_SUFIJOS)]
    if modelo is not None:
        exacto = [f for f in candidatos if f.stem.lower() == f"{base}_{modelo}".lower()]
        if exacto:
            return exacto[0], ""
        return None, f"no existe {nombrebase}_{modelo}.tif en {directorio}"
    if len(candidatos) == 1:
        return candidatos[0], ""
    if not candidatos:
        return None, f"no hay {nombrebase}_*.tif en {directorio}"
    nombres = [f.stem[len(nombrebase) + 1:] for f in candidatos]
    return None, f"{len(candidatos)} modelos para {nombrebase} en {directorio}: {nombres} -> define MODELO_A/MODELO_B"


# ───────────────────────────────────────────────────────────────────
# CARGA
# ───────────────────────────────────────────────────────────────────

def cargar_referencia(ruta):
    """Máscara uint8 0/1/255 + meta. Misma función que usa el generador del dataset."""
    gt, info = wc.extraer_mascara_quemado(str(ruta), umbral_rojo=UMBRAL_ROJO, umbral_gb=UMBRAL_GB)
    _, meta = wc.leer_raster(str(ruta))
    print(f"  [ESA]      {info['pixeles_quemados']:>10,} px quemados  "
          f"({100 * info['fraccion_quemada']:.2f}% de lo válido)")
    return gt, meta


def cargar_inferencia(ruta, forma_ref, meta_ref, etiqueta):
    """Máscara uint8 0/1/255 alineada a la rejilla de la referencia."""
    arr, meta = wc.leer_raster(str(ruta))
    m = arr[0]
    if m.dtype.kind == "f":
        m = np.nan_to_num(m, nan=VALOR_SIN_DATO if VALOR_SIN_DATO is not None else 0)
    m = m.astype(np.int64)
    sin_dato = (m == VALOR_SIN_DATO) if VALOR_SIN_DATO is not None else np.zeros(m.shape, bool)
    otros = np.setdiff1d(np.unique(m[~sin_dato]), [0, VALOR_QUEMADO])
    if otros.size:
        print(f"  ⚠ [{etiqueta}] valores inesperados {otros[:10].tolist()}: se cuentan como quemado si > 0")
    mascara = ((m > 0) & ~sin_dato).astype(np.uint8)
    mascara[sin_dato] = wc.IGNORE_INDEX

    if not wc._mismo_grid(meta, mascara.shape, meta_ref, forma_ref):
        print(f"  [{etiqueta}] Alineando {mascara.shape} -> {tuple(forma_ref)} (vecino más cercano)")
        mascara = wc.remuestrear(mascara, meta, forma_ref, meta_ref, vecino_mas_cercano=True,
                                 nodata=wc.IGNORE_INDEX)
    n = int((mascara == 1).sum())
    v = int((mascara != wc.IGNORE_INDEX).sum())
    print(f"  [{etiqueta}] {n:>10,} px quemados  ({100 * n / max(v, 1):.2f}% de lo válido)")
    return mascara


# ───────────────────────────────────────────────────────────────────
# MÉTRICAS
# ───────────────────────────────────────────────────────────────────

def conteos(ref, pred, validos):
    return {"vp": int((ref & pred & validos).sum()), "fp": int((~ref & pred & validos).sum()),
            "fn": int((ref & ~pred & validos).sum()), "vn": int((~ref & ~pred & validos).sum())}


def metricas_desde_conteos(c):
    vp, fp, fn, vn = c["vp"], c["fp"], c["fn"], c["vn"]
    total = vp + fp + fn + vn
    n_ref, n_pred = vp + fn, vp + fp

    def div(a, b):
        return a / b if b > 0 else 0.0

    precision, recall = div(vp, vp + fp), div(vp, vp + fn)
    return dict(
        **c, total=total, n_ref=n_ref, n_pred=n_pred,
        pixeles_error=fp + fn,
        pct_error=div(fp + fn, n_ref) * 100 if n_ref > 0 else div(fp + fn, total) * 100,
        precision=precision, recall=recall,
        f1=div(2 * precision * recall, precision + recall),
        iou=div(vp, vp + fp + fn),
        exactitud=div(vp + vn, total),
        especificidad=div(vn, vn + fp),
    )


# ───────────────────────────────────────────────────────────────────
# IMÁGENES
# ───────────────────────────────────────────────────────────────────

def mascara_a_rgb(mascara, validos):
    """Fondo blanco, quemado rojo, sin dato gris."""
    rgb = np.full((*mascara.shape, 3), 255, dtype=np.uint8)
    rgb[mascara] = [255, 0, 0]
    rgb[~validos] = GRIS_SIN_DATO
    return rgb


def imagen_diferencias(ref, pred, validos):
    """VP rojo · FN azul oscuro · FP azul claro · VN blanco · sin dato gris."""
    img = np.full((*ref.shape, 3), 255, dtype=np.uint8)
    img[ref & pred] = [255, 0, 0]
    img[ref & ~pred] = [0, 0, 150]
    img[~ref & pred] = [0, 0, 255]
    img[~validos] = GRIS_SIN_DATO
    return img


# ───────────────────────────────────────────────────────────────────
# FIGURAS
# ───────────────────────────────────────────────────────────────────

def _panel(ax, img_rgb, titulo, subtitulo, color_titulo):
    ax.imshow(img_rgb, interpolation="nearest")
    ax.set_title(titulo, fontsize=11, fontweight="bold", color=color_titulo, pad=5)
    ax.set_xlabel(subtitulo, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _barras(ax, ma, mb, nombre_a, nombre_b, titulo="Comparación de resultados por modelo"):
    nombres = [n for n, _ in METRICAS]
    keys = [k for _, k in METRICAS]
    x, w = np.arange(len(nombres)), 0.36

    def dibujar(vals, offset, color, label):
        barras = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85,
                        edgecolor="black", linewidth=0.6)
        for bar, v in zip(barras, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=8.5)

    dibujar([ma[k] * 100 for k in keys], -w / 2, COLORES["A"], f"Modelo A: {nombre_a}")
    dibujar([mb[k] * 100 for k in keys], +w / 2, COLORES["B"], f"Modelo B: {nombre_b}")
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Porcentaje (%)", fontsize=11)
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.axhline(80, color="green", linestyle="--", alpha=0.45, linewidth=1.2, label="Excelente ≥80%")
    ax.axhline(60, color="orange", linestyle="--", alpha=0.45, linewidth=1.2, label="Aceptable ≥60%")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def visualizar(nombrebase, rgb_ref, da, db, nombre_a, nombre_b, ruta_salida):
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[3, 3, 2.2], hspace=0.40, wspace=0.08)

    ma, mb = da["m"], db["m"]
    _panel(fig.add_subplot(gs[0, 0]), rgb_ref, "Píxeles etiquetados ESA",
           f"Píxeles quemados: {ma['n_ref']:,} px", "black")
    _panel(fig.add_subplot(gs[0, 1]), da["rgb"], nombre_a, f"Detectados: {ma['n_pred']:,} px", COLORES["A"])
    _panel(fig.add_subplot(gs[0, 2]), db["rgb"], nombre_b, f"Detectados: {mb['n_pred']:,} px", COLORES["B"])

    ax_ley = fig.add_subplot(gs[1, 0])
    ax_ley.axis("off")
    parches = [
        mpatches.Patch(color="red", label="Verdadero Positivo (VP)\nQuemado real, detectado"),
        mpatches.Patch(color="#000096", label="Falso Negativo (FN)\nQuemado real, NO detectado"),
        mpatches.Patch(color="blue", label="Falso Positivo (FP)\nDetectado sin ser quemado"),
        mpatches.Patch(facecolor="white", edgecolor="gray", label="Verdadero Negativo (VN)\nFondo correcto"),
        mpatches.Patch(color=np.array(GRIS_SIN_DATO) / 255, label="Sin dato\n(no entra en las métricas)"),
    ]
    ax_ley.legend(handles=parches, loc="center", fontsize=10, title="DIFERENCIAS", title_fontsize=11,
                  framealpha=0.95, edgecolor="gray")

    for col, d, nombre, clave in ((1, da, nombre_a, "A"), (2, db, nombre_b, "B")):
        m = d["m"]
        _panel(fig.add_subplot(gs[1, col]), d["diff"], f"Diferencias — {nombre} vs ESA",
               f"F1={m['f1'] * 100:.1f}%   IoU={m['iou'] * 100:.1f}%   Recall={m['recall'] * 100:.1f}%   "
               f"Precisión={m['precision'] * 100:.1f}%", COLORES[clave])

    _barras(fig.add_subplot(gs[2, :]), ma, mb, nombre_a, nombre_b)
    fig.suptitle(f"Comparación de modelos MbUN — {nombrebase}", fontsize=15, fontweight="bold", y=1.01)
    fig.savefig(str(ruta_salida), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figura guardada: {ruta_salida}")


def figura_global(ma, mb, n, ruta_salida):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    _barras(ax, ma, mb, NOMBRE_MODELO_A, NOMBRE_MODELO_B,
            titulo=f"Métricas globales ({n} escenas, píxeles acumulados)")
    fig.savefig(str(ruta_salida), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura global: {ruta_salida}")


# ───────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────

def imprimir_tabla(ma, mb):
    print(f"\n  {'Métrica':<16} {'Modelo A':>10} {'Modelo B':>10}")
    print(f"  {'─' * 38}")
    for lbl, k in METRICAS:
        print(f"  {lbl:<16} {ma[k] * 100:>9.2f}% {mb[k] * 100:>9.2f}%")


def main():
    for d, nombre in [(DIR_BURNED, "DIR_BURNED"), (DIR_INFERENCIAS_A, "DIR_INFERENCIAS_A"),
                      (DIR_INFERENCIAS_B, "DIR_INFERENCIAS_B")]:
        if not d.is_dir():
            sys.exit(f"[ERROR] Directorio no encontrado: {d}  ({nombre})")
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    referencias = buscar_burned(DIR_BURNED)
    if not referencias:
        sys.exit(f"[ERROR] No hay archivos *{SUFIJO_BURNED}.tif/.tiff en: {DIR_BURNED}")

    sep = "=" * 62
    print(f"\n{sep}\nCOMPARADOR MbUN — INFERENCIAS VS ESA")
    print(f"  Modelo A      : {NOMBRE_MODELO_A}  [{MODELO_A or 'auto'}]")
    print(f"  Modelo B      : {NOMBRE_MODELO_B}  [{MODELO_B or 'auto'}]")
    print(f"  Referencia    : {DIR_BURNED}")
    print(f"  Inferencias A : {DIR_INFERENCIAS_A}")
    print(f"  Inferencias B : {DIR_INFERENCIAS_B}")
    print(f"  Salida        : {DIR_SALIDA}")
    print(f"  Escenas       : {len(referencias)}\n{sep}")

    filas = []
    acum = {"A": dict(vp=0, fp=0, fn=0, vn=0), "B": dict(vp=0, fp=0, fn=0, vn=0)}
    exitos = omitidos = errores = 0

    for nombrebase, ruta_ref in referencias:
        print(f"\n{'─' * 62}\nProcesando: {nombrebase}\n{'─' * 62}")
        ruta_a, msg_a = buscar_inferencia(DIR_INFERENCIAS_A, nombrebase, MODELO_A)
        ruta_b, msg_b = buscar_inferencia(DIR_INFERENCIAS_B, nombrebase, MODELO_B)
        if ruta_a is None or ruta_b is None:
            for msg in (msg_a, msg_b):
                if msg:
                    print(f"  [FALTA] {msg}")
            print(f"  [OMITIDO] {nombrebase}")
            omitidos += 1
            continue

        try:
            gt, meta_ref = cargar_referencia(ruta_ref)
            print(f"  A: {ruta_a.name}   B: {ruta_b.name}")
            pa = cargar_inferencia(ruta_a, gt.shape, meta_ref, "Modelo A")
            pb = cargar_inferencia(ruta_b, gt.shape, meta_ref, "Modelo B")
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            errores += 1
            continue

        validos = (gt != wc.IGNORE_INDEX) & (pa != wc.IGNORE_INDEX) & (pb != wc.IGNORE_INDEX)
        if not validos.any():
            print("  [OMITIDO] ningún píxel válido en común")
            omitidos += 1
            continue
        print(f"  Píxeles válidos en común: {int(validos.sum()):,} ({100 * validos.mean():.1f}%)")
        ref, ma_b, mb_b = gt == 1, pa == 1, pb == 1

        ca, cb = conteos(ref, ma_b, validos), conteos(ref, mb_b, validos)
        for k in ca:
            acum["A"][k] += ca[k]
            acum["B"][k] += cb[k]
        ma, mb = metricas_desde_conteos(ca), metricas_desde_conteos(cb)

        da = {"rgb": mascara_a_rgb(ma_b, validos), "diff": imagen_diferencias(ref, ma_b, validos), "m": ma}
        db = {"rgb": mascara_a_rgb(mb_b, validos), "diff": imagen_diferencias(ref, mb_b, validos), "m": mb}
        visualizar(nombrebase, mascara_a_rgb(ref, validos), da, db, NOMBRE_MODELO_A, NOMBRE_MODELO_B,
                   DIR_SALIDA / f"{nombrebase}_comparacion.png")
        imprimir_tabla(ma, mb)
        for clave, m, ruta in (("A", ma, ruta_a), ("B", mb, ruta_b)):
            filas.append({"escena": nombrebase, "modelo": clave, "archivo": ruta.name,
                          **{k: m[k] for k in ("vp", "fp", "fn", "vn", "n_ref", "n_pred")},
                          **{k: round(m[k], 6) for _, k in METRICAS}})
        exitos += 1

    if exitos:
        gma, gmb = metricas_desde_conteos(acum["A"]), metricas_desde_conteos(acum["B"])
        for clave, m in (("A", gma), ("B", gmb)):
            filas.append({"escena": "GLOBAL", "modelo": clave, "archivo": "",
                          **{k: m[k] for k in ("vp", "fp", "fn", "vn", "n_ref", "n_pred")},
                          **{k: round(m[k], 6) for _, k in METRICAS}})
        ruta_csv = DIR_SALIDA / "metricas.csv"
        with open(ruta_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            wr.writeheader()
            wr.writerows(filas)
        print(f"\n{sep}\n  GLOBAL ({exitos} escenas, píxeles acumulados)")
        imprimir_tabla(gma, gmb)
        print(f"\n  CSV: {ruta_csv}")
        if FIGURA_GLOBAL:
            figura_global(gma, gmb, exitos, DIR_SALIDA / "comparacion_global.png")

    print(f"\n{sep}\n  RESUMEN FINAL")
    print(f"  Completadas : {exitos}")
    print(f"  Omitidas    : {omitidos}")
    print(f"  Con errores : {errores}")
    print(f"  Salida      : {DIR_SALIDA.resolve()}\n{sep}\n")
    if errores or not exitos:
        sys.exit(1)


if __name__ == "__main__":
    main()
