"""
Comparador MbUN — Inferencias vs Ground Truth
==============================================
Compara dos modelos MbUN contra las imágenes etiquetadas (burned).

Estructura de archivos esperada:
  DIR_BURNED      : <nombrebase>_burned.tiff        (RGB, incendios en ROJO)
  DIR_INFERENCIAS_A : <nombrebase>_final.tif        (binario 0/1, 1 banda)
  DIR_INFERENCIAS_B : <nombrebase>_final.tif        (binario 0/1, 1 banda)

El script empareja automáticamente por nombrebase y ejecuta una figura
comparativa + resumen de consola por cada par de modelos.

Uso:
  python comparador_mbun.py \
      --dir_inf_a /ruta/inferencias_modelo_A \
      --dir_inf_b /ruta/inferencias_modelo_B

  O bien edita las constantes de configuración al inicio del script.
"""

import sys
import re
from pathlib import Path

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN  ← edita estos valores
# ═══════════════════════════════════════════════════════════════════
DIR_BURNED      = Path("/home/liese2/SPRI_AI_project/Inferencias/Burned")  # ground truth _burned.tiff
DIR_INFERENCIAS_A = Path("/home/liese2/SPRI_AI_project/Inferencias/Output_Inferencias_Mobile-UNet/Output_MbUN_17_exp1")  # archivos _final.tif del modelo A
DIR_INFERENCIAS_B = Path("/home/liese2/SPRI_AI_project/Inferencias/Output_Inferencias_UNet/Output_UN_52")  # archivos _final.tif del modelo B
DIR_SALIDA      = Path("/home/liese2/SPRI_AI_project/Inferencias/Resultados7_UNvsMbUN_ex1")         # carpeta de salida

NOMBRE_MODELO_A = "Mobile-UNet"    # Nombre para mostrar del modelo A
NOMBRE_MODELO_B = "UNet"    # Nombre para mostrar del modelo B
UMBRAL_ROJO = 255               # umbral canal R para detectar incendios en ground truth
# ═══════════════════════════════════════════════════════════════════


COLORES = {'A': '#2980b9', 'B': '#c0392b'}


# ───────────────────────────────────────────────────────────────────
# CARGA
# ───────────────────────────────────────────────────────────────────

def cargar_geotiff(ruta):
    ds = gdal.Open(str(ruta))
    if ds is None:
        raise ValueError(f"No se pudo abrir: {ruta}")
    bandas = [ds.GetRasterBand(i).ReadAsArray() for i in range(1, ds.RasterCount + 1)]
    img = np.stack(bandas, axis=-1) if len(bandas) > 1 else bandas[0]
    ds = None
    return img


# ───────────────────────────────────────────────────────────────────
# DETECCIÓN
# ───────────────────────────────────────────────────────────────────

def detectar_burned(imagen, umbral=200):
    """Ground truth RGB: incendios donde R > umbral, G < 100, B < 100."""
    if imagen.ndim == 2:
        mascara = imagen > umbral
    else:
        r, g, b = imagen[:, :, 0], imagen[:, :, 1], imagen[:, :, 2]
        mascara = (r > umbral) & (g < 100) & (b < 100)
    print(f"  [Burned]  {mascara.sum():>10,} px  ({mascara.sum()/mascara.size*100:.2f}%)")
    return mascara


def detectar_binaria(imagen, nombre="Inferencia"):
    """Inferencia binaria 1 banda: 0 = fondo, 1 = incendio."""
    if imagen.ndim == 3:
        imagen = imagen[:, :, 0]
    mascara = imagen.astype(bool)
    print(f"  [{nombre}]  {mascara.sum():>10,} px  ({mascara.sum()/mascara.size*100:.2f}%)")
    return mascara


def redimensionar(imagen, shape_ref, nombre):
    if imagen.shape[:2] != shape_ref[:2]:
        print(f"  [{nombre}] Redimensionando {imagen.shape[:2]} → {shape_ref[:2]}")
        interp = cv2.INTER_NEAREST
        if imagen.ndim == 2:
            imagen = cv2.resize(imagen, (shape_ref[1], shape_ref[0]), interpolation=interp)
        else:
            imagen = cv2.resize(imagen, (shape_ref[1], shape_ref[0]), interpolation=interp)
    return imagen


def mascara_a_rgb(mascara):
    """Booleano → RGB: fondo blanco, incendios rojos."""
    rgb = np.full((*mascara.shape, 3), 255, dtype=np.uint8)
    rgb[mascara] = [255, 0, 0]
    return rgb


def imagen_diferencias(mascara_ref, mascara_pred):
    """
    VP  = rojo        (incendio real, detectado)
    FN  = azul oscuro (incendio real, NO detectado)
    FP  = azul claro  (detectado sin ser incendio)
    VN  = blanco      (fondo correcto)
    """
    img = np.full((*mascara_ref.shape, 3), 255, dtype=np.uint8)
    img[ mascara_ref &  mascara_pred] = [255,   0,   0]
    img[ mascara_ref & ~mascara_pred] = [  0,   0, 150]
    img[~mascara_ref &  mascara_pred] = [  0,   0, 255]
    return img


# ───────────────────────────────────────────────────────────────────
# MÉTRICAS
# ───────────────────────────────────────────────────────────────────

def calcular_metricas(ref, pred):
    vp = int(( ref &  pred).sum())
    fp = int((~ref &  pred).sum())
    fn = int(( ref & ~pred).sum())
    vn = int((~ref & ~pred).sum())
    total = ref.size
    n_ref = int(ref.sum())

    precision     = vp / (vp + fp)          if (vp + fp) > 0 else 0.0
    recall        = vp / (vp + fn)          if (vp + fn) > 0 else 0.0
    f1            = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    exactitud     = (vp + vn) / total
    especificidad = vn / (vn + fp)          if (vn + fp) > 0 else 0.0
    pct_error     = (fp + fn) / n_ref * 100 if n_ref > 0 else (fp + fn) / total * 100

    return dict(
        vp=vp, fp=fp, fn=fn, vn=vn, total=total,
        n_ref=n_ref, n_pred=int(pred.sum()),
        pixeles_error=fp+fn, pct_error=pct_error,
        precision=precision, recall=recall, f1=f1,
        exactitud=exactitud, especificidad=especificidad,
    )


# ───────────────────────────────────────────────────────────────────
# FIGURA
# ───────────────────────────────────────────────────────────────────

def visualizar(nombrebase, rgb_burned, da, db, nombre_a, nombre_b, ruta_salida=None):
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            height_ratios=[3, 3, 2.2],
                            hspace=0.40, wspace=0.08)

    # Fila 0: imágenes procesadas
    _panel(fig.add_subplot(gs[0, 0]), rgb_burned,
           'Pixeles etiquetados ESA',
           f'Píxeles de incendio: {da["m"]["n_ref"]:,} px', 'black')

    _panel(fig.add_subplot(gs[0, 1]), da['rgb'],
           f'{nombre_a}',
           f'Incendios detectados: {da["m"]["n_pred"]:,} px', COLORES['A'])

    _panel(fig.add_subplot(gs[0, 2]), db['rgb'],
           f'{nombre_b}',
           f'Incendios detectados: {db["m"]["n_pred"]:,} px', COLORES['B'])

    # Fila 1: leyenda + diferencias
    ax_ley = fig.add_subplot(gs[1, 0])
    ax_ley.axis('off')
    patches = [
        mpatches.Patch(color='red',     label='Verdadero Positivo (VP)\nIncendio real, detectado'),
        mpatches.Patch(color='#000096', label='Falso Negativo (FN)\nIncendio real, NO detectado'),
        mpatches.Patch(color='blue',    label='Falso Positivo (FP)\nDetectado sin ser incendio'),
        mpatches.Patch(facecolor='white', edgecolor='gray',
                       label='Verdadero Negativo (VN)\nFondo correcto'),
    ]
    ax_ley.legend(handles=patches, loc='center', fontsize=10,
                  title='DIFERENCIAS', title_fontsize=11,
                  framealpha=0.95, edgecolor='gray')

    ma = da['m']
    _panel(fig.add_subplot(gs[1, 1]), da['diff'],
           f'Diferencias — {nombre_a} vs ESA',
           f'F1={ma["f1"]*100:.1f}%   Recall={ma["recall"]*100:.1f}%   '
           f'Precisión={ma["precision"]*100:.1f}%', COLORES['A'])

    mb = db['m']
    _panel(fig.add_subplot(gs[1, 2]), db['diff'],
           f'Diferencias — {nombre_b} vs ESA',
           f'F1={mb["f1"]*100:.1f}%   Recall={mb["recall"]*100:.1f}%   '
           f'Precisión={mb["precision"]*100:.1f}%', COLORES['B'])

    # Fila 2: barras comparativas
    _barras(fig.add_subplot(gs[2, :]), ma, mb, nombre_a, nombre_b)

    fig.suptitle(f'Comparación de modelos MbUN — {nombrebase}',
                 fontsize=15, fontweight='bold', y=1.01)

    if ruta_salida:
        fig.savefig(str(ruta_salida), dpi=200, bbox_inches='tight')
        print(f"  Figura guardada: {ruta_salida}")

    plt.close(fig)


def _panel(ax, img_rgb, titulo, subtitulo, color_titulo):
    ax.imshow(img_rgb)
    ax.set_title(titulo, fontsize=11, fontweight='bold', color=color_titulo, pad=5)
    ax.set_xlabel(subtitulo, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _barras(ax, ma, mb, nombre_a, nombre_b):
    nombres = ['Precisión', 'Recall', 'F1-Score', 'Exactitud', 'Especificidad']
    keys    = ['precision', 'recall', 'f1', 'exactitud', 'especificidad']
    x, w = np.arange(len(nombres)), 0.32

    def dibujar(vals, offset, color, label):
        bars = ax.bar(x + offset, vals, w, label=label,
                      color=color, alpha=0.85, edgecolor='black', linewidth=0.6)
        for bar, v in zip(bars, vals):
            if v > 3:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=8.5)

    dibujar([ma[k]*100 for k in keys], -w/2, COLORES['A'], f'Modelo A: {nombre_a}')
    dibujar([mb[k]*100 for k in keys], +w/2, COLORES['B'], f'Modelo B: {nombre_b}')

    ax.set_xticks(x); ax.set_xticklabels(nombres, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel('Porcentaje (%)', fontsize=11)
    ax.set_title('Comparación de resultados por modelo', fontsize=12, fontweight='bold')
    ax.axhline(80, color='green',  linestyle='--', alpha=0.45, linewidth=1.2, label='Excelente ≥80%')
    ax.axhline(60, color='orange', linestyle='--', alpha=0.45, linewidth=1.2, label='Aceptable ≥60%')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)


# ───────────────────────────────────────────────────────────────────
# BÚSQUEDA DE ARCHIVOS
# ───────────────────────────────────────────────────────────────────

def buscar_burned(directorio: Path) -> list[Path]:
    """Devuelve todos los *_Burned.tiff del directorio."""
    return sorted(directorio.glob("*_Burned.tiff"))


def buscar_inferencia(directorio: Path, nombrebase: str) -> Path | None:
    """
    Busca <nombrebase>_final.tif (insensible a mayúsculas).
    nombrebase es la parte antes de '_burned'.
    """
    patron = re.compile(
        rf'^{re.escape(nombrebase)}_best\.tif$',
        re.IGNORECASE
    )
    for f in directorio.iterdir():
        if patron.match(f.name):
            return f
    return None


# ───────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comparador MbUN: dos inferencias binarias vs ground truth burned'
    )
    parser.add_argument('--dir_inf_a', default=None,
                        help='Directorio con inferencias del modelo A (*_final.tif)')
    parser.add_argument('--dir_inf_b', default=None,
                        help='Directorio con inferencias del modelo B (*_final.tif)')
    parser.add_argument('--dir_burned', default=None,
                        help='Directorio con los _burned.tiff')
    parser.add_argument('--dir_salida', default=None,
                        help='Directorio de salida')
    parser.add_argument('--nombre_a', default=None,
                        help='Nombre para mostrar del modelo A')
    parser.add_argument('--nombre_b', default=None,
                        help='Nombre para mostrar del modelo B')
    parser.add_argument('--umbral_rojo', type=int, default=None,
                        help='Umbral canal R para ground truth (default 200)')
    args = parser.parse_args()

    # Prioridad: argumento CLI > constante del script
    dir_burned   = Path(args.dir_burned)   if args.dir_burned   else DIR_BURNED
    dir_inf_a    = Path(args.dir_inf_a)    if args.dir_inf_a    else DIR_INFERENCIAS_A
    dir_inf_b    = Path(args.dir_inf_b)    if args.dir_inf_b    else DIR_INFERENCIAS_B
    dir_salida   = Path(args.dir_salida)   if args.dir_salida   else DIR_SALIDA
    nombre_a     = args.nombre_a           if args.nombre_a     else NOMBRE_MODELO_A
    nombre_b     = args.nombre_b           if args.nombre_b     else NOMBRE_MODELO_B
    umbral       = args.umbral_rojo        if args.umbral_rojo  else UMBRAL_ROJO

    # Validaciones
    for d, nombre in [(dir_burned, 'DIR_BURNED'), 
                      (dir_inf_a, 'DIR_INFERENCIAS_A'),
                      (dir_inf_b, 'DIR_INFERENCIAS_B')]:
        if not d.is_dir():
            sys.exit(f"[ERROR] Directorio no encontrado: {d}  ({nombre})")

    dir_salida.mkdir(parents=True, exist_ok=True)

    archivos_burned = buscar_burned(dir_burned)
    if not archivos_burned:
        sys.exit(f"[ERROR] No se encontraron archivos *_burned.tiff en: {dir_burned}")

    sep = "=" * 62
    print(f"\n{sep}")
    print("COMPARADOR MbUN — INFERENCIAS VS GROUND TRUTH")
    print(f"  Modelo A        : {nombre_a}")
    print(f"  Modelo B        : {nombre_b}")
    print(f"  Ground truth    : {dir_burned}")
    print(f"  Inferencias A   : {dir_inf_a}")
    print(f"  Inferencias B   : {dir_inf_b}")
    print(f"  Salida          : {dir_salida}")
    print(f"  Imágenes burned : {len(archivos_burned)}")
    print(f"{sep}\n")

    exitos = errores = 0

    for ruta_burned in archivos_burned:
        # nombrebase = todo antes de '_burned'
        nombrebase = ruta_burned.stem.replace("_Burned", "")
        print(f"\n{'─'*62}")
        print(f"Procesando: {nombrebase}")
        print(f"{'─'*62}")

        # Buscar inferencias
        ruta_a = buscar_inferencia(dir_inf_a, nombrebase)
        ruta_b = buscar_inferencia(dir_inf_b, nombrebase)

        falta = []
        if ruta_a is None:
            falta.append(f"  [FALTA] {nombrebase}_best.tif en {dir_inf_a}")
        if ruta_b is None:
            falta.append(f"  [FALTA] {nombrebase}_best.tif en {dir_inf_b}")

        if falta:
            print("\n".join(falta))
            print(f"  [OMITIDO] {nombrebase} — no se encontraron ambas inferencias")
            errores += 1
            continue

        # Cargar
        print(f"Cargando ground truth : {ruta_burned.name}")
        img_burned = cargar_geotiff(ruta_burned)
        mascara_gt = detectar_burned(img_burned, umbral)
        rgb_burned  = mascara_a_rgb(mascara_gt)

        print(f"Cargando modelo A     : {ruta_a.name}")
        img_a = cargar_geotiff(ruta_a)
        img_a = redimensionar(img_a, img_burned.shape, f"ModeloA")
        mascara_a = detectar_binaria(img_a, f"Modelo A ({nombre_a})")

        print(f"Cargando modelo B     : {ruta_b.name}")
        img_b = cargar_geotiff(ruta_b)
        img_b = redimensionar(img_b, img_burned.shape, f"ModeloB")
        mascara_b = detectar_binaria(img_b, f"Modelo B ({nombre_b})")

        # Métricas y diferencias
        ma = calcular_metricas(mascara_gt, mascara_a)
        mb = calcular_metricas(mascara_gt, mascara_b)

        da = {'rgb': mascara_a_rgb(mascara_a), 'diff': imagen_diferencias(mascara_gt, mascara_a), 'm': ma}
        db = {'rgb': mascara_a_rgb(mascara_b), 'diff': imagen_diferencias(mascara_gt, mascara_b), 'm': mb}

        # Figura
        ruta_fig = dir_salida / f"{nombrebase}_comparacion_mbun_best.png"
        visualizar(
            nombrebase=nombrebase,
            rgb_burned=rgb_burned,
            da=da, db=db,
            nombre_a=nombre_a,
            nombre_b=nombre_b,
            ruta_salida=ruta_fig
        )

        # Resumen en consola
        print(f"\n  {'Métrica':<16} {'Modelo A':>10} {'Modelo B':>10}")
        print(f"  {'─'*38}")
        for lbl, k in [('Precisión','precision'), ('Recall','recall'), ('F1-Score','f1'),
                        ('Exactitud','exactitud'), ('Especificidad','especificidad')]:
            print(f"  {lbl:<16} {ma[k]*100:>9.2f}% {mb[k]*100:>9.2f}%")

        exitos += 1

    # Resumen final
    print(f"\n{sep}")
    print(f"  RESUMEN FINAL")
    print(f"  Completadas : {exitos}")
    print(f"  Con errores : {errores}")
    print(f"  Salida      : {dir_salida.resolve()}")
    print(f"{sep}\n")

    if errores:
        sys.exit(1)


if __name__ == "__main__":
    main()
