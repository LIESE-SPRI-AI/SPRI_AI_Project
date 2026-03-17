"""
Extractor de Firmas Espectrales - Imágenes Satelitales de Incendios Forestales
===============================================================================
Procesa imágenes TIFF de 4 bandas (uint16) en un directorio de entrada
y guarda las gráficas de firmas espectrales en un directorio de salida.

Estructura esperada:
    INPUT_DIR/   → imágenes .tif / .tiff de 4 bandas uint16
    OUTPUT_DIR/  → gráficas PNG generadas automáticamente

Uso:
    python firmas_espectrales.py
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

INPUT_DIR  = "/home/liese2/SPRI_AI_project/Dataset/Entrada_firmas"   # directorio con las imágenes .tif
OUTPUT_DIR = "/home/liese2/SPRI_AI_project/Dataset/Salida_firmas"      # directorio donde se guardan las gráficas

# Nombres de bandas según el sensor más común (ajustar si es necesario)
# Orden asumido: Banda1=Azul, Banda2=Verde, Banda3=Rojo, Banda4=NIR
BAND_NAMES  = ["Azul (B)", "Verde (G)", "Rojo (R)", "NIR"]
BAND_COLORS = ["#4169E1", "#2E8B57", "#CC2200", "#8B008B"]

# Factor de escala para normalizar uint16 → reflectancia [0, 1]
# Sentinel-2 = 10000 | Landsat 8/9 = 65535 | PlanetScope = 10000
SCALE_FACTOR = 10000


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def listar_imagenes(directorio: str) -> list[str]:
    """Devuelve la lista de rutas de imágenes .tif/.tiff en el directorio."""
    extensiones = {".tif", ".tiff"}
    archivos = [
        os.path.join(directorio, f)
        for f in sorted(os.listdir(directorio))
        if os.path.splitext(f)[1].lower() in extensiones
    ]
    return archivos


def leer_imagen(ruta: str) -> tuple[np.ndarray, dict]:
    """
    Lee una imagen raster multi-banda y devuelve:
        - array float32 de forma (bandas, filas, cols)
        - metadata del archivo
    """
    with rasterio.open(ruta) as src:
        data = src.read().astype(np.float32)
        meta = {
            "crs":       src.crs,
            "transform": src.transform,
            "nodata":    src.nodata,
            "n_bands":   src.count,
            "width":     src.width,
            "height":    src.height,
        }
    return data, meta


def normalizar(image: np.ndarray, nodata=None) -> np.ndarray:
    """Normaliza uint16 → reflectancia [0, 1] enmascarando nodata."""
    img = image / SCALE_FACTOR
    if nodata is not None:
        img[image == nodata] = np.nan
    return np.clip(img, 0, 1)


def dividir_en_zonas(image: np.ndarray, n_zonas: int = 9) -> dict[str, np.ndarray]:
    """
    Divide la imagen en n_zonas regiones cuadradas y calcula
    la firma espectral (mediana) de cada una.
    """
    _, H, W = image.shape
    grid = int(np.sqrt(n_zonas))
    h_step, w_step = H // grid, W // grid

    zonas = {}
    for i in range(grid):
        for j in range(grid):
            r0, r1 = i * h_step, (i + 1) * h_step
            c0, c1 = j * w_step, (j + 1) * w_step
            parche = image[:, r0:r1, c0:c1]
            etiqueta = f"Zona ({i+1},{j+1})"
            # mediana por banda ignorando NaN
            zonas[etiqueta] = np.array([
                np.nanmedian(parche[b]) for b in range(parche.shape[0])
            ])
    return zonas


def calcular_indices(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Calcula índices espectrales relevantes para incendios forestales.
    Asume orden: B0=Azul, B1=Verde, B2=Rojo, B3=NIR
    """
    eps = 1e-10
    B, G, R, NIR = image[0], image[1], image[2], image[3]

    NDVI = (NIR - R)   / (NIR + R   + eps)   # vegetación
    NDWI = (G   - NIR) / (G   + NIR + eps)   # agua / humedad
    BAI  = 1.0 / ((0.1 - R)**2 + (0.06 - NIR)**2 + eps)  # área quemada
    EVI  = 2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1 + eps)

    return {"NDVI": NDVI, "NDWI": NDWI, "BAI": BAI, "EVI": EVI}


def estadisticas_banda(image: np.ndarray) -> list[dict]:
    """Calcula estadísticas básicas por banda."""
    stats = []
    for b in range(image.shape[0]):
        banda = image[b][~np.isnan(image[b])]
        stats.append({
            "min":    float(np.min(banda)),
            "max":    float(np.max(banda)),
            "mean":   float(np.mean(banda)),
            "median": float(np.median(banda)),
            "std":    float(np.std(banda)),
            "p5":     float(np.percentile(banda, 5)),
            "p95":    float(np.percentile(banda, 95)),
        })
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE GRÁFICAS
# ─────────────────────────────────────────────────────────────────────────────

def graficar_firmas(nombre: str, zonas: dict, stats: list[dict],
                    indices: dict, image: np.ndarray,
                    ruta_salida: str) -> None:
    """
    Crea una figura con 4 paneles:
      1. Firmas espectrales por zona
      2. Boxplot de reflectancia por banda
      3. Mapa RGB (bandas Rojo, Verde, Azul) o composición disponible
      4. Mapa de NDVI
    """
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    fig.suptitle(
        f"Análisis Espectral — {nombre}",
        fontsize=16, color="#1a1a1a", fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax_firma  = fig.add_subplot(gs[0, 0])
    ax_box    = fig.add_subplot(gs[0, 1])
    ax_rgb    = fig.add_subplot(gs[1, 0])
    ax_ndvi   = fig.add_subplot(gs[1, 1])

    for ax in [ax_firma, ax_box, ax_rgb, ax_ndvi]:
        ax.set_facecolor("#f5f5f5")
        ax.tick_params(colors="#333")
        for spine in ax.spines.values():
            spine.set_edgecolor("#bbb")

    # ── Panel 1: Firmas espectrales ──────────────────────────────────────────
    cmap_zonas = plt.cm.tab20
    for idx, (etiqueta, firma) in enumerate(zonas.items()):
        color = cmap_zonas(idx / len(zonas))
        ax_firma.plot(BAND_NAMES, firma, marker="o", linewidth=1.8,
                      color=color, label=etiqueta, alpha=0.85)

    # Firma global (mediana de toda la imagen)
    firma_global = np.array([
        np.nanmedian(image[b]) for b in range(image.shape[0])
    ])
    ax_firma.plot(BAND_NAMES, firma_global, marker="D", linewidth=2.5,
                  color="#1a1a1a", linestyle="--", label="Global (mediana)", zorder=5)

    ax_firma.set_title("Firmas Espectrales por Zona", color="#1a1a1a", fontsize=11)
    ax_firma.set_xlabel("Banda espectral", color="#444")
    ax_firma.set_ylabel("Reflectancia", color="#444")
    ax_firma.legend(fontsize=6.5, ncol=2, framealpha=0.7,
                    labelcolor="#1a1a1a", facecolor="white")
    ax_firma.grid(True, alpha=0.4, color="#aaa")
    ax_firma.set_ylim(0, max(1.0, firma_global.max() * 1.2))

    # ── Panel 2: Boxplot por banda ───────────────────────────────────────────
    datos_box = []
    for b in range(image.shape[0]):
        banda = image[b].flatten()
        datos_box.append(banda[~np.isnan(banda)])

    bp = ax_box.boxplot(datos_box, patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], BAND_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color("#555")

    ax_box.set_xticks(range(1, len(BAND_NAMES) + 1))
    ax_box.set_xticklabels(BAND_NAMES, color="#1a1a1a", fontsize=9)
    ax_box.set_title("Distribución de Reflectancia por Banda", color="#1a1a1a", fontsize=11)
    ax_box.set_ylabel("Reflectancia", color="#444")
    ax_box.grid(True, axis="y", alpha=0.4, color="#aaa")

    # ── Panel 3: Composición RGB ─────────────────────────────────────────────
    # Usar bandas R=2, G=1, B=0 (índices 0-based)
    def stretch(band, p_low=2, p_high=98):
        lo, hi = np.nanpercentile(band, p_low), np.nanpercentile(band, p_high)
        out = (band - lo) / (hi - lo + 1e-10)
        return np.clip(out, 0, 1)

    rgb = np.dstack([
        stretch(image[2]),   # Rojo
        stretch(image[1]),   # Verde
        stretch(image[0]),   # Azul
    ])
    rgb = np.nan_to_num(rgb, nan=0)
    ax_rgb.imshow(rgb, interpolation="bilinear")
    ax_rgb.set_title("Composición RGB (R-G-B)", color="#1a1a1a", fontsize=11)
    ax_rgb.axis("off")

    # ── Panel 4: Mapa NDVI ───────────────────────────────────────────────────
    ndvi = indices["NDVI"]
    ndvi_clipped = np.clip(ndvi, -1, 1)
    im = ax_ndvi.imshow(ndvi_clipped, cmap="RdYlGn", vmin=-1, vmax=1,
                         interpolation="bilinear")
    cbar = plt.colorbar(im, ax=ax_ndvi, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="#333")
    cbar.set_label("NDVI", color="#1a1a1a")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#333")
    ax_ndvi.set_title("NDVI  (Rojo=quemado / Verde=vegetación)", color="#1a1a1a", fontsize=11)
    ax_ndvi.axis("off")

    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def graficar_estadisticas_globales(resumen: list[dict], ruta_salida: str) -> None:
    """Gráfica resumen con estadísticas de todas las imágenes procesadas."""
    if not resumen:
        return

    nombres = [r["nombre"] for r in resumen]
    n = len(nombres)
    x = np.arange(n)
    ancho = 0.18

    fig, axes = plt.subplots(1, len(BAND_NAMES), figsize=(5 * len(BAND_NAMES), 6),
                              facecolor="white")
    fig.suptitle("Comparativa de Reflectancia Media por Imagen",
                 fontsize=14, color="#1a1a1a", fontweight="bold")

    for b_idx, (ax, bname, bcolor) in enumerate(zip(axes, BAND_NAMES, BAND_COLORS)):
        medias = [r["stats"][b_idx]["mean"] for r in resumen]
        stds   = [r["stats"][b_idx]["std"]  for r in resumen]

        ax.bar(x, medias, yerr=stds, color=bcolor, alpha=0.8,
               capsize=4, error_kw=dict(ecolor="#333", elinewidth=1))
        ax.set_facecolor("#f5f5f5")
        ax.set_title(bname, color="#1a1a1a")
        ax.set_xticks(x)
        ax.set_xticklabels(nombres, rotation=45, ha="right", fontsize=7, color="#1a1a1a")
        ax.set_ylabel("Reflectancia media", color="#444")
        ax.tick_params(colors="#333")
        ax.grid(True, axis="y", alpha=0.4, color="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#bbb")

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → Resumen global guardado: {os.path.basename(ruta_salida)}")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def procesar_directorio(input_dir: str, output_dir: str) -> None:
    """Procesa todas las imágenes del directorio de entrada."""

    # Validar directorio de entrada
    if not os.path.isdir(input_dir):
        print(f"[ERROR] El directorio de entrada no existe: {input_dir}")
        sys.exit(1)

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    imagenes = listar_imagenes(input_dir)
    if not imagenes:
        print(f"[AVISO] No se encontraron imágenes .tif/.tiff en: {input_dir}")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"  Imágenes encontradas : {len(imagenes)}")
    print(f"  Entrada              : {os.path.abspath(input_dir)}")
    print(f"  Salida               : {os.path.abspath(output_dir)}")
    print(f"{'='*60}\n")

    resumen_global = []

    for idx, ruta in enumerate(imagenes, start=1):
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        print(f"[{idx}/{len(imagenes)}] Procesando: {nombre}")

        # Leer imagen
        try:
            image_raw, meta = leer_imagen(ruta)
        except Exception as e:
            print(f"  [ERROR] No se pudo leer la imagen: {e}")
            continue

        if meta["n_bands"] < 4:
            print(f"  [AVISO] La imagen tiene {meta['n_bands']} banda(s), se esperan 4. Saltando.")
            continue

        # Usar solo las 4 primeras bandas
        image_raw = image_raw[:4]

        # Normalizar
        image_norm = normalizar(image_raw, nodata=meta["nodata"])

        # Calcular firmas por zona, índices y estadísticas
        zonas  = dividir_en_zonas(image_norm, n_zonas=9)
        stats  = estadisticas_banda(image_norm)
        indices = calcular_indices(image_norm)

        # Guardar gráfica individual
        ruta_salida = os.path.join(output_dir, f"{nombre}_firmas.png")
        try:
            graficar_firmas(nombre, zonas, stats, indices, image_norm, ruta_salida)
            print(f"  → Gráfica guardada : {os.path.basename(ruta_salida)}")
        except Exception as e:
            print(f"  [ERROR] No se pudo generar la gráfica: {e}")
            continue

        # Acumular para resumen global
        resumen_global.append({"nombre": nombre[:20], "stats": stats})

        # Mostrar estadísticas en consola
        print(f"  → Estadísticas por banda:")
        for b, (bname, st) in enumerate(zip(BAND_NAMES, stats)):
            print(f"     {bname:20s}: media={st['mean']:.4f}  std={st['std']:.4f}  "
                  f"[{st['p5']:.4f} – {st['p95']:.4f}]")

    # Gráfica resumen comparativa (si hay más de 1 imagen)
    if len(resumen_global) > 1:
        ruta_resumen = os.path.join(output_dir, "_resumen_comparativo.png")
        graficar_estadisticas_globales(resumen_global, ruta_resumen)

    print(f"\n✔  Proceso completado. Resultados en: {os.path.abspath(output_dir)}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    procesar_directorio(INPUT_DIR, OUTPUT_DIR)