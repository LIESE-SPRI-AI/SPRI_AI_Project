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
from matplotlib.patches import Rectangle
import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

INPUT_DIR  = "/home/liese2/SPRI_AI_project/Dataset/Entrada_firmas"   # directorio con las imágenes .tif
OUTPUT_DIR = "/home/liese2/SPRI_AI_project/Dataset/Salida_firmas2"      # directorio donde se guardan las gráficas

# Nombres de bandas según el sensor más común (ajustar si es necesario)
# Orden asumido: Banda1=Azul, Banda2=Verde, Banda3=Rojo, Banda4=NIR
BAND_NAMES  = ["Azul", "Verde", "Rojo", "NIR"]
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


def dividir_en_zonas(image: np.ndarray, n_zonas: int = 9) -> dict[str, dict]:
    """
    Divide la imagen en n_zonas regiones cuadradas y calcula
    la firma espectral (mediana y desviación estándar) de cada una.
    También retorna las coordenadas de cada zona para referencia.
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
            etiqueta = f"Zona {i*grid + j + 1}"
            
            # Calcular mediana y desviación estándar por banda ignorando NaN
            mediana = np.array([
                np.nanmedian(parche[b]) for b in range(parche.shape[0])
            ])
            desviacion = np.array([
                np.nanstd(parche[b]) for b in range(parche.shape[0])
            ])
            
            # Calcular percentiles para mejor visualización de la variabilidad
            p25 = np.array([
                np.nanpercentile(parche[b], 25) for b in range(parche.shape[0])
            ])
            p75 = np.array([
                np.nanpercentile(parche[b], 75) for b in range(parche.shape[0])
            ])
            
            zonas[etiqueta] = {
                "mediana": mediana,
                "std": desviacion,
                "p25": p25,
                "p75": p75,
                "coords": (r0, r1, c0, c1),
                "pos": (i, j)
            }
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


def visualizar_mapa_zonas(ax, image_shape: tuple, zonas: dict) -> None:
    """Visualiza la división de zonas en un mapa."""
    H, W = image_shape
    grid = int(np.sqrt(len(zonas)))
    h_step, w_step = H // grid, W // grid
    
    # Dibujar cuadrícula
    for i in range(grid + 1):
        ax.axhline(y=i * h_step, color='white', linewidth=2, alpha=0.8)
    for j in range(grid + 1):
        ax.axvline(x=j * w_step, color='white', linewidth=2, alpha=0.8)
    
    # Numerar zonas
    for idx, (etiqueta, datos) in enumerate(zonas.items()):
        i, j = datos["pos"]
        center_x = (j * w_step + (j + 1) * w_step) / 2
        center_y = (i * h_step + (i + 1) * h_step) / 2
        ax.text(center_x, center_y, str(idx + 1), 
               color='yellow', fontsize=14, fontweight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle="circle,pad=0.4", facecolor='black', alpha=0.8))
    
    ax.set_title("NDVI - División en Zonas", fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')


# ─────────────────────────────────────────────────────────────────────────────
# GENERACIÓN DE GRÁFICAS MEJORADA
# ─────────────────────────────────────────────────────────────────────────────

def graficar_firmas(nombre: str, zonas: dict, stats: list[dict],
                    indices: dict, image: np.ndarray,
                    ruta_salida: str) -> None:
    """
    Crea una figura con múltiples paneles:
      - Mapa de NDVI con división de zonas (panel grande)
      - Gráficas individuales para cada zona organizadas en grid
      - Gráfica del promedio global con desviación estándar
    """
    num_zonas = len(zonas)
    
    # Determinar el grid óptimo para las subgráficas
    grid_cols = 3  # Fijo 3 columnas para mejor organización
    grid_rows = (num_zonas + grid_cols - 1) // grid_cols + 1  # +1 para el promedio global
    
    # Calcular tamaño de figura basado en el número de filas
    fig_height = 5 * grid_rows
    fig_width = 18
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    
    # Título principal
    fig.suptitle(
        f"Análisis Espectral por Zonas — {nombre}",
        fontsize=20, color="#1a1a1a", fontweight="bold", y=0.98
    )
    
    # Crear grid de subplots con espacios adecuados
    gs = gridspec.GridSpec(grid_rows, grid_cols, figure=fig, 
                           hspace=0.5, wspace=0.3,
                           height_ratios=[1]*grid_rows)
    
    # Panel 1: Mapa NDVI con división de zonas (ocupa la primera fila completa)
    ax_ndvi = fig.add_subplot(gs[0, :])
    
    # Mostrar NDVI
    ndvi = indices["NDVI"]
    ndvi_clipped = np.clip(ndvi, -1, 1)
    im = ax_ndvi.imshow(ndvi_clipped, cmap="RdYlGn", vmin=-1, vmax=1,
                        interpolation="bilinear", aspect='auto')
    
    # Añadir división de zonas
    visualizar_mapa_zonas(ax_ndvi, ndvi.shape, zonas)
    
    # Añadir colorbar
    cbar = plt.colorbar(im, ax=ax_ndvi, fraction=0.015, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#333")
    cbar.set_label("NDVI", color="#1a1a1a", fontweight="bold", fontsize=11)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#333", fontsize=9)
    
    # Crear gráficas individuales para cada zona en las filas siguientes
    x_positions = range(len(BAND_NAMES))
    
    for idx, (etiqueta, datos) in enumerate(zonas.items()):
        # Calcular posición en el grid (fila 1 en adelante)
        row = 1 + (idx // grid_cols)
        col = idx % grid_cols
        
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#f8f9fa")
        ax.set_xlim(-0.5, 3.5)
        
        # Color basado en el NDVI medio de la zona
        zona_ndvi = np.nanmean(ndvi[datos["coords"][0]:datos["coords"][1], 
                                     datos["coords"][2]:datos["coords"][3]])
        
        if zona_ndvi > 0.5:
            color_fondo = "#d4edda"  # Verde claro para vegetación densa
        elif zona_ndvi > 0.2:
            color_fondo = "#fff3cd"  # Amarillo claro para vegetación moderada
        else:
            color_fondo = "#f8d7da"  # Rojo claro para suelo quemado/desnudo
        
        ax.set_facecolor(color_fondo)
        
        # Graficar mediana con área de desviación estándar
        mediana = datos["mediana"]
        std = datos["std"]
        
        # Área de desviación estándar
        ax.fill_between(x_positions, mediana - std, mediana + std,
                        color="#6c757d", alpha=0.2)
        
        # Línea de mediana
        ax.plot(x_positions, mediana, 'o-', linewidth=2.5, 
               color='#2c3e50', markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        
        # Barras de error en los puntos
        ax.errorbar(x_positions, mediana, yerr=std, fmt='none',
                   ecolor='#2c3e50', capsize=4, capthick=1.5, elinewidth=1.5, alpha=0.7)
        
        # Configurar el gráfico
        ax.set_title(f"{etiqueta}\nNDVI: {zona_ndvi:.3f}", fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel("Banda", fontsize=9, labelpad=5)
        ax.set_ylabel("Reflectancia", fontsize=9, labelpad=5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(BAND_NAMES, fontsize=8, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#666')
        ax.tick_params(labelsize=8)
        
        # Añadir texto con estadísticas
        ax.text(0.05, 0.95, f"σ = {np.mean(std):.3f}", 
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor='#ccc'))
    
    # Panel para el promedio global (en la última fila, centrado)
    last_row = grid_rows - 1
    last_col_start = (grid_cols - 1) // 2
    ax_global = fig.add_subplot(gs[last_row, last_col_start:last_col_start+2])
    ax_global.set_facecolor("#e9ecef")
    ax_global.set_xlim(-0.5, 3.5)
    
    # Calcular firma global
    firma_global_mediana = np.array([
        np.nanmedian(image[b]) for b in range(image.shape[0])
    ])
    firma_global_std = np.array([
        np.nanstd(image[b][~np.isnan(image[b])]) for b in range(image.shape[0])
    ])
    
    # Graficar promedio global
    ax_global.fill_between(x_positions, 
                           firma_global_mediana - firma_global_std,
                           firma_global_mediana + firma_global_std,
                           color='#495057', alpha=0.2)
    
    ax_global.plot(x_positions, firma_global_mediana, 'o-', linewidth=3,
                  color='#000000', markersize=10, markeredgecolor='white', markeredgewidth=2,
                  label='Promedio global')
    
    ax_global.errorbar(x_positions, firma_global_mediana, yerr=firma_global_std,
                      fmt='none', ecolor='#000000', capsize=5, capthick=2,
                      elinewidth=2, alpha=0.8)
    
    # Configurar gráfico global
    ax_global.set_title("FIRMA ESPECTRAL GLOBAL\n(Toda la imagen)", 
                       fontsize=13, fontweight='bold', color='#2c3e50', pad=12)
    ax_global.set_xlabel("Banda espectral", fontsize=10, fontweight='bold', labelpad=8)
    ax_global.set_ylabel("Reflectancia", fontsize=10, fontweight='bold', labelpad=8)
    ax_global.set_xticks(x_positions)
    ax_global.set_xticklabels(BAND_NAMES, fontsize=9, fontweight='bold')
    ax_global.set_ylim(0, 1)
    ax_global.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='#666')
    
    # Añadir estadísticas globales
    stats_text = f"NDVI global: {np.nanmean(ndvi):.3f}\n"
    stats_text += f"σ promedio: {np.mean(firma_global_std):.3f}"
    ax_global.text(0.05, 0.95, stats_text, transform=ax_global.transAxes,
                  fontsize=10, verticalalignment='top', horizontalalignment='left',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                           alpha=0.9, edgecolor='#999', linewidth=1))
    
    # Ajustar layout para evitar superposiciones
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def graficar_estadisticas_globales(resumen: list[dict], ruta_salida: str) -> None:
    """Gráfica resumen con estadísticas de todas las imágenes procesadas."""
    if not resumen:
        return

    nombres = [r["nombre"] for r in resumen]
    n = len(nombres)
    x = np.arange(n)
    
    # Crear figura más grande si hay muchas imágenes
    fig_width = max(5 * len(BAND_NAMES), 8)
    fig_height = 8
    
    fig, axes = plt.subplots(1, len(BAND_NAMES), figsize=(fig_width, fig_height),
                              facecolor="white")
    fig.suptitle("Comparativa de Reflectancia Media por Imagen",
                 fontsize=16, color="#1a1a1a", fontweight="bold", y=0.98)

    for b_idx, (ax, bname, bcolor) in enumerate(zip(axes, BAND_NAMES, BAND_COLORS)):
        medias = [r["stats"][b_idx]["mean"] for r in resumen]
        stds   = [r["stats"][b_idx]["std"]  for r in resumen]

        bars = ax.bar(x, medias, yerr=stds, color=bcolor, alpha=0.8,
                     capsize=8, capthick=2, error_kw=dict(ecolor="#333", elinewidth=2))
        ax.set_facecolor("#f8f9fa")
        ax.set_title(bname, color="#1a1a1a", fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(nombres, rotation=45, ha="right", fontsize=8, color="#1a1a1a")
        ax.set_ylabel("Reflectancia media", color="#444", fontsize=10)
        ax.tick_params(colors="#333", labelsize=9)
        ax.grid(True, axis="y", alpha=0.3, color="#aaa", linestyle="--")
        ax.set_ylim(0, 1.1)
        for spine in ax.spines.values():
            spine.set_edgecolor("#bbb")
            spine.set_linewidth(1)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
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
        ruta_salida = os.path.join(output_dir, f"{nombre}_firmas_zonas.png")
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