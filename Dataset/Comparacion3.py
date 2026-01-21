import cv2
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

def cargar_imagen_geotiff(ruta):
    """Carga una imagen GeoTIFF conservando la información de georreferenciación."""
    ds = gdal.Open(ruta)
    if ds is None:
        raise ValueError(f"No se pudo abrir la imagen: {ruta}")
    
    # Leer todas las bandas
    bandas = []
    for i in range(1, ds.RasterCount + 1):
        banda = ds.GetRasterBand(i).ReadAsArray()
        bandas.append(banda)
    
    # Convertir a array numpy (H, W, C)
    if len(bandas) > 1:
        imagen = np.stack(bandas, axis=-1)
    else:
        imagen = bandas[0]
    
    # Obtener información de georreferenciación
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    
    ds = None
    return imagen, geotransform, projection

def analizar_imagen(imagen, nombre="Imagen"):
    """Analiza una imagen para entender su estructura."""
    print(f"\n=== ANÁLISIS DE {nombre.upper()} ===")
    print(f"Dimensiones: {imagen.shape}")
    print(f"Tipo de datos: {imagen.dtype}")
    print(f"Valor mínimo: {imagen.min()}")
    print(f"Valor máximo: {imagen.max()}")
    print(f"Valor medio: {imagen.mean():.2f}")
    
    if len(imagen.shape) == 3:
        print(f"Canales: {imagen.shape[2]}")
        for i in range(min(3, imagen.shape[2])):
            print(f"  Canal {i}: min={imagen[:,:,i].min()}, max={imagen[:,:,i].max()}, mean={imagen[:,:,i].mean():.2f}")
    
    # Contar colores más comunes
    if len(imagen.shape) == 3 and imagen.shape[2] >= 3:
        # Buscar píxeles rojos (R alto, G y B bajos)
        r = imagen[:,:,0]
        g = imagen[:,:,1]
        b = imagen[:,:,2]
        
        rojos = ((r > 200) & (g < 100) & (b < 100)).sum()
        blancos = ((r > 240) & (g > 240) & (b > 240)).sum()
        negros = ((r < 10) & (g < 10) & (b < 10)).sum()
        
        total = r.size
        print(f"\nDistribución de colores:")
        print(f"  Píxeles rojos: {rojos} ({rojos/total*100:.2f}%)")
        print(f"  Píxeles blancos: {blancos} ({blancos/total*100:.2f}%)")
        print(f"  Píxeles negros: {negros} ({negros/total*100:.2f}%)")
    
    return imagen

def extraer_capa_incendios(imagen, umbral_rojo=200):
    """
    Extrae la capa de incendios de una imagen RGB.
    Detecta píxeles rojos donde R > umbral_rojo y G, B bajos.
    Devuelve imagen con fondo blanco y incendios rojos.
    """
    if len(imagen.shape) == 2:
        # Si es una sola banda, convertir a 3 bandas
        imagen = np.stack([imagen, imagen, imagen], axis=-1)
    
    # Crear una nueva imagen con fondo blanco
    imagen_con_fondo_blanco = np.ones_like(imagen) * 255  # Fondo blanco (255,255,255)
    
    # Separar canales
    if imagen.shape[2] >= 3:
        r = imagen[:, :, 0]
        g = imagen[:, :, 1]
        b = imagen[:, :, 2]
    else:
        raise ValueError("La imagen debe tener al menos 3 canales (RGB)")
    
    # Crear máscara para píxeles rojos (fuego)
    mascara_incendios = (r > umbral_rojo) & (g < 100) & (b < 100)
    
    # Poner los incendios en rojo sobre fondo blanco
    imagen_con_fondo_blanco[mascara_incendios, 0] = 255  # Canal R
    imagen_con_fondo_blanco[mascara_incendios, 1] = 0    # Canal G  
    imagen_con_fondo_blanco[mascara_incendios, 2] = 0    # Canal B
    
    return imagen_con_fondo_blanco.astype(np.uint8)

def procesar_prediccion(imagen_prediccion):
    """
    Procesa la imagen de predicción:
    1. Detecta incendios basados en colores rojos
    2. Devuelve imagen con fondo blanco y incendios rojos
    """
    if len(imagen_prediccion.shape) == 2:
        # Convertir imagen de 1 banda a 3 bandas
        imagen_prediccion = np.stack([imagen_prediccion, imagen_prediccion, imagen_prediccion], axis=-1)
    
    # Crear una nueva imagen con fondo blanco
    imagen_con_fondo_blanco = np.ones_like(imagen_prediccion) * 255  # Fondo blanco
    
    # Detectar incendios en la predicción
    if imagen_prediccion.shape[2] >= 3:
        r = imagen_prediccion[:, :, 0]
        g = imagen_prediccion[:, :, 1]
        b = imagen_prediccion[:, :, 2]
        
        # Detectar píxeles rojos (incendios)
        # Ajusta estos umbrales según tu imagen específica
        mascara_rojos = (r > 200) & (g < 50) & (b < 50)
        
        # Poner los incendios en rojo sobre fondo blanco
        imagen_con_fondo_blanco[mascara_rojos, 0] = 255  # Canal R
        imagen_con_fondo_blanco[mascara_rojos, 1] = 0    # Canal G
        imagen_con_fondo_blanco[mascara_rojos, 2] = 0    # Canal B
    
    return imagen_con_fondo_blanco.astype(np.uint8)

def comparar_imagenes(ruta_original, ruta_prediccion, umbral_rojo=200):
    """
    Compara dos imágenes de incendios y genera una imagen de diferencias.
    
    Args:
        ruta_original: Ruta a la imagen original con detección de rojos
        ruta_prediccion: Ruta a la imagen de predicción
        umbral_rojo: Umbral para detectar píxeles rojos (0-255)
    
    Returns:
        imagen_diferencias: Imagen RGB con diferencias marcadas
        estadisticas: Diccionario con estadísticas de comparación
        geotransform: Información de georreferenciación
        projection: Proyección de la imagen
    """
    
    # Cargar imágenes
    print(f"Cargando imagen original: {ruta_original}")
    img_original, geotransform, projection = cargar_imagen_geotiff(ruta_original)
    
    print(f"Cargando imagen de predicción: {ruta_prediccion}")
    img_prediccion, _, _ = cargar_imagen_geotiff(ruta_prediccion)
    
    # Analizar las imágenes para entender su estructura
    img_original = analizar_imagen(img_original, "Original")
    img_prediccion = analizar_imagen(img_prediccion, "Predicción")
    
    print(f"Dimensiones original: {img_original.shape}")
    print(f"Dimensiones predicción: {img_prediccion.shape}")
    
    # Guardar copias originales para referencia
    img_original_original = img_original.copy()
    img_prediccion_original = img_prediccion.copy()
    
    # Verificar que las dimensiones coincidan
    if img_original.shape[:2] != img_prediccion.shape[:2]:
        print("Ajustando dimensiones de la predicción...")
        img_prediccion = cv2.resize(img_prediccion, 
                                    (img_original.shape[1], img_original.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
    
    # Procesar imágenes para tener fondo blanco y incendios rojos
    print("\nProcesando imagen original con fondo blanco...")
    img_original_procesada = extraer_capa_incendios(img_original, umbral_rojo)
    
    print("Procesando imagen de predicción con fondo blanco...")
    img_prediccion_procesada = procesar_prediccion(img_prediccion)
    
    # Extraer máscaras de incendios de las imágenes procesadas
    # (Rojo puro: R=255, G=0, B=0)
    mascara_original = (img_original_procesada[:, :, 0] == 255) & \
                       (img_original_procesada[:, :, 1] == 0) & \
                       (img_original_procesada[:, :, 2] == 0)
    
    mascara_prediccion = (img_prediccion_procesada[:, :, 0] == 255) & \
                         (img_prediccion_procesada[:, :, 1] == 0) & \
                         (img_prediccion_procesada[:, :, 2] == 0)
    
    # Convertir máscaras a booleanas
    mascara_original_bool = mascara_original.astype(bool)
    mascara_prediccion_bool = mascara_prediccion.astype(bool)
    
    # Para visualización, usar las imágenes con fondo blanco
    img_original_vis = img_original_procesada
    img_prediccion_vis = img_prediccion_procesada
    
    # Calcular diferencias
    # Falsos Negativos: Incendios en original pero no en predicción
    falsos_negativos = mascara_original_bool & ~mascara_prediccion_bool
    
    # Falsos Positivos: Incendios en predicción pero no en original
    falsos_positivos = ~mascara_original_bool & mascara_prediccion_bool
    
    # Verdaderos Positivos: Incendios correctamente detectados
    verdaderos_positivos = mascara_original_bool & mascara_prediccion_bool
    
    # Verdaderos Negativos: Fondo correctamente identificado
    verdaderos_negativos = ~mascara_original_bool & ~mascara_prediccion_bool
    
    # Crear imagen de diferencias con fondo blanco
    imagen_diferencias = np.ones((img_original.shape[0], img_original.shape[1], 3), dtype=np.uint8) * 255
    
    # Aplicar colores a las diferencias sobre fondo blanco
    # Verdaderos Positivos: ROJO (incendios detectados correctamente)
    imagen_diferencias[verdaderos_positivos] = [255, 0, 0]  # Rojo puro
    
    # Falsos Negativos: AZUL OSCURO (incendios no detectados)
    imagen_diferencias[falsos_negativos] = [0, 0, 150]  # Azul oscuro
    
    # Falsos Positivos: AZUL BRILLANTE (detectados erróneamente)
    imagen_diferencias[falsos_positivos] = [0, 0, 255]  # Azul brillante
    
    # Calcular estadísticas detalladas
    estadisticas = calcular_estadisticas_detalladas(
        mascara_original_bool, mascara_prediccion_bool,
        falsos_negativos, falsos_positivos,
        verdaderos_positivos, verdaderos_negativos
    )
    
    return imagen_diferencias, estadisticas, geotransform, projection, img_original_vis, img_prediccion_vis

def calcular_estadisticas_detalladas(mascara_original, mascara_prediccion,
                                   falsos_negativos, falsos_positivos,
                                   verdaderos_positivos, verdaderos_negativos):
    """
    Calcula estadísticas detalladas de la comparación.
    """
    
    total_pixeles = mascara_original.size
    
    # Conteos básicos
    pixeles_incendio_original = np.sum(mascara_original)
    pixeles_incendio_prediccion = np.sum(mascara_prediccion)
    pixeles_errados = np.sum(falsos_negativos) + np.sum(falsos_positivos)
    
    if pixeles_incendio_original > 0:
        porcentaje_error = (pixeles_errados / pixeles_incendio_original) * 100
    else:
        porcentaje_error = (pixeles_errados / total_pixeles) * 100
    
    # Calcular métricas de precisión
    verdaderos_pos = np.sum(verdaderos_positivos)
    falsos_pos = np.sum(falsos_positivos)
    falsos_neg = np.sum(falsos_negativos)
    verdaderos_neg = np.sum(verdaderos_negativos)
    
    # Precisión (Precision)
    if (verdaderos_pos + falsos_pos) > 0:
        precision = verdaderos_pos / (verdaderos_pos + falsos_pos)
    else:
        precision = 0
    
    # Recall (Sensibilidad)
    if (verdaderos_pos + falsos_neg) > 0:
        recall = verdaderos_pos / (verdaderos_pos + falsos_neg)
    else:
        recall = 0
    
    # F1-Score (Media armónica de Precisión y Recall)
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    # Exactitud (Accuracy)
    exactitud = (verdaderos_pos + verdaderos_neg) / total_pixeles
    
    # Especificidad
    if (verdaderos_neg + falsos_pos) > 0:
        especificidad = verdaderos_neg / (verdaderos_neg + falsos_pos)
    else:
        especificidad = 0
    
    estadisticas = {
        'total_pixeles': total_pixeles,
        'pixeles_incendio_original': pixeles_incendio_original,
        'pixeles_incendio_prediccion': pixeles_incendio_prediccion,
        'pixeles_errados': pixeles_errados,
        'porcentaje_error': porcentaje_error,
        
        # Matriz de confusión
        'verdaderos_positivos': verdaderos_pos,
        'falsos_positivos': falsos_pos,
        'falsos_negativos': falsos_neg,
        'verdaderos_negativos': verdaderos_neg,
        
        # Métricas principales
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'exactitud': exactitud,
        'especificidad': especificidad,
        
        # Porcentajes
        'porcentaje_incendio_original': (pixeles_incendio_original / total_pixeles) * 100,
        'porcentaje_incendio_prediccion': (pixeles_incendio_prediccion / total_pixeles) * 100,
        'porcentaje_vp': (verdaderos_pos / total_pixeles) * 100,
        'porcentaje_fp': (falsos_pos / total_pixeles) * 100,
        'porcentaje_fn': (falsos_neg / total_pixeles) * 100,
        'porcentaje_vn': (verdaderos_neg / total_pixeles) * 100,
    }
    
    return estadisticas

def crear_histograma_estadisticas(estadisticas, ruta_salida=None):
    """
    Crea un histograma con las principales estadísticas.
    """
    
    # Preparar datos para el histograma
    categorias = ['Precisión', 'Recall', 'F1-Score', 'Exactitud']
    valores = [
        estadisticas['precision'] * 100,
        estadisticas['recall'] * 100,
        estadisticas['f1_score'] * 100,
        estadisticas['exactitud'] * 100
    ]
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Histograma de métricas principales
    ax1 = plt.subplot(2, 2, 1)
    bars = ax1.bar(categorias, valores, color=['blue', 'green', 'red', 'orange'])
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Porcentaje (%)')
    ax1.set_title('Métricas de Rendimiento')
    ax1.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{valor:.1f}%', ha='center', va='bottom')
    
    # 2. Matriz de confusión en gráfico de barras
    ax2 = plt.subplot(2, 2, 2)
    categorias_conf = ['VP', 'FP', 'FN', 'VN']
    valores_conf = [
        estadisticas['verdaderos_positivos'],
        estadisticas['falsos_positivos'],
        estadisticas['falsos_negativos'],
        estadisticas['verdaderos_negativos']
    ]
    porcentajes_conf = [
        estadisticas['porcentaje_vp'],
        estadisticas['porcentaje_fp'],
        estadisticas['porcentaje_fn'],
        estadisticas['porcentaje_vn']
    ]
    
    bars2 = ax2.bar(categorias_conf, valores_conf, 
                    color=['green', 'red', 'orange', 'blue'])
    ax2.set_ylabel('Número de Píxeles')
    ax2.set_title('Matriz de Confusión (Conteos)')
    ax2.grid(True, alpha=0.3)
    
    # Añadir porcentajes en las barras
    for bar, valor, porcentaje in zip(bars2, valores_conf, porcentajes_conf):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(valores_conf)*0.01,
                f'{porcentaje:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 3. Distribución de incendios vs no incendios
    ax3 = plt.subplot(2, 2, 3)
    labels = ['Incendios\n(Original)', 'Incendios\n(Predicción)', 'No Incendios']
    sizes = [
        estadisticas['pixeles_incendio_original'],
        estadisticas['pixeles_incendio_prediccion'],
        estadisticas['total_pixeles'] - estadisticas['pixeles_incendio_original']
    ]
    colors = ['red', 'orange', 'green']
    
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90)
    ax3.set_title('Distribución de Píxeles')
    
    # 4. Resumen numérico
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    resumen_texto = (
        f"RESUMEN ESTADÍSTICO\n"
        f"{'='*30}\n"
        f"Total píxeles: {estadisticas['total_pixeles']:,}\n"
        f"Píxeles incendio original: {estadisticas['pixeles_incendio_original']:,}\n"
        f"Píxeles incendio predicción: {estadisticas['pixeles_incendio_prediccion']:,}\n"
        f"Píxeles errados: {estadisticas['pixeles_errados']:,}\n"
        f"Porcentaje error: {estadisticas['porcentaje_error']:.2f}%\n\n"
        f"MÉTRICAS:\n"
        f"Precisión: {estadisticas['precision']*100:.2f}%\n"
        f"Recall: {estadisticas['recall']*100:.2f}%\n"
        f"F1-Score: {estadisticas['f1_score']*100:.2f}%\n"
        f"Exactitud: {estadisticas['exactitud']*100:.2f}%\n"
        f"Especificidad: {estadisticas['especificidad']*100:.2f}%"
    )
    
    ax4.text(0.1, 0.5, resumen_texto, fontsize=11, 
            verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if ruta_salida:
        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
        print(f"Histograma guardado: {ruta_salida}")
    
    return fig

def visualizar_tres_imagenes(imagen_original, imagen_prediccion, imagen_diferencias, estadisticas, titulo="Comparación de Imágenes"):
    """
    Visualiza las tres imágenes juntas en una sola figura.
    
    Args:
        imagen_original: Imagen original procesada (con fondo blanco)
        imagen_prediccion: Imagen de predicción procesada (con fondo blanco)
        imagen_diferencias: Imagen de diferencias generada (con fondo blanco)
        estadisticas: Diccionario con estadísticas
        titulo: Título para la figura
    """
    
    # Crear figura grande
    fig = plt.figure(figsize=(18, 10))
    
    # Configurar la cuadrícula de subplots
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1])
    
    # 1. Imagen Original Procesada (con fondo blanco)
    ax1 = fig.add_subplot(gs[0, 0])
    if len(imagen_original.shape) == 3 and imagen_original.shape[2] >= 3:
        # Convertir de BGR a RGB para matplotlib
        display_img = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2RGB)
        ax1.imshow(display_img)
    else:
        ax1.imshow(imagen_original, cmap='gray')
    ax1.set_title('IMAGEN ORIGINAL PROCESADA', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'Dimensiones: {imagen_original.shape[1]}×{imagen_original.shape[0]}', fontsize=10)
    ax1.text(0.02, 0.98, f'Incendios: {estadisticas["pixeles_incendio_original"]:,} pix',
             transform=ax1.transAxes, fontsize=10, color='red',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             verticalalignment='top')
    ax1.axis('off')
    
    # 2. Imagen de Predicción Procesada (con fondo blanco)
    ax2 = fig.add_subplot(gs[0, 1])
    if len(imagen_prediccion.shape) == 3 and imagen_prediccion.shape[2] >= 3:
        # Convertir de BGR a RGB para matplotlib
        display_img = cv2.cvtColor(imagen_prediccion, cv2.COLOR_BGR2RGB)
        ax2.imshow(display_img)
    else:
        ax2.imshow(imagen_prediccion, cmap='gray')
    ax2.set_title('IMAGEN PREDICCIÓN PROCESADA', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'Dimensiones: {imagen_prediccion.shape[1]}×{imagen_prediccion.shape[0]}', fontsize=10)
    ax2.text(0.02, 0.98, f'Incendios: {estadisticas["pixeles_incendio_prediccion"]:,} pix',
             transform=ax2.transAxes, fontsize=10, color='red',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             verticalalignment='top')
    ax2.axis('off')
    
    # 3. Imagen de Diferencias (con fondo blanco)
    ax3 = fig.add_subplot(gs[0, 2])
    # Convertir de BGR a RGB para matplotlib
    display_dif = cv2.cvtColor(imagen_diferencias, cv2.COLOR_BGR2RGB)
    ax3.imshow(display_dif)
    ax3.set_title('IMAGEN DE DIFERENCIAS', fontsize=14, fontweight='bold')
    ax3.set_xlabel(f'Errores: {estadisticas["pixeles_errados"]:,} pix ({estadisticas["porcentaje_error"]:.2f}%)', 
                  fontsize=10)
    ax3.axis('off')
    
    # 4. Leyenda de colores ampliada
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis('off')
    
    # Crear un cuadro de leyenda visual
    leyenda_texto = (
        f"LEYENDA DE COLORES\n"
        f"{'='*20}\n\n"
        f"EN IMAGEN DE DIFERENCIAS (FONDO BLANCO):\n"
        f"• Rojo: Verdaderos Positivos\n"
        f"  (detectados correctamente)\n"
        f"• Azul oscuro: Falsos Negativos\n"
        f"  (incendios reales no detectados)\n"
        f"• Azul brillante: Falsos Positivos\n"
        f"  (detectados erróneamente)\n"
        f"• Blanco: Fondo sin cambios\n\n"
        f"IMÁGENES PROCESADAS:\n"
        f"• Fondo: Blanco (255,255,255)\n"
        f"• Incendios: Rojo (255,0,0)\n\n"
        f"{'='*20}\n"
        f"RESUMEN ESTADÍSTICO:\n\n"
        f"Precisión: {estadisticas['precision']*100:.2f}%\n"
        f"Recall: {estadisticas['recall']*100:.2f}%\n"
        f"F1-Score: {estadisticas['f1_score']*100:.2f}%\n"
        f"Exactitud: {estadisticas['exactitud']*100:.2f}%\n\n"
        f"VP: {estadisticas['verdaderos_positivos']:,}\n"
        f"FP: {estadisticas['falsos_positivos']:,}\n"
        f"FN: {estadisticas['falsos_negativos']:,}"
    )
    
    ax4.text(0.05, 0.95, leyenda_texto, fontsize=11, 
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 5. Histograma de métricas (en la fila inferior)
    ax5 = fig.add_subplot(gs[1, :])
    
    # Preparar datos para el histograma simple
    metricas = ['Precisión', 'Recall', 'F1-Score', 'Exactitud', 'Especificidad']
    valores = [
        estadisticas['precision'] * 100,
        estadisticas['recall'] * 100,
        estadisticas['f1_score'] * 100,
        estadisticas['exactitud'] * 100,
        estadisticas['especificidad'] * 100
    ]
    
    # Crear barras con colores según el valor
    colors = []
    for val in valores:
        if val >= 80:
            colors.append('green')
        elif val >= 60:
            colors.append('yellow')
        else:
            colors.append('red')
    
    bars = ax5.bar(metricas, valores, color=colors, edgecolor='black')
    ax5.set_ylabel('Porcentaje (%)', fontsize=12)
    ax5.set_title('MÉTRICAS DE RENDIMIENTO', fontsize=14, fontweight='bold')
    ax5.set_ylim(0, 100)
    ax5.grid(True, axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{valor:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Añadir líneas de referencia
    ax5.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Excelente (≥80%)')
    ax5.axhline(y=60, color='yellow', linestyle='--', alpha=0.5, label='Aceptable (≥60%)')
    ax5.legend(loc='upper right')
    
    # Configurar título general
    fig.suptitle(titulo, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def guardar_visualizacion(fig, ruta_salida):
    """
    Guarda la visualización en un archivo.
    """
    fig.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    print(f"Visualización guardada: {ruta_salida}")

def guardar_resultados(ruta_salida, imagen_diferencias, estadisticas, geotransform=None, projection=None):
    """
    Guarda la imagen de diferencias y un reporte de estadísticas.
    """
    # Guardar imagen de diferencias
    if ruta_salida:
        # La imagen ya está en formato correcto (BGR para OpenCV)
        # Pero OpenCV espera imágenes en formato BGR, así que está bien
        cv2.imwrite(ruta_salida, imagen_diferencias)
        print(f"Imagen de diferencias guardada: {ruta_salida}")
    
    # Guardar estadísticas en archivo de texto
    ruta_base = Path(ruta_salida).stem
    ruta_estadisticas = f"{ruta_base}_estadisticas.txt"
    ruta_histograma = f"{ruta_base}_histograma.png"
    
    # Crear y guardar histograma
    fig = crear_histograma_estadisticas(estadisticas, ruta_histograma)
    
    with open(ruta_estadisticas, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("REPORTE DETALLADO DE COMPARACIÓN DE DETECCIÓN DE INCENDIOS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. ESTADÍSTICAS GENERALES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total de píxeles: {estadisticas['total_pixeles']:,}\n")
        f.write(f"Píxeles de incendio (original): {estadisticas['pixeles_incendio_original']:,} ")
        f.write(f"({estadisticas['porcentaje_incendio_original']:.2f}% del total)\n")
        f.write(f"Píxeles de incendio (predicción): {estadisticas['pixeles_incendio_prediccion']:,} ")
        f.write(f"({estadisticas['porcentaje_incendio_prediccion']:.2f}% del total)\n")
        f.write(f"Píxeles errados: {estadisticas['pixeles_errados']:,}\n")
        f.write(f"Porcentaje de error: {estadisticas['porcentaje_error']:.2f}%\n\n")
        
        f.write("2. MATRIZ DE CONFUSIÓN:\n")
        f.write("-" * 40 + "\n")
        f.write("                         Predicción\n")
        f.write("                  Positivo     Negativo\n")
        f.write("Real   Positivo   %7d      %7d\n" % (estadisticas['verdaderos_positivos'], 
                                                    estadisticas['falsos_negativos']))
        f.write("       Negativo   %7d      %7d\n\n" % (estadisticas['falsos_positivos'], 
                                                      estadisticas['verdaderos_negativos']))
        
        f.write("Distribución porcentual:\n")
        f.write(f"  Verdaderos Positivos: {estadisticas['porcentaje_vp']:.2f}%\n")
        f.write(f"  Falsos Positivos: {estadisticas['porcentaje_fp']:.2f}%\n")
        f.write(f"  Falsos Negativos: {estadisticas['porcentaje_fn']:.2f}%\n")
        f.write(f"  Verdaderos Negativos: {estadisticas['porcentaje_vn']:.2f}%\n\n")
        
        f.write("3. MÉTRICAS DE PRECISIÓN:\n")
        f.write("-" * 40 + "\n")
        f.write("Fórmulas de cálculo:\n")
        f.write("  Precisión = VP / (VP + FP) = De los detectados, cuántos eran reales\n")
        f.write("  Recall    = VP / (VP + FN) = De los reales, cuántos detectamos\n")
        f.write("  F1-Score  = 2 * (Precisión * Recall) / (Precisión + Recall)\n")
        f.write("  Exactitud = (VP + VN) / Total = Porcentaje total de aciertos\n")
        f.write("  Especificidad = VN / (VN + FP) = Capacidad de detectar no incendios\n\n")
        
        f.write("Resultados:\n")
        f.write(f"  Precisión:     {estadisticas['precision']:.4f} ({estadisticas['precision']*100:.2f}%)\n")
        f.write(f"  Recall:        {estadisticas['recall']:.4f} ({estadisticas['recall']*100:.2f}%)\n")
        f.write(f"  F1-Score:      {estadisticas['f1_score']:.4f} ({estadisticas['f1_score']*100:.2f}%)\n")
        f.write(f"  Exactitud:     {estadisticas['exactitud']:.4f} ({estadisticas['exactitud']*100:.2f}%)\n")
        f.write(f"  Especificidad: {estadisticas['especificidad']:.4f} ({estadisticas['especificidad']*100:.2f}%)\n\n")
        
        f.write("4. LEYENDA DE COLORES EN IMAGEN DE DIFERENCIAS:\n")
        f.write("-" * 40 + "\n")
        f.write("NOTA: Todas las imágenes tienen fondo blanco (255,255,255)\n\n")
        f.write("• Rojo (255,0,0): Verdaderos Positivos (incendios detectados correctamente)\n")
        f.write("• Azul oscuro (0,0,150): Falsos Negativos (incendios reales no detectados)\n")
        f.write("• Azul brillante (0,0,255): Falsos Positivos (detectados erróneamente)\n")
        f.write("• Blanco (255,255,255): Fondo sin cambios\n\n")
        
        f.write("5. PROCESAMIENTO APLICADO:\n")
        f.write("-" * 40 + "\n")
        f.write("• Imagen original: Se extrajeron píxeles rojos (R>200, G<100, B<100)\n")
        f.write("• Imagen predicción: Se extrajeron píxeles rojos (R>200, G<50, B<50)\n")
        f.write("• Ambas imágenes convertidas a fondo blanco con incendios rojos\n")
        f.write("• Imagen de diferencias creada sobre fondo blanco\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("=" * 70 + "\n")
    
    print(f"Reporte de estadísticas guardado: {ruta_estadisticas}")
    
    return ruta_estadisticas, ruta_histograma

def main():
    parser = argparse.ArgumentParser(description='Compara imágenes de detección de incendios')
    parser.add_argument('--original', required=True, help='Ruta a la imagen original')
    parser.add_argument('--prediccion', required=True, help='Ruta a la imagen de predicción')
    parser.add_argument('--salida', default='diferencias.png', help='Ruta para guardar imagen de diferencias')
    parser.add_argument('--umbral_rojo', type=int, default=200, help='Umbral para detectar píxeles rojos (0-255)')
    parser.add_argument('--visualizar', action='store_true', help='Mostrar visualización de resultados')
    parser.add_argument('--guardar_visualizacion', default='', help='Ruta para guardar visualización (si no se especifica, no guarda)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("COMPARADOR DE DETECCIÓN DE INCENDIOS - VERSIÓN MEJORADA")
    print("Imágenes procesadas con FONDO BLANCO")
    print("=" * 70)
    
    try:
        # Comparar las imágenes
        img_diferencias, estadisticas, geotransform, projection, img_original, img_prediccion = comparar_imagenes(
            args.original, args.prediccion, args.umbral_rojo
        )
        
        # Guardar resultados
        ruta_estadisticas, ruta_histograma = guardar_resultados(
            args.salida, img_diferencias, estadisticas, geotransform, projection
        )
        
        # Mostrar resumen detallado
        print("\n" + "=" * 70)
        print("RESUMEN DETALLADO DE RESULTADOS")
        print("=" * 70)
        print(f"Total píxeles analizados: {estadisticas['total_pixeles']:,}")
        print(f"Píxeles de incendio en original: {estadisticas['pixeles_incendio_original']:,}")
        print(f"Píxeles de incendio en predicción: {estadisticas['pixeles_incendio_prediccion']:,}")
        print(f"Píxeles errados: {estadisticas['pixeles_errados']:,} ({estadisticas['porcentaje_error']:.2f}%)")
        print(f"\nMÉTRICAS PRINCIPALES:")
        print(f"  Precisión:   {estadisticas['precision']*100:.2f}%")
        print(f"  Recall:      {estadisticas['recall']*100:.2f}%")
        print(f"  F1-Score:    {estadisticas['f1_score']*100:.2f}%")
        print(f"  Exactitud:   {estadisticas['exactitud']*100:.2f}%")
        print(f"  Especificidad: {estadisticas['especificidad']*100:.2f}%")
        
        # Visualizar si se solicita
        if args.visualizar:
            # Crear título personalizado
            titulo = f"Comparación de Detección de Incendios\nOriginal: {Path(args.original).name} | Predicción: {Path(args.prediccion).name}"
            
            # Visualizar las tres imágenes
            fig = visualizar_tres_imagenes(
                img_original, 
                img_prediccion, 
                img_diferencias, 
                estadisticas,
                titulo
            )
            
            # Guardar visualización si se especifica
            if args.guardar_visualizacion:
                guardar_visualizacion(fig, args.guardar_visualizacion)
        
        print("\n" + "=" * 70)
        print("PROCESO COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print(f"Archivos generados:")
        print(f"  1. Imagen de diferencias: {args.salida}")
        print(f"  2. Reporte estadístico: {ruta_estadisticas}")
        print(f"  3. Histograma: {ruta_histograma}")
        if args.guardar_visualizacion:
            print(f"  4. Visualización completa: {args.guardar_visualizacion}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
        
