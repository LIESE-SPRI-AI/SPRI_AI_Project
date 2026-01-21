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

def extraer_capa_incendios(imagen, umbral_rojo=200):
    """
    Extrae la capa de incendios de una imagen RGB.
    Detecta píxeles rojos donde R > umbral_rojo y G, B bajos.
    """
    if len(imagen.shape) == 2:
        # Si es una sola banda, convertir a 3 bandas
        imagen = np.stack([imagen, imagen, imagen], axis=-1)
    
    # Separar canales
    if imagen.shape[2] >= 3:
        r = imagen[:, :, 0]
        g = imagen[:, :, 1]
        b = imagen[:, :, 2]
    else:
        raise ValueError("La imagen debe tener al menos 3 canales (RGB)")
    
    # Crear máscara para píxeles rojos (fuego)
    mascara_incendios = (r > umbral_rojo) & (g < 100) & (b < 100)
    
    return mascara_incendios.astype(np.uint8) * 255

def procesar_prediccion(imagen_prediccion):
    """
    Procesa la imagen de predicción:
    1. Quita el fondo negro (píxeles con valor 0 en todas las bandas)
    2. Detecta incendios basados en colores rojos
    """
    if len(imagen_prediccion.shape) == 2:
        # Convertir imagen de 1 banda a 3 bandas
        imagen_prediccion = np.stack([imagen_prediccion, imagen_prediccion, imagen_prediccion], axis=-1)
    
    # Crear máscara para fondo negro (píxeles con valor 0 en todas las bandas)
    mascara_fondo = np.all(imagen_prediccion == 0, axis=2)
    
    # Detectar incendios en la predicción
    if imagen_prediccion.shape[2] >= 3:
        r = imagen_prediccion[:, :, 0]
        g = imagen_prediccion[:, :, 1]
        b = imagen_prediccion[:, :, 2]
        
        # Detectar píxeles rojos (incendios)
        mascara_rojos = (r > 150) & (g < 50) & (b < 50)
        
        # Combinar máscaras
        mascara_final = mascara_rojos & ~mascara_fondo
    
    return mascara_final.astype(np.uint8) * 255

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
    
    print(f"Dimensiones original: {img_original.shape}")
    print(f"Dimensiones predicción: {img_prediccion.shape}")
    
    # Verificar que las dimensiones coincidan
    if img_original.shape[:2] != img_prediccion.shape[:2]:
        print("Ajustando dimensiones de la predicción...")
        img_prediccion = cv2.resize(img_prediccion, 
                                    (img_original.shape[1], img_original.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
    
    # Extraer capas de incendios
    print("Extrayendo capa de incendios de la imagen original...")
    mascara_original = extraer_capa_incendios(img_original, umbral_rojo)
    
    print("Procesando imagen de predicción...")
    mascara_prediccion = procesar_prediccion(img_prediccion)
    
    # Convertir máscaras a booleanas (0 o 1)
    mascara_original_bool = mascara_original > 0
    mascara_prediccion_bool = mascara_prediccion > 0
    
    # Calcular diferencias
    # Falsos Negativos: Incendios en original pero no en predicción
    falsos_negativos = mascara_original_bool & ~mascara_prediccion_bool
    
    # Falsos Positivos: Incendios en predicción pero no en original
    falsos_positivos = ~mascara_original_bool & mascara_prediccion_bool
    
    # Verdaderos Positivos: Incendios correctamente detectados
    verdaderos_positivos = mascara_original_bool & mascara_prediccion_bool
    
    # Verdaderos Negativos: Fondo correctamente identificado
    verdaderos_negativos = ~mascara_original_bool & ~mascara_prediccion_bool
    
    # Crear imagen de diferencias con colores CORREGIDOS
    imagen_diferencias = np.zeros((img_original.shape[0], img_original.shape[1], 3), dtype=np.uint8)
    
    # Fondo: Convertir imagen original a escala de grises
    if len(img_original.shape) == 3 and img_original.shape[2] >= 3:
        fondo_gris = cv2.cvtColor(img_original[:, :, :3], cv2.COLOR_RGB2GRAY)
        imagen_diferencias[:, :, 0] = fondo_gris
        imagen_diferencias[:, :, 1] = fondo_gris
        imagen_diferencias[:, :, 2] = fondo_gris
    else:
        fondo_gris = img_original
        imagen_diferencias[:, :, 0] = fondo_gris
        imagen_diferencias[:, :, 1] = fondo_gris
        imagen_diferencias[:, :, 2] = fondo_gris
    
    # CORREGIDO: Ahora rojo es rojo y azul es azul
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
    
    return imagen_diferencias, estadisticas, geotransform, projection

def calcular_estadisticas_detalladas(mascara_original, mascara_prediccion,
                                   falsos_negativos, falsos_positivos,
                                   verdaderos_positivos, verdaderos_negativos):
    """
    Calcula estadísticas detalladas de la comparación.
    
    Explicación de cómo se calculan las estadísticas:
    1. Matriz de confusión:
       - VP: Pixeles donde ambas imágenes muestran incendio (correcto)
       - FP: Pixeles donde predicción dice incendio pero original no (error tipo I)
       - FN: Pixeles donde original dice incendio pero predicción no (error tipo II)
       - VN: Pixeles donde ambas dicen no incendio (correcto)
    
    2. Métricas derivadas:
       - Precisión = VP / (VP + FP) → De los detectados, cuántos eran reales
       - Recall = VP / (VP + FN) → De los reales, cuántos detectamos
       - F1-Score = 2 * (Precisión * Recall) / (Precisión + Recall) → Media armónica
       - Exactitud = (VP + VN) / Total → Porcentaje total de aciertos
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
    
    Args:
        estadisticas: Diccionario con las estadísticas calculadas
        ruta_salida: Ruta para guardar el histograma (opcional)
    
    Returns:
        fig: Figura de matplotlib
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

def guardar_resultados(ruta_salida, imagen_diferencias, estadisticas, geotransform=None, projection=None):
    """
    Guarda la imagen de diferencias y un reporte de estadísticas.
    """
    # Guardar imagen de diferencias
    if ruta_salida:
        # Convertir a BGR para OpenCV si es necesario
        if len(imagen_diferencias.shape) == 3 and imagen_diferencias.shape[2] == 3:
            imagen_guardar = cv2.cvtColor(imagen_diferencias, cv2.COLOR_RGB2BGR)
            cv2.imwrite(ruta_salida, imagen_guardar)
            print(f"Imagen de diferencias guardada: {ruta_salida}")
        else:
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
        
        f.write("4. INTERPRETACIÓN:\n")
        f.write("-" * 40 + "\n")
        f.write("• Alta Precisión + Alto Recall: Modelo excelente\n")
        f.write("• Alta Precisión + Bajo Recall: Detecta poco pero seguro\n")
        f.write("• Baja Precisión + Alto Recall: Detecta mucho pero con errores\n")
        f.write("• F1-Score > 0.7: Buen rendimiento\n")
        f.write("• F1-Score > 0.9: Excelente rendimiento\n\n")
        
        f.write("5. LEYENDA DE COLORES EN IMAGEN DE DIFERENCIAS:\n")
        f.write("-" * 40 + "\n")
        f.write("• Rojo (255,0,0): Verdaderos Positivos (incendios detectados correctamente)\n")
        f.write("• Azul oscuro (0,0,150): Falsos Negativos (incendios reales no detectados)\n")
        f.write("• Azul brillante (0,0,255): Falsos Positivos (detectados erróneamente)\n")
        f.write("• Escala de grises: Fondo sin cambios\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("=" * 70 + "\n")
    
    print(f"Reporte de estadísticas guardado: {ruta_estadisticas}")
    
    return ruta_estadisticas, ruta_histograma

def visualizar_resultados(img_original, img_prediccion, img_diferencias, estadisticas):
    """
    Visualiza las imágenes y resultados.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Imagen original
    if len(img_original.shape) == 3 and img_original.shape[2] >= 3:
        axes[0, 0].imshow(cv2.cvtColor(img_original[:, :, :3], cv2.COLOR_BGR2RGB))
    else:
        axes[0, 0].imshow(img_original, cmap='gray')
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].set_xlabel(f'Píxeles incendio: {estadisticas["pixeles_incendio_original"]:,}')
    axes[0, 0].axis('off')
    
    # Imagen de predicción
    if len(img_prediccion.shape) == 3 and img_prediccion.shape[2] >= 3:
        axes[0, 1].imshow(cv2.cvtColor(img_prediccion[:, :, :3], cv2.COLOR_BGR2RGB))
    else:
        axes[0, 1].imshow(img_prediccion, cmap='gray')
    axes[0, 1].set_title('Imagen de Predicción')
    axes[0, 1].set_xlabel(f'Píxeles incendio: {estadisticas["pixeles_incendio_prediccion"]:,}')
    axes[0, 1].axis('off')
    
    # Imagen de diferencias
    axes[0, 2].imshow(cv2.cvtColor(img_diferencias, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Mapa de Diferencias')
    axes[0, 2].set_xlabel(f'Píxeles errados: {estadisticas["pixeles_errados"]:,} ({estadisticas["porcentaje_error"]:.2f}%)')
    axes[0, 2].axis('off')
    
    # Crear histograma en el subplot inferior
    crear_histograma_estadisticas(estadisticas)
    
    # Eliminar los ejes vacíos restantes
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compara imágenes de detección de incendios')
    parser.add_argument('--original', required=True, help='Ruta a la imagen original')
    parser.add_argument('--prediccion', required=True, help='Ruta a la imagen de predicción')
    parser.add_argument('--salida', default='diferencias.png', help='Ruta para guardar imagen de diferencias')
    parser.add_argument('--umbral_rojo', type=int, default=200, help='Umbral para detectar píxeles rojos (0-255)')
    parser.add_argument('--visualizar', action='store_true', help='Mostrar visualización de resultados')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("COMPARADOR DE DETECCIÓN DE INCENDIOS - VERSIÓN MEJORADA")
    print("=" * 70)
    
    try:
        # Comparar las imágenes
        img_diferencias, estadisticas, geotransform, projection = comparar_imagenes(
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
            # Cargar imágenes para visualización
            img_orig, _, _ = cargar_imagen_geotiff(args.original)
            img_pred, _, _ = cargar_imagen_geotiff(args.prediccion)
            visualizar_resultados(img_orig, img_pred, img_diferencias, estadisticas)
        
        print("\n" + "=" * 70)
        print("PROCESO COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print(f"Archivos generados:")
        print(f"  1. Imagen de diferencias: {args.salida}")
        print(f"  2. Reporte estadístico: {ruta_estadisticas}")
        print(f"  3. Histograma: {ruta_histograma}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()