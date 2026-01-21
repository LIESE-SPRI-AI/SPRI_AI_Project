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
    imagen = np.stack(bandas, axis=-1)
    
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

def comparar_imagenes(ruta_original, ruta_prediccion, umbral_rojo=200):
    """
    Compara dos imágenes de incendios y genera una imagen de diferencias.
    
    Args:
        ruta_original: Ruta a la imagen original con detección de rojos
        ruta_prediccion: Ruta a la imagen de predicción
        umbral_rojo: Umbral para detectar píxeles rojos (0-255)
    
    Returns:
        imagen_diferencias: Imagen RGB con diferencias marcadas en azul
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
    
    print("Extrayendo capa de incendios de la predicción...")
    mascara_prediccion = extraer_capa_incendios(img_prediccion, umbral_rojo)
    
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
    
    # Crear imagen de diferencias
    imagen_diferencias = np.zeros_like(img_original)
    
    # Copiar la imagen original en escala de grises para fondo
    if len(img_original.shape) == 3 and img_original.shape[2] >= 3:
        fondo_gris = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
        imagen_diferencias[:, :, 0] = fondo_gris
        imagen_diferencias[:, :, 1] = fondo_gris
        imagen_diferencias[:, :, 2] = fondo_gris
    else:
        imagen_diferencias[:, :, :] = img_original[:, :, :3]
    
    # Marcar diferencias en azul
    # Falsos Negativos: Azul oscuro (0, 0, 100)
    imagen_diferencias[falsos_negativos] = [0, 0, 100]
    
    # Falsos Positivos: Azul brillante (0, 0, 255)
    imagen_diferencias[falsos_positivos] = [0, 0, 255]
    
    # Verdaderos Positivos: Mantener rojo original (255, 0, 0)
    imagen_diferencias[verdaderos_positivos] = [255, 0, 0]
    
    # Calcular estadísticas
    total_pixeles = img_original.shape[0] * img_original.shape[1]
    pixeles_incendio_original = np.sum(mascara_original_bool)
    pixeles_incendio_prediccion = np.sum(mascara_prediccion_bool)
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
    
    # Precisión, Recall y F1-Score
    if (verdaderos_pos + falsos_pos) > 0:
        precision = verdaderos_pos / (verdaderos_pos + falsos_pos)
    else:
        precision = 0
    
    if (verdaderos_pos + falsos_neg) > 0:
        recall = verdaderos_pos / (verdaderos_pos + falsos_neg)
    else:
        recall = 0
    
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    exactitud = (verdaderos_pos + verdaderos_neg) / total_pixeles
    
    estadisticas = {
        'total_pixeles': total_pixeles,
        'pixeles_incendio_original': pixeles_incendio_original,
        'pixeles_incendio_prediccion': pixeles_incendio_prediccion,
        'pixeles_errados': pixeles_errados,
        'porcentaje_error': porcentaje_error,
        'verdaderos_positivos': verdaderos_pos,
        'falsos_positivos': falsos_pos,
        'falsos_negativos': falsos_neg,
        'verdaderos_negativos': verdaderos_neg,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'exactitud': exactitud
    }
    
    return imagen_diferencias, estadisticas, geotransform, projection

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
    ruta_estadisticas = ruta_salida.replace('.png', '_estadisticas.txt').replace('.tif', '_estadisticas.txt')
    
    with open(ruta_estadisticas, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE COMPARACIÓN DE DETECCIÓN DE INCENDIOS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("ESTADÍSTICAS GENERALES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total de píxeles: {estadisticas['total_pixeles']:,}\n")
        f.write(f"Píxeles de incendio (original): {estadisticas['pixeles_incendio_original']:,} ({estadisticas['pixeles_incendio_original']/estadisticas['total_pixeles']*100:.2f}%)\n")
        f.write(f"Píxeles de incendio (predicción): {estadisticas['pixeles_incendio_prediccion']:,} ({estadisticas['pixeles_incendio_prediccion']/estadisticas['total_pixeles']*100:.2f}%)\n")
        f.write(f"Píxeles errados: {estadisticas['pixeles_errados']:,}\n")
        f.write(f"Porcentaje de error: {estadisticas['porcentaje_error']:.2f}%\n\n")
        
        f.write("MATRIZ DE CONFUSIÓN:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Verdaderos Positivos (VP): {estadisticas['verdaderos_positivos']:,}\n")
        f.write(f"Falsos Positivos (FP): {estadisticas['falsos_positivos']:,}\n")
        f.write(f"Falsos Negativos (FN): {estadisticas['falsos_negativos']:,}\n")
        f.write(f"Verdaderos Negativos (VN): {estadisticas['verdaderos_negativos']:,}\n\n")
        
        f.write("MÉTRICAS DE PRECISIÓN:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Precisión (VP/(VP+FP)): {estadisticas['precision']:.4f} ({estadisticas['precision']*100:.2f}%)\n")
        f.write(f"Recall (VP/(VP+FN)): {estadisticas['recall']:.4f} ({estadisticas['recall']*100:.2f}%)\n")
        f.write(f"F1-Score: {estadisticas['f1_score']:.4f}\n")
        f.write(f"Exactitud: {estadisticas['exactitud']:.4f} ({estadisticas['exactitud']*100:.2f}%)\n\n")
        
        f.write("LEYENDA DE COLORES EN IMAGEN DE DIFERENCIAS:\n")
        f.write("-" * 40 + "\n")
        f.write("• Rojo (255,0,0): Incendios detectados correctamente\n")
        f.write("• Azul oscuro (0,0,100): Falsos Negativos (incendios no detectados)\n")
        f.write("• Azul brillante (0,0,255): Falsos Positivos (detectados erróneamente)\n")
        f.write("• Escala de grises: Fondo sin cambios\n")
    
    print(f"Reporte de estadísticas guardado: {ruta_estadisticas}")
    
    return ruta_estadisticas

def visualizar_resultados(img_original, img_prediccion, img_diferencias, estadisticas):
    """
    Visualiza las imágenes y resultados.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Imagen original
    axes[0, 0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Imagen Original')
    axes[0, 0].set_xlabel(f'Píxeles incendio: {estadisticas["pixeles_incendio_original"]:,}')
    axes[0, 0].axis('off')
    
    # Imagen de predicción
    axes[0, 1].imshow(cv2.cvtColor(img_prediccion, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Imagen de Predicción')
    axes[0, 1].set_xlabel(f'Píxeles incendio: {estadisticas["pixeles_incendio_prediccion"]:,}')
    axes[0, 1].axis('off')
    
    # Imagen de diferencias
    axes[1, 0].imshow(cv2.cvtColor(img_diferencias, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Diferencias (Azul = Errores)')
    axes[1, 0].set_xlabel(f'Píxeles errados: {estadisticas["pixeles_errados"]:,} ({estadisticas["porcentaje_error"]:.2f}%)')
    axes[1, 0].axis('off')
    
    # Leyenda
    axes[1, 1].axis('off')
    leyenda_texto = (
        f"RESULTADOS:\n"
        f"Precisión: {estadisticas['precision']*100:.2f}%\n"
        f"Recall: {estadisticas['recall']*100:.2f}%\n"
        f"F1-Score: {estadisticas['f1_score']:.4f}\n"
        f"Exactitud: {estadisticas['exactitud']*100:.2f}%\n\n"
        f"LEYENDA:\n"
        f"• Azul Detección correcta\n"
        f"• Rojo oscuro: Falsos Negativos\n"
        f"• Rojo brillante: Falsos Positivos"
    )
    axes[1, 1].text(0.1, 0.5, leyenda_texto, fontsize=12, 
                   verticalalignment='center', transform=axes[1, 1].transAxes)
    
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
    
    print("=" * 60)
    print("COMPARADOR DE DETECCIÓN DE INCENDIOS")
    print("=" * 60)
    
    try:
        # Comparar las imágenes
        img_diferencias, estadisticas, geotransform, projection = comparar_imagenes(
            args.original, args.prediccion, args.umbral_rojo
        )
        
        # Guardar resultados
        ruta_estadisticas = guardar_resultados(args.salida, img_diferencias, estadisticas, geotransform, projection)
        
        # Mostrar resumen
        print("\n" + "=" * 60)
        print("RESUMEN DE RESULTADOS")
        print("=" * 60)
        print(f"Porcentaje de error: {estadisticas['porcentaje_error']:.2f}%")
        print(f"Píxeles errados: {estadisticas['pixeles_errados']:,} de {estadisticas['total_pixeles']:,}")
        print(f"Precisión: {estadisticas['precision']*100:.2f}%")
        print(f"Recall: {estadisticas['recall']*100:.2f}%")
        print(f"F1-Score: {estadisticas['f1_score']:.4f}")
        
        # Visualizar si se solicita
        if args.visualizar:
            # Cargar imágenes para visualización
            img_orig, _, _ = cargar_imagen_geotiff(args.original)
            img_pred, _, _ = cargar_imagen_geotiff(args.prediccion)
            visualizar_resultados(img_orig, img_pred, img_diferencias, estadisticas)
        
        print("\n" + "=" * 60)
        print("PROCESO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()