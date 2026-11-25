# Mobile-UNet 27: cambios respecto a la versión 26

## Cómo se usa ahora
1. Copia `.env.ejemplo` como `.env` y pon `BASE_DIR=/home/liese2/SPRI_AI_project` (sin `/` al final).
2. Edita el bloque `CONFIGURACIÓN` que está al inicio de cada script. Ya no hay argumentos por terminal.
3. Ejecuta los scripts en este orden: `python3 revisar_zips.py`, luego `python3 generate_wildfire_dataset.py`, luego `./run_training.sh` y al final `python3 predict_segmentation.py` o `python3 inferencias.py`.
4. `python3 chec_quant.py` revisa si hay saturación en cada punto de cuantización y `python3 mem_map.py` calcula la memoria a 16 bits.

Rutas por defecto, construidas a partir de BASE_DIR:
| Script | Variable | Ruta |
|---|---|---|
| generate / revisar_zips | INPUT_DIR | `$BASE_DIR/Dataset/Raw` |
| generate | OUTPUT_DIR | `$BASE_DIR/Dataset/Dataset_v3` |
| train / chec_quant | DATA_DIR | `$BASE_DIR/Dataset/Dataset_v3` |
| train | OUT_DIR | `<carpeta del código>/runs` |
| predict | ENTRADAS | `$BASE_DIR/Inferencias/Input/*_Merged.tif` |
| predict | DIR_SALIDA | `$BASE_DIR/Inferencias/Predicciones` |
| inferencias | DIR_SALIDA | `$BASE_DIR/Inferencias/Output_Inferencias_Mobile-UNet/Output_MbUN_27` |

## Cambios pedidos
- **Aumentaciones.** Se aplican sobre los DN, antes de calcular los índices. Incluyen volteos y rot90 (las 8 orientaciones), escala aleatoria entre 0.8 y 1.25, brillo global con variación por banda, bruma (un offset aditivo que es mayor en el azul), ruido y desenfoque. Cada una tiene su probabilidad `AUG_P_*` al inicio del script de entrenamiento.
- **Cuantización a 16 bits.** Los formatos Qm.n se declaran al inicio (`Q_*_FRAC`). Ahora `int_bits` se calcula como BITS − FRAC − signo. Se agregaron puntos de cuantización en la entrada, el bias, la salida de cada upconv, la suma residual y los logits. Antes esos puntos se quedaban en float.
  - entrada Q3.12 · pesos Q2.13 · ReLU6 UQ3.13 · proyección, residual, upconv y logits Q4.11 · bias Q4.11
  - El redondeo es `half_up` (floor(x+0.5)), que es lo que normalmente se implementa en HDL.
  - `QUANT_8BIT_LEGACY` reproduce exactamente el modelo anterior. Los checkpoints viejos se cargan con esa configuración.
- **Índices y bandas.** Los índices se calculan al vuelo con `construir_entrada()`, igual en el entrenamiento que en la predicción, así que el dataset no cambia cuando se cambian los índices.
  - Por defecto `INDICES = ("NDVI", "BAI")`, lo que da 6 canales de entrada. Los dos se calculan solo con R y NIR, así que son compatibles con la cámara RGB+IR.
  - También están disponibles GNDVI y SAVI. NBR, NBR2 y NDMI necesitan B11/B12 (SWIR), que se activan en `BANDAS_DATASET` del generador. La cámara no tiene SWIR.
- **Ctrl+C.** Ya no escribe `final.pth`, así que con `REANUDAR = "auto"` la siguiente ejecución continúa la corrida interrumpida.
- **Configuración al inicio.** Cada script declara al principio sus rutas, canales de entrada y salida, bits, hiperparámetros y aumentaciones. Esos valores se guardan en el checkpoint, y predict, chec_quant y mem_map los leen de ahí.

## Errores encontrados y corregidos
1. `run_training.sh`: cada reintento después de una caída empezaba un entrenamiento nuevo desde cero. Ahora `REANUDAR = "auto"` continúa desde `last.pth`. Además, un error de configuración (código 2) ya no se reintenta indefinidamente, y `echo` sin `-e` imprimía `\n` como texto.
2. Aumentaciones: el brillo y el ruido se aplicaban sobre la entrada ya normalizada. Con z-score, multiplicar por una ganancia desplazaba la media de forma incorrecta. Si se hubiera agregado NDVI con ese esquema, el índice habría quedado inconsistente con las bandas.
3. `fake_quantize`: el parámetro `int_bits` se ignoraba y `total_bits=8` estaba fijo. La ReLU que según el código era "int_bits=3, frac 4" en realidad era UQ4.4, con rango [0, 16).
4. El modelo no era completamente de punto fijo: la entrada, la salida de las ConvTranspose2d, los bias, la suma residual y los logits quedaban en float. Lo simulado no coincidía con lo que haría la FPGA.
5. `revisar_zips.py` usaba `--min-fire 0.035` y el generador usaba `0.03`, así que los conteos de ventanas no coincidían. Ahora revisar_zips importa los parámetros del generador. El valor unificado es 0.035.
6. `inferencias.py` solo buscaba `weights/model_*.pth` y nunca encontraba los `runs/*/best.pth` del entrenamiento nuevo. Además lanzaba un proceso por cada par imagen×modelo y recargaba el modelo cada vez, tenía un ciclo de validación vacío (`for … pass`) y su docstring no coincidía con el nombre de salida real.
7. `mem_map.py` quitaba el prefijo `module.` antes de desempaquetar `model_state_dict`, por lo que no funcionaba con checkpoints de DataParallel. También tenía `sys.path` fijo en `/home/liese2/...` e `in_channels=4` fijo.
8. `chec_quant.py` solo revisaba las salidas de FakeQuantAct con entrada aleatoria en [0,1], no medía la saturación y el docstring mencionaba otro nombre de archivo.

## Pendientes (no corregidos)
- **Las BatchNorm no están plegadas.** Al exportar a la FPGA hay que plegar cada BN en su convolución y volver a cuantizar. Los pesos plegados no son exactamente los que vio el QAT. Lo correcto sería hacer QAT con BN plegada o agregar un script de exportación que pliegue BN y verifique el resultado contra el modelo.
- **Tamaño del modelo.** Con BASE_CHANNELS=64 y EXPAND_RATIO=6 el modelo tiene alrededor de 21.8 M parámetros, unos 44 MB a 16 bits (antes eran 22 MB a 8 bits). Casi todo está en dec4, que expande 1024 canales a 6144. Con BASE_CHANNELS=32 serían unos 5.5 M parámetros.
- **BAI depende de la calibración.** Usa reflectancias absolutas (0.1 y 0.06). Con la cámara habrá que calibrar a reflectancia o quitar BAI. NDVI es más robusto porque es una razón entre bandas.
- **Diferencia de sensor.** El NIR de Sentinel-2 (B08, 842 nm) no es igual al IR de la OX05B1S. Los índices entrenados con Sentinel-2 no se transfieren directamente a la cámara.
