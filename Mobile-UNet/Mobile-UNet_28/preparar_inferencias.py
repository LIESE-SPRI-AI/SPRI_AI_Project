#!/usr/bin/env python3
"""
preparar_inferencias.py
=======================
Genera las entradas de inferencias.py y comparador_mbun.py a partir de los zips de EO Browser,
con EXACTAMENTE el mismo preprocesamiento que generate_wildfire_dataset.py:

  DIR_MERGED/<nombrebase>_Merged.tif   bandas ORDEN_BANDAS en DN (reflectancia*10000), uint16
  DIR_BURNED/<nombrebase>_Burned.tif   referencia 0 = no quemado, 1 = quemado, 255 = sin dato

Por qué: los *_Merged.tif del generador viejo se hacían con gdal_translate -scale min max 0 65535
POR BANDA. El modelo actual se entrena con DN sin estirar, así que esas entradas llegan con otro
brillo y otras razones entre bandas (NDVI y BAI cambian) y aparecen falsos positivos.

Toda la configuración está en el bloque CONFIGURACIÓN.
"""

import os
import sys
import tempfile

import wildfire_common as wc

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
ZIPS = [wc.ruta_proyecto("Inferencias", "Zips", "*.zip")]       # rutas o patrones glob
DIR_MERGED = wc.ruta_proyecto("Inferencias", "Input")             # DIR_IMAGENES de inferencias.py
DIR_BURNED = wc.ruta_proyecto("Inferencias", "Burned")            # DIR_BURNED de comparador_mbun.py
ORDEN_BANDAS = ("B04", "B03", "B02", "B08")                       # = ORDEN_BANDAS_TIFF de inferencias.py
ESCALA = "auto"                                                   # igual que el generador
UMBRAL_ROJO = 0.75                                                # igual que el generador
UMBRAL_GB = 0.2
SOBRESCRIBIR = False
# =============================================================================


def main():
    import glob
    zips = sorted({p for patron in ZIPS for p in glob.glob(patron)})
    if not zips:
        sys.exit(f"No se encontraron zips con {ZIPS}")
    os.makedirs(DIR_MERGED, exist_ok=True)
    os.makedirs(DIR_BURNED, exist_ok=True)
    errores = 0
    for i, ruta in enumerate(zips, 1):
        base = wc.nombre_seguro(os.path.splitext(os.path.basename(ruta))[0])
        sal_img = os.path.join(DIR_MERGED, f"{base}_Merged.tif")
        sal_ref = os.path.join(DIR_BURNED, f"{base}_Burned.tif")
        if not SOBRESCRIBIR and os.path.exists(sal_img):
            print(f"[{i}/{len(zips)}] {base}: ya existe, se omite")
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="wf_inf_") as tmp:
                wc.extraer_zip_seguro(ruta, tmp)
                hallados = wc.buscar_archivos_escena(tmp, bandas=ORDEN_BANDAS)
                faltan = [b for b in ORDEN_BANDAS if b not in hallados["bandas"]]
                if faltan:
                    raise ValueError(f"faltan bandas {faltan}")
                dn, meta, nodata, info = wc.leer_stack_bandas(hallados["bandas"], escala=ESCALA, orden=ORDEN_BANDAS)
                wc.escribir_raster(sal_img, wc.a_uint16(dn), meta)
                msg = f"escala {info['modo_escala']}, p99 DN {[round(v) for v in info['p99_dn']]}"
                if hallados["mascara"]:
                    mascara, info_m = wc.extraer_mascara_quemado(hallados["mascara"], nodata.shape, meta,
                                                                 umbral_rojo=UMBRAL_ROJO, umbral_gb=UMBRAL_GB)
                    mascara[nodata] = wc.IGNORE_INDEX
                    wc.escribir_raster(sal_ref, mascara, meta, nodata=wc.IGNORE_INDEX)
                    msg += f", {100 * info_m['fraccion_quemada']:.2f}% quemado"
                else:
                    msg += ", sin capa Burned Area (no hay referencia)"
            print(f"[{i}/{len(zips)}] {base}: {msg}")
        except Exception as e:
            errores += 1
            print(f"[{i}/{len(zips)}] {base}: ERROR {type(e).__name__}: {e}")
    print(f"\nEntradas en {DIR_MERGED} | referencias en {DIR_BURNED}" + (f" | {errores} con error" if errores else ""))
    if errores:
        sys.exit(1)


if __name__ == "__main__":
    main()
