#!/usr/bin/env python3
"""
revisar_zips.py
===============
Revisa una carpeta de zips de EO Browser antes de generar el dataset: qué zips traen
las bandas requeridas (B02, B03, B04, B08 + capa Burned Area Detection), cuántos sitios
hay en total y cuántos están disponibles.

Usa exactamente la misma lógica de búsqueda Y LOS MISMOS PARÁMETROS que
generate_wildfire_dataset.py (se importan de ahí: INPUT_DIR, BANDAS_DATASET, TILE, OVERLAP,
MIN_FIRE...), así que "disponible" aquí significa que el generador lo va a poder procesar.
Antes cada script tenía sus propios valores por defecto (p. ej. min-fire 0.03 vs 0.035).

  python3 revisar_zips.py      (configuración en el bloque CONFIGURACIÓN)

Modo rápido (por defecto): lista el contenido de cada zip y lee solo las cabeceras de los
  archivos (tamaño, ubicación). Detecta zips corruptos, capas faltantes y duplicados.
Modo profundo (PROFUNDO = True): además lee las bandas y la máscara completas, cuenta píxeles
  quemados y estima cuántos recortes positivos dará cada zip con los mismos parámetros
  del generador (TILE, OVERLAP, MIN_FIRE, MAX_NODATA).

Un "sitio" es un grupo de zips cuyas huellas geográficas se traslapan (p. ej. el mismo
incendio en varias fechas). Es la misma agrupación que usa el generador para el split.

Salida en SALIDA (por defecto <carpeta del código>/reporte_zips):
  resumen.txt           lo mismo que se imprime en pantalla
  reporte_zips.csv      un renglón por zip: estado, motivo, bandas, fecha, sitio, ...
  reporte_sitios.csv    un renglón por sitio: zips, disponibles, fechas, centro
  zips_disponibles.txt  rutas de los zips utilizables (sin duplicados; incluye los que solo dan negativos)
                        -> LISTA_ZIPS en generate_wildfire_dataset.py
  zips_con_problemas.txt  rutas y motivo de los que no
"""

import csv
import io
import os
import sys
import tempfile
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import wildfire_common as wc
import generate_wildfire_dataset as gen

if wc.HAS_RASTERIO:
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    from rasterio.io import MemoryFile

ESTADOS_OK = ("disponible",)
DESCRIPCION_ESTADO = {
    "disponible": "disponibles para el dataset",
    "solo_negativos": "sin píxeles quemados (solo darían negativos)",
    "sin_positivos": "con quemado pero ninguna ventana llega a --min-fire",
    "sin_mascara": "sin capa Burned Area Detection",
    "faltan_bandas": "les faltan bandas",
    "ilegible": "archivo dañado o ilegible dentro del zip",
    "zip_corrupto": "zip corrupto",
    "error": "error inesperado",
}


# =============================================================================
# CONFIGURACIÓN
# =============================================================================
INPUT_DIR = gen.INPUT_DIR                     # mismos zips que el generador
SALIDA = os.path.join(wc.DIR_CODIGO, "reporte_zips")
RECURSIVO = gen.RECURSIVO
PROFUNDO = False                              # leer bandas y máscara completas (más lento)
WORKERS = gen.WORKERS
# Parámetros compartidos con el generador (cámbialos allá para que ambos coincidan)
BANDAS = tuple(gen.BANDAS_DATASET)
PALABRAS_MASCARA = gen.PALABRAS_MASCARA
TILE, OVERLAP, MIN_FIRE, MAX_NODATA = gen.TILE, gen.OVERLAP, gen.MIN_FIRE, gen.MAX_NODATA
UMBRAL_ROJO, UMBRAL_GB, ESCALA = gen.UMBRAL_ROJO, gen.UMBRAL_GB, gen.ESCALA
# =============================================================================


# ---------------------------------------------------------------------------
# Acceso al contenido de los zips (sin descomprimir todo)
# ---------------------------------------------------------------------------

def listar_entradas_zip(ruta):
    """Lista los archivos del zip (y de zips anidados un nivel) con tamaño y CRC."""
    entradas = []
    with zipfile.ZipFile(ruta) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if info.filename.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(zf.read(info))) as zi:
                    for i2 in zi.infolist():
                        if not i2.is_dir():
                            entradas.append({"nombre": f"{info.filename}!/{i2.filename}", "tam": i2.file_size,
                                             "crc": i2.CRC, "ref": (info.filename, i2.filename)})
            else:
                entradas.append({"nombre": info.filename, "tam": info.file_size, "crc": info.CRC,
                                 "ref": (info.filename,)})
    return entradas


def leer_miembros(ruta, refs):
    """refs: {nombre: ref} -> {nombre: bytes}. Verifica el CRC al leer."""
    salida, internos = {}, {}
    with zipfile.ZipFile(ruta) as zf:
        try:
            for nombre, ref in refs.items():
                if len(ref) == 1:
                    salida[nombre] = zf.read(ref[0])
                else:
                    if ref[0] not in internos:
                        internos[ref[0]] = zipfile.ZipFile(io.BytesIO(zf.read(ref[0])))
                    salida[nombre] = internos[ref[0]].read(ref[1])
        finally:
            for z in internos.values():
                z.close()
    return salida


def _forma(shape, axes):
    if len(shape) == 2:
        return shape[0], shape[1], 1
    if axes.endswith("S"):
        return shape[0], shape[1], shape[2]
    return shape[-2], shape[-1], shape[0]


def _cabecera_rasterio(origen):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with (origen.open() if isinstance(origen, MemoryFile) else rasterio.open(origen)) as s:
            georef = s.crs is not None
            return {"alto": s.height, "ancho": s.width, "canales": s.count, "dtype": s.dtypes[0],
                    "meta": {"georef": georef, "transform": s.transform if georef else None,
                             "crs": s.crs if georef else None},
                    "crs": s.crs.to_string() if georef else ""}


def cabecera_miembro(ruta_zip, nombre, ref):
    """Tamaño, tipo y georreferencia de un archivo dentro del zip, leyendo lo mínimo posible."""
    if wc.HAS_RASTERIO and len(ref) == 1:
        try:  # GDAL lee solo la cabecera directamente del zip
            return _cabecera_rasterio(f"/vsizip/{os.path.abspath(ruta_zip)}/{ref[0]}")
        except Exception:
            pass
    datos = leer_miembros(ruta_zip, {nombre: ref})[nombre]
    if wc.HAS_RASTERIO:
        with MemoryFile(datos) as mf:
            return _cabecera_rasterio(mf)
    ext = os.path.splitext(nombre)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(io.BytesIO(datos)) as tf:
            serie = tf.series[0]
            alto, ancho, canales = _forma(serie.shape, serie.axes)
            dtype = str(serie.dtype)
    else:
        from PIL import Image
        with Image.open(io.BytesIO(datos)) as im:
            ancho, alto = im.size
            canales, dtype = len(im.getbands()), im.mode
    return {"alto": alto, "ancho": ancho, "canales": canales, "dtype": dtype,
            "meta": {"georef": False, "transform": None, "crs": None}, "crs": ""}


# ---------------------------------------------------------------------------
# Revisión de un zip
# ---------------------------------------------------------------------------

def revisar_zip(ruta, cfg):
    reg = {
        "zip": os.path.basename(ruta), "ruta": ruta, "tamano_mb": round(os.path.getsize(ruta) / 1e6, 2),
        "estado": "error", "motivo": "", "fecha": "", "alto": "", "ancho": "", "crs": "",
        "bounds_wgs84": "", "centro_lat": "", "centro_lon": "", "bandas_ok": "", "faltan": "",
        "archivo_mascara": "", "otras_bandas": "", "n_archivos": 0, "avisos": "",
        "modo_escala": "", "pixeles_quemados": "", "fraccion_quemada": "",
        "ventanas_positivas": "", "ventanas_negativas": "", "ventanas_bajas": "", "ventanas_nodata": "",
        "_huella_b04": None,
    }
    avisos = []
    try:
        try:
            entradas = listar_entradas_zip(ruta)
        except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError) as e:
            reg["estado"], reg["motivo"] = "zip_corrupto", f"{type(e).__name__}: {e}"
            return reg

        por_nombre = {e["nombre"]: e for e in entradas}
        nombres = list(por_nombre)
        reg["n_archivos"] = len(nombres)
        cls = wc.clasificar_rutas(nombres, cfg["palabras_mascara"], bandas=cfg["bandas"])
        avisos += cls["avisos"]
        bandas, mascara = cls["bandas"], cls["mascara"]
        faltan = [b for b in cfg["bandas"] if b not in bandas]
        reg["bandas_ok"] = ";".join(b for b in cfg["bandas"] if b in bandas)
        reg["faltan"] = ";".join(faltan)
        reg["archivo_mascara"] = mascara or ""
        extra = sorted(wc.bandas_presentes(nombres) - set(cfg["bandas"]))
        reg["otras_bandas"] = ";".join(extra)
        reg["fecha"] = wc.fecha_de_nombre(bandas.get("B04") or mascara or (nombres[0] if nombres else ""))
        if "B04" in bandas:
            e = por_nombre[bandas["B04"]]
            reg["_huella_b04"] = (e["crc"], e["tam"])

        listado = ", ".join(os.path.basename(n) for n in nombres[:10]) + (" ..." if len(nombres) > 10 else "")
        if mascara is None or faltan:
            partes = []
            if mascara is None:
                partes.append("no hay capa de área quemada")
            if faltan:
                partes.append(f"faltan {', '.join(faltan)}")
            reg["estado"] = "sin_mascara" if mascara is None else "faltan_bandas"
            reg["motivo"] = "; ".join(partes) + f". Contiene: {listado or '(vacío)'}"
        else:
            reg["estado"] = "disponible"

        # --- Cabeceras: tamaño, ubicación y legibilidad ---
        necesarios = {b: bandas[b] for b in cfg["bandas"] if b in bandas}
        if mascara:
            necesarios["mascara"] = mascara
        cabeceras = {}
        for clave, nombre in necesarios.items():
            problema = None
            if por_nombre[nombre]["tam"] == 0:
                problema = f"{os.path.basename(nombre)} está vacío (0 bytes)"
            else:
                try:
                    cabeceras[clave] = cabecera_miembro(ruta, nombre, por_nombre[nombre]["ref"])
                except Exception as e:
                    problema = f"no se pudo leer {os.path.basename(nombre)}: {type(e).__name__}: {e}"
            if problema:
                if reg["estado"] == "disponible":
                    reg["estado"], reg["motivo"] = "ilegible", problema
                else:
                    avisos.append(problema)

        ref = cabeceras.get("B04") or next((c for k, c in cabeceras.items() if k != "mascara"), None) \
            or cabeceras.get("mascara")
        if ref is not None:
            reg["alto"], reg["ancho"], reg["crs"] = ref["alto"], ref["ancho"], ref["crs"]
            caja = wc.bounds_wgs84(ref["meta"], ref["alto"], ref["ancho"])
            if caja is not None:
                reg["bounds_wgs84"] = ";".join(f"{v:.6f}" for v in caja)
                reg["centro_lon"] = round((caja[0] + caja[2]) / 2, 5)
                reg["centro_lat"] = round((caja[1] + caja[3]) / 2, 5)
            formas = {k: (c["alto"], c["ancho"]) for k, c in cabeceras.items()}
            distintas = sorted({k for k, f in formas.items() if f != (ref["alto"], ref["ancho"])})
            if distintas:
                avisos.append(f"tamaño distinto en {distintas} (el generador las remuestrea)")
            if min(ref["alto"], ref["ancho"]) < cfg["tile"]:
                avisos.append(f"escena de {ref['ancho']}x{ref['alto']} px, menor que el recorte (se rellena)")
            if "mascara" in cabeceras and cabeceras["mascara"]["canales"] < 3:
                avisos.append("la capa de quemado tiene menos de 3 canales")

        # --- Revisión profunda: lo mismo que hará el generador, sin escribir nada ---
        if cfg["profundo"] and reg["estado"] == "disponible":
            revisar_contenido(ruta, necesarios, por_nombre, cfg, reg)

        reg["avisos"] = " | ".join(avisos + ([reg["avisos"]] if reg["avisos"] else []))
        return reg
    except Exception as e:
        reg["estado"], reg["motivo"] = "error", f"{type(e).__name__}: {e}"
        reg["avisos"] = " | ".join(avisos)
        return reg


def revisar_contenido(ruta, necesarios, por_nombre, cfg, reg):
    try:
        datos = leer_miembros(ruta, {n: por_nombre[n]["ref"] for n in necesarios.values()})
    except Exception as e:
        reg["estado"], reg["motivo"] = "ilegible", f"Error al descomprimir (CRC/datos): {type(e).__name__}: {e}"
        return
    with tempfile.TemporaryDirectory(prefix="wf_rev_") as tmp:
        rutas = {}
        for clave, nombre in necesarios.items():
            destino = os.path.join(tmp, f"{clave}_" + wc.nombre_seguro(os.path.basename(nombre)) +
                                   os.path.splitext(nombre)[1].lower())
            with open(destino, "wb") as f:
                f.write(datos[nombre])
            rutas[clave] = destino
        del datos
        try:
            dn, meta, nodata, info_b = wc.leer_stack_bandas({b: rutas[b] for b in cfg["bandas"]},
                                                            escala=cfg["escala"], orden=cfg["bandas"])
            mascara, info_m = wc.extraer_mascara_quemado(rutas["mascara"], nodata.shape, meta,
                                                         umbral_rojo=cfg["umbral_rojo"], umbral_gb=cfg["umbral_gb"])
        except Exception as e:
            reg["estado"], reg["motivo"] = "ilegible", f"{type(e).__name__}: {e}"
            return
    del dn
    mascara[nodata] = wc.IGNORE_INDEX
    tam = cfg["tile"]
    alto, ancho = mascara.shape
    real = np.ones((alto, ancho), dtype=bool)
    pad_h, pad_w = max(0, tam - alto), max(0, tam - ancho)
    if pad_h or pad_w:
        mascara = np.pad(mascara, ((0, pad_h), (0, pad_w)), constant_values=wc.IGNORE_INDEX)
        real = np.pad(real, ((0, pad_h), (0, pad_w)), constant_values=False)
    pos, neg, bajos, n_nodata = wc.clasificar_ventanas(mascara, real, tam, cfg["overlap"],
                                                       cfg["min_fire"], cfg["max_nodata"])
    reg.update({
        "modo_escala": info_b["modo_escala"], "pixeles_quemados": info_m["pixeles_quemados"],
        "fraccion_quemada": round(info_m["fraccion_quemada"], 6),
        "ventanas_positivas": len(pos), "ventanas_negativas": len(neg),
        "ventanas_bajas": len(bajos), "ventanas_nodata": n_nodata,
    })
    if info_m["pixeles_quemados"] == 0:
        reg["estado"], reg["motivo"] = "solo_negativos", "la capa no tiene píxeles quemados"
    elif not pos:
        reg["estado"] = "sin_positivos"
        reg["motivo"] = (f"{info_m['pixeles_quemados']} px quemados pero ninguna ventana llega a "
                         f"{100 * cfg['min_fire']:.1f}% (prueba INCLUIR_BAJOS o bajar MIN_FIRE)")
    if (~nodata).sum() == 0:
        reg["estado"], reg["motivo"] = "ilegible", "las bandas no tienen datos válidos (todo en 0)"


# ---------------------------------------------------------------------------
# Sitios y duplicados
# ---------------------------------------------------------------------------

def _caja(reg):
    return tuple(float(v) for v in reg["bounds_wgs84"].split(";")) if reg["bounds_wgs84"] else None


def _iou_cajas(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def asignar_sitios_y_duplicados(regs):
    cajas = {r["zip"] + "|" + r["ruta"]: _caja(r) for r in regs}
    raiz = wc.agrupar_por_footprint(cajas)
    grupos = {}
    for r in regs:
        grupos.setdefault(raiz[r["zip"] + "|" + r["ruta"]], []).append(r)
    orden = sorted(grupos.values(), key=lambda g: min(x["zip"] for x in g))
    for i, g in enumerate(orden, 1):
        for r in g:
            r["sitio"] = f"S{i:03d}"
            r["sitio_con_ubicacion"] = bool(r["bounds_wgs84"])

    # Solo importan duplicados entre zips que el generador sí usaría
    utilizables = ESTADOS_OK + ("solo_negativos", "sin_positivos")
    for r in regs:
        r["duplicado_de"] = ""

    # Duplicados exactos: misma banda B04 (CRC y tamaño) en dos zips
    vistos = {}
    for r in sorted(regs, key=lambda x: x["zip"]):
        h = r.get("_huella_b04")
        if h is None or r["estado"] not in utilizables:
            continue
        if h in vistos:
            r["duplicado_de"] = vistos[h]
        else:
            vistos[h] = r["zip"]
    # Posibles duplicados: mismo sitio, misma fecha y casi la misma huella
    for g in orden:
        g = sorted((r for r in g if r["estado"] in utilizables), key=lambda x: x["zip"])
        for i, a in enumerate(g):
            for b in g[i + 1:]:
                if b["duplicado_de"] or not (a["fecha"] and a["fecha"] == b["fecha"]):
                    continue
                ca, cb = _caja(a), _caja(b)
                if ca and cb and _iou_cajas(ca, cb) > 0.9:
                    b["duplicado_de"] = f"{a['zip']} (misma fecha y huella)"
    return orden


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def escribir_csv(ruta, filas, campos):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)


def main():
    t0 = time.time()
    if not os.path.isdir(INPUT_DIR):
        sys.exit(f"ERROR: no existe INPUT_DIR = {INPUT_DIR}")

    if RECURSIVO:
        zips = [os.path.join(c, a) for c, _, archivos in os.walk(INPUT_DIR)
                for a in archivos if a.lower().endswith(".zip")]
    else:
        zips = [os.path.join(INPUT_DIR, a) for a in os.listdir(INPUT_DIR) if a.lower().endswith(".zip")]
    zips.sort()
    if not zips:
        sys.exit(f"No hay archivos .zip en {INPUT_DIR}")

    cfg = {"palabras_mascara": PALABRAS_MASCARA, "profundo": PROFUNDO, "tile": TILE, "bandas": BANDAS,
           "overlap": OVERLAP, "min_fire": MIN_FIRE, "max_nodata": MAX_NODATA,
           "umbral_rojo": UMBRAL_ROJO, "umbral_gb": UMBRAL_GB, "escala": ESCALA}

    modo = "profundo" if PROFUNDO else "rápido"
    print(f"Revisando {len(zips)} zips en {INPUT_DIR} (modo {modo}, {WORKERS} worker(s))")
    regs = []

    def mostrar(i, r):
        marca = "✓" if r["estado"] in ESTADOS_OK else ("·" if r["estado"] in ("solo_negativos", "sin_positivos") else "✗")
        extra = f"{r['ventanas_positivas']} ventanas positivas" if PROFUNDO and r["estado"] == "disponible" \
            else r["motivo"][:110]
        print(f"  [{i}/{len(zips)}] {marca} {r['zip']}: {r['estado']}" + (f" - {extra}" if extra else ""), flush=True)

    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futuros = [ex.submit(revisar_zip, z, cfg) for z in zips]
            for i, fut in enumerate(as_completed(futuros), 1):
                regs.append(fut.result())
                mostrar(i, regs[-1])
    else:
        for i, z in enumerate(zips, 1):
            regs.append(revisar_zip(z, cfg))
            mostrar(i, regs[-1])
    regs.sort(key=lambda r: r["zip"])

    sitios = asignar_sitios_y_duplicados(regs)

    # --- Tabla por sitio ---
    filas_sitio = []
    for g in sitios:
        cajas = [_caja(r) for r in g if _caja(r)]
        fila = {
            "sitio": g[0]["sitio"], "zips": len(g),
            "disponibles": sum(r["estado"] in ESTADOS_OK and not r["duplicado_de"] for r in g),
            "solo_negativos": sum(r["estado"] == "solo_negativos" for r in g),
            "con_problemas": sum(r["estado"] not in ESTADOS_OK + ("solo_negativos", "sin_positivos") for r in g),
            "duplicados": sum(bool(r["duplicado_de"]) for r in g),
            "fechas": ";".join(sorted({r["fecha"] for r in g if r["fecha"]})),
            "con_ubicacion": bool(cajas),
            "centro_lat": "", "centro_lon": "", "bounds_wgs84": "",
            "ventanas_positivas": sum(int(r["ventanas_positivas"] or 0) for r in g
                                      if r["estado"] in ESTADOS_OK and not r["duplicado_de"]) if PROFUNDO else "",
            "lista_zips": ";".join(r["zip"] for r in g),
        }
        if cajas:
            w, s = min(c[0] for c in cajas), min(c[1] for c in cajas)
            e, n = max(c[2] for c in cajas), max(c[3] for c in cajas)
            fila.update({"bounds_wgs84": f"{w:.6f};{s:.6f};{e:.6f};{n:.6f}",
                         "centro_lat": round((s + n) / 2, 5), "centro_lon": round((w + e) / 2, 5)})
        filas_sitio.append(fila)

    # --- Escribir archivos ---
    os.makedirs(SALIDA, exist_ok=True)
    escribir_csv(os.path.join(SALIDA, "reporte_zips.csv"), regs,
                 ["zip", "estado", "motivo", "sitio", "duplicado_de", "fecha", "alto", "ancho", "tamano_mb",
                  "bandas_ok", "faltan", "archivo_mascara", "otras_bandas", "crs", "centro_lat", "centro_lon",
                  "bounds_wgs84", "modo_escala", "pixeles_quemados", "fraccion_quemada", "ventanas_positivas",
                  "ventanas_negativas", "ventanas_bajas", "ventanas_nodata", "n_archivos", "avisos", "ruta"])
    escribir_csv(os.path.join(SALIDA, "reporte_sitios.csv"), filas_sitio,
                 ["sitio", "zips", "disponibles", "solo_negativos", "con_problemas", "duplicados", "fechas",
                  "ventanas_positivas", "con_ubicacion", "centro_lat", "centro_lon", "bounds_wgs84", "lista_zips"])
    disponibles = [r for r in regs if r["estado"] in ESTADOS_OK and not r["duplicado_de"]]
    # Para el generador también sirven las escenas sin incendio (aportan negativos)
    utilizables = [r for r in regs if r["estado"] in ESTADOS_OK + ("solo_negativos", "sin_positivos")
                   and not r["duplicado_de"]]
    with open(os.path.join(SALIDA, "zips_disponibles.txt"), "w") as f:
        f.writelines(r["ruta"] + "\n" for r in utilizables)
    with open(os.path.join(SALIDA, "zips_con_problemas.txt"), "w") as f:
        f.writelines(f"{r['ruta']}\t{r['estado']}\t{r['motivo']}\n" for r in regs if r["estado"] not in ESTADOS_OK)

    # --- Resumen ---
    L = []
    L.append("=" * 72)
    L.append(f"REPORTE DE ZIPS ({modo}) - {time.strftime('%Y-%m-%d %H:%M')}")
    L.append(f"Carpeta: {os.path.abspath(INPUT_DIR)}")
    L.append("=" * 72)
    L.append(f"Zips encontrados: {len(regs)}")
    conteo = {}
    for r in regs:
        conteo[r["estado"]] = conteo.get(r["estado"], 0) + 1
    for estado in DESCRIPCION_ESTADO:
        if conteo.get(estado):
            marca = "✓" if estado in ESTADOS_OK else ("·" if estado in ("solo_negativos", "sin_positivos") else "✗")
            desc = DESCRIPCION_ESTADO[estado]
            if estado == "disponible":
                desc = (f"con las {len(BANDAS)} bandas, capa de quemado y ventanas positivas" if PROFUNDO
                        else f"con las {len(BANDAS)} bandas y la capa de quemado legibles")
            L.append(f"  {marca} {conteo[estado]:5d}  {desc}")
    dups = [r for r in regs if r["duplicado_de"]]
    if dups:
        L.append(f"  ⚠ {len(dups):5d}  duplicados de otro zip (no se cuentan como disponibles)")
    if PROFUNDO:
        L.append(f"  => {len(disponibles)} zips con incendio utilizables"
                 + (f" (+{len(utilizables) - len(disponibles)} que solo aportan negativos)"
                    if len(utilizables) > len(disponibles) else ""))
    else:
        L.append(f"  => {len(disponibles)} zips utilizables (estructura completa; usa --profundo para "
                 "verificar que tengan píxeles quemados)")

    faltantes = {}
    for r in regs:
        for b in filter(None, r["faltan"].split(";")):
            faltantes[b] = faltantes.get(b, 0) + 1
    if faltantes:
        L.append("  Bandas faltantes: " + ", ".join(f"{b} en {n} zips" for b, n in sorted(faltantes.items())))

    con_ubic = [f for f in filas_sitio if f["con_ubicacion"]]
    sin_ubic = [f for f in filas_sitio if not f["con_ubicacion"]]
    sitios_disp = [f for f in filas_sitio if f["disponibles"] > 0]
    L.append("")
    L.append("Sitios (zips agrupados por ubicación; varias fechas del mismo lugar = 1 sitio):")
    L.append(f"  totales:      {len(filas_sitio)}")
    L.append(f"  disponibles:  {len(sitios_disp)}  (al menos un zip utilizable)")
    if PROFUNDO:
        solo_neg = [f for f in filas_sitio if f["disponibles"] == 0 and f["solo_negativos"] > 0]
        L.append(f"  solo negativos: {len(solo_neg)}")
    if sin_ubic:
        L.append(f"  ⚠ {len(sin_ubic)} sin ubicación (sin georreferencia legible): cada zip cuenta como un sitio")
    multi = [f for f in con_ubic if f["zips"] > 1]
    if multi:
        L.append(f"  {len(multi)} sitios tienen más de un zip (máx. {max(f['zips'] for f in multi)})")
    if not wc.HAS_RASTERIO:
        L.append("  ⚠ rasterio no está instalado: no se puede leer la ubicación, los sitios = zips")

    if PROFUNDO and disponibles:
        tot_pos = sum(int(r["ventanas_positivas"] or 0) for r in disponibles)
        tot_neg = sum(int(r["ventanas_negativas"] or 0) for r in disponibles)
        L.append("")
        L.append(f"Ventanas estimadas ({TILE}px, traslape {OVERLAP}px, mín. {100 * MIN_FIRE:.1f}% quemado):")
        L.append(f"  positivas: {tot_pos:,}   negativas candidatas: {tot_neg:,}")
        modos = {}
        for r in disponibles:
            modos[r["modo_escala"]] = modos.get(r["modo_escala"], 0) + 1
        L.append(f"  formato de bandas: {', '.join(f'{k}: {v}' for k, v in sorted(modos.items()))}")

    extra = {}
    for r in regs:
        for b in filter(None, r["otras_bandas"].split(";")):
            extra[b] = extra.get(b, 0) + 1
    if extra:
        L.append("")
        L.append("Otras bandas presentes: " + ", ".join(f"{b} en {n}" for b, n in sorted(extra.items())))

    malos = [r for r in regs if r["estado"] not in ESTADOS_OK]
    if malos:
        L.append("")
        L.append("Zips con problemas (todos en zips_con_problemas.txt):")
        for r in malos[:25]:
            L.append(f"  - {r['zip']} [{r['estado']}]: {r['motivo'][:150]}")
        if len(malos) > 25:
            L.append(f"  ... y {len(malos) - 25} más")
    if dups:
        L.append("")
        L.append("Duplicados:")
        for r in dups[:15]:
            L.append(f"  - {r['zip']} = {r['duplicado_de']}")
    L.append("")
    L.append(f"Reporte en {os.path.abspath(SALIDA)}  ({time.time() - t0:.1f}s)")
    L.append("=" * 72)

    texto = "\n".join(L)
    print("\n" + texto)
    with open(os.path.join(SALIDA, "resumen.txt"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")


if __name__ == "__main__":
    main()
