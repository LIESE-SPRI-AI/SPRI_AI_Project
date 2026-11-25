"""
wildfire_common.py
==================
Funciones compartidas por generate_wildfire_dataset.py, trainModelWildfire3D4B.py
y predict_segmentation.py. Tenerlas en un solo lugar garantiza que el
preprocesamiento sea IDÉNTICO al generar el dataset, al entrenar y al predecir.

Convenciones del proyecto
-------------------------
* Orden de bandas de entrada: B04, B03, B02, B08  ->  R, G, B, NIR.
* Imágenes guardadas en DN de Sentinel-2 L2A: reflectancia * 10000 (uint16).
  Ya NO se estira cada banda con su min/max (eso rompía la relación entre
  bandas y hacía que el mismo valor físico valiera distinto en cada escena).
* Máscaras de 1 banda uint8:  0 = sin quemado, 1 = quemado, 255 = sin dato
  (se ignora en la pérdida y en las métricas).
* Las máscaras "antiguas" de 3 bandas (rojo = incendio) se siguen leyendo.

No depende de torch: se puede usar y probar sin GPU.
"""

import math
import os
import re
import shutil
import warnings
import zipfile

import numpy as np

try:
    import rasterio
    from rasterio.enums import ColorInterp, Resampling
    from rasterio.errors import NotGeoreferencedWarning
    from rasterio.warp import reproject, transform_bounds
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform
    HAS_RASTERIO = True
except ImportError:  # modo degradado: sin georreferencia (útil para pruebas)
    HAS_RASTERIO = False

BAND_ORDER = ("B04", "B03", "B02", "B08")          # R, G, B, NIR
BAND_EXTS = (".tif", ".tiff", ".jp2")
MASK_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2")
IGNORE_INDEX = 255
DEFAULT_MASK_KEYWORDS = ("burnedarea", "burntarea", "burnedareadetection", "burned", "burnt")

# Factores para convertir lo que venga en el archivo a DN (reflectancia*10000)
ESCALAS = {
    "dn": 1.0,                      # ya viene en DN (EO Browser TIFF 16 bits "Raw")
    "reflectancia": 10000.0,        # float 0..1 (EO Browser TIFF 32 bits float)
    "8bit": 10000.0 / 255.0,        # uint8 0..255 (pierde rango dinámico, no recomendado)
    "16bit_completo": 10000.0 / 65535.0,  # 0..1 mapeado a 0..65535
}


# ---------------------------------------------------------------------------
# Utilidades de nombres y archivos
# ---------------------------------------------------------------------------

def normalizar_nombre(texto):
    """Minúsculas y sin separadores: 'Burned_Area-Detection' -> 'burnedareadetection'."""
    return re.sub(r"[^a-z0-9]", "", texto.lower())


def nombre_seguro(texto):
    """Convierte un nombre de zip en un identificador seguro para archivos."""
    limpio = re.sub(r"[^A-Za-z0-9_-]+", "_", texto).strip("_")
    return limpio or "escena"


def _regex_banda(banda):
    # 'B02' no debe coincidir con 'B021' ni con 'B8A'; tolera 'B2' en algunos productos
    num = banda[1:]
    variantes = {num, num.lstrip("0") or "0"}
    alternativas = "|".join(sorted(variantes, key=len, reverse=True))
    return re.compile(r"(?<![a-z0-9])b(?:" + alternativas + r")(?![0-9a-z])")


def listar_archivos(raiz):
    """Lista recursiva (los zips de EO Browser a veces traen subcarpetas)."""
    salida = []
    for carpeta, _, archivos in os.walk(raiz):
        for a in archivos:
            if a.startswith(".") or a.endswith(".aux.xml"):
                continue
            salida.append(os.path.join(carpeta, a))
    return sorted(salida)


def _es_basura(ruta):
    base = os.path.basename(ruta)
    return (not base or base.startswith(".") or base.endswith(".aux.xml")
            or "__MACOSX" in ruta.replace("\\", "/").split("/"))


def clasificar_rutas(archivos, palabras_mascara=DEFAULT_MASK_KEYWORDS, bandas=BAND_ORDER):
    """
    Decide, solo por nombre, cuál archivo es cada banda y cuál es la capa de área quemada.
    Sirve igual para rutas en disco que para nombres dentro de un zip.

    Devuelve dict con:
      'bandas': {'B04': ruta, ...}  (solo las encontradas)
      'mascara': ruta o None
      'avisos': lista de textos
    """
    archivos = [a for a in archivos if not _es_basura(a)]
    avisos = []
    claves = [normalizar_nombre(k) for k in palabras_mascara]

    candidatos_mascara = []
    for ruta in archivos:
        base = os.path.basename(ruta)
        ext = os.path.splitext(base)[1].lower()
        if ext in MASK_EXTS and any(k in normalizar_nombre(base) for k in claves):
            candidatos_mascara.append(ruta)
    # Preferir TIFF sobre JPG/PNG (sin compresión con pérdida)
    candidatos_mascara.sort(key=lambda r: (os.path.splitext(r)[1].lower() not in (".tif", ".tiff"), r))
    mascara = candidatos_mascara[0] if candidatos_mascara else None
    if len(candidatos_mascara) > 1:
        avisos.append(f"{len(candidatos_mascara)} archivos de máscara; se usa {os.path.basename(mascara)}")

    encontradas = {}
    for banda in bandas:
        rx = _regex_banda(banda)
        cands = []
        for ruta in archivos:
            if ruta in candidatos_mascara:
                continue
            base = os.path.basename(ruta)
            ext = os.path.splitext(base)[1].lower()
            if ext in BAND_EXTS and rx.search(base.lower()):
                cands.append(ruta)
        if not cands:
            continue
        # Preferir las versiones "Raw" (valores físicos) sobre visualizaciones
        cands.sort(key=lambda r: ("raw" not in normalizar_nombre(os.path.basename(r)), r))
        encontradas[banda] = cands[0]
        if len(cands) > 1:
            avisos.append(f"{banda}: {len(cands)} candidatos, se usa {os.path.basename(cands[0])}")

    return {"bandas": encontradas, "mascara": mascara, "avisos": avisos}


def buscar_archivos_escena(raiz, palabras_mascara=DEFAULT_MASK_KEYWORDS, bandas=BAND_ORDER):
    """Como clasificar_rutas, pero recorriendo una carpeta (incluye 'archivos' para diagnosticar)."""
    archivos = listar_archivos(raiz)
    resultado = clasificar_rutas(archivos, palabras_mascara, bandas)
    resultado["archivos"] = [os.path.relpath(a, raiz) for a in archivos]
    return resultado


_RX_BANDAS_S2 = re.compile(r"(?<![a-z0-9])(b0?[1-9]|b1[0-2]|b8a|scl|clm|cld|snw)(?![0-9a-z])")


def bandas_presentes(archivos):
    """Todas las bandas Sentinel-2 reconocibles por nombre (B01..B12, B8A, SCL, CLM...)."""
    encontradas = set()
    for a in archivos:
        if _es_basura(a):
            continue
        for m in _RX_BANDAS_S2.findall(os.path.basename(a).lower()):
            m = m.upper()
            if m[0] == "B" and len(m) == 2:
                m = "B0" + m[1]
            encontradas.add(m)
    return encontradas


def fecha_de_nombre(nombre):
    """Fecha de adquisición a partir del nombre de EO Browser (AAAA-MM-DD) o SAFE (AAAAMMDDT)."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", nombre)
    if m:
        return m.group(1)
    m = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})T\d", nombre)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""


def extraer_zip_seguro(ruta_zip, destino, anidados=True):
    """Extrae un zip evitando rutas fuera de 'destino' (zip slip). Extrae zips anidados (1 nivel)."""
    destino_abs = os.path.abspath(destino)
    with zipfile.ZipFile(ruta_zip, "r") as zf:
        for miembro in zf.infolist():
            nombre = miembro.filename
            # Los ':' de los nombres de EO Browser dan problemas en algunos sistemas
            nombre_local = nombre.replace(":", "_")
            ruta_final = os.path.abspath(os.path.join(destino_abs, nombre_local))
            if not ruta_final.startswith(destino_abs + os.sep) and ruta_final != destino_abs:
                raise ValueError(f"Ruta sospechosa dentro del zip: {nombre}")
            if miembro.is_dir():
                os.makedirs(ruta_final, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(ruta_final), exist_ok=True)
            with zf.open(miembro) as src, open(ruta_final, "wb") as dst:
                shutil.copyfileobj(src, dst)
    if anidados:
        for ruta in listar_archivos(destino_abs):
            if ruta.lower().endswith(".zip"):
                sub = os.path.splitext(ruta)[0]
                os.makedirs(sub, exist_ok=True)
                extraer_zip_seguro(ruta, sub, anidados=False)
                os.remove(ruta)


# ---------------------------------------------------------------------------
# Lectura / escritura raster
# ---------------------------------------------------------------------------

def _meta_vacia(dtype):
    return {"transform": None, "crs": None, "nodata": None, "dtype": str(dtype),
            "georef": False, "alpha": None}


def leer_raster(ruta):
    """
    Lee un raster como (C, H, W) en su tipo original.
    meta: transform, crs, nodata, dtype, georef, alpha (índice de la banda alfa o None)
    """
    ext = os.path.splitext(ruta)[1].lower()
    if HAS_RASTERIO:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(ruta) as src:
                arr = src.read()
                alpha = None
                try:
                    ci = list(src.colorinterp)
                    if ColorInterp.alpha in ci:
                        alpha = ci.index(ColorInterp.alpha)
                except Exception:
                    pass
                georef = src.crs is not None
                meta = {
                    "transform": src.transform if georef else None,
                    "crs": src.crs if georef else None,
                    "nodata": src.nodata,
                    "dtype": str(arr.dtype),
                    "georef": georef,
                    "alpha": alpha,
                }
        return arr, meta

    # --- Respaldo sin rasterio (sin georreferencia) ---
    if ext in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(ruta) as tf:
            serie = tf.series[0]
            arr = serie.asarray()
            ejes = serie.axes
        if arr.ndim == 2:
            arr = arr[None]
        elif ejes.endswith("S"):
            arr = np.moveaxis(arr, -1, 0)
        return np.ascontiguousarray(arr), _meta_vacia(arr.dtype)
    from PIL import Image
    with Image.open(ruta) as im:
        modo = im.mode
        arr = np.array(im)
    arr = arr[None] if arr.ndim == 2 else np.moveaxis(arr, -1, 0)
    meta = _meta_vacia(arr.dtype)
    if modo in ("RGBA", "LA"):
        meta["alpha"] = arr.shape[0] - 1
    return np.ascontiguousarray(arr), meta


def escribir_raster(ruta, arr, meta=None, nodata=None):
    """Escribe (C, H, W) o (H, W) como GeoTIFF comprimido (georreferenciado si meta lo permite)."""
    if arr.ndim == 2:
        arr = arr[None]
    meta = meta or {}
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    if HAS_RASTERIO:
        perfil = {
            "driver": "GTiff", "height": arr.shape[1], "width": arr.shape[2],
            "count": arr.shape[0], "dtype": arr.dtype.name, "compress": "deflate",
        }
        if arr.dtype.kind in "iu":
            perfil["predictor"] = 2
        if meta.get("georef") and meta.get("transform") is not None:
            perfil["crs"] = meta["crs"]
            perfil["transform"] = meta["transform"]
        if nodata is not None:
            perfil["nodata"] = nodata
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(ruta, "w", **perfil) as dst:
                dst.write(arr)
        return
    import tifffile
    tifffile.imwrite(ruta, arr, photometric="minisblack", planarconfig="separate",
                     compression="zlib")


def transform_de_ventana(meta, x, y, ancho, alto):
    """Meta georreferenciada para un recorte que empieza en columna x, fila y."""
    nueva = dict(meta)
    if HAS_RASTERIO and meta.get("georef") and meta.get("transform") is not None:
        nueva["transform"] = window_transform(Window(x, y, ancho, alto), meta["transform"])
    return nueva


def bounds_wgs84(meta, alto, ancho):
    """(oeste, sur, este, norte) en EPSG:4326, o None si no hay georreferencia."""
    if not (HAS_RASTERIO and meta.get("georef") and meta.get("transform") is not None):
        return None
    t = meta["transform"]
    xs = [t.c, t.c + t.a * ancho]
    ys = [t.f, t.f + t.e * alto]
    try:
        return tuple(transform_bounds(meta["crs"], "EPSG:4326", min(xs), min(ys), max(xs), max(ys)))
    except Exception:
        return None


def _mismo_grid(meta_a, forma_a, meta_b, forma_b):
    if tuple(forma_a) != tuple(forma_b):
        return False
    if meta_a.get("georef") and meta_b.get("georef"):
        if meta_a["crs"] != meta_b["crs"]:
            return False
        return meta_a["transform"].almost_equals(meta_b["transform"], precision=1e-6)
    return True


def remuestrear(arr2d, meta_src, forma_dst, meta_dst, vecino_mas_cercano, nodata=None):
    """Lleva una banda 2D a la rejilla de referencia (reproyecta si hay georreferencia)."""
    import cv2
    alto, ancho = forma_dst
    georef = (HAS_RASTERIO and meta_src.get("georef") and meta_dst.get("georef"))
    if georef:
        relleno = nodata if nodata is not None else 0
        dst = np.full((alto, ancho), relleno, dtype=arr2d.dtype)
        reproject(
            source=arr2d, destination=dst,
            src_transform=meta_src["transform"], src_crs=meta_src["crs"],
            dst_transform=meta_dst["transform"], dst_crs=meta_dst["crs"],
            resampling=Resampling.nearest if vecino_mas_cercano else Resampling.bilinear,
            src_nodata=nodata, dst_nodata=nodata,
        )
        return dst
    interp = cv2.INTER_NEAREST if vecino_mas_cercano else cv2.INTER_LINEAR
    if arr2d.dtype == np.float64:
        arr2d = arr2d.astype(np.float32)
    return cv2.resize(arr2d, (ancho, alto), interpolation=interp)


# ---------------------------------------------------------------------------
# Bandas -> DN
# ---------------------------------------------------------------------------

def detectar_escala(stack, validos, dtype_original):
    """Decide cómo convertir a DN (reflectancia*10000) a partir del tipo y del percentil 99."""
    muestra = stack[:, validos] if validos is not None and validos.any() else stack.reshape(stack.shape[0], -1)
    muestra = muestra.reshape(-1)
    if muestra.size > 2_000_000:
        muestra = muestra[:: muestra.size // 2_000_000 + 1]
    muestra = muestra[np.isfinite(muestra)]
    p99 = float(np.percentile(muestra, 99)) if muestra.size else 0.0
    kind = np.dtype(dtype_original).kind
    if kind == "f":
        return ("reflectancia" if p99 <= 3.0 else "dn"), p99
    if np.dtype(dtype_original) == np.uint8:
        return "8bit", p99
    return ("16bit_completo" if p99 > 20000 else "dn"), p99


def leer_stack_bandas(rutas_bandas, escala="auto", orden=BAND_ORDER):
    """
    Lee las 4 bandas, las alinea a la rejilla de B04 y las convierte a DN (float32).

    Devuelve (dn (4,H,W) float32, meta_referencia, nodata (H,W) bool, info dict)
    """
    faltan = [b for b in orden if b not in rutas_bandas]
    if faltan:
        raise ValueError(f"Faltan bandas: {faltan}")

    ref_arr, ref_meta = leer_raster(rutas_bandas[orden[0]])
    forma = ref_arr.shape[1:]
    dtype_original = ref_arr.dtype
    capas, dtypes, remuestreadas = [], [], []
    for banda in orden:
        if banda == orden[0]:
            arr, meta = ref_arr, ref_meta
        else:
            arr, meta = leer_raster(rutas_bandas[banda])
        dtypes.append(str(arr.dtype))
        capa = arr[0].astype(np.float32)
        if not _mismo_grid(meta, arr.shape[1:], ref_meta, forma):
            capa = remuestrear(capa, meta, forma, ref_meta, vecino_mas_cercano=False,
                               nodata=meta.get("nodata"))
            remuestreadas.append(banda)
        capas.append(capa)
    stack = np.stack(capas, axis=0)

    no_finito = ~np.isfinite(stack).all(axis=0)
    stack[:, no_finito] = 0
    nodata = (stack == 0).all(axis=0) | no_finito
    if ref_meta.get("nodata") is not None:
        nodata |= (stack == ref_meta["nodata"]).all(axis=0)

    if escala == "auto":
        modo, p99 = detectar_escala(stack, ~nodata, dtype_original)
    else:
        if escala not in ESCALAS:
            raise ValueError(f"Escala desconocida: {escala}")
        modo, p99 = escala, None
    dn = stack * ESCALAS[modo]
    dn[:, nodata] = 0

    info = {
        "modo_escala": modo,
        "p99_original": p99,
        "dtypes": dtypes,
        "bandas_remuestreadas": remuestreadas,
        "p99_dn": [float(np.percentile(dn[i][~nodata], 99)) if (~nodata).any() else 0.0
                   for i in range(dn.shape[0])],
    }
    if len(set(dtypes)) > 1:
        info["aviso"] = f"Tipos de dato distintos entre bandas: {dtypes}"
    return dn.astype(np.float32), ref_meta, nodata, info


def leer_tiff_4bandas(ruta, escala="auto"):
    """Lee un TIFF ya apilado (R,G,B,NIR) y lo convierte a DN."""
    arr, meta = leer_raster(ruta)
    if arr.shape[0] < 4:
        raise ValueError(f"{ruta} tiene {arr.shape[0]} bandas, se esperaban 4 (R,G,B,NIR)")
    stack = arr[:4].astype(np.float32)
    no_finito = ~np.isfinite(stack).all(axis=0)
    stack[:, no_finito] = 0
    nodata = (stack == 0).all(axis=0) | no_finito
    if escala == "auto":
        modo, p99 = detectar_escala(stack, ~nodata, arr.dtype)
    else:
        modo, p99 = escala, None
    dn = stack * ESCALAS[modo]
    return dn.astype(np.float32), meta, nodata, {"modo_escala": modo, "p99_original": p99}


def a_uint16(dn):
    return np.clip(np.rint(dn), 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# Máscara de área quemada (capa Burned Area Detection de EO Browser)
# ---------------------------------------------------------------------------

def _a_unidad(banda, dtype):
    """Escala FIJA a [0,1] según el tipo (nunca min/max por escena)."""
    b = banda.astype(np.float32)
    if dtype == np.uint8:
        return b / 255.0
    if dtype == np.uint16:
        return b / 65535.0
    if np.dtype(dtype).kind == "f":
        return np.nan_to_num(b, nan=0.0)
    return b / float(np.iinfo(dtype).max)


def extraer_mascara_quemado(ruta, forma_ref=None, meta_ref=None, umbral_rojo=0.75, umbral_gb=0.2):
    """
    El script de Sentinel Hub pinta lo quemado como rojo puro [1,0,0] y el resto en
    color verdadero con ganancia 2.5. Un píxel natural con R>=0.75 y G,B<=0.2 exigiría
    B04 >= 5 veces B03, lo que prácticamente no ocurre, así que el criterio es seguro y
    tolera compresión JPG/PNG.

    Devuelve (mascara uint8 0/1/255, info dict)
    """
    arr, meta = leer_raster(ruta)
    dtype = arr.dtype
    info = {"canales": int(arr.shape[0]), "dtype": str(dtype), "alpha": False, "remuestreada": False}

    if arr.shape[0] >= 3:
        r = _a_unidad(arr[0], dtype)
        g = _a_unidad(arr[1], dtype)
        b = _a_unidad(arr[2], dtype)
        quemado = (r >= umbral_rojo) & (g <= umbral_gb) & (b <= umbral_gb)
        validos = np.ones(quemado.shape, dtype=bool)
        idx_alpha = meta.get("alpha")
        if idx_alpha is None and arr.shape[0] == 4:
            idx_alpha = 3  # EO Browser exporta dataMask como 4.a banda
        if idx_alpha is not None and idx_alpha < arr.shape[0]:
            validos = _a_unidad(arr[idx_alpha], dtype) > 0.5
            info["alpha"] = True
    else:
        m = arr[0]
        unicos = set(np.unique(m).tolist())
        if 1 in unicos and unicos <= {0, 1, IGNORE_INDEX}:   # formato del proyecto 0/1/255
            quemado, validos = m == 1, m != IGNORE_INDEX
        elif len(unicos) <= 2:                               # binaria 0/X
            quemado, validos = m > 0, np.ones(m.shape, dtype=bool)
        else:
            raise ValueError(f"Máscara de {arr.shape[0]} banda(s) con {len(unicos)} valores distintos; "
                             "se esperaba RGB(A) o binaria")

    mascara = quemado.astype(np.uint8)
    mascara[~validos] = IGNORE_INDEX

    if forma_ref is not None and meta_ref is not None and not _mismo_grid(meta, mascara.shape, meta_ref, forma_ref):
        mascara = remuestrear(mascara, meta, forma_ref, meta_ref, vecino_mas_cercano=True, nodata=IGNORE_INDEX)
        info["remuestreada"] = True

    v = mascara != IGNORE_INDEX
    info["fraccion_quemada"] = float((mascara == 1).sum() / max(v.sum(), 1))
    info["pixeles_quemados"] = int((mascara == 1).sum())
    return mascara, info


def cargar_mascara_entrenamiento(ruta):
    """
    Máscara para entrenar/evaluar -> uint8 (H,W) con 0/1/255.
      * 1 banda: formato nuevo (0, 1, 255=ignorar)
      * 3-4 bandas: formato antiguo (rojo > 127 = incendio)
    """
    arr, _ = leer_raster(ruta)
    if arr.shape[0] == 1:
        m = arr[0]
        if m.dtype != np.uint8:
            m = np.clip(m, 0, 255).astype(np.uint8)
        return m
    rojo = arr[0]
    if rojo.dtype == np.uint16:
        rojo = (rojo // 257).astype(np.uint8)
    elif rojo.dtype.kind == "f":
        rojo = (np.clip(rojo, 0, 1) * 255).astype(np.uint8)
    return (rojo > 127).astype(np.uint8)


# ---------------------------------------------------------------------------
# Normalización de entrada al modelo
# ---------------------------------------------------------------------------

def normalizar(dn, norm):
    """dn (C,H,W) float32 en DN -> entrada del modelo. 'norm' viaja dentro del checkpoint."""
    tipo = norm.get("tipo", "div")
    if tipo == "div":
        return np.clip(dn / float(norm.get("valor", 12500.0)), 0.0, 1.0).astype(np.float32)
    if tipo == "zscore":
        media = np.asarray(norm["media"], dtype=np.float32)[:, None, None]
        desv = np.asarray(norm["desv"], dtype=np.float32)[:, None, None]
        return ((dn - media) / np.maximum(desv, 1e-6)).astype(np.float32)
    raise ValueError(f"Normalización desconocida: {tipo}")


# ---------------------------------------------------------------------------
# Teselado
# ---------------------------------------------------------------------------

def inicios_ventanas(longitud, tam, paso):
    """Posiciones de inicio que cubren [0, longitud) con ventanas de 'tam' y paso 'paso'."""
    if longitud <= tam:
        return [0]
    paso = max(1, int(paso))
    inicios = list(range(0, longitud - tam + 1, paso))
    if inicios[-1] != longitud - tam:
        inicios.append(longitud - tam)
    return inicios


def clasificar_ventanas(mascara, real, tam, traslape, min_fire, max_nodata):
    """
    Recorre la máscara (0/1/255) con ventanas de 'tam' y paso tam-traslape.
    'real' marca los píxeles que existen (False en el relleno de escenas chicas).

    Devuelve (positivos, negativos, bajos, n_descartadas_nodata); cada ventana es
    (y, x, quemados, n_validos, fraccion_quemada, fraccion_nodata, tipo).
    """
    hp, wp = mascara.shape
    paso = max(1, tam - traslape)
    positivos, negativos, bajos = [], [], []
    n_nodata = 0
    for y in inicios_ventanas(hp, tam, paso):
        for x in inicios_ventanas(wp, tam, paso):
            m = mascara[y:y + tam, x:x + tam]
            n_real = int(real[y:y + tam, x:x + tam].sum())
            if n_real == 0:
                continue
            n_validos = int((m != IGNORE_INDEX).sum())
            frac_nodata = 1.0 - n_validos / n_real
            if frac_nodata > max_nodata or n_validos == 0:
                n_nodata += 1
                continue
            quemados = int((m == 1).sum())
            frac = quemados / n_validos
            if frac >= min_fire:
                positivos.append((y, x, quemados, n_validos, frac, frac_nodata, "positivo"))
            elif quemados == 0:
                negativos.append((y, x, quemados, n_validos, frac, frac_nodata, "negativo"))
            else:
                bajos.append((y, x, quemados, n_validos, frac, frac_nodata, "bajo"))
    return positivos, negativos, bajos, n_nodata


def agrupar_por_footprint(cajas):
    """
    Union-find: elementos cuyas cajas (oeste, sur, este, norte) se intersectan forman un sitio.
    cajas: {id: caja o None}. Los que no tienen caja quedan solos. Devuelve {id: id_raiz}.
    """
    padre = {i: i for i in cajas}

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    lista = [(i, c) for i, c in cajas.items() if c is not None]
    for a in range(len(lista)):
        ia, (w1, s1, e1, n1) = lista[a]
        for b in range(a + 1, len(lista)):
            ib, (w2, s2, e2, n2) = lista[b]
            if w1 < e2 and w2 < e1 and s1 < n2 and s2 < n1:
                ra, rb = raiz(ia), raiz(ib)
                if ra != rb:
                    padre[ra] = rb
    return {i: raiz(i) for i in cajas}


def ventana_ponderacion(tam, sigma_rel=0.25, minimo=1e-3):
    """Ventana gaussiana 2D: el centro del parche pesa más que los bordes."""
    c = (tam - 1) / 2.0
    eje = np.arange(tam, dtype=np.float32) - c
    g = np.exp(-(eje ** 2) / (2 * (sigma_rel * tam) ** 2))
    w = np.outer(g, g)
    w /= w.max()
    return np.maximum(w, minimo).astype(np.float32)


def predecir_ventanas(imagen, fn_inferencia, tam=128, traslape=64, lote=16,
                      ponderar=True, progreso=None):
    """
    Predicción por ventanas deslizantes sobre una imagen completa.

    imagen: (C,H,W) float32 ya normalizada
    fn_inferencia: recibe (N,C,tam,tam) float32 y devuelve (N,tam,tam) prob. de incendio
    Devuelve prob (H,W) float32.
    """
    c, alto, ancho = imagen.shape
    pad_h, pad_w = max(0, tam - alto), max(0, tam - ancho)
    if pad_h or pad_w:
        modo = "reflect" if (pad_h < alto and pad_w < ancho) else "edge"
        imagen = np.pad(imagen, ((0, 0), (0, pad_h), (0, pad_w)), mode=modo)
    _, hp, wp = imagen.shape

    paso = max(1, tam - traslape)
    posiciones = [(y, x) for y in inicios_ventanas(hp, tam, paso) for x in inicios_ventanas(wp, tam, paso)]
    peso = ventana_ponderacion(tam) if ponderar else np.ones((tam, tam), np.float32)
    acum = np.zeros((hp, wp), np.float32)
    suma_pesos = np.zeros((hp, wp), np.float32)

    total = len(posiciones)
    for i0 in range(0, total, lote):
        bloque = posiciones[i0:i0 + lote]
        batch = np.stack([imagen[:, y:y + tam, x:x + tam] for y, x in bloque]).astype(np.float32)
        probs = fn_inferencia(batch)
        for (y, x), p in zip(bloque, probs):
            acum[y:y + tam, x:x + tam] += p * peso
            suma_pesos[y:y + tam, x:x + tam] += peso
        if progreso is not None:
            progreso(min(i0 + lote, total), total)

    prob = acum / np.maximum(suma_pesos, 1e-8)
    return prob[:alto, :ancho]


# ---------------------------------------------------------------------------
# Métricas (acumuladas sobre todo el conjunto, no promedio por imagen)
# ---------------------------------------------------------------------------

def conteos_confusion(pred, objetivo):
    """pred bool (H,W); objetivo uint8 0/1/255. Devuelve (tp, fp, fn, tn)."""
    validos = objetivo != IGNORE_INDEX
    t = objetivo == 1
    p = pred.astype(bool)
    tp = int((p & t & validos).sum())
    fp = int((p & ~t & validos).sum())
    fn = int((~p & t & validos).sum())
    tn = int((~p & ~t & validos).sum())
    return tp, fp, fn, tn


def metricas(tp, fp, fn, tn=0):
    def div(a, b):
        return a / b if b > 0 else float("nan")
    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    return {
        "iou": div(tp, tp + fp + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "precision": precision,
        "recall": recall,
        "exactitud": div(tp + tn, tp + fp + fn + tn),
    }


def fmt(v, dec=4):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{dec}f}"
