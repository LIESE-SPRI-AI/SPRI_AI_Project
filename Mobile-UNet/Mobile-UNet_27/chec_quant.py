#!/usr/bin/env python3
"""
chec_quant.py
=============
Verifica la cuantización del modelo:
  1. Que pesos, bias y activaciones caigan EXACTAMENTE en la rejilla 2^-frac_bits
     (no solo que existan los módulos FakeQuantAct / QATConv2d).
  2. Saturación: qué fracción de valores se recorta en cada punto de cuantización y cuál es
     el máximo |x| antes de cuantizar. Si un punto satura mucho, hay que bajar frac_bits
     (más bits enteros); si el máximo es muy chico, se pueden subir.

Usa datos reales del dataset (si existe) preparados con construir_entrada(), con las bandas,
índices y normalización guardados en el checkpoint.
"""

import glob
import json
import math
import os
import random

import numpy as np
import torch

import wildfire_common as wc
import Wildfire_models as wm

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
CHECKPOINT = "auto"          # "auto" = último runs/*/best.pth | ruta a .pth | None = pesos aleatorios
DIR_RUNS = os.path.join(wc.DIR_CODIGO, "runs")
USAR_DATOS_REALES = True
DATA_DIR = wc.ruta_proyecto("Dataset", "Dataset_v3")
LISTA = "valid.txt"
N_MUESTRAS = 16
IMAGE_SIZE = None            # None = el del checkpoint (o 128)
EPS = 1e-5                   # tolerancia por error de punto flotante (float32)
SATURACION_AVISO = 1e-3      # avisar si se recorta más del 0.1 % de los valores
SEED = 0
# =============================================================================


def cargar_modelo():
    ruta = CHECKPOINT
    if ruta == "auto":
        cands = glob.glob(os.path.join(DIR_RUNS, "*", "best.pth"))
        ruta = max(cands, key=os.path.getmtime) if cands else None
    if ruta is None:
        print("Sin checkpoint: pesos recién inicializados (solo para probar el script)")
        return wm.MbUNet2D(), {}
    try:
        state = torch.load(ruta, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 1.13
        state = torch.load(ruta, map_location="cpu")
    meta = {}
    if isinstance(state, dict) and "model_state_dict" in state:
        meta = state.get("meta") or {}
        state = state["model_state_dict"]
    model = wm.modelo_desde_meta(meta)
    model.load_state_dict(state)
    print(f"Checkpoint: {ruta}")
    return model, meta


def lote_de_entrada(model, meta):
    cfg = model.config
    tam = IMAGE_SIZE or int(meta.get("image_size", 128))
    bandas = tuple(meta.get("bandas_modelo") or meta.get("orden_bandas") or wc.BAND_ORDER)
    indices = tuple(meta.get("indices") or ())
    norm = meta.get("norm", {"tipo": "div", "valor": 12500.0})
    lista = os.path.join(DATA_DIR, LISTA)
    if USAR_DATOS_REALES and os.path.isfile(lista):
        orden_ds = wc.BAND_ORDER
        ruta_cfg = os.path.join(DATA_DIR, "config.json")
        if os.path.isfile(ruta_cfg):
            with open(ruta_cfg) as f:
                orden_ds = tuple(json.load(f).get("orden_bandas") or wc.BAND_ORDER)
        with open(lista) as f:
            nombres = [l.strip() for l in f if l.strip()]
        random.Random(SEED).shuffle(nombres)
        orden = wc.bandas_requeridas(bandas, indices)
        sel = [list(orden_ds).index(b) for b in orden]
        xs = []
        for n in nombres[:N_MUESTRAS]:
            arr, _ = wc.leer_raster(os.path.join(DATA_DIR, "Images", n))
            dn = arr[sel].astype(np.float32)[:, :tam, :tam]
            xs.append(wc.construir_entrada(dn, orden, bandas, indices, norm))
        if xs and all(x.shape[1:] == (tam, tam) for x in xs):
            print(f"Entrada: {len(xs)} recortes reales de {lista}")
            return torch.from_numpy(np.stack(xs))
    # Respaldo: bandas en [0,1] (norm div) e índices en [-1,1]
    print("Entrada: aleatoria (bandas U[0,1], índices U[-1,1])")
    g = torch.Generator().manual_seed(SEED)
    x = torch.rand(N_MUESTRAS, cfg["in_channels"], tam, tam, generator=g)
    x[:, len(bandas):] = x[:, len(bandas):] * 2 - 1
    return x


def en_rejilla(x, fmt):
    _, _, scale = wm.limites(fmt)
    xs = x.detach().double() / scale
    err = (xs - torch.round(xs)).abs().max().item()
    return err < EPS, err


def saturacion(x, fmt):
    qmin, qmax, scale = wm.limites(fmt)
    xs = x.detach() / scale
    return ((xs < qmin) | (xs > qmax + 0.5)).float().mean().item(), x.detach().abs().max().item()


def bits_enteros_necesarios(max_abs):
    return max(0, math.ceil(math.log2(max_abs + 1e-12))) if max_abs > 0 else 0


def main():
    torch.manual_seed(SEED)
    model, meta = cargar_modelo()
    model.eval()
    print(f"Cuantización: {wm.resumen_quant(model.config['quant'])}")
    x = lote_de_entrada(model, meta)
    filas, todo_ok = [], True

    # --- Pesos y bias ---
    for nombre, mod in model.named_modules():
        if not isinstance(mod, (wm.QATConv2d, wm.QATConvTranspose2d)):
            continue
        for pn, par, fmt in (("weight", mod.weight, mod.w_fmt), ("bias", mod.bias, mod.b_fmt)):
            if par is None or fmt is None:
                continue
            q = wm.fake_quantize(par, fmt, mod.redondeo)
            ok, err = en_rejilla(q, fmt)
            sat, mx = saturacion(par, fmt)
            filas.append((f"{nombre}.{pn}", wm.nombre_formato(fmt), ok, err, sat, mx, fmt))

    # --- Activaciones: entrada (antes) y salida (después) de cada FakeQuantAct ---
    capturas = {}

    def gancho(nombre):
        def h(mod, inp, out):
            c = capturas.setdefault(nombre, {"fmt": mod.fmt, "sat": [], "max": 0.0, "ok": True, "err": 0.0})
            s, m = saturacion(inp[0], mod.fmt)
            ok, err = en_rejilla(out, mod.fmt)
            c["sat"].append(s)
            c["max"] = max(c["max"], m)
            c["ok"] &= ok
            c["err"] = max(c["err"], err)
        return h

    hs = [m.register_forward_hook(gancho(n)) for n, m in model.named_modules()
          if isinstance(m, wm.FakeQuantAct) and m.fmt is not None]
    with torch.no_grad():
        for i in range(0, x.shape[0], 4):
            model(x[i:i + 4])
    for h in hs:
        h.remove()
    for nombre, c in capturas.items():
        filas.append((nombre, wm.nombre_formato(c["fmt"]), c["ok"], c["err"], float(np.mean(c["sat"])), c["max"],
                      c["fmt"]))

    print("\n" + "=" * 112)
    print(f"{'punto':42s} {'formato':>8s} {'rejilla':>8s} {'err':>9s} {'saturado':>10s} {'max|x|':>10s}  aviso")
    print("=" * 112)
    for nombre, fmtn, ok, err, sat, mx, fmt in filas:
        todo_ok &= ok
        aviso = ""
        if sat > SATURACION_AVISO:
            aviso = f"satura: necesita ~{bits_enteros_necesarios(mx)} bits enteros"
        elif mx > 0 and bits_enteros_necesarios(mx) + 1 < wm.int_bits(fmt):
            aviso = f"holgado: con {bits_enteros_necesarios(mx)} bits enteros alcanza"
        print(f"{nombre:42s} {fmtn:>8s} {'OK' if ok else 'FALLA':>8s} {err:9.1e} {100 * sat:9.4f}% {mx:10.4f}  {aviso}")
    print("=" * 112)
    print("RESULTADO:", "todo en la rejilla de punto fijo" if todo_ok else "HAY VALORES FUERA DE LA REJILLA")
    print("Nota: BatchNorm no está plegada; al exportar a FPGA pliega BN en los pesos y vuelve a verificar.")


if __name__ == "__main__":
    main()
