#!/usr/bin/env python3
"""
mem_map.py
==========
Reporte de la arquitectura y de la memoria que ocupa el modelo en punto fijo:
  * estructura del modelo (con el formato Qm.n de cada capa),
  * resumen de torchinfo (si está instalado),
  * memoria de pesos por capa con los bits de la cuantización del checkpoint,
  * tamaño de las activaciones por capa (batch 1) para dimensionar buffers en la FPGA.

La arquitectura, canales y cuantización se leen del checkpoint (no hay que declararlos aquí).
"""

import glob
import os
import sys

import torch

import wildfire_common as wc
import Wildfire_models as wm

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
CHECKPOINT = "auto"          # "auto" = último runs/*/best.pth | ruta a .pth | None = modelo nuevo por defecto
DIR_RUNS = os.path.join(wc.DIR_CODIGO, "runs")
IMAGE_SIZE = None            # None = el del checkpoint (o 128)
SALIDA = os.path.join(wc.DIR_CODIGO, "model_report.txt")
USAR_TORCHINFO = True
# =============================================================================


def cargar():
    ruta = CHECKPOINT
    if ruta == "auto":
        cands = glob.glob(os.path.join(DIR_RUNS, "*", "best.pth"))
        ruta = max(cands, key=os.path.getmtime) if cands else None
        if ruta is None:
            print("⚠ No hay runs/*/best.pth: se usa un modelo nuevo con la configuración por defecto")
    if ruta is None:
        return wm.MbUNet2D(), {}, "(modelo nuevo)"
    try:
        state = torch.load(ruta, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 1.13
        state = torch.load(ruta, map_location="cpu")
    meta = {}
    if isinstance(state, dict) and "model_state_dict" in state:     # primero desempaquetar...
        meta = state.get("meta") or {}
        state = state["model_state_dict"]
    if any(k.startswith("module.") for k in state):                # ...luego quitar 'module.'
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model = wm.modelo_desde_meta(meta)
    model.load_state_dict(state)
    return model, meta, ruta


def bits_de(fmt):
    return fmt["bits"] if fmt else 32


def main():
    model, meta, origen = cargar()
    model.eval()
    cfg = model.config
    q = cfg["quant"]
    tam = IMAGE_SIZE or int(meta.get("image_size", 128))
    entrada = (1, cfg["in_channels"], tam, tam)

    lineas = []
    p = lineas.append
    p(f"Checkpoint: {origen}")
    p(f"Canales: {meta.get('canales') or cfg['in_channels']} -> {cfg['out_channels']} | base {cfg['base_channels']}"
      f" | expansión {cfg['expand_ratio']} | entrada {entrada}")
    p(f"Cuantización: {wm.resumen_quant(q)}")
    p("")
    p(str(model))

    # --- Pesos por capa ---
    p("")
    p("=" * 92)
    p(f"{'capa':45s} {'formato':>10s} {'parámetros':>12s} {'bytes':>12s}")
    p("=" * 92)
    total_p = total_b = 0
    for nombre, mod in model.named_modules():
        propios = list(mod.named_parameters(recurse=False))
        if not propios:
            continue
        for pn, par in propios:
            if isinstance(mod, (wm.QATConv2d, wm.QATConvTranspose2d)):
                fmt = mod.w_fmt if pn == "weight" else mod.b_fmt
            else:
                fmt = None   # BatchNorm: se pliega en la conv al exportar (no ocupa memoria propia)
            bytes_ = par.numel() * bits_de(fmt) / 8
            total_p += par.numel()
            total_b += bytes_
            p(f"{nombre + '.' + pn:45s} {wm.nombre_formato(fmt):>10s} {par.numel():12,d} {bytes_:12,.0f}")
    p("-" * 92)
    p(f"{'TOTAL':45s} {'':>10s} {total_p:12,d} {total_b:12,.0f}  ({total_b / 1e6:.2f} MB, BN incluida en float32)")

    # --- Activaciones por capa (batch 1) ---
    salidas = []

    def gancho(nombre):
        def h(mod, inp, out):
            fmt = mod.fmt if isinstance(mod, wm.FakeQuantAct) else None
            salidas.append((nombre, tuple(out.shape), out.numel(), wm.nombre_formato(fmt),
                            out.numel() * bits_de(fmt) / 8))
        return h

    hs = [m.register_forward_hook(gancho(n)) for n, m in model.named_modules()
          if isinstance(m, wm.FakeQuantAct) and m.fmt is not None]
    with torch.no_grad():
        model(torch.zeros(entrada))
    for h in hs:
        h.remove()
    p("")
    p("=" * 92)
    p("Activaciones cuantizadas (batch 1)")
    p(f"{'punto':45s} {'forma':>22s} {'formato':>10s} {'bytes':>12s}")
    p("=" * 92)
    for nombre, forma, n, fmt, b in salidas:
        p(f"{nombre:45s} {str(forma):>22s} {fmt:>10s} {b:12,.0f}")
    if salidas:
        mayor = max(salidas, key=lambda s: s[4])
        p(f"Activación más grande: {mayor[0]} = {mayor[4] / 1e6:.2f} MB")

    texto = "\n".join(lineas)
    with open(SALIDA, "w", encoding="utf-8") as f:
        f.write(texto + "\n")
        if USAR_TORCHINFO:
            try:
                from torchinfo import summary
                s = summary(model, input_size=entrada, col_names=["input_size", "output_size", "num_params",
                                                                    "params_percent", "trainable"],
                            col_width=20, depth=4, row_settings=["var_names"], verbose=0)
                f.write("\n" + str(s) + "\n")
            except ImportError:
                f.write("\n(torchinfo no instalado: pip install torchinfo)\n")
    print(f"Pesos: {total_p:,} parámetros = {total_b / 1e6:.2f} MB | reporte en {SALIDA}")


if __name__ == "__main__":
    sys.exit(main())
