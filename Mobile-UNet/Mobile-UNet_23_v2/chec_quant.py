"""
Verifica que las salidas del modelo realmente caigan en la rejilla de
cuantizacion (multiplos exactos de 'scale' = 2**-frac_bits), no solo que
los modulos FakeQuantAct/QATConv2d/QATConvTranspose2d existan en el modelo.

Uso:
    python3 verify_quantization.py
    python3 verify_quantization.py --checkpoint weights/model_best.pth
"""
import argparse
import torch
import torch.nn as nn

from Wildfire_models import UNet2D, FakeQuantAct, QATConv2d, QATConvTranspose2d, fake_quantize

EPS = 1e-5  # tolerancia por error de punto flotante (float32), no por diseno


def scale_of(module):
    if isinstance(module, FakeQuantAct):
        return 2.0 ** (-module.frac_bits)
    if isinstance(module, (QATConv2d, QATConvTranspose2d)):
        return 2.0 ** (-module.weight_frac_bits)
    return None


def check_on_grid(x, scale):
    """Regresa (esta_en_grid: bool, max_error_absoluto: float, n_valores_unicos: int)"""
    x_scaled = x.detach() / scale
    residual = (x_scaled - torch.round(x_scaled)).abs()
    max_err = residual.max().item()
    n_unique = torch.unique(torch.round(x_scaled)).numel()
    return max_err < EPS, max_err, n_unique


def verify_activations(model, x):
    """Cuelga hooks en cada FakeQuantAct y revisa su SALIDA (ya deberia estar cuantizada)."""
    results = []

    def make_hook(name, module):
        def hook(mod, inp, out):
            scale = scale_of(module)
            ok, max_err, n_unique = check_on_grid(out, scale)
            results.append((name, 'activation', ok, max_err, n_unique, scale))
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantAct):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    model.eval()
    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    return results


def verify_weights(model):
    """Recalcula fake_quantize(peso) y confirma que caiga en grid (esto valida la funcion,
    no requiere forward -- los pesos SIEMPRE se recuantizan en cada forward dentro de
    QATConv2d/QATConvTranspose2d.forward, nunca se guardan ya cuantizados)."""
    results = []
    for name, module in model.named_modules():
        if isinstance(module, (QATConv2d, QATConvTranspose2d)):
            scale = scale_of(module)
            w_q = fake_quantize(module.weight, frac_bits=module.weight_frac_bits, signed=True)
            ok, max_err, n_unique = check_on_grid(w_q, scale)
            results.append((name, 'weight', ok, max_err, n_unique, scale))
    return results


def print_report(results, title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    all_ok = True
    for name, kind, ok, max_err, n_unique, scale in results:
        status = "OK" if ok else "FALLA"
        if not ok:
            all_ok = False
        print(f"[{status:5s}] {name:35s} ({kind:10s}) "
              f"scale={scale:.6f}  max_err={max_err:.2e}  valores_unicos={n_unique}")
    print(f"\n-> {'TODO CUANTIZADO CORRECTAMENTE' if all_ok else 'HAY CAPAS QUE NO ESTAN EN EL GRID'}")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None, help='ruta a .pth (model_best.pth, checkpoint.pth, etc.)')
    parser.add_argument('--image-size', type=int, default=128)
    args = parser.parse_args()

    model = UNet2D(in_channels=4, out_channels=2)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu')
        state_dict = state['model_state_dict'] if 'model_state_dict' in state else state
        model.load_state_dict(state_dict)
        print(f"Checkpoint cargado desde: {args.checkpoint}")
    else:
        print("Sin checkpoint -- usando pesos recien inicializados (solo para probar el script)")

    # Simula el rango real [0,1] que produce tu SegmentationDataset (banda/12500, clip)
    x = torch.rand(1, 4, args.image_size, args.image_size)

    weight_results = verify_weights(model)
    act_results = verify_activations(model, x)

    ok1 = print_report(weight_results, "PESOS (Conv2d / ConvTranspose2d cuantizados)")
    ok2 = print_report(act_results, "ACTIVACIONES (salidas de FakeQuantAct)")

    print(f"\n{'='*70}")
    print("RESULTADO FINAL:", "PASA" if (ok1 and ok2) else "REVISAR")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()