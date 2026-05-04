import torch
import torch.nn as nn

# Primero, necesitas tener definida la clase del modelo (la misma arquitectura)
# Asumiendo que ya tienes las clases definidas en tu script

# Cargar el modelo entrenado
def load_and_analyze(model_class, checkpoint_path, device='cpu', **model_kwargs):
    """
    Carga un modelo entrenado y analiza su estructura
    
    Args:
        model_class: Clase del modelo (ej: WildFireNet2DV3L_3x3_Residual)
        checkpoint_path: Ruta al archivo .pth
        device: 'cpu' o 'cuda'
        **model_kwargs: Argumentos para inicializar el modelo
    """
    
    # 1. Crear la instancia del modelo
    model = model_class(**model_kwargs)
    
    # 2. Cargar los pesos entrenados
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Manejar diferentes formatos de checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Cargado state_dict desde checkpoint con 'model_state_dict'")
            # Si hay más información útil
            if 'epoch' in checkpoint:
                print(f"   Época: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"   Loss: {checkpoint['loss']}")
            if 'accuracy' in checkpoint:
                print(f"   Accuracy: {checkpoint['accuracy']}")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Cargado state_dict directamente")
    else:
        # Si es solo el state_dict
        model.load_state_dict(checkpoint)
        print("✅ Cargado state_dict")
    
    model = model.to(device)
    model.eval()  # Modo evaluación
    
    return model

# Ejemplo de uso
model = load_and_analyze(
    WildFireNet2DV3L_3x3_Residual,
    'tu_modelo_entrenado.pth',
    input_channels=5,
    num_classes=2,
    dims=(32, 64)
)