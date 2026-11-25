#!/bin/bash
# Reintenta el entrenamiento si se cae. Con REANUDAR = "auto" en trainModelWildfire3D4B.py
# cada reintento continúa desde el last.pth de la corrida que se cayó (antes empezaba de cero).
# La configuración (épocas, bandas, cuantización...) está al inicio del script de Python.
cd "$(dirname "$0")" || exit 1
while true; do
    python3 trainModelWildfire3D4B.py
    codigo=$?
    if [ $codigo -eq 0 ]; then
        echo -e "\nEntrenamiento completado exitosamente."
        break
    fi
    if [ $codigo -eq 2 ]; then     # error de configuración (rutas, bandas...): reintentar no sirve
        echo -e "\nError de configuración; revisa el bloque CONFIGURACIÓN del script."
        break
    fi
    if [ $codigo -eq 130 ]; then   # Ctrl+C
        echo -e "\nCancelado por el usuario."
        break
    fi
    echo -e "\nNo me la hagas jochis (código $codigo). Reintentando en 30s...\n"
    sleep 30
done
