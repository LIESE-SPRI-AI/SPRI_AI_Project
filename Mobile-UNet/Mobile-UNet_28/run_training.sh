#!/bin/bash
# Reintenta el entrenamiento si se cae. Con --reanudar auto (por defecto) cada reintento continúa
# desde el last.pth de la corrida que se cayó. Los argumentos se pasan tal cual al script de Python:
#   ./run_training.sh --optimizador sgd --lr 0.02 --scheduler plateau
# (python3 trainModelWildfire3D4B.py -h para ver todos)
cd "$(dirname "$0")" || exit 1
while true; do
    python3 trainModelWildfire3D4B.py "$@"
    codigo=$?
    if [ $codigo -eq 0 ]; then
        echo -e "\nEntrenamiento completado exitosamente."
        break
    fi
    if [ $codigo -eq 2 ]; then     # error de configuración o de argumentos: reintentar no sirve
        echo -e "\nError de configuración; revisa los argumentos (python3 trainModelWildfire3D4B.py -h)."
        break
    fi
    if [ $codigo -eq 130 ]; then   # Ctrl+C
        echo -e "\nCancelado por el usuario."
        break
    fi
    echo -e "\nNo me la hagas jochis (código $codigo). Reintentando en 30s...\n"
    sleep 30
done
