#!/bin/bash
while true; do
    python3 trainModelWildfire3D4B.py --epochs 15

    if [ $? -eq 0 ]; then
        echo "Entrenamiento completado exitosamente."
        break
    fi

    echo "No me la hagas jochis. Reintentando en 30s..."
    sleep 30
done