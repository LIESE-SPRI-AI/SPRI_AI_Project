#!/bin/bash
while true; do
    python3 trainModelWildfire3D4B.py --epochs 500

    if [ $? -eq 0 ]; then
        echo "\nEntrenamiento completado exitosamente."
        break
    fi

    echo $'\nNo me la hagas jochis. Reintentando en 30s...\n'
    sleep 30
done
