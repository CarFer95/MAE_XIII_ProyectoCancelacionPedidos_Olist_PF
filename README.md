# Proyecto Cancelaciones Olist – Sprint 3 (Medallion)

Este paquete contiene la reorganización del notebook **Copia de Sprint3 (1).ipynb**
en una estructura modular orientada a la metodología **Medallion (Bronze / Silver / Gold)**,
además de parametrización vía variables de entorno.

## Estructura

- `data/raw` y `data/processed`: carpetas listas para que ubiques tus datos si lo deseas.
- `notebooks/Copia_de_Sprint3_original.ipynb`: notebook original de referencia.
- `src/bronze/bronze_layer.py`: BLOQUES 1–6 (Sprint 1) → ingesta, master table y feature engineering.
- `src/silver/silver_layer.py`: limpieza, splits temporales y baseline limpio (Sprint 2).
- `src/gold/gold_layer.py`: modelos avanzados (Random Forest, XGBoost), tuning, validación,
  export del modelo final y generación de figuras (Sprint 3).
- `src/config/config.py`: clase `Settings` que lee variables de entorno (.env).
- `src/utils/paths.py`: helper simple para rutas.
- `main_pipeline.py`: orquestador estilo Medallion.
- `.env.example`: plantilla de variables de entorno.
- `requirements.txt`: dependencias principales de Python.

La lógica de los bloques originales se mantiene, simplemente se ha redistribuido en capas.
