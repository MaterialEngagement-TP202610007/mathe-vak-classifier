"""
predict_local.py
================
Prueba local del modelo serializado (Notion - Fase 2, tarea 4).

  - Carga vak_model_<ver>.pkl, scaler.pkl y label_encoder.pkl con joblib.load().
  - Toma una fila del dataset, construye el vector con la MISMA funcion que usara
    Lambda (preprocess.build_feature_vector) y predice.
  - Sirve como sanity-check de que la ruta de inferencia es identica a la de Lambda.

Uso:
    python predict_local.py [--row N]
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np

from preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    VAK_CLASSES,
    build_feature_vector,
    load_dataset,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_PATH = os.path.join(HERE, "data", "dataset_simulado.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int, default=0, help="Indice de fila del dataset a predecir.")
    args = parser.parse_args()

    with open(os.path.join(MODELS_DIR, "model_metadata.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    model = joblib.load(os.path.join(MODELS_DIR, meta["model_file"]))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"Artefactos cargados OK. Clases: {list(label_encoder.classes_)}")

    df = load_dataset(DATA_PATH)
    sample = df.iloc[args.row]
    raw = {col: sample[col] for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES}

    vector = build_feature_vector(raw, scaler)
    proba = model.predict_proba(vector)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = label_encoder.classes_[pred_idx]

    print(f"\nFila #{args.row}  (etiqueta real: {sample['target_vak_label']})")
    print(f"Prediccion: {pred_label}")
    print("Confianza por clase:")
    for i, cls in enumerate(VAK_CLASSES):
        print(f"  {cls:<12}: {proba[i] * 100:5.2f}%")

    assert vector.shape[1] == meta["n_features"], "El vector no coincide con n_features del modelo"
    print(f"\nVector de {vector.shape[1]} features consistente con el modelo. OK.")


if __name__ == "__main__":
    main()
