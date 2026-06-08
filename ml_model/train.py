"""
train.py
========
Entrena el clasificador XGBoost VAK (version beta) sobre el dataset simulado.

Segun Notion - Fase 2:
  - Hiperparametros iniciales: n_estimators=100, max_depth=4, learning_rate=0.1,
    objective=multi:softprob, num_class=3.
  - Opcional: balanceo de clases con SMOTE si el dataset esta desbalanceado.
  - Serializa modelo (vak_model_v1.pkl), scaler.pkl, label_encoder.pkl y
    model_metadata.json en ml_model/models/.

Uso:
    python train.py [--no-smote] [--dataset ruta.csv]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from preprocess import (
    VAK_CLASSES,
    get_feature_columns,
    load_dataset,
    preprocess_training,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "dataset_simulado.csv")
MODELS_DIR = os.path.join(HERE, "models")

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
MODEL_FILE = f"vak_model_{MODEL_VERSION}.pkl"

# Hiperparametros iniciales (Notion - Fase 2).
XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)


def maybe_smote(X_train, y_train, enabled: bool):
    """Aplica SMOTE para balancear clases si esta habilitado y hay desbalance."""
    counts = Counter(y_train.tolist())
    balanced = len(set(counts.values())) == 1
    if not enabled or balanced:
        return X_train, y_train, False
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("[WARN] imbalanced-learn no instalado; se omite SMOTE.")
        return X_train, y_train, False

    # k_neighbors no puede superar (min_clase - 1).
    min_class = min(counts.values())
    k = max(1, min(5, min_class - 1))
    sm = SMOTE(random_state=42, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res, True


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo VAK XGBoost.")
    parser.add_argument("--dataset", default=DATA_PATH, help="Ruta al CSV del dataset.")
    parser.add_argument("--no-smote", action="store_true", help="Desactiva el balanceo SMOTE.")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"[1/5] Cargando dataset: {args.dataset}")
    df = load_dataset(args.dataset)
    print(f"      Registros: {len(df)} | Distribucion VAK: {dict(Counter(df['target_vak_label']))}")

    print("[2/5] Preprocesando (one-hot + scaler + split estratificado 80/20)...")
    prep = preprocess_training(df)
    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]
    scaler, label_encoder = prep["scaler"], prep["label_encoder"]
    feature_columns = prep["feature_columns"]
    print(f"      Train: {X_train.shape} | Test: {X_test.shape} | Features: {len(feature_columns)}")

    print("[3/5] Balanceo de clases (SMOTE)...")
    X_train, y_train, smote_applied = maybe_smote(X_train, y_train, enabled=not args.no_smote)
    print(f"      SMOTE aplicado: {smote_applied} | Train balanceado: {dict(Counter(y_train.tolist()))}")

    print("[4/5] Entrenando XGBoost...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    # Metricas rapidas sobre el conjunto de prueba (el detalle va en evaluate.py).
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2])
    print(f"      Accuracy test: {acc:.4f} | F1 macro: {f1_macro:.4f}")
    for cls, f1c in zip(VAK_CLASSES, f1_per_class):
        print(f"        F1 {cls:<12}: {f1c:.4f}")

    print("[5/5] Serializando artefactos en models/...")
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    joblib.dump(model, model_path)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    metadata = {
        "model_version": MODEL_VERSION,
        "model_file": MODEL_FILE,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "XGBoost (XGBClassifier)",
        "objective": XGB_PARAMS["objective"],
        "hyperparameters": {k: v for k, v in XGB_PARAMS.items()},
        "n_samples_total": int(len(df)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "smote_applied": bool(smote_applied),
        "classes": VAK_CLASSES,  # indice = etiqueta codificada (Visual=0, Auditivo=1, Kinestesico=2)
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "metrics": {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_per_class": {cls: float(v) for cls, v in zip(VAK_CLASSES, f1_per_class)},
        },
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\nArtefactos guardados:")
    print(f"  - {model_path}  ({size_mb:.2f} MB)")
    print(f"  - {os.path.join(MODELS_DIR, 'scaler.pkl')}")
    print(f"  - {os.path.join(MODELS_DIR, 'label_encoder.pkl')}")
    print(f"  - {meta_path}")
    if size_mb >= 10:
        print(f"[WARN] El .pkl supera 10MB ({size_mb:.2f}MB); revisar limite de Lambda.")
    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()
