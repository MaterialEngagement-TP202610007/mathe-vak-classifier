"""
evaluate.py
===========
Evalua el modelo VAK entrenado y genera reportes (Notion - Fase 2, tarea 3).

Genera y documenta:
  - Accuracy global
  - Precision / Recall / F1 por clase (Visual, Auditivo, Kinestesico)
  - Matriz de confusion  -> reports/confusion_matrix.png
  - Feature importance    -> reports/feature_importance.png
  - Reporte de clasificacion -> reports/classification_report.json

Reutiliza el MISMO split estratificado (random_state fijo) que train.py para que
el conjunto de prueba sea identico.

Uso:
    python evaluate.py
"""

from __future__ import annotations

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # backend sin display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from preprocess import VAK_CLASSES, load_dataset, preprocess_training

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "dataset_simulado.csv")
MODELS_DIR = os.path.join(HERE, "models")
REPORTS_DIR = os.path.join(HERE, "reports")

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")


def _load_metadata():
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), encoding="utf-8") as fh:
        return json.load(fh)


def plot_confusion_matrix(cm, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=VAK_CLASSES,
        yticklabels=VAK_CLASSES,
        cbar=False,
    )
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.title("Matriz de Confusion - Clasificador VAK")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_feature_importance(model, feature_columns, path, top_n=20):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    names = [feature_columns[i] for i in order]
    values = importances[order]

    plt.figure(figsize=(8, max(4, 0.35 * len(names))))
    sns.barplot(x=values, y=names, color="#4C72B0")
    plt.xlabel("Importancia")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance (top {len(names)}) - Clasificador VAK")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    metadata = _load_metadata()
    model_path = os.path.join(MODELS_DIR, metadata["model_file"])

    print(f"Cargando modelo: {model_path}")
    model = joblib.load(model_path)
    feature_columns = metadata["feature_columns"]

    # Reconstruir el MISMO test set (mismo random_state que train.py).
    df = load_dataset(DATA_PATH)
    prep = preprocess_training(df)
    X_test, y_test = prep["X_test"], prep["y_test"]

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    print("\n=== Metricas globales ===")
    print(f"Accuracy global : {acc:.4f}")
    print(f"F1 macro        : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\n=== Por clase ===")
    print(f"{'Clase':<14}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    for i, cls in enumerate(VAK_CLASSES):
        print(f"{cls:<14}{precision[i]:>10.4f}{recall[i]:>10.4f}{f1[i]:>10.4f}")

    report = classification_report(
        y_test, y_pred, labels=[0, 1, 2], target_names=VAK_CLASSES,
        output_dict=True, zero_division=0,
    )
    report["accuracy_global"] = float(acc)

    # Matriz de confusion.
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path)

    # Feature importance.
    fi_path = os.path.join(REPORTS_DIR, "feature_importance.png")
    plot_feature_importance(model, feature_columns, fi_path)

    # Guardar reporte JSON.
    report_path = os.path.join(REPORTS_DIR, "classification_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("\n=== Reportes generados ===")
    print(f"  - {cm_path}")
    print(f"  - {fi_path}")
    print(f"  - {report_path}")

    # Verificacion contra metricas objetivo del Notion.
    print("\n=== Verificacion de metricas objetivo (beta) ===")
    print(f"  Accuracy > 80%          : {'OK' if acc > 0.80 else 'NO'} ({acc:.2%})")
    for i, cls in enumerate(VAK_CLASSES):
        ok = f1[i] > 0.75
        print(f"  F1 {cls:<12} > 0.75 : {'OK' if ok else 'NO'} ({f1[i]:.3f})")


if __name__ == "__main__":
    main()
