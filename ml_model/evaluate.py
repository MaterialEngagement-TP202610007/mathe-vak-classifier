"""Evaluate the trained VAK model and write reports/.

Rebuilds the same stratified test split as train.py (fixed random_state) and
produces a confusion matrix, feature importance plot and classification report.

Usage:
    python evaluate.py
"""

from __future__ import annotations

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
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


def plot_confusion_matrix(cm, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=VAK_CLASSES, yticklabels=VAK_CLASSES, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - VAK Classifier")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_feature_importance(model, feature_columns, path):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    names = [feature_columns[i] for i in order]
    values = importances[order]

    plt.figure(figsize=(8, max(4, 0.5 * len(names))))
    sns.barplot(x=values, y=names, color="#4C72B0")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance - VAK Classifier")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), encoding="utf-8") as fh:
        metadata = json.load(fh)

    model_path = os.path.join(MODELS_DIR, metadata["model_file"])
    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)
    feature_columns = metadata["feature_columns"]

    df = load_dataset(DATA_PATH)
    prep = preprocess_training(df)
    X_test, y_test = prep["X_test"], prep["y_test"]
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    print("\n=== Global metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 macro : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\n=== Per class ===")
    print(f"{'Class':<14}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    for i, cls in enumerate(VAK_CLASSES):
        print(f"{cls:<14}{precision[i]:>10.4f}{recall[i]:>10.4f}{f1[i]:>10.4f}")

    report = classification_report(
        y_test, y_pred, labels=[0, 1, 2], target_names=VAK_CLASSES,
        output_dict=True, zero_division=0,
    )
    report["accuracy_global"] = float(acc)

    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    fi_path = os.path.join(REPORTS_DIR, "feature_importance.png")
    report_path = os.path.join(REPORTS_DIR, "classification_report.json")
    plot_confusion_matrix(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]), cm_path)
    plot_feature_importance(model, feature_columns, fi_path)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("\n=== Reports written ===")
    for p in (cm_path, fi_path, report_path):
        print(f"  - {p}")

    print("\n=== Target metrics (beta) ===")
    print(f"  Accuracy > 80%       : {'OK' if acc > 0.80 else 'NO'} ({acc:.2%})")
    for i, cls in enumerate(VAK_CLASSES):
        print(f"  F1 {cls:<12} > 0.75 : {'OK' if f1[i] > 0.75 else 'NO'} ({f1[i]:.3f})")


if __name__ == "__main__":
    main()
