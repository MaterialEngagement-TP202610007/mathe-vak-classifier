"""Train the VAK XGBoost classifier on the simulated dataset.

Usage:
    python train.py [--no-smote] [--dataset path.csv]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone

import joblib
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from preprocess import VAK_CLASSES, load_dataset, preprocess_training

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "dataset_simulado.csv")
MODELS_DIR = os.path.join(HERE, "models")

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
MODEL_FILE = f"vak_model_{MODEL_VERSION}.pkl"

# Serialized without scikit-learn, for the Lambda inference path.
BOOSTER_FILE = f"vak_model_{MODEL_VERSION}.json"
SCALER_PARAMS_FILE = "scaler_params.json"

XGB_PARAMS = dict(
    n_estimators=180,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=2,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)


def maybe_smote(X_train, y_train, enabled: bool):
    counts = Counter(y_train.tolist())
    if not enabled or len(set(counts.values())) == 1:
        return X_train, y_train, False
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("[WARN] imbalanced-learn not installed; skipping SMOTE.")
        return X_train, y_train, False

    k = max(1, min(5, min(counts.values()) - 1))
    X_res, y_res = SMOTE(random_state=42, k_neighbors=k).fit_resample(X_train, y_train)
    return X_res, y_res, True


def main():
    parser = argparse.ArgumentParser(description="Train the VAK XGBoost model.")
    parser.add_argument("--dataset", default=DATA_PATH, help="Path to the dataset CSV.")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE balancing.")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"[1/5] Loading dataset: {args.dataset}")
    df = load_dataset(args.dataset)
    print(f"      Rows: {len(df)} | VAK distribution: {dict(Counter(df['target_vak_label']))}")

    print("[2/5] Preprocessing (scaler + stratified 80/20 split)...")
    prep = preprocess_training(df)
    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]
    scaler, label_encoder = prep["scaler"], prep["label_encoder"]
    feature_columns = prep["feature_columns"]
    print(f"      Train: {X_train.shape} | Test: {X_test.shape} | Features: {len(feature_columns)}")

    print("[3/5] Class balancing (SMOTE)...")
    X_train, y_train, smote_applied = maybe_smote(X_train, y_train, enabled=not args.no_smote)
    print(f"      SMOTE applied: {smote_applied} | Train: {dict(Counter(y_train.tolist()))}")

    print("[4/5] Training XGBoost...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2])
    print(f"      Accuracy test: {acc:.4f} | F1 macro: {f1_macro:.4f}")
    for cls, f1c in zip(VAK_CLASSES, f1_per_class):
        print(f"        F1 {cls:<12}: {f1c:.4f}")

    print("[5/5] Serializing artifacts to models/...")
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    joblib.dump(model, model_path)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))

    # Inference artifacts for Lambda. The deployment package cannot ship
    # scikit-learn (it would exceed the 250MB unzipped limit), so the booster
    # goes out in XGBoost's native format and the scaler as plain mean/scale
    # vectors. See verify_parity.py for the equivalence check.
    model.get_booster().save_model(os.path.join(MODELS_DIR, BOOSTER_FILE))

    scaler_params = {
        "features": feature_columns,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(os.path.join(MODELS_DIR, SCALER_PARAMS_FILE), "w", encoding="utf-8") as fh:
        json.dump(scaler_params, fh, indent=2)

    metadata = {
        "model_version": MODEL_VERSION,
        "model_file": MODEL_FILE,
        "booster_file": BOOSTER_FILE,
        "scaler_params_file": SCALER_PARAMS_FILE,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "XGBoost (XGBClassifier)",
        "objective": XGB_PARAMS["objective"],
        "hyperparameters": dict(XGB_PARAMS),
        "n_samples_total": int(len(df)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "smote_applied": bool(smote_applied),
        "classes": VAK_CLASSES,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "metrics": {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_per_class": {cls: float(v) for cls, v in zip(VAK_CLASSES, f1_per_class)},
        },
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\nArtifacts saved ({size_mb:.2f} MB model).")
    if size_mb >= 10:
        print(f"[WARN] Model .pkl over 10MB ({size_mb:.2f}MB); check Lambda limit.")
    print("Training complete.")


if __name__ == "__main__":
    main()
