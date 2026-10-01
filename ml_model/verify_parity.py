"""Verify the sklearn-free inference path matches the original one.

The Lambda package cannot ship scikit-learn (it would push the deployment
package past the 250MB unzipped limit), so inference loads the XGBoost booster
in its native JSON format and applies the StandardScaler as raw arithmetic from
scaler_params.json.

This script proves both paths produce identical probabilities:

    reference : joblib(vak_model_vN.pkl).predict_proba(scaler.transform(X))
    lambda    : Booster(vak_model_vN.json).predict(DMatrix((X - mean) / scale))

Run it after train.py and before building the deployment package.

Usage:
    python verify_parity.py [--rows N] [--tolerance 1e-6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np
import xgboost as xgb

from preprocess import NUMERIC_FEATURES, load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_PATH = os.path.join(HERE, "data", "dataset_simulado.csv")

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")


def load_reference():
    """The training-time path: pickled estimator + pickled scaler."""
    model = joblib.load(os.path.join(MODELS_DIR, f"vak_model_{MODEL_VERSION}.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return lambda X: model.predict_proba(scaler.transform(X))


def load_lambda():
    """The inference path shipped to Lambda: no scikit-learn involved."""
    booster = xgb.Booster()
    booster.load_model(os.path.join(MODELS_DIR, f"vak_model_{MODEL_VERSION}.json"))

    with open(os.path.join(MODELS_DIR, "scaler_params.json"), encoding="utf-8") as fh:
        params = json.load(fh)

    if params["features"] != NUMERIC_FEATURES:
        raise ValueError(
            "Feature order mismatch between scaler_params.json and preprocess.py:\n"
            f"  scaler_params.json : {params['features']}\n"
            f"  preprocess.py      : {NUMERIC_FEATURES}"
        )

    mean = np.asarray(params["mean"], dtype=float)
    scale = np.asarray(params["scale"], dtype=float)
    return lambda X: booster.predict(xgb.DMatrix((X - mean) / scale))


def main():
    parser = argparse.ArgumentParser(description="Check inference-path parity.")
    parser.add_argument("--rows", type=int, default=0, help="Rows to check (0 = all).")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Max allowed delta.")
    args = parser.parse_args()

    missing = [
        name
        for name in (
            f"vak_model_{MODEL_VERSION}.pkl",
            f"vak_model_{MODEL_VERSION}.json",
            "scaler.pkl",
            "scaler_params.json",
        )
        if not os.path.exists(os.path.join(MODELS_DIR, name))
    ]
    if missing:
        print(f"[FAIL] Missing artifacts in models/: {missing}")
        print("       Run train.py first.")
        sys.exit(1)

    df = load_dataset(DATA_PATH)
    X = df[NUMERIC_FEATURES].values.astype(float)
    if args.rows:
        X = X[: args.rows]

    reference, lambda_path = load_reference(), load_lambda()
    proba_ref = np.asarray(reference(X), dtype=float)
    proba_lambda = np.asarray(lambda_path(X), dtype=float)

    if proba_ref.shape != proba_lambda.shape:
        print(f"[FAIL] Shape mismatch: {proba_ref.shape} vs {proba_lambda.shape}")
        sys.exit(1)

    delta = np.abs(proba_ref - proba_lambda)
    max_delta = float(delta.max())
    argmax_agreement = float((proba_ref.argmax(axis=1) == proba_lambda.argmax(axis=1)).mean())

    print(f"Rows checked      : {X.shape[0]}")
    print(f"Max abs delta     : {max_delta:.3e} (tolerance {args.tolerance:.0e})")
    print(f"Predicted class   : {argmax_agreement:.2%} agreement")

    worst = int(delta.max(axis=1).argmax())
    print(f"Worst row ({worst}):")
    print(f"  reference : {np.round(proba_ref[worst], 8).tolist()}")
    print(f"  lambda    : {np.round(proba_lambda[worst], 8).tolist()}")

    if max_delta > args.tolerance or argmax_agreement < 1.0:
        print("\n[FAIL] Inference paths diverge. Do NOT deploy this package.")
        sys.exit(1)

    print("\n[OK] Both inference paths are identical.")


if __name__ == "__main__":
    main()
