"""Local prediction sanity check against the serialized model.

Builds the feature vector with the same function Lambda uses and prints the
full response contract.

Usage:
    python predict_local.py [--row N]
"""

from __future__ import annotations

import argparse
import json
import os

import joblib

from preprocess import NUMERIC_FEATURES, build_feature_vector, build_response, load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_PATH = os.path.join(HERE, "data", "dataset_simulado.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", type=int, default=0, help="Dataset row index to predict.")
    args = parser.parse_args()

    with open(os.path.join(MODELS_DIR, "model_metadata.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    model = joblib.load(os.path.join(MODELS_DIR, meta["model_file"]))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    df = load_dataset(DATA_PATH)
    sample = df.iloc[args.row]
    raw = {col: sample[col] for col in NUMERIC_FEATURES}

    vector = build_feature_vector(raw, scaler)
    proba = model.predict_proba(vector)[0]
    response = build_response(proba)

    print(f"Row #{args.row}  (actual label: {sample['target_vak_label']})")
    print(json.dumps(response, indent=2, ensure_ascii=False))

    assert vector.shape[1] == meta["n_features"], "Vector width does not match n_features"


if __name__ == "__main__":
    main()
