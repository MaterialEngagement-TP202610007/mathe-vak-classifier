"""VAK feature schema and inference helpers.

Single source of truth for feature order, label encoding and the JSON response
contract. Imported by train.py / evaluate.py and mirrored in lambda_function.py,
which cannot import this module because Lambda ships without pandas.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ID_COLUMN = "student_id"
TARGET_COLUMN = "target_vak_label"

# All features are numeric and StandardScaled. Order is fixed.
NUMERIC_FEATURES = [
    "visual_score",
    "auditory_score",
    "kinesthetic_score",
    "response_consistency",
    "avg_response_time",
    "total_changes",
    "total_backtracks",
]

# Class labels, as in the CSV and the API response.
# Index is the encoded class: Visual=0, Auditory=1, Kinesthetic=2.
VAK_CLASSES = ["Visual", "Auditory", "Kinesthetic"]

# Profile type is decided by the gap between the top two class probabilities.
CLEAR_MARGIN = 0.30   # >= -> "clear"
MIXED_MARGIN = 0.12   # in [MIXED, CLEAR) -> "tendency"; below -> "mixed"


def get_feature_columns() -> list[str]:
    return list(NUMERIC_FEATURES)


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    expected = set([ID_COLUMN, TARGET_COLUMN] + NUMERIC_FEATURES)
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df.drop_duplicates().dropna(subset=[TARGET_COLUMN])
    for col in NUMERIC_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df.reset_index(drop=True)


def preprocess_training(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """Build train/test matrices, the scaler and the label encoder."""
    feature_columns = get_feature_columns()
    X_full = df[feature_columns].values.astype(float)

    # Force the class order (LabelEncoder.fit would sort alphabetically).
    class_to_idx = {label: idx for idx, label in enumerate(VAK_CLASSES)}
    y_full = df[TARGET_COLUMN].astype(str).map(class_to_idx)
    if y_full.isna().any():
        bad = sorted(df.loc[y_full.isna(), TARGET_COLUMN].unique())
        raise ValueError(f"Unknown VAK labels in dataset: {bad}")
    y_full = y_full.astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_state, stratify=y_full,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(VAK_CLASSES)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
    }


def build_feature_vector(raw: dict, scaler: StandardScaler) -> np.ndarray:
    """Turn a raw feature dict into a scaled (1, n_features) vector."""
    numeric = np.array([float(raw[col]) for col in NUMERIC_FEATURES], dtype=float)
    return scaler.transform(numeric.reshape(1, -1))


def profile_type(top_prob: float, second_prob: float) -> str:
    gap = top_prob - second_prob
    if gap >= CLEAR_MARGIN:
        return "clear"
    if gap >= MIXED_MARGIN:
        return "tendency"
    return "mixed"


def build_response(proba, classifier_type: str = "xgboost") -> dict:
    """Build the API contract from a 3-class probability vector."""
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(proba)[::-1]
    top, second = int(order[0]), int(order[1])
    profile = profile_type(proba[top], proba[second])

    confidence = {VAK_CLASSES[i]: round(float(proba[i]) * 100, 2) for i in range(len(VAK_CLASSES))}
    return {
        "predominant_style": VAK_CLASSES[top],
        "secondary_style": VAK_CLASSES[second],
        "confidence": confidence,
        "predominant_confidence": confidence[VAK_CLASSES[top]],
        "profile_type": profile,
        "is_mixed_profile": profile == "mixed",
        "classifier_type": classifier_type,
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = load_dataset(os.path.join(here, "data", "simulated_dataset.csv"))
    prep = preprocess_training(data)
    print(f"Rows: {len(data)} | Features: {len(prep['feature_columns'])}")
    print(f"X_train: {prep['X_train'].shape} | X_test: {prep['X_test'].shape}")
    print(f"Classes: {list(prep['label_encoder'].classes_)}")
