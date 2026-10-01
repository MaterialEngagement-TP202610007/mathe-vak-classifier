"""Train the production VAK model (v2) on the real pilot data, drop-in compatible with v1.

Same 7 inputs as vak_model_v1.pkl (scores included, same names and order), same classes
(Visual=0, Auditory=1, Kinesthetic=2), same scaler.pkl, plain XGBClassifier with
predict_proba. Label: argmax of the three scores (ties dropped). Hyperparameters are the
same as real_v2. CV is reported only as concordance with the declared preference.

Usage:
    python train_prod_real.py [--data path/resultados.csv]

Outputs (aggregates only, no rows or ids):
    models/vak_model_v2.pkl, models/model_metadata_v2.json, reports/metrics_v2_prod.json
The source CSV contains personal data of minors and must never be committed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from preprocess import NUMERIC_FEATURES, VAK_CLASSES
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "mathe-export-20260924-1013", "resultados.csv")
SEED = 42

# Same hyperparameters as real_v2 (train_real_cv.py), fixed a priori.
XGB_PARAMS = dict(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=1.0,
    min_child_weight=3,
    reg_lambda=1.0,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=SEED,
    n_jobs=1,
)
MODELS_DIR = os.path.join(HERE, "models")
REPORTS_DIR = os.path.join(HERE, "reports")
MODEL_FILE = "vak_model_v2.pkl"
K_FOLDS = 5

# Backend payload mapping (complete-questionnaire.use-case.ts): CSV column -> model feature.
CSV_TO_FEATURE = {
    "visualScore": "visual_score",
    "auditoryScore": "auditory_score",
    "kinestheticScore": "kinesthetic_score",
    "responseConsistency": "response_consistency",
    "avgQuestionTime": "avg_response_time",
    "totalChanges": "total_changes",
    "totalReviews": "total_backtracks",
}
NOTE = ("Las metricas miden concordancia con la preferencia declarada por el\n"
        "estudiante (puntaje directo), no capacidad predictiva independiente. La\n"
        "evaluacion predictiva con variables de comportamiento esta en real_v2.")


def load(path: str):
    df = pd.read_csv(path, usecols=["classifierType", *CSV_TO_FEATURE])
    df = df[df.classifierType == "xgboost"].rename(columns=CSV_TO_FEATURE)
    scores = df[["visual_score", "auditory_score", "kinesthetic_score"]]
    scores.columns = VAK_CLASSES
    tie = scores.eq(scores.max(axis=1), axis=0).sum(axis=1) > 1
    kept = df[~tie]
    y = scores[~tie].idxmax(axis=1).map({c: i for i, c in enumerate(VAK_CLASSES)}).to_numpy()
    return kept[NUMERIC_FEATURES].to_numpy(dtype=float), y, len(df), int(tie.sum())


def fit(X, y):
    Xs, ys = SMOTE(random_state=SEED).fit_resample(X, y)
    return XGBClassifier(**XGB_PARAMS).fit(Xs, ys)


def compat_check(model_path: str):
    """Load like the Lambda (joblib model + scaler.pkl) and run the real handler on a fake payload."""
    sys.path.insert(0, os.path.join(HERE, "lambda"))
    import lambda_function as lf
    lf._MODEL = joblib.load(model_path)
    lf._SCALER = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    event = {"body": json.dumps({"features": {
        "visual_score": 6, "auditory_score": 3, "kinesthetic_score": 1, "response_consistency": 0.4,
        "avg_response_time": 21.5, "total_changes": 2, "total_backtracks": 1}})}
    out = lf.lambda_handler(event, None)
    print("compat: statusCode", out["statusCode"], "| body", out["body"])
    assert out["statusCode"] == 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    args = ap.parse_args()

    v1_meta = json.load(open(os.path.join(MODELS_DIR, "model_metadata.json")))
    assert NUMERIC_FEATURES == v1_meta["feature_columns"], "feature order differs from v1"
    assert VAK_CLASSES == v1_meta["classes"], "class order differs from v1"
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

    X_raw, y, n_rows, n_ties = load(args.data)
    X = scaler.transform(X_raw)
    dist = {VAK_CLASSES[i]: int((y == i).sum()) for i in range(3)}
    print(f"filas xgboost={n_rows} | empates descartados={n_ties} | n final={len(y)} | clases={dist}")

    acc, f1 = [], []
    for tr, te in StratifiedKFold(K_FOLDS, shuffle=True, random_state=SEED).split(X, y):
        pred = fit(X[tr], y[tr]).predict(X[te])
        acc.append(accuracy_score(y[te], pred))
        f1.append(f1_score(y[te], pred, average="macro"))
    ms = lambda a: {"mean": round(float(np.mean(a)), 4), "sd": round(float(np.std(a, ddof=1)), 4)}
    cv = {"scheme": f"StratifiedKFold(k={K_FOLDS}, shuffle=True, random_state={SEED}), SMOTE inside train folds",
          "accuracy": ms(acc), "f1_macro": ms(f1)}
    print("CV concordancia:", json.dumps(cv))

    model = fit(X, y)
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    joblib.dump(model, model_path)

    meta = {
        "model_version": "v2", "model_file": MODEL_FILE,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data": "piloto institucional, resultados.csv rows with classifierType=xgboost",
        "label": "argmax(visual_score, auditory_score, kinesthetic_score); ties dropped",
        "n_rows_xgboost": n_rows, "n_ties_dropped": n_ties, "n_final": int(len(y)), "class_distribution": dist,
        "feature_columns": NUMERIC_FEATURES, "n_features": len(NUMERIC_FEATURES), "classes": VAK_CLASSES,
        "scaler": "scaler.pkl from v1 (unchanged, transform only)", "smote": "applied to full set for final fit",
        "hyperparameters": XGB_PARAMS, "cv_metrics": cv, "note": NOTE,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata_v2.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(REPORTS_DIR, "metrics_v2_prod.json"), "w") as f:
        json.dump({k: meta[k] for k in ("model_version", "trained_at", "n_rows_xgboost", "n_ties_dropped",
                                         "n_final", "class_distribution", "cv_metrics", "note")},
                  f, indent=2, ensure_ascii=False)

    compat_check(model_path)


if __name__ == "__main__":
    main()
