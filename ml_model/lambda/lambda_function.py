"""AWS Lambda VAK inference handler.

Cold start downloads the model, scaler and label encoder from S3 (cached in /tmp
and module globals while warm). Each POST parses the feature dict, scales it,
predicts and returns the JSON contract with per-class confidence and profile type.

CRITICAL: feature order and the response logic below must stay byte-identical to
preprocess.py. Lambda cannot import preprocess.py (no pandas in the package).

Env vars: S3_BUCKET, S3_PREFIX, MODEL_FILE.
"""

from __future__ import annotations

import json
import os

import boto3
import joblib
import numpy as np

NUMERIC_FEATURES = [
    "visual_score",
    "auditory_score",
    "kinesthetic_score",
    "response_consistency",
    "avg_response_time",
    "total_changes",
    "total_backtracks",
]

VAK_CLASSES = ["Visual", "Auditory", "Kinesthetic"]

CLEAR_MARGIN = 0.30
MIXED_MARGIN = 0.12

S3_BUCKET = os.environ.get("S3_BUCKET", "material-engagement-models")
S3_PREFIX = os.environ.get("S3_PREFIX", "vak/v1").strip("/")
MODEL_FILE = os.environ.get("MODEL_FILE", "vak_model_v1.pkl")

_MODEL = None
_SCALER = None


def _download(s3, filename):
    key = f"{S3_PREFIX}/{filename}" if S3_PREFIX else filename
    local = os.path.join("/tmp", filename)
    if not os.path.exists(local):
        s3.download_file(S3_BUCKET, key, local)
    return local


def _load_artifacts():
    global _MODEL, _SCALER
    if _MODEL is not None:
        return
    s3 = boto3.client("s3")
    _MODEL = joblib.load(_download(s3, MODEL_FILE))
    _SCALER = joblib.load(_download(s3, "scaler.pkl"))


def build_feature_vector(raw: dict) -> np.ndarray:
    numeric = np.array([float(raw[col]) for col in NUMERIC_FEATURES], dtype=float)
    return _SCALER.transform(numeric.reshape(1, -1))


def build_response(proba) -> dict:
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(proba)[::-1]
    top, second = int(order[0]), int(order[1])

    gap = proba[top] - proba[second]
    profile = "clear" if gap >= CLEAR_MARGIN else "tendency" if gap >= MIXED_MARGIN else "mixed"

    confidence = {VAK_CLASSES[i]: round(float(proba[i]) * 100, 2) for i in range(len(VAK_CLASSES))}
    return {
        "predominant_style": VAK_CLASSES[top],
        "secondary_style": VAK_CLASSES[second],
        "confidence": confidence,
        "predominant_confidence": confidence[VAK_CLASSES[top]],
        "profile_type": profile,
        "is_mixed_profile": profile == "mixed",
        "classifier_type": "xgboost",
    }


def _parse_body(event):
    if "body" in event and event["body"] is not None:
        body = event["body"]
        if isinstance(body, str):
            body = json.loads(body)
        return body.get("features", body)
    return event.get("features", event)


def _response(status, payload):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, ensure_ascii=False),
    }


def lambda_handler(event, context):
    try:
        _load_artifacts()
    except Exception as exc:
        return _response(503, {"error": "model_unavailable", "detail": str(exc)})

    try:
        raw = _parse_body(event)
        missing = [c for c in NUMERIC_FEATURES if c not in raw]
        if missing:
            return _response(400, {"error": "missing_features", "missing": missing})

        proba = _MODEL.predict_proba(build_feature_vector(raw))[0]
        return _response(200, build_response(proba))
    except Exception as exc:
        return _response(500, {"error": "prediction_error", "detail": str(exc)})
