"""AWS Lambda VAK inference handler.

Cold start downloads the booster and the scaler parameters from S3 (cached in
/tmp and module globals while warm). Each POST parses the feature dict, scales
it, predicts and returns the JSON contract with per-class confidence and profile
type.

The deployment package ships neither scikit-learn nor joblib: together they push
the unzipped package past Lambda's 250MB limit. Instead the model travels in
XGBoost's native JSON format and the StandardScaler as plain mean/scale vectors,
applied here as (x - mean) / scale. verify_parity.py proves the two paths agree.

CRITICAL: feature order and the response logic below must stay byte-identical to
preprocess.py. Lambda cannot import preprocess.py (no pandas in the package).
The feature order is checked against scaler_params.json at load time, so a
divergence fails loudly instead of silently mispredicting.

Env vars: S3_BUCKET, S3_PREFIX, MODEL_FILE, SCALER_FILE.
"""

from __future__ import annotations

import json
import os

import boto3
import numpy as np
import xgboost as xgb

NUMERIC_FEATURES = [
    "visual_score",
    "auditory_score",
    "kinesthetic_score",
    "response_consistency",
    "avg_response_time",
    "total_changes",
    "total_backtracks",
]

VAK_DISPLAY = ["Visual", "Auditivo", "Kinestesico"]

CLEAR_MARGIN = 0.30
MIXED_MARGIN = 0.12

S3_BUCKET = os.environ.get("S3_BUCKET", "material-engagement-models")
S3_PREFIX = os.environ.get("S3_PREFIX", "vak/v1").strip("/")
MODEL_FILE = os.environ.get("MODEL_FILE", "vak_model_v1.json")
SCALER_FILE = os.environ.get("SCALER_FILE", "scaler_params.json")

_BOOSTER = None
_MEAN = None
_SCALE = None


def _download(s3, filename):
    key = f"{S3_PREFIX}/{filename}" if S3_PREFIX else filename
    local = os.path.join("/tmp", filename)
    if not os.path.exists(local):
        s3.download_file(S3_BUCKET, key, local)
    return local


def _load_artifacts():
    global _BOOSTER, _MEAN, _SCALE
    if _BOOSTER is not None:
        return
    s3 = boto3.client("s3")

    booster = xgb.Booster()
    booster.load_model(_download(s3, MODEL_FILE))

    with open(_download(s3, SCALER_FILE), encoding="utf-8") as fh:
        params = json.load(fh)
    if params["features"] != NUMERIC_FEATURES:
        raise ValueError(
            f"feature order mismatch: {SCALER_FILE} has {params['features']}, "
            f"handler expects {NUMERIC_FEATURES}"
        )

    _MEAN = np.asarray(params["mean"], dtype=float)
    _SCALE = np.asarray(params["scale"], dtype=float)
    _BOOSTER = booster


def build_feature_vector(raw: dict) -> np.ndarray:
    numeric = np.array([float(raw[col]) for col in NUMERIC_FEATURES], dtype=float)
    return ((numeric - _MEAN) / _SCALE).reshape(1, -1)


def build_response(proba) -> dict:
    proba = np.asarray(proba, dtype=float)
    order = np.argsort(proba)[::-1]
    top, second = int(order[0]), int(order[1])

    gap = proba[top] - proba[second]
    tipo = "claro" if gap >= CLEAR_MARGIN else "tendencia" if gap >= MIXED_MARGIN else "mixto"

    confianza = {VAK_DISPLAY[i]: round(float(proba[i]) * 100, 2) for i in range(len(VAK_DISPLAY))}
    return {
        "estilo_predominante": VAK_DISPLAY[top],
        "estilo_secundario": VAK_DISPLAY[second],
        "confianza": confianza,
        "confianza_predominante": confianza[VAK_DISPLAY[top]],
        "tipo_perfil": tipo,
        "es_perfil_mixto": tipo == "mixto",
        "clasificador_tipo": "xgboost",
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
    except Exception as exc:  # noqa: BLE001
        return _response(503, {"error": "modelo_no_disponible", "detail": str(exc)})

    try:
        raw = _parse_body(event)
        missing = [c for c in NUMERIC_FEATURES if c not in raw]
        if missing:
            return _response(400, {"error": "features_faltantes", "missing": missing})

        proba = _BOOSTER.predict(xgb.DMatrix(build_feature_vector(raw)))[0]
        return _response(200, build_response(proba))
    except Exception as exc:  # noqa: BLE001
        return _response(500, {"error": "error_prediccion", "detail": str(exc)})
