"""
lambda_function.py
==================
AWS Lambda para predecir el estilo de aprendizaje VAK en tiempo real
(Notion - Fase 3, tarea 2).

Ciclo de vida:
  - Cold start: descarga vak_model_<ver>.pkl, scaler.pkl y label_encoder.pkl desde S3
    (cacheados en /tmp y en memoria global mientras el contenedor siga "warm").
  - POST: parsea las features, normaliza, predice y retorna JSON con la clase y los
    porcentajes de confianza por clase.

REGLA CRITICA: el orden del vector de features DEBE ser identico al de entrenamiento.
Aqui se replica byte a byte la logica de `preprocess.build_feature_vector`. El orden
canonico tambien viene en model_metadata.json (feature_columns) como verificacion.

Variables de entorno esperadas en Lambda:
  S3_BUCKET, S3_PREFIX, MODEL_FILE

Dependencias del deployment package: xgboost, scikit-learn, numpy, joblib, boto3.
"""

from __future__ import annotations

import json
import os

import boto3
import joblib
import numpy as np

# --------------------------------------------------------------------------- #
# Esquema de features (replica EXACTA de preprocess.py — NO modificar el orden)
# --------------------------------------------------------------------------- #

NUMERIC_FEATURES = [
    "visual_score",
    "auditory_score",
    "kinesthetic_score",
    "avg_response_time",
    "total_quest_time",
    "n_response_changes",
    "n_clicks",
    "engagement_level",
    "completion_level",
    "content_repetition",
    "response_consistency",
    "age",
    "number_of_sessions",
    "session_duration",
    "usage_frequency",
]

CATEGORICAL_VALUES = {
    "options_selection": ["A", "B", "C", "D"],
    "preferred_content_type": ["audio", "exercise", "image", "text", "video"],
    "navigation_sequence": ["back_and_forth", "linear", "random", "skip_and_return"],
    "academic_grade": [
        "1ro Primaria", "2do Primaria", "3ro Primaria", "4to Primaria",
        "5to Primaria", "6to Primaria", "1ro Secundaria", "2do Secundaria",
        "3ro Secundaria", "4to Secundaria", "5to Secundaria",
    ],
}
CATEGORICAL_FEATURES = list(CATEGORICAL_VALUES.keys())
VAK_CLASSES = ["Visual", "Auditivo", "Kinestesico"]

# --------------------------------------------------------------------------- #
# Configuracion S3
# --------------------------------------------------------------------------- #

S3_BUCKET = os.environ.get("S3_BUCKET", "material-engagement-models")
S3_PREFIX = os.environ.get("S3_PREFIX", "vak/v1").strip("/")
MODEL_FILE = os.environ.get("MODEL_FILE", "vak_model_v1.pkl")

# Cache global entre invocaciones warm.
_MODEL = None
_SCALER = None
_LABEL_ENCODER = None


def _download(s3, filename):
    key = f"{S3_PREFIX}/{filename}" if S3_PREFIX else filename
    local = os.path.join("/tmp", filename)
    if not os.path.exists(local):
        s3.download_file(S3_BUCKET, key, local)
    return local


def _load_artifacts():
    """Carga modelo, scaler y label encoder (una sola vez por contenedor)."""
    global _MODEL, _SCALER, _LABEL_ENCODER
    if _MODEL is not None:
        return
    s3 = boto3.client("s3")
    _MODEL = joblib.load(_download(s3, MODEL_FILE))
    _SCALER = joblib.load(_download(s3, "scaler.pkl"))
    _LABEL_ENCODER = joblib.load(_download(s3, "label_encoder.pkl"))


def build_feature_vector(raw: dict) -> np.ndarray:
    """Replica EXACTA de preprocess.build_feature_vector (numerico escalado + one-hot)."""
    numeric = np.array([float(raw[col]) for col in NUMERIC_FEATURES], dtype=float)
    numeric = _SCALER.transform(numeric.reshape(1, -1))[0]

    onehot = []
    for cat in CATEGORICAL_FEATURES:
        value = str(raw.get(cat, ""))
        for candidate in CATEGORICAL_VALUES[cat]:
            onehot.append(1.0 if value == candidate else 0.0)

    vector = np.concatenate([numeric, np.array(onehot, dtype=float)])
    return vector.reshape(1, -1)


def _parse_body(event):
    """Extrae el dict de features del evento (Function URL / API GW / invocacion directa)."""
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

        vector = build_feature_vector(raw)
        proba = _MODEL.predict_proba(vector)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = VAK_CLASSES[pred_idx]

        confidences = {VAK_CLASSES[i]: round(float(proba[i]) * 100, 2) for i in range(len(VAK_CLASSES))}

        return _response(200, {
            "estilo_predominante": pred_label,
            "confianza": confidences,            # porcentajes por clase
            "confianza_predominante": confidences[pred_label],
            "clasificador_tipo": "xgboost",
        })
    except Exception as exc:  # noqa: BLE001
        return _response(500, {"error": "error_prediccion", "detail": str(exc)})
