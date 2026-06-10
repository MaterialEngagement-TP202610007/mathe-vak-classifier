"""Local HTTP server to test the VAK classifier with Postman or curl (no AWS).

Loads the model from models/ and returns the exact same JSON contract as
lambda_function.py. Standard library only.

Usage:
    python serve_local.py [--port 8000] [--host 127.0.0.1]

Endpoints:
    GET  /         healthcheck
    POST /predict  features in the JSON body (root or under "features")

Example body:
    {
      "visual_score": 7, "auditory_score": 2, "kinesthetic_score": 1,
      "response_consistency": 0.84, "avg_response_time": 18.6,
      "total_changes": 2, "total_backtracks": 1
    }
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib

from preprocess import NUMERIC_FEATURES, VAK_DISPLAY, build_feature_vector, build_response

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

_MODEL = None
_SCALER = None
_META = None


def _load_artifacts():
    global _MODEL, _SCALER, _META
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), encoding="utf-8") as fh:
        _META = json.load(fh)
    _MODEL = joblib.load(os.path.join(MODELS_DIR, _META["model_file"]))
    _SCALER = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"Artifacts loaded from {MODELS_DIR}")
    print(f"  model   : {_META['model_file']}")
    print(f"  features: {_META['n_features']} -> {NUMERIC_FEATURES}")


def _predict(raw: dict) -> dict:
    proba = _MODEL.predict_proba(build_feature_vector(raw, _SCALER))[0]
    return build_response(proba)


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/predict"):
            self._send(200, {
                "status": "ok",
                "modelo": _META["model_file"],
                "clases": VAK_DISPLAY,
                "features": NUMERIC_FEATURES,
                "hint": "POST to /predict with the features in the JSON body.",
            })
        else:
            self._send(404, {"error": "ruta_no_encontrada", "path": self.path})

    def do_POST(self):
        if self.path.rstrip("/") != "/predict":
            self._send(404, {"error": "ruta_no_encontrada", "path": self.path})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw_body or b"{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"error": "json_invalido", "detail": str(exc)})
            return

        raw = data.get("features", data)
        missing = [c for c in NUMERIC_FEATURES if c not in raw]
        if missing:
            self._send(400, {"error": "features_faltantes", "missing": missing})
            return

        try:
            self._send(200, _predict(raw))
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": "error_prediccion", "detail": str(exc)})

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} - {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="Local VAK inference server.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    _load_artifacts()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\nListening on http://{args.host}:{args.port}")
    print(f"  GET  /         healthcheck")
    print(f"  POST /predict  prediction")
    print("Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
