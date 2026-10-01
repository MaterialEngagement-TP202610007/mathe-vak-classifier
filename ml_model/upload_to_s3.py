"""
upload_to_s3.py
===============
Sube los artefactos del modelo a AWS S3 (Notion - Fase 3, tarea 1).

Sube:
  - models/vak_model_<version>.pkl
  - models/scaler.pkl
  - models/label_encoder.pkl
  - models/model_metadata.json

Configuracion via .env:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_REGION, S3_BUCKET, S3_PREFIX

Uso:
    python upload_to_s3.py [--create-bucket]
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

load_dotenv(os.path.join(HERE, ".env"))

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
BUCKET = os.environ.get("S3_BUCKET", "material-engagement-models")
PREFIX = os.environ.get("S3_PREFIX", f"vak/{MODEL_VERSION}").strip("/")
REGION = os.environ.get("AWS_REGION", "us-east-1")

ARTIFACTS = [
    f"vak_model_{MODEL_VERSION}.pkl",
    f"vak_model_{MODEL_VERSION}.json",  # booster loaded by Lambda (no sklearn)
    "scaler.pkl",
    "scaler_params.json",               # mean/scale loaded by Lambda (no sklearn)
    "label_encoder.pkl",
    "model_metadata.json",
]


def ensure_bucket(s3, create: bool):
    import botocore

    try:
        s3.head_bucket(Bucket=BUCKET)
        print(f"Bucket '{BUCKET}' encontrado.")
        return
    except botocore.exceptions.ClientError:
        if not create:
            print(f"[ERROR] El bucket '{BUCKET}' no existe. Usa --create-bucket para crearlo.")
            sys.exit(1)

    print(f"Creando bucket '{BUCKET}' en {REGION}...")
    if REGION == "us-east-1":
        s3.create_bucket(Bucket=BUCKET)
    else:
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )
    print("Bucket creado.")


def main():
    parser = argparse.ArgumentParser(description="Sube artefactos del modelo a S3.")
    parser.add_argument("--create-bucket", action="store_true", help="Crea el bucket si no existe.")
    args = parser.parse_args()

    try:
        import boto3
    except ImportError:
        print("[ERROR] boto3 no esta instalado. pip install boto3")
        sys.exit(1)

    # Validar credenciales basicas.
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        print("[ERROR] Faltan credenciales AWS en .env (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).")
        sys.exit(1)

    # Validar que existan los artefactos.
    missing = [a for a in ARTIFACTS if not os.path.exists(os.path.join(MODELS_DIR, a))]
    if missing:
        print(f"[ERROR] Faltan artefactos en models/: {missing}. Ejecuta train.py primero.")
        sys.exit(1)

    s3 = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )
    ensure_bucket(s3, args.create_bucket)

    for artifact in ARTIFACTS:
        local = os.path.join(MODELS_DIR, artifact)
        key = f"{PREFIX}/{artifact}" if PREFIX else artifact
        print(f"Subiendo {artifact} -> s3://{BUCKET}/{key}")
        s3.upload_file(local, BUCKET, key)

    print("\nUpload completado. Configura en Lambda:")
    print(f"  S3_BUCKET={BUCKET}")
    print(f"  S3_PREFIX={PREFIX}")
    print(f"  MODEL_FILE=vak_model_{MODEL_VERSION}.json")
    print("  SCALER_FILE=scaler_params.json")


if __name__ == "__main__":
    main()
