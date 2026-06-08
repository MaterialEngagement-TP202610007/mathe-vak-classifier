"""
preprocess.py
=============
Pipeline de preprocesamiento para el clasificador VAK (Visual / Auditivo / Kinestesico).

Este modulo es la UNICA fuente de verdad del esquema de features. Es importado por
`train.py` y `evaluate.py`, y su logica de inferencia (`build_feature_vector`) se replica
de forma identica en `lambda/lambda_function.py`.

REGLA CRITICA (ver Notion): el orden del vector de features debe ser identico en
entrenamiento y en prediccion. Cualquier cambio de orden produce predicciones
incorrectas SIN generar errores. Por eso el orden se deriva siempre de las constantes
NUMERIC_FEATURES + CATEGORICAL_FEATURES/CATEGORICAL_VALUES y se persiste en
model_metadata.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Esquema del dataset
# --------------------------------------------------------------------------- #

ID_COLUMN = "student_id"
TARGET_COLUMN = "target_vak_label"

# Variables numericas (se normalizan con StandardScaler).
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

# Variables categoricas (se codifican one-hot con valores FIJOS y conocidos).
# El listado de valores se fija aqui para garantizar reproducibilidad entre
# entrenamiento e inferencia, incluso si un batch no contiene todas las categorias.
CATEGORICAL_VALUES = {
    "options_selection": ["A", "B", "C", "D"],
    "preferred_content_type": ["audio", "exercise", "image", "text", "video"],
    "navigation_sequence": ["back_and_forth", "linear", "random", "skip_and_return"],
    "academic_grade": [
        "1ro Primaria",
        "2do Primaria",
        "3ro Primaria",
        "4to Primaria",
        "5to Primaria",
        "6to Primaria",
        "1ro Secundaria",
        "2do Secundaria",
        "3ro Secundaria",
        "4to Secundaria",
        "5to Secundaria",
    ],
}
CATEGORICAL_FEATURES = list(CATEGORICAL_VALUES.keys())

# Mapeo explicito de la etiqueta VAK -> entero (Visual=0, Auditivo=1, Kinestesico=2).
# Coincide con el orden indicado en el Notion.
VAK_CLASSES = ["Visual", "Auditivo", "Kinestesico"]


def get_feature_columns() -> list[str]:
    """Devuelve el orden CANONICO de columnas del vector de features.

    numericas (en orden) + one-hot de cada categorica (en orden de CATEGORICAL_VALUES).
    """
    columns = list(NUMERIC_FEATURES)
    for cat in CATEGORICAL_FEATURES:
        for value in CATEGORICAL_VALUES[cat]:
            columns.append(f"{cat}__{value}")
    return columns


# --------------------------------------------------------------------------- #
# Carga y limpieza
# --------------------------------------------------------------------------- #

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Carga el dataset desde CSV y aplica limpieza basica."""
    df = pd.read_csv(csv_path)

    expected = set([ID_COLUMN, TARGET_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"El dataset no contiene las columnas esperadas: {sorted(missing)}")

    # Limpieza: eliminar duplicados exactos y filas sin etiqueta.
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET_COLUMN])

    # Imputacion simple de nulos numericos con la mediana (robusto a outliers).
    for col in NUMERIC_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Nulos categoricos -> primera categoria (valor por defecto determinista).
    for col in CATEGORICAL_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(CATEGORICAL_VALUES[col][0])

    return df.reset_index(drop=True)


def _one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding determinista basado en CATEGORICAL_VALUES.

    Garantiza que SIEMPRE existan todas las columnas one-hot esperadas, en el mismo
    orden, sin importar que categorias aparezcan en este subconjunto de datos.
    """
    frames = [df[NUMERIC_FEATURES].copy()]
    for cat in CATEGORICAL_FEATURES:
        for value in CATEGORICAL_VALUES[cat]:
            col_name = f"{cat}__{value}"
            frames.append((df[cat] == value).astype(int).rename(col_name))
    out = pd.concat(frames, axis=1)
    # Reordenar al orden canonico por seguridad.
    return out[get_feature_columns()]


# --------------------------------------------------------------------------- #
# Preprocesamiento de entrenamiento
# --------------------------------------------------------------------------- #

def preprocess_training(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Construye matrices de entrenamiento/prueba, scaler y label encoder.

    Pasos (segun Notion - Fase 2):
      1. One-hot de categoricas + features numericas.
      2. LabelEncoder de la etiqueta VAK.
      3. train_test_split estratificado 80/20.
      4. StandardScaler ajustado SOLO con el train, aplicado a columnas numericas.

    Returns dict con X_train, X_test, y_train, y_test, scaler, label_encoder,
    feature_columns.
    """
    feature_columns = get_feature_columns()

    X_full = _one_hot(df)

    # Codificacion de la etiqueta con orden FIJO Visual=0, Auditivo=1, Kinestesico=2.
    # OJO: LabelEncoder.fit() ordena alfabeticamente; para respetar el mapeo del Notion
    # fijamos classes_ manualmente y codificamos por indice en VAK_CLASSES.
    # inverse_transform(classes_[i]) sigue funcionando para decodificar.
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(VAK_CLASSES)
    class_to_idx = {label: idx for idx, label in enumerate(VAK_CLASSES)}
    y_full = df[TARGET_COLUMN].astype(str).map(class_to_idx)
    if y_full.isna().any():
        bad = sorted(df.loc[y_full.isna(), TARGET_COLUMN].unique())
        raise ValueError(f"Etiquetas VAK no reconocidas en el dataset: {bad}")
    y_full = y_full.astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X_full.values.astype(float),
        y_full,
        test_size=test_size,
        random_state=random_state,
        stratify=y_full,
    )

    # StandardScaler solo sobre las columnas numericas (las primeras N).
    n_numeric = len(NUMERIC_FEATURES)
    scaler = StandardScaler()
    X_train[:, :n_numeric] = scaler.fit_transform(X_train[:, :n_numeric])
    X_test[:, :n_numeric] = scaler.transform(X_test[:, :n_numeric])

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
    }


# --------------------------------------------------------------------------- #
# Transformacion de inferencia (replicada en Lambda)
# --------------------------------------------------------------------------- #

def build_feature_vector(raw: dict, scaler: StandardScaler) -> np.ndarray:
    """Convierte un dict de features crudas en un vector listo para `model.predict`.

    `raw` debe contener las claves de NUMERIC_FEATURES y CATEGORICAL_FEATURES.
    Devuelve un np.ndarray de forma (1, n_features) en el orden canonico, con las
    columnas numericas ya normalizadas por `scaler`.

    Esta funcion es la referencia que `lambda_function.py` replica byte a byte.
    """
    n_numeric = len(NUMERIC_FEATURES)

    numeric = np.array([float(raw[col]) for col in NUMERIC_FEATURES], dtype=float)
    numeric = scaler.transform(numeric.reshape(1, -1))[0]

    onehot = []
    for cat in CATEGORICAL_FEATURES:
        value = str(raw.get(cat, ""))
        for candidate in CATEGORICAL_VALUES[cat]:
            onehot.append(1.0 if value == candidate else 0.0)

    vector = np.concatenate([numeric, np.array(onehot, dtype=float)])
    return vector.reshape(1, -1)


if __name__ == "__main__":
    # Smoke test: carga el dataset y muestra el esquema resultante.
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "data", "dataset_simulado.csv")
    data = load_dataset(path)
    prep = preprocess_training(data)
    print(f"Registros cargados      : {len(data)}")
    print(f"Total features          : {len(prep['feature_columns'])}")
    print(f"X_train shape           : {prep['X_train'].shape}")
    print(f"X_test shape            : {prep['X_test'].shape}")
    print(f"Clases (LabelEncoder)   : {list(prep['label_encoder'].classes_)}")
    print(json.dumps(prep["feature_columns"], indent=2, ensure_ascii=False))
