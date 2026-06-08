# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ml_model/` is the ML module of **Material Engagement** — a beta XGBoost classifier that
labels a student's learning style as **Visual / Auditivo / Kinestesico** (VAK) from a
questionnaire + behavioral features. It trains on a **simulated** dataset and is meant to be
served via **AWS Lambda + S3** for an Express.js backend to consume. The data, comments, and
user-facing strings are in **Spanish**; keep that language when editing.

The plan/spec lives in the root file
`notion-Material-Engagement-...md` (the Notion export). When in doubt about scope, phases,
hyperparameters, or target metrics, that file is the source of truth.

## Environment & commands

Python deps are installed in a venv at `ml_model/venv` (built with `python3.13`; Lambda
runtime target is 3.10). Always run from inside `ml_model/`:

```bash
cd ml_model
source venv/bin/activate            # prompt shows (venv); deactivate to exit

python train.py                     # train + serialize to models/
python train.py --no-smote          # disable SMOTE class balancing
python train.py --dataset PATH.csv  # train on a CSV elsewhere (default: data/dataset_simulado.csv)
python evaluate.py                  # metrics + reports/ (loads the saved .pkl; run train.py first)
python predict_local.py --row N     # predict on row N (0-499) of the CSV; sanity-checks inference path
python fallback.py                  # demo the non-ML simple-score fallback
python upload_to_s3.py --create-bucket   # push artifacts to S3 (needs AWS creds in .env)
python db.py --check                # verify PostgreSQL connection (optional)

# Run the exploration notebook end-to-end:
cd notebooks && ../venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace exploracion_dataset.ipynb
```

There is no test suite, linter, or build step. `predict_local.py` and `evaluate.py` are the
de-facto verification tools after any change to the pipeline.

## Architecture: how the pieces fit

The pipeline is a straight line: **CSV → preprocess → XGBoost → .pkl artifacts → Lambda**.

- **`preprocess.py` is the single source of truth for the feature schema.** It defines
  `NUMERIC_FEATURES` (15, StandardScaled), `CATEGORICAL_VALUES` (4 features, one-hot encoded
  with a FIXED value list), the VAK label order, and `get_feature_columns()` which produces the
  **canonical 39-column feature order**. `train.py` and `evaluate.py` import from it.
- **`train.py`** does: load → one-hot + scale + stratified 80/20 split → SMOTE (training set
  only) → `XGBClassifier` (`multi:softprob`, `num_class=3`, n_estimators=100, max_depth=4,
  lr=0.1) → serialize `vak_model_v1.pkl`, `scaler.pkl`, `label_encoder.pkl`,
  `model_metadata.json` into `models/`.
- **`evaluate.py`** reloads the saved model, rebuilds the *same* test split (fixed
  `random_state=42`), and writes `reports/` (confusion matrix, feature importance,
  classification report).
- **`lambda/lambda_function.py`** is the production inference path: downloads artifacts from S3
  (cached in `/tmp` + module globals while warm), transforms an incoming feature dict, and
  returns JSON with per-class confidence and `clasificador_tipo: "xgboost"`.
- **`fallback.py`** is a deliberate non-ML classifier (predominant raw VAK score) returning the
  same JSON contract with `clasificador_tipo: "puntaje_simple"`. It exists so the backend always
  has a fallback when Lambda is down. The `clasificador_tipo` field is how callers distinguish
  ML output from fallback output.

## Critical invariants (break these and predictions go silently wrong)

- **Feature order must be byte-identical between training and inference.** It is derived from
  `preprocess.get_feature_columns()` and persisted in `model_metadata.json → feature_columns`.
  `lambda/lambda_function.py` **manually re-declares** `NUMERIC_FEATURES`/`CATEGORICAL_VALUES`
  and replicates `build_feature_vector` (it can't import `preprocess.py` — Lambda ships without
  pandas). **If you change the feature schema in `preprocess.py`, you must mirror the exact same
  change in `lambda_function.py`.**
- **VAK label encoding is forced to Visual=0, Auditivo=1, Kinestesico=2.** `LabelEncoder.fit()`
  would sort alphabetically (Auditivo=0) and break this, so `preprocess.py` sets
  `label_encoder.classes_` manually and encodes via index into `VAK_CLASSES`. Don't replace this
  with a plain `.fit()`.
- **The scaler is fit on numeric columns only** (the first 15 of the 39), then one-hot columns
  are appended unscaled. Inference applies the same partial scaling.

## Dataset gotcha (this has bitten the user)

`train.py` reads **`ml_model/data/dataset_simulado.csv`** and nothing else. To train on new
data, replace *that exact file* (or pass `--dataset`). A second copy may exist under the
repo-root `dataset/` folder — that one is **not** used. To confirm new data was loaded, check the
distribution line `train.py` prints (`Registros: N | Distribucion VAK: {...}`) or
`model_metadata.json → n_samples_total`; do **not** judge by the report PNGs, which look
identical whenever accuracy stays near 100% (the simulated data is highly separable, so ~100%
accuracy is expected, not a bug).

## Not in scope here

Phase 4 (Express.js backend integration, Gemini feedback, PostgreSQL result storage) lives in a
separate backend project. This module delivers the deployable model, the JSON contract, and the
reference fallback only. `db.py` is a thin optional helper, not a running service.
