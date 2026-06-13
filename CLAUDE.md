# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ml_model/` is the ML module of **Material Engagement** — a beta XGBoost classifier that
labels a student's learning style as **Visual / Auditory / Kinesthetic** (VAK) from VAK scores
plus behavioral signals. It trains on a **simulated** dataset and is meant to be served via
**AWS Lambda + S3** for an Express.js backend to consume. **Everything is in English** — code,
comments, the request body and the JSON response keys/values. Keep it that way when editing.

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
python train.py --dataset PATH.csv  # train on a CSV elsewhere (default: data/simulated_dataset.csv)
python evaluate.py                  # metrics + reports/ (loads the saved .pkl; run train.py first)
python predict_local.py --row N     # predict on row N of the CSV; sanity-checks inference path
python serve_local.py               # local HTTP server for Postman/curl (POST /predict)
python fallback.py                  # demo the non-ML simple-score fallback
python upload_to_s3.py --create-bucket   # push artifacts to S3 (needs AWS creds in .env)
python db.py --check                # verify PostgreSQL connection (optional)
```

There is no test suite, linter, or build step. `predict_local.py`, `serve_local.py` and
`evaluate.py` are the de-facto verification tools after any change to the pipeline.

## Architecture: how the pieces fit

The pipeline is a straight line: **CSV → preprocess → XGBoost → .pkl artifacts → Lambda**.

- **`preprocess.py` is the single source of truth for the feature schema and the response
  contract.** It defines `NUMERIC_FEATURES` (the **7 numeric features**, all StandardScaled),
  the class order, profile-type thresholds, and `build_response()` (the JSON contract).
  `train.py` and `evaluate.py` import from it. There are **no categorical / one-hot features**.
- **`train.py`** does: load → scale + stratified 80/20 split → SMOTE (training set only) →
  `XGBClassifier` (`multi:softprob`, `num_class=3`) → serialize `vak_model_v1.pkl`, `scaler.pkl`,
  `label_encoder.pkl`, `model_metadata.json` into `models/`.
- **`evaluate.py`** reloads the saved model, rebuilds the *same* test split (fixed
  `random_state=42`), and writes `reports/` (confusion matrix, feature importance,
  classification report).
- **`lambda/lambda_function.py`** is the production inference path: downloads artifacts from S3
  (cached in `/tmp` + module globals while warm), scales the incoming feature dict and returns
  the JSON contract with `classifier_type: "xgboost"`.
- **`serve_local.py`** is the same inference path but as a local HTTP server (stdlib only, loads
  from `models/` instead of S3). It exists so the model can be tested in Postman/curl before AWS.
- **`fallback.py`** is a deliberate non-ML classifier (predominant raw VAK score) returning the
  same JSON contract with `classifier_type: "simple_score"`. It exists so the backend always
  has a fallback when Lambda is down. The `classifier_type` field is how callers distinguish
  ML output from fallback output.

## Feature schema & JSON contract

**The 7 features (English keys, this exact order):** `visual_score`, `auditory_score`,
`kinesthetic_score`, `response_consistency`, `avg_response_time`, `total_changes`,
`total_backtracks`. The CSV adds `student_id` and the label column `target_vak_label`, whose
values are **English**: `Visual`, `Auditory`, `Kinesthetic`.

**Request** (POST body, features at root or nested under `"features"`):

```json
{ "features": { "visual_score": 7, "auditory_score": 2, "kinesthetic_score": 1,
  "response_consistency": 0.84, "avg_response_time": 18.6, "total_changes": 2,
  "total_backtracks": 1 } }
```

**Response** (all keys and values in English; styles use the class names):

```json
{ "predominant_style": "Visual", "secondary_style": "Auditory",
  "confidence": { "Visual": 91.20, "Auditory": 7.45, "Kinesthetic": 1.35 },
  "predominant_confidence": 91.20, "profile_type": "clear",
  "is_mixed_profile": false, "classifier_type": "xgboost" }
```

`profile_type` is decided by the gap between the top-1 and top-2 class probabilities
(`preprocess.CLEAR_MARGIN` / `MIXED_MARGIN`): gap ≥ 0.30 → `clear`; 0.12–0.30 → `tendency`;
< 0.12 → `mixed` (and `is_mixed_profile: true`).

## Critical invariants (break these and predictions go silently wrong)

- **Feature order must be byte-identical between training and inference.** It is derived from
  `preprocess.get_feature_columns()` and persisted in `model_metadata.json → feature_columns`.
  `lambda/lambda_function.py` **manually re-declares** `NUMERIC_FEATURES` and replicates
  `build_feature_vector` **and** `build_response` (it can't import `preprocess.py` — Lambda ships
  without pandas). **If you change the feature schema or response logic in `preprocess.py`, mirror
  the exact same change in `lambda_function.py`.**
- **One label vocabulary.** `VAK_CLASSES = Visual / Auditory / Kinesthetic` are the CSV values
  and the names used in the JSON response. Index is the encoded class: Visual=0, Auditory=1,
  Kinesthetic=2. `lambda_function.py` and `fallback.py` re-declare the same `VAK_CLASSES`.
- **Class encoding is forced by index, not by `LabelEncoder.fit()`** (which would sort
  alphabetically). `preprocess.py` sets `label_encoder.classes_` manually and maps via
  `VAK_CLASSES`. Don't replace this with a plain `.fit()`.
- **The scaler is fit on all 7 numeric features.** Inference applies the same scaling.

## Dataset gotcha (this has bitten the user)

`train.py` reads **`ml_model/data/simulated_dataset.csv`** and nothing else. To train on new
data, replace *that exact file* (or pass `--dataset`) — keeping the **same 7 feature columns and
the same English label values**. If you add/remove/rename a column you must also update
`NUMERIC_FEATURES` in both `preprocess.py` and `lambda_function.py`. A second copy may exist
under the repo-root `dataset/` folder — that one is **not** used. To confirm new data was loaded,
check the distribution line `train.py` prints (`Rows: N | VAK distribution: {...}`) or
`model_metadata.json → n_samples_total`.

Because the simulated data is highly separable (each style has its own score high and the others
near zero), the model is very confident and rarely outputs `mixed`. The behavioral features
(`avg_response_time`, `total_changes`) mainly break ties when the three scores are close.

## Not in scope here

Phase 4 (Express.js backend integration, Gemini feedback, PostgreSQL result storage) lives in a
separate backend project. This module delivers the deployable model, the JSON contract, and the
reference fallback only. `db.py` is a thin optional helper, not a running service.
