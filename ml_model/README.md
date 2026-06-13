# Material Engagement — ML Module (VAK Classifier)

Beta classifier built on a **simulated dataset**. It labels a student's learning
style as **Visual / Auditory / Kinesthetic** with **XGBoost** and is served via
**AWS Lambda + S3** for an Express.js backend to consume.

> The simulated dataset does **not** replace real data: it demonstrates the
> end-to-end flow. The final model will be retrained with pilot data from the
> Claretiano School. This repo does **not** generate the dataset (it already
> lives in `data/`).

## Structure

```
ml_model/
├── data/
│   └── simulated_dataset.csv       # existing dataset
├── models/
│   ├── vak_model_v1.pkl            # serialized XGBoost model
│   ├── scaler.pkl                  # StandardScaler (numeric features)
│   ├── label_encoder.pkl           # LabelEncoder (Visual=0, Auditory=1, Kinesthetic=2)
│   └── model_metadata.json         # version, metrics, date, n_samples, feature_columns
├── reports/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── classification_report.json
├── notebooks/
│   └── dataset_exploration.ipynb   # EDA + dataset validation
├── lambda/
│   ├── lambda_function.py          # real-time prediction handler
│   └── requirements.txt            # deployment package deps
├── preprocess.py                   # feature schema + transforms (source of truth)
├── train.py                        # training + serialization
├── evaluate.py                     # metrics + confusion matrix + feature importance
├── predict_local.py                # local inference check (same path as Lambda)
├── serve_local.py                  # local HTTP server for Postman/curl
├── fallback.py                     # simple-score fallback classifier
├── upload_to_s3.py                 # uploads artifacts to S3
├── db.py                           # PostgreSQL connection/load (optional)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Stack

Python 3.10+ · XGBoost 2+ · scikit-learn 1.4+ · pandas · numpy · joblib ·
imbalanced-learn (SMOTE) · matplotlib/seaborn · boto3 · psycopg2 · python-dotenv.

## Setup

```bash
cd ml_model
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in PostgreSQL / AWS credentials
```

## Workflow

```bash
# 1. Train (StandardScaler + stratified 80/20 split + SMOTE + XGBoost)
python train.py                 # use --no-smote to disable balancing

# 2. Evaluate (writes reports/ and checks target metrics)
python evaluate.py

# 3. Test inference locally (same vector as Lambda)
python predict_local.py --row 0
python serve_local.py           # HTTP server for Postman/curl

# 4. Upload artifacts to S3
python upload_to_s3.py --create-bucket
```

## Model

- **Algorithm:** XGBoost `multi:softprob`, `num_class=3`, `n_estimators=180`,
  `max_depth=4`, `learning_rate=0.08`.
- **Features (7, numeric, StandardScaled):** `visual_score`, `auditory_score`,
  `kinesthetic_score`, `response_consistency`, `avg_response_time`,
  `total_changes`, `total_backtracks`.
- **Label:** Visual=0, Auditory=1, Kinesthetic=2.

### Prediction contract (Lambda)

`POST` with the 7 features at the root or nested under `"features"`:

```json
{ "features": { "visual_score": 7, "auditory_score": 2, "kinesthetic_score": 1,
  "response_consistency": 0.84, "avg_response_time": 18.6, "total_changes": 2,
  "total_backtracks": 1 } }
```

Response:

```json
{
  "predominant_style": "Visual",
  "secondary_style": "Auditory",
  "confidence": { "Visual": 91.20, "Auditory": 7.45, "Kinesthetic": 1.35 },
  "predominant_confidence": 91.20,
  "profile_type": "clear",
  "is_mixed_profile": false,
  "classifier_type": "xgboost"
}
```

`profile_type` comes from the gap between the top two class probabilities:
gap ≥ 0.30 → `clear`; 0.12–0.30 → `tendency`; < 0.12 → `mixed`.

If Lambda is unavailable, the backend uses `fallback.classify_by_score`
(`classifier_type = "simple_score"`), which returns the same contract.

## Target metrics (beta) vs. obtained

| Metric | Target | Obtained |
|---|---|---|
| Global accuracy | > 80% | ~98% |
| F1 Visual / Auditory / Kinesthetic | > 0.75 | high |
| `.pkl` size | < 10MB | ~0.24MB |

> The high scores come from the strong separability of the **simulated** dataset
> (the VAK scores almost fully determine the label). Real metrics will be measured
> with pilot data.

## Critical rules

- **The feature vector order must be identical** in `train.py` and Lambda. It is
  derived from `preprocess.get_feature_columns()` and persisted in
  `model_metadata.json → feature_columns`. `lambda_function.py` mirrors that logic.
- The **scaler** and **label_encoder** travel with the model; Lambda applies the
  same scaling before predicting.
- The **simple-score fallback** is always available as a backup.
- **Version everything:** each `.pkl` has its `model_metadata.json` with accuracy,
  training date and sample count.

## AWS deployment

1. `python upload_to_s3.py --create-bucket` uploads the `.pkl` files and metadata.
2. Create a Lambda function (Python 3.10 runtime, 30s timeout, 512MB) with the
   deployment package from `lambda/requirements.txt`.
3. Lambda environment variables: `S3_BUCKET`, `S3_PREFIX`, `MODEL_FILE`.
4. Create a **Function URL** so Express.js can `POST` to it.

> Phase 4 (Express.js integration) lives in the backend project; this module
> delivers the deployable model, the JSON contract and the reference fallback.
