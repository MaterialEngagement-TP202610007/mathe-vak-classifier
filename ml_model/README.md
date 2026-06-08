# Material Engagement — Módulo ML (Clasificador VAK)

Versión Beta sobre **dataset simulado**. Clasifica el estilo de aprendizaje de un
estudiante en **Visual / Auditivo / Kinestésico** con **XGBoost**, exponiendo el
modelo vía **AWS Lambda + S3** para que el backend Express.js lo consuma.

> El dataset simulado **NO** reemplaza datos reales: demuestra el flujo de extremo a
> extremo. El modelo definitivo se reentrenará con datos de la fase piloto del
> Colegio Claretiano. Este repo **no genera** el dataset (ya existe en `data/`).

---

## Estructura

```
ml_model/
├── data/
│   └── dataset_simulado.csv        # dataset existente (500 registros)
├── models/
│   ├── vak_model_v1.pkl            # modelo XGBoost serializado
│   ├── scaler.pkl                  # StandardScaler (features numéricas)
│   ├── label_encoder.pkl           # LabelEncoder VAK (Visual=0, Auditivo=1, Kinestesico=2)
│   └── model_metadata.json         # versión, métricas, fecha, nº muestras, feature_columns
├── reports/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── classification_report.json
│   └── *.png                       # figuras de exploración
├── notebooks/
│   └── exploracion_dataset.ipynb   # EDA + validación del dataset
├── lambda/
│   ├── lambda_function.py          # handler de predicción en tiempo real
│   └── requirements.txt            # deps del deployment package
├── preprocess.py                   # esquema de features + transformación (fuente de verdad)
├── train.py                        # entrenamiento + serialización
├── evaluate.py                     # métricas + matriz de confusión + feature importance
├── predict_local.py                # prueba local de inferencia (mismo path que Lambda)
├── fallback.py                     # clasificador de respaldo por puntaje simple
├── upload_to_s3.py                 # sube artefactos a S3
├── db.py                           # conexión/carga PostgreSQL (opcional)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Tecnologías

Python 3.10+ · XGBoost 2+ · scikit-learn 1.4+ · pandas · numpy · joblib ·
imbalanced-learn (SMOTE) · matplotlib/seaborn · boto3 · psycopg2 · python-dotenv.

## Puesta en marcha

```bash
cd ml_model
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # rellenar credenciales PostgreSQL / AWS
```

## Flujo de trabajo

```bash
# 1. Entrenar (one-hot + StandardScaler + split estratificado 80/20 + SMOTE + XGBoost)
python train.py                 # usa --no-smote para desactivar el balanceo

# 2. Evaluar (genera reports/ y verifica métricas objetivo)
python evaluate.py

# 3. Probar inferencia local (mismo vector que Lambda)
python predict_local.py --row 0

# 4. Subir artefactos a S3
python upload_to_s3.py --create-bucket
```

## Modelo

- **Algoritmo:** XGBoost `multi:softprob`, `num_class=3`,
  `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`.
- **Features (39):** 15 numéricas normalizadas + one-hot de 4 categóricas
  (`options_selection`, `preferred_content_type`, `navigation_sequence`,
  `academic_grade`).
- **Etiqueta:** Visual=0, Auditivo=1, Kinestésico=2.

### Contrato de predicción (Lambda)

`POST` con `{"features": { ...15 numéricas + 4 categóricas... }}` →

```json
{
  "estilo_predominante": "Visual",
  "confianza": { "Visual": 99.86, "Auditivo": 0.06, "Kinestesico": 0.08 },
  "confianza_predominante": 99.86,
  "clasificador_tipo": "xgboost"
}
```

Si Lambda no está disponible, el backend usa `fallback.classify_by_score`
(`clasificador_tipo = "puntaje_simple"`), que devuelve el mismo contrato.

## Métricas objetivo (beta) vs. obtenidas

| Métrica | Objetivo | Obtenido |
|---|---|---|
| Accuracy global | > 80% | 100% |
| F1 Visual / Auditivo / Kinestésico | > 0.75 | 1.00 / 1.00 / 1.00 |
| Tamaño del `.pkl` | < 10MB | ~0.24MB |

> El 100% se debe a la fuerte separabilidad del dataset **simulado**
> (los puntajes VAK determinan casi por completo la etiqueta). Las métricas reales se
> evaluarán con datos del piloto.

## ⚠️ Reglas críticas

- **El orden del vector de features debe ser idéntico** en `train.py` y en Lambda.
  Se deriva de `preprocess.get_feature_columns()` y se persiste en
  `model_metadata.json → feature_columns`. `lambda_function.py` replica esa lógica.
- El **scaler** y el **label_encoder** viajan junto al modelo; Lambda aplica la misma
  normalización antes de predecir.
- El **fallback por puntaje simple** está siempre disponible como respaldo.
- **Versionar todo:** cada `.pkl` tiene su `model_metadata.json` con accuracy, fecha de
  entrenamiento y nº de muestras.

## Despliegue en AWS (Fase 3)

1. `python upload_to_s3.py --create-bucket` sube los `.pkl` y el metadata.
2. Crear función Lambda (runtime Python 3.10, timeout 30s, memoria 512MB) con el
   deployment package de `lambda/requirements.txt`.
3. Variables de entorno en Lambda: `S3_BUCKET`, `S3_PREFIX`, `MODEL_FILE`.
4. Crear **Function URL** para que Express.js haga `POST`.

## Cobertura de historias de usuario

| HU | Cómo se cubre |
|---|---|
| HU-30, HU-52 | Lambda devuelve estilo + porcentajes de confianza por clase |
| HU-28 | Contrato listo para que el backend genere feedback (Gemini/predefinido) |
| HU-44 | `target_vak_label` etiquetada en el dataset; pipeline de reentrenamiento |
| HU-47, HU-53 | `fallback.py` (puntaje simple) cuando XGBoost/IA no está disponible |
| HU-54 | Lambda responde errores JSON comprensibles (400/500/503) |
| HU-39 | `db.py` carga/almacena en PostgreSQL |
| HU-55 | `.pkl` ligero + caché warm en Lambda para latencia baja |

> **Fase 4 (integración Express.js)** vive en el backend; este módulo entrega el
> modelo desplegable, el contrato JSON y el fallback de referencia que el backend
> reproduce.
