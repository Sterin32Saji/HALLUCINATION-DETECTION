# Custom Detector Flow

This document explains how the independent custom hallucination detector works and how to execute the full pipeline end-to-end.

## 1. Objective

Build a **standalone supervised hallucination detector** that predicts `human` labels directly from RAG experiment fields:

- question
- answer
- evidence
- rag_answer
- question_type
- difficulty
- source_doc

This detector does **not** use outputs from baseline, RAGAS, SelfCheckGPT, or retrieval similarity detectors.

## 2. Code Structure

- `utils.py`
  - dataset loading (`.jsonl` / `.csv`)
  - schema checks
  - label normalization (`human`, `human_label`, `label`, `target`, `hallucination` -> `human`)
- `feature_extractor.py`
  - semantic, lexical, technical-consistency, and metadata feature engineering
- `dataset_builder.py`
  - builds engineered feature dataset from raw RAG data
- `train_detector.py`
  - trains and compares Logistic Regression / Random Forest / Gradient Boosting
  - selects best model by F1
  - saves model and evaluation artifacts
- `predict.py`
  - loads saved model and runs inference on new raw data

## 3. End-to-End Execution

Run commands from:

`notebooks/final_all_dataset/custom_detector`

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2: Build Feature Dataset

```bash
python dataset_builder.py \
  --input ./human_label.jsonl \
  --output artifacts/features.csv
```

Expected output:

- `artifacts/features.csv`

### Step 3: Train and Evaluate Models

```bash
python train_detector.py \
  --features artifacts/features.csv \
  --output-dir artifacts
```

This automatically:

1. Splits data into train/test (`train_test_split`, stratified).
2. Encodes metadata (`question_type`, `difficulty`, `source_doc`) with one-hot encoding.
3. Scales numeric features.
4. Trains:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
5. Evaluates each model on test data.
6. Selects best model by F1 score.
7. Saves all artifacts.

### Step 4: Predict with Trained Detector

```bash
python predict.py \
  --model artifacts/custom_hallucination_detector.pkl \
  --input ./human_label.jsonl \
  --output artifacts/predictions.jsonl
```

Expected output:

- `artifacts/predictions.jsonl`
- Includes `custom_prediction` and probabilities (if available).

## 4. Feature Engineering Flow

For each sample:

1. Semantic features
   - cosine(answer, rag_answer)
   - cosine(evidence, rag_answer)
2. Lexical grounding features
   - TF-IDF keyword overlap
   - Jaccard/token overlap
   - coverage ratio
   - unsupported word ratio
   - length ratio and token counts
3. Technical consistency features
   - extract entities, numbers, dates, technical terms (regex + spaCy)
   - compare reference text (`answer + evidence`) vs `rag_answer`
   - compute overlap/missing/extra ratios
4. Metadata features
   - question_type, difficulty, source_doc

Output is one numeric/categorical feature row per sample, then used for supervised training.

## 5. Training/Evaluation Artifacts

Generated in `artifacts/`:

- `features.csv`
- `model_comparison.csv`
- `training_summary.json`
- `custom_hallucination_detector.pkl`
- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.csv`
- `predictions.jsonl`

## 6. Current Run Results (This Workspace)

From the latest training run:

- Best model: `logistic_regression`
- Accuracy: `0.7333`
- Precision: `0.7778`
- Recall: `0.7778`
- F1: `0.7778`
- ROC AUC: `0.7870`

Reference files:

- `artifacts/training_summary.json`
- `artifacts/model_comparison.csv`

## 7. How to Interpret Predictions

- `custom_prediction = 1`: predicted hallucinated
- `custom_prediction = 0`: predicted not hallucinated
- `custom_probability_hallucinated`: confidence for hallucination class (if model supports probabilities)

## 8. Troubleshooting

1. Error: missing label column
   - Ensure dataset includes one of: `human`, `human_label`, `label`, `target`, `hallucination`.
2. spaCy model missing
   - Run: `python -m spacy download en_core_web_sm`
3. Slow first run
   - SentenceTransformer downloads model weights once, then caches locally.
4. HF Hub warning
   - Optional: set `HF_TOKEN` for higher rate limits.

## 9. Dissertation Reporting Guidance

Use this pipeline in your dissertation as an **independent grounded-consistency detector**:

- Not an ensemble of baseline detectors.
- Trained directly from manually labeled RAG data.
- Interpretable via feature importance + technical mismatch features.
- Comparable to baseline methods using the same evaluation metrics (Accuracy/Precision/Recall/F1, confusion matrix, ROC).

## 10. Node Backend + Vanilla Frontend Debug Page

For practical testing/debugging, this project also includes:

- `node_backend/` (Node.js API)
- `serve_predict.py` (Python bridge called by Node)
- `frontend/` (vanilla HTML/CSS/JS page inside `custom_detector`)

### Start Node Backend

From `notebooks/final_all_dataset/custom_detector/node_backend`:

```bash
npm install
npm run dev
```

Default server:

- `http://127.0.0.1:5055`

Node backend now serves the local vanilla frontend from `custom_detector/frontend`.

### Available Node API Endpoints

- `GET /api/health`
- `POST /api/predict` (single JSON sample)
- `POST /api/predict-batch` (array of JSON samples)
- `POST /api/predict-jsonl` (multipart file: `input_file`)

### Open Vanilla Frontend Page

1. Start Node backend from `custom_detector/node_backend`.
2. Open in browser:
   - `http://127.0.0.1:5055/`

This page submits raw sample JSON/JSONL directly to the Node backend and displays the model output for debugging.
