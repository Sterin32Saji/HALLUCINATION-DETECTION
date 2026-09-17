# Custom Hallucination Detector (Independent)

This module builds a **fully independent hallucination detector** for RAG outputs.
It does **not** use `baseline`, `ragas`, `selfcheck`, or `similarity` detector outputs as model inputs.

It learns directly from your RAG experiment schema:

- `question`
- `answer` (ground truth)
- `evidence` (retrieved context)
- `rag_answer` (generated answer)
- `question_type`
- `difficulty`
- `source_doc`
- `human` (label)

## File Layout

- `feature_extractor.py`: semantic, lexical, technical-consistency, and metadata feature engineering
- `dataset_builder.py`: converts raw dataset (`.jsonl`/`.csv`) into feature dataset (`.csv`)
- `train_detector.py`: trains multiple models, compares them, evaluates, and saves best model
- `predict.py`: runs inference with saved detector
- `utils.py`: shared dataset I/O helpers

## Why This Is Independent And Novel

This detector is not an ensemble over prior detector scores. It constructs a new feature space from:

1. **Semantic consistency** between `answer/evidence` and `rag_answer`
2. **Lexical grounding** and unsupported token analysis
3. **Technical consistency** via generic extraction + cross-text comparison
4. **Metadata priors** (`question_type`, `difficulty`, `source_doc`)

This forms a direct supervised hallucination detector over your annotation labels, suitable for comparative dissertation evaluation.

## Feature Groups And Rationale

### 1) Semantic Features

- `semantic_answer_rag`: cosine similarity between ground truth answer and rag answer
- `semantic_evidence_rag`: cosine similarity between evidence and rag answer

Why useful:
- Hallucinations often reduce semantic alignment with both canonical answer and retrieved context.

### 2) Lexical Features

- `keyword_overlap` (TF-IDF top-term overlap)
- `jaccard_similarity`
- `token_overlap`
- `coverage_ratio`
- `unsupported_word_ratio`
- `length_ratio`
- token counts (`answer/evidence/rag_answer`)

Why useful:
- Hallucinations introduce unsupported vocabulary and drift from source terminology.
- Coverage and unsupported ratios quantify grounding vs invention behavior.

### 3) Technical Consistency Features

Generic extraction (no hardcoded topic-specific entities) using regex + spaCy + numeric/date parsing:

- RFC references
- NIST-style document identifiers
- AWS/Amazon service-style entities
- HTTP-like status codes
- Version patterns
- Acronyms / protocol-like tokens
- named entities
- numbers
- dates

Compared between reference text (`answer + evidence`) and `rag_answer`:

- `entity_overlap`, `entity_missing`, `entity_extra`
- `number_match`, `number_mismatch`, `number_extra`
- `date_match`, `date_mismatch`, `date_extra`
- `technical_term_overlap`, `technical_term_missing`, `technical_term_extra`
- `technical_consistency_mismatch`

Why useful:
- Technical hallucinations frequently appear as wrong numbers, versions, standards, acronyms, or invented entities.
- Match/mismatch signals are interpretable and dissertation-friendly.

### 4) Metadata Features

- `question_type`, `difficulty`, `source_doc` are one-hot encoded in training.

Why useful:
- Hallucination rates are often conditioned by question complexity and domain-specific documents.

## Training Pipeline

`train_detector.py` trains and compares:

- Logistic Regression
- Random Forest
- Gradient Boosting

Selection criterion:
- **Best F1 score** on test set.

Metrics produced:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix plot
- ROC Curve plot (if probability output available)
- Classification Report (JSON)
- Feature importance CSV

Artifacts saved:

- `custom_hallucination_detector.pkl`
- `model_comparison.csv`
- `training_summary.json`
- `confusion_matrix.png`
- `roc_curve.png` (when available)
- `feature_importance.csv`

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Build features from raw dataset:

```bash
python dataset_builder.py \
  --input ../balanced_experiment_dataset-human-label.jsonl \
  --output artifacts/features.csv
```

Train and evaluate:

```bash
python train_detector.py \
  --features artifacts/features.csv \
  --output-dir artifacts
```

Predict on new data:

```bash
python predict.py \
  --model artifacts/custom_hallucination_detector.pkl \
  --input ../balanced_experiment_dataset-human-label.jsonl \
  --output artifacts/predictions.jsonl
```

## Dissertation Positioning Notes

This detector can be reported as a **feature-driven grounded consistency model**.

Suggested comparative framing:

- Baseline methods (RAGAS/SelfCheck/etc.) are detector-specific heuristics.
- Your custom detector is a supervised, multi-signal model trained from labeled RAG behavior.
- It supports ablation (semantic-only vs lexical-only vs technical-only vs full model), interpretability via feature importance, and generalization evaluation across document types.
