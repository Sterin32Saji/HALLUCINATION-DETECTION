# Next Steps After Preparing The 300-Question Sets

Once these five files are complete and aligned:

- `150_question_from_each/baseline_score_filtered.jsonl`
- `150_question_from_each/ragas_hallucination_results_filtered.jsonl`
- `150_question_from_each/selfcheck_dataset_filtered.jsonl`
- `150_question_from_each/similarity_dataset_filtered.jsonl`
- `150_question_from_each/human_label.jsonl`

the next step is not to label more data. The next step is to turn them into one clean evaluation dataset and run the comparison in a controlled way.

The main implementation now lives in:

- `300_question_from_each/code/pipeline.py`

All generated outputs are written under:

- `300_question_from_each/analysis_results`

## 1. Verify Alignment Across All Files

Make sure all five files refer to the same 300 question IDs.

Checks to perform:

- each file should contain 300 records
- no duplicate `id` values inside a file
- the same `id` set should appear across baseline, ragas, selfcheck, similarity, and human label files
- `human_label.jsonl` should be treated as the ground-truth file

If one method is missing an ID that exists in the human file, fix that before continuing.

## 2. Build One Merged Master Dataset

Create a merged file where each row contains:

- shared metadata: `id`, `question`, `answer`, `evidence`, `question_type`, `difficulty`, `chunk_id`, `source_doc`, `rag_answer`
- baseline output: `hallucination`
- ragas output: `faithfulness` or derived label
- selfcheck output: score or derived label
- similarity output: score or derived label
- human annotation: `human_label`
- optional human annotation detail: `hallucination_type`, `reason` (don't do this)

This merged file becomes the main evaluation table for the dissertation.

## 3. Standardize Each Method Into A Comparable Prediction

Your four methods must produce comparable outputs against the human label.

Typical setup:

- baseline: already binary
- custom detector: binary prediction and optional probability
- ragas: convert score to binary using a fixed threshold
- selfcheck: convert score to binary using a fixed threshold
- similarity: convert score to binary using a fixed threshold

Important:

- choose thresholds once and document them clearly
- do not tune thresholds on the final test set only to improve results

## 4. Split Data Properly For The Custom Model

If you are retraining the custom detector with the 300-question human-labelled set, use a proper split.

Recommended split:

- 70% train
- 15% validation
- 15% test

For 300 examples, that gives approximately:

- 210 train
- 45 validation
- 45 test

Use stratification by `human_label`. If possible, also preserve balance across difficulty levels.

## 5. Retrain The Custom Detector

Retrain the custom model using the expanded human-labelled dataset.

Expected actions:

- rebuild features for all 300 labelled records
- train on the training split
- use the validation split for threshold or model selection if needed
- evaluate once on the held-out test split
- save the updated model, metrics, confusion matrix, and feature importance

This becomes the strongest version of the custom detector in the dissertation.

## 6. Compare All Methods On The Same Held-Out Test Set

Run every method on the exact same test examples:

- baseline
- ragas
- selfcheck
- similarity
- custom detector

Report at minimum:

- accuracy
- precision
- recall
- F1-score
- confusion matrix

If possible, also report ROC-AUC for methods that produce scores or probabilities.

## 7. Run Statistical Analysis

After computing final predictions on the held-out set:

- run bootstrap confidence intervals for key metrics such as F1
- run McNemar's test for paired classifier comparison

This helps you justify whether the custom detector is only numerically better or whether the difference is statistically convincing.

## 8. Produce Final Outputs For The Dissertation

Prepare these final artifacts:

- merged master dataset
- final model artifact
- evaluation tables
- statistical test results
- final charts for metrics and confusion matrices

## 9. Final Dissertation Framing

The cleanest argument is:

- human labels are the gold standard
- the 300-question dataset is the core benchmark
- all methods are evaluated on the same held-out subset
- the custom detector is the project's main contribution
- the baseline, ragas, selfcheck, and similarity methods are comparison methods

## Suggested Immediate Order

1. Verify all IDs and counts across the five files.
2. Run `300_question_from_each/code/pipeline.py` to build the merged dataset and outputs.
3. Retrain the custom detector using the 300 labelled examples.
4. Evaluate all methods on the same held-out test split.
5. Run bootstrap and McNemar analysis.
6. Write the final results and discussion from those outputs.