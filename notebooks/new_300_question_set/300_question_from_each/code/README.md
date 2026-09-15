# 300-Question Analysis Code

This folder contains the code used to process the 300-question benchmark set.

## Main Script

- `pipeline.py`: runs the full workflow end to end.

## What The Pipeline Does

1. Verifies that baseline, RAGAS, selfcheck, similarity, and human files contain the same 300 IDs.
2. Builds the merged raw dataset and the binary comparison table.
3. Generates custom hallucination features using the existing feature extractor from the main project.
4. Trains the custom detector with a train/validation/test split.
5. Trains the hybrid detector from baseline, RAGAS, selfcheck, and similarity outputs.
6. Evaluates all methods on the same held-out test set.
7. Writes bootstrap confidence intervals and McNemar results.

## Output Locations

All outputs are written to:

- `analysis_results/data`
- `analysis_results/models`
- `analysis_results/plots`
- `analysis_results/reports`

## Run

```bash
python pipeline.py
```

Run it from this folder or from the project root using the full path.