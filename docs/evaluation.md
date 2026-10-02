# GENROGRAM — Evaluation Metrics & Protocol

## Evaluation Metrics

Evaluation is performed exclusively on the untouched test set (15% of GTZAN tracks):
- Overall Accuracy
- Macro Precision, Recall, and F1-Score
- Per-class Precision, Recall, and F1-Score
- Normalized & Unnormalized Confusion Matrix
- Confidence Distribution Analysis

## Outputs

All metrics are serialized to JSON (`models/evaluation_results.json`) and visualization assets are stored as SVG/PNG graphs in `models/plots/`.
