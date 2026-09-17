# Results — Model Training & Evaluation

## Setup

- Features: 46 columns from `src/preprocessing.py` (`load_and_preprocess`).
- Split: stratified 80/20 train/test (`random_state=42`), preserving the
  ~73.5% / 26.5% class balance in both splits.
- Models compared: Logistic Regression (`class_weight="balanced"`),
  XGBoost (`scale_pos_weight` set to the train-fold class ratio), and a
  small PyTorch MLP (64 → 32 → 16 → 1, ReLU + dropout).
- Cross-validation: 5-fold stratified CV on the training set only (test set
  held out entirely until final evaluation), scaler refit per fold to avoid
  leakage.

## 5-Fold Stratified CV Results (training set, mean ± std)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.518 ± 0.016 | 0.797 ± 0.037 | 0.628 ± 0.021 | 0.846 ± 0.011 |
| XGBoost | 0.535 ± 0.009 | 0.761 ± 0.025 | 0.628 ± 0.015 | 0.840 ± 0.010 |
| PyTorch MLP | 0.640 ± 0.024 | 0.535 ± 0.026 | 0.583 ± 0.020 | 0.836 ± 0.011 |

## Held-Out Test Set Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | 0.499 | **0.791** | 0.612 | **0.842** |
| XGBoost | 0.519 | 0.770 | **0.620** | 0.835 |
| PyTorch MLP | **0.621** | 0.535 | 0.575 | 0.825 |

CV and test results agree closely (no meaningful overfitting to the CV
folds), and all three models land in a fairly tight ROC-AUC band
(0.82–0.85) — the dataset's signal is largely linear/additive (contract
type, tenure), which is why Logistic Regression is competitive with the
more complex models rather than trailing them.

## Confusion Matrix — Logistic Regression (test set, n=1,409)

| | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 738 | 297 |
| **Actual: Churn** | 78 | 296 |

Out of 374 actual churners in the test set, the model catches 296 (79.1%
recall) at the cost of 297 false alarms among the 1,035 non-churners
(28.7% false-positive rate).

## Error Analysis

Misclassification rate broken down by `tenure_bucket × contract` segment
(see `reports/error_segment_breakdown.csv` for the full table):

| Segment | Error Rate | n |
|---|---|---|
| 49+ months, Month-to-month | 43.8% | 64 |
| 13-24 months, Month-to-month | 42.7% | 157 |
| 25-48 months, Month-to-month | 41.4% | 152 |
| 0-12 months, Month-to-month | 38.8% | 400 |
| 25-48 months, One year | 11.4% | 105 |
| 49+ months, Two year | 2.7% | 258 |
| 0-12 / 13-24 months, Two year | 0.0% | 13 / 18 |

**Finding: errors cluster almost entirely in `Month-to-month` contracts**,
regardless of tenure — error rates there sit around 40% across every
tenure bucket, while `Two year` contracts are predicted almost perfectly
(0–4% error). This makes sense: `Contract` is the single strongest
correlate of churn found in EDA, but it isn't a *perfect* predictor —
plenty of month-to-month customers stay, and plenty of long-tenure
customers still churn, so this is exactly the segment where the other,
weaker features (pricing, service mix, payment method) have to do the
discriminating work and the model has the least to go on. This is a real
finding, not a modeling artifact: it says month-to-month customers are a
fundamentally harder-to-predict population, and a retention team should
expect the model's flags to be noisier (more false positives per true
positive) specifically within that segment.

## Model Choice Justification

**Logistic Regression was selected as the best model, evaluated primarily
on ROC-AUC (0.842 on the test set).** ROC-AUC was chosen as the primary
metric — rather than accuracy or a threshold-locked F1 — because the
intended use case is a *ranked risk list* a retention team works down with
a limited budget, and ROC-AUC measures ranking quality independent of
where a cutoff is later drawn. Accuracy would be actively misleading here:
a model that always predicts "no churn" scores ~73.5% accuracy while
catching zero at-risk customers, which is worse than useless for this
business problem.

Among the three models, Logistic Regression edged out XGBoost on ROC-AUC
(0.842 vs. 0.835) and clearly beat it on **recall** (0.791 vs. 0.770) — and
recall is the metric with the most direct business cost here: a false
negative is a churner who leaves with no retention attempt at all (fully
lost revenue), while a false positive only costs one unnecessary retention
offer to a customer who would have stayed anyway (comparatively cheap).
The PyTorch MLP posted the best precision (0.621) but at the cost of
missing nearly half of all actual churners (0.535 recall) — the wrong
tradeoff when the cost of a miss so clearly outweighs the cost of a false
alarm. Logistic Regression is also the simplest and most interpretable of
the three (coefficients map directly to feature effects), which is a
genuine secondary advantage for a retention team that will want to
understand *why* a customer was flagged, not just that they were.
