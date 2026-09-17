# Model Card — Churn Prediction

## Model

**Logistic Regression** (`sklearn.linear_model.LogisticRegression`,
`class_weight="balanced"`, `max_iter=1000`), trained on 46 one-hot encoded /
numeric features from `src/preprocessing.py`, with features standardized
via `StandardScaler` before fitting.

Saved artifacts (in this directory):
- `best_model.joblib` — the fitted `LogisticRegression` model
- `scaler.joblib` — the `StandardScaler` fit on the training set (required
  at inference time to transform raw features before prediction)
- `feature_names.json` — ordered list of the 46 feature columns the model
  expects, for aligning new input at inference time
- `best_model_name.txt` — the winning model's name, for the serving layer
  to know which loading path to use

## Why This Model Won

Three models were compared with 5-fold stratified cross-validation on the
training set, then re-evaluated on a held-out 20% test set: Logistic
Regression, XGBoost, and a small PyTorch MLP (3 hidden layers). Full
numbers are in [RESULTS.md](../RESULTS.md).

**Selection metric: ROC-AUC.** For a churn-prediction service, the output
that matters is a *ranked risk score* the retention team uses to prioritize
outreach with a limited budget, not a fixed yes/no cutoff — ROC-AUC
measures ranking quality independent of any one threshold, which is a
better fit than accuracy (misleading under class imbalance) or F1 at the
default 0.5 threshold (arbitrary threshold choice this early).

Logistic Regression won on ROC-AUC (0.8416 on the test set vs. 0.8350 for
XGBoost and 0.8246 for the MLP) and also had the best **recall** (0.79 vs.
0.77 for XGBoost and 0.53 for the MLP) — recall matters here because a
missed churner (false negative) is a lost customer with no retention
attempt, while a false positive just costs one unnecessary outreach. The
MLP overfit its higher precision at the cost of much lower recall, missing
nearly half of actual churners — the wrong tradeoff for this use case. On
a small (7K rows), mostly-linear-signal, feature-engineered tabular
dataset, a well-regularized linear model was able to match or beat the
more complex models rather than underfitting relative to them, which is a
known pattern on datasets this size and shape.

## Known Limitations

- Precision is modest (~0.50 at the default 0.5 threshold) — about half of
  customers flagged as at-risk are false positives. This is an acceptable
  tradeoff for high-recall retention targeting but should be tuned (e.g.
  via threshold adjustment or a cost-sensitive objective) if outreach cost
  per customer is high.
- Trained on a single historical snapshot (~7K customers, one point in
  time) — no temporal validation; performance on genuinely new data drift
  is unverified.
