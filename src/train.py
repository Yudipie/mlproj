"""Train and compare Logistic Regression, XGBoost, and a small PyTorch MLP
on the Telco churn dataset using 5-fold stratified cross-validation, then
fit the best model on the full training set and save it.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluate import (
    compute_metrics,
    error_analysis,
    plot_confusion_matrix,
    print_metrics_table,
    summarize_cv_results,
)
from src.preprocessing import load_and_preprocess

RANDOM_STATE = 42
N_SPLITS = 5
DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"


# ---------------------------------------------------------------------------
# PyTorch MLP
# ---------------------------------------------------------------------------
class ChurnMLP(nn.Module):
    """Small 3-hidden-layer MLP for binary churn classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, epochs: int = 60,
              lr: float = 1e-3, seed: int = RANDOM_STATE) -> ChurnMLP:
    torch.manual_seed(seed)
    model = ChurnMLP(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        optimizer.step()

    return model


def predict_mlp(model: ChurnMLP, X: np.ndarray, threshold: float = 0.5):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        proba = torch.sigmoid(logits).numpy()
    return (proba >= threshold).astype(int), proba


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def cross_validate_all(X: pd.DataFrame, y: pd.Series) -> dict:
    """5-fold stratified CV for all three models. Returns
    {model_name: {metric: [fold_scores]}}.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    X_arr, y_arr = X.values, y.values

    results = {
        "LogisticRegression": {"precision": [], "recall": [], "f1": [], "roc_auc": []},
        "XGBoost": {"precision": [], "recall": [], "f1": [], "roc_auc": []},
        "PyTorch MLP": {"precision": [], "recall": [], "f1": [], "roc_auc": []},
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), start=1):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_scaled, X_val_scaled = scaler.transform(X_tr), scaler.transform(X_val)

        # Logistic Regression (needs scaled features)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        lr.fit(X_tr_scaled, y_tr)
        pred, proba = lr.predict(X_val_scaled), lr.predict_proba(X_val_scaled)[:, 1]
        for k, v in compute_metrics(y_val, pred, proba).items():
            results["LogisticRegression"][k].append(v)

        # XGBoost (tree-based, no scaling needed)
        neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        xgb_clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=neg / pos, eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
        xgb_clf.fit(X_tr, y_tr)
        pred, proba = xgb_clf.predict(X_val), xgb_clf.predict_proba(X_val)[:, 1]
        for k, v in compute_metrics(y_val, pred, proba).items():
            results["XGBoost"][k].append(v)

        # PyTorch MLP (needs scaled features)
        mlp = train_mlp(X_tr_scaled, y_tr)
        pred, proba = predict_mlp(mlp, X_val_scaled)
        for k, v in compute_metrics(y_val, pred, proba).items():
            results["PyTorch MLP"][k].append(v)

        print(f"Fold {fold}/{N_SPLITS} done.")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    X, y = load_and_preprocess(DATA_PATH)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("Running 5-fold stratified cross-validation on training set...")
    cv_results = cross_validate_all(X_train, y_train)
    cv_summary = summarize_cv_results(cv_results)
    print_metrics_table(cv_summary, title="5-Fold Stratified CV Results (mean +/- std)")
    cv_summary.to_csv(os.path.join(REPORTS_DIR, "cv_results.csv"))

    # --- Fit final models on the full training set, evaluate on the held-out test set ---
    scaler = StandardScaler().fit(X_train.values)
    X_train_scaled = scaler.transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    final_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    final_lr.fit(X_train_scaled, y_train.values)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    final_xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=neg / pos, eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    final_xgb.fit(X_train.values, y_train.values)

    final_mlp = train_mlp(X_train_scaled, y_train.values)

    test_results = {}
    preds = {}

    pred, proba = final_lr.predict(X_test_scaled), final_lr.predict_proba(X_test_scaled)[:, 1]
    test_results["LogisticRegression"] = compute_metrics(y_test.values, pred, proba)
    preds["LogisticRegression"] = pred

    pred, proba = final_xgb.predict(X_test.values), final_xgb.predict_proba(X_test.values)[:, 1]
    test_results["XGBoost"] = compute_metrics(y_test.values, pred, proba)
    preds["XGBoost"] = pred

    pred, proba = predict_mlp(final_mlp, X_test_scaled)
    test_results["PyTorch MLP"] = compute_metrics(y_test.values, pred, proba)
    preds["PyTorch MLP"] = pred

    test_df = pd.DataFrame(test_results).T
    print_metrics_table(test_df, title="Held-Out Test Set Results")
    test_df.to_csv(os.path.join(REPORTS_DIR, "test_results.csv"))

    # --- Pick the best model by ROC-AUC (threshold-independent ranking quality) ---
    best_model_name = test_df["roc_auc"].idxmax()
    print(f"\nBest model by ROC-AUC: {best_model_name}")

    best_model = {"LogisticRegression": final_lr, "XGBoost": final_xgb, "PyTorch MLP": final_mlp}[best_model_name]
    best_pred = preds[best_model_name]

    cm = plot_confusion_matrix(
        y_test.values, best_pred, best_model_name,
        os.path.join(REPORTS_DIR, "confusion_matrix_best_model.png"),
    )
    print("\nConfusion matrix:\n", cm)

    sample, segment_breakdown = error_analysis(X_test, y_test, best_pred, n_examples=10)
    sample.to_csv(os.path.join(REPORTS_DIR, "misclassified_examples.csv"))
    segment_breakdown.to_csv(os.path.join(REPORTS_DIR, "error_segment_breakdown.csv"))
    print("\nError rate by segment (tenure bucket x contract):\n", segment_breakdown.round(3))

    # --- Save the best model ---
    if best_model_name == "PyTorch MLP":
        torch.save(final_mlp.state_dict(), os.path.join(MODELS_DIR, "best_model.pt"))
        joblib.dump({"input_dim": X_train.shape[1]}, os.path.join(MODELS_DIR, "best_model_meta.joblib"))
    else:
        joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))

    # Scaler + feature names are needed at inference time for LR and MLP,
    # and feature alignment is needed for all three.
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_model_name)

    print(f"\nSaved best model ({best_model_name}) to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
