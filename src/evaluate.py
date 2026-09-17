"""Reusable evaluation utilities: metrics table, confusion matrix, error analysis."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute precision, recall, F1, and ROC-AUC for one set of predictions."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def summarize_cv_results(cv_results: dict) -> pd.DataFrame:
    """cv_results: {model_name: {metric_name: [fold_scores...]}} -> summary DataFrame
    with mean +/- std per model/metric.
    """
    rows = []
    for model_name, metrics in cv_results.items():
        row = {"model": model_name}
        for metric_name, scores in metrics.items():
            scores = np.array(scores)
            row[f"{metric_name}_mean"] = scores.mean()
            row[f"{metric_name}_std"] = scores.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def print_metrics_table(df: pd.DataFrame, title: str = "Metrics"):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print(df.round(4).to_string())


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return cm


def error_analysis(X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray,
                    n_examples: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sample of misclassified rows, segment-level error-rate breakdown).

    Segments examined: tenure bucket (reconstructed from one-hot columns) and
    contract type (reconstructed from one-hot columns) — the two strongest
    churn correlates found in EDA.
    """
    misclassified_mask = y_test.values != y_pred
    misclassified = X_test.loc[misclassified_mask].copy()
    misclassified["actual"] = y_test.values[misclassified_mask]
    misclassified["predicted"] = y_pred[misclassified_mask]
    sample = misclassified.sample(n=min(n_examples, len(misclassified)), random_state=42)

    # Reconstruct human-readable segment labels from one-hot columns for the
    # error breakdown.
    analysis_df = X_test.copy()
    analysis_df["misclassified"] = misclassified_mask

    tenure_cols = [c for c in analysis_df.columns if c.startswith("tenure_bucket_")]
    contract_cols = [c for c in analysis_df.columns if c.startswith("Contract_")]

    def reconstruct(df, cols, prefix):
        labels = pd.Series("unknown", index=df.index)
        for c in cols:
            label = c[len(prefix):]
            labels.loc[df[c] == 1] = label
        return labels

    analysis_df["tenure_bucket"] = reconstruct(analysis_df, tenure_cols, "tenure_bucket_")
    analysis_df["contract"] = reconstruct(analysis_df, contract_cols, "Contract_")

    segment_breakdown = (
        analysis_df.groupby(["tenure_bucket", "contract"])["misclassified"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "error_rate", "count": "n"})
        .sort_values("error_rate", ascending=False)
    )

    return sample, segment_breakdown
