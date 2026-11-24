import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)


LABELS = ["Supported", "Contradicted", "Unsupported"]


def results_to_dataframe(results):
    columns = [
        "Claim", "Status", "NLI", "Decision Method", "NLI Confidence", "Similarity",
        "Numeric Check", "Citation", "Explanation"
    ]
    rows = []
    for result in results:
        rows.append({
            "Claim": result.get("claim", ""),
            "Status": result.get("status", "Verification Error"),
            "NLI": result.get("nli_label", "error"),
            "Decision Method": result.get("decision_method", "NLI"),
            "NLI Confidence": result.get("confidence", 0.0),
            "Similarity": result.get("similarity", 0.0),
            "Numeric Check": result.get("numeric_check", "Not evaluated"),
            "Citation": result.get("citation_status", "Not evaluated"),
            "Explanation": result.get("explanation", "")
        })
    return pd.DataFrame(rows, columns=columns)


def calculate_metrics(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("No evaluation examples were supplied.")

    unknown_true = sorted(set(y_true) - set(LABELS))
    unknown_pred = sorted(set(y_pred) - set(LABELS))
    if unknown_true or unknown_pred:
        raise ValueError(
            f"Labels must be one of {LABELS}. "
            f"Unknown gold labels: {unknown_true}; unknown predictions: {unknown_pred}."
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
        for i, label in enumerate(LABELS)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(sum(f1) / len(LABELS)),
        "confusion_matrix": matrix,
        "labels": LABELS,
        "per_class": per_class,
        "evaluated": len(y_true)
    }
