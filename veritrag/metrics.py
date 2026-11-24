
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)


LABELS = [
    "Supported",
    "Contradicted",
    "Unsupported"
]


def results_to_dataframe(results):

    rows = []

    for result in results:

        rows.append({

            "Claim": result["claim"],

            "Status": result["status"],

            "NLI": result["nli_label"],

            "Confidence": result["confidence"],

            "Similarity": result["similarity"],

            "Numeric Check": result["numeric_check"],

            "Citation": result["citation_status"]

        })

    return pd.DataFrame(rows)


def calculate_metrics(y_true, y_pred):

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    return {

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "confusion_matrix": matrix,

        "labels": LABELS,

        "per_class": {

            label: {

                "precision": precision[i],

                "recall": recall[i],

                "f1": f1[i],

                "support": support[i]

            }

            for i, label in enumerate(LABELS)

        }

    }
