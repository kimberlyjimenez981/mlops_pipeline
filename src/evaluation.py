"""Model evaluation functions for ML pipeline."""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred, average: str = "weighted") -> Dict[str, float]:
    """Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multiclass metrics

    Returns:
        metrics: Dictionary mapping metric name to value
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    logger.info(f"Metrics calculated: {metrics}")
    return metrics


def generate_evaluation_report(
    y_true, y_pred, target_names: Optional[List[str]] = None
) -> Dict:
    """Generate a detailed evaluation report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Class names for display

    Returns:
        report: Dictionary with full evaluation details
    """
    metrics = calculate_metrics(y_true, y_pred)

    clf_report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred).tolist()

    report = {
        "summary_metrics": metrics,
        "classification_report": clf_report,
        "confusion_matrix": cm,
    }

    logger.info("Evaluation report generated")
    return report


def save_evaluation_report(report: Dict, path: str) -> None:
    """Save evaluation report to disk as JSON.

    Args:
        report: Evaluation report dictionary
        path: File path to save report
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved to {path}")
