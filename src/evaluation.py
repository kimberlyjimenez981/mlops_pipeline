"""Model evaluation and validation functions."""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test set.

    Args:
        model: Trained scikit-learn model
        X_test: Test features
        y_test: Test target

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def cross_validate_model(model, X, y, cv=5):
    """Evaluate model using stratified k-fold cross-validation.

    Args:
        model: Scikit-learn model (unfitted)
        X: Features DataFrame
        y: Target Series
        cv: Number of cross-validation folds

    Returns:
        cv_results: Dictionary with mean and std of each metric
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scoring_metrics = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    cv_results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
        key = metric.replace("_weighted", "")
        cv_results[f"cv_{key}_mean"] = float(np.mean(scores))
        cv_results[f"cv_{key}_std"] = float(np.std(scores))
        logger.info(f"CV {metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    return cv_results


def get_feature_importance(model, feature_names):
    """Extract feature importances from a tree-based model.

    Args:
        model: Trained tree-based model with feature_importances_ attribute
        feature_names: List of feature names

    Returns:
        importance_df: DataFrame sorted by importance descending
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_ attribute")
        return None

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info(f"Top features: {importance_df['feature'].tolist()[:5]}")
    return importance_df


def validate_model(model, X_test, y_test, min_accuracy=0.7):
    """Run model validation checks and raise if thresholds are not met.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        min_accuracy: Minimum required accuracy (default 0.7)

    Raises:
        ValueError: If validation checks fail
    """
    metrics = evaluate_model(model, X_test, y_test)

    if metrics["accuracy"] < min_accuracy:
        raise ValueError(
            f"Model accuracy {metrics['accuracy']:.4f} is below "
            f"minimum threshold {min_accuracy}"
        )

    random_baseline = 1.0 / len(np.unique(y_test))
    if metrics["accuracy"] <= random_baseline:
        raise ValueError(
            f"Model accuracy {metrics['accuracy']:.4f} is not better "
            f"than random baseline {random_baseline:.4f}"
        )

    logger.info("Model validation passed all checks")
    return metrics


def generate_performance_report(model, X_test, y_test, feature_names=None):
    """Generate a comprehensive performance report.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        feature_names: Optional list of feature names

    Returns:
        report: Dictionary containing all evaluation results
    """
    y_pred = model.predict(X_test)

    metrics = evaluate_model(model, X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    report = {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
    }

    if feature_names is not None:
        importance_df = get_feature_importance(model, feature_names)
        if importance_df is not None:
            report["feature_importances"] = importance_df.to_dict(orient="records")

    logger.info("Performance report generated")
    return report
