"""Run multiple MLflow experiments with varied hyperparameters."""

import sys
import os
import logging

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import load_data, preprocess_data
from src.evaluation import evaluate_model, cross_validate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "iris_classification"

# 7 distinct experiment configurations varying n_estimators, max_depth,
# test_size and normalization to exceed the 5-run minimum.
EXPERIMENT_CONFIGS = [
    {
        "run_name": "exp_01_baseline",
        "n_estimators": 100,
        "max_depth": 10,
        "test_size": 0.2,
        "normalize": True,
        "random_state": 42,
    },
    {
        "run_name": "exp_02_shallow_tree",
        "n_estimators": 100,
        "max_depth": 3,
        "test_size": 0.2,
        "normalize": True,
        "random_state": 42,
    },
    {
        "run_name": "exp_03_more_estimators",
        "n_estimators": 200,
        "max_depth": 10,
        "test_size": 0.2,
        "normalize": True,
        "random_state": 42,
    },
    {
        "run_name": "exp_04_large_test_split",
        "n_estimators": 100,
        "max_depth": 10,
        "test_size": 0.3,
        "normalize": True,
        "random_state": 42,
    },
    {
        "run_name": "exp_05_no_normalize",
        "n_estimators": 100,
        "max_depth": 10,
        "test_size": 0.2,
        "normalize": False,
        "random_state": 42,
    },
    {
        "run_name": "exp_06_few_estimators",
        "n_estimators": 50,
        "max_depth": 5,
        "test_size": 0.25,
        "normalize": True,
        "random_state": 42,
    },
    {
        "run_name": "exp_07_deep_tree",
        "n_estimators": 150,
        "max_depth": None,
        "test_size": 0.2,
        "normalize": True,
        "random_state": 42,
    },
]


def run_experiment(cfg):
    """Execute a single experiment run and return the run ID and metrics.

    Args:
        cfg: Experiment configuration dictionary

    Returns:
        run_id: MLflow run ID
        metrics: Evaluation metrics dictionary
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        # Load and preprocess data
        X, y = load_data(source="sklearn", name="iris")

        X_processed, y, *_ = preprocess_data(X, y, normalize=cfg["normalize"])

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y,
            test_size=cfg["test_size"],
            random_state=cfg["random_state"],
        )

        # Log parameters
        mlflow.log_params({
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "test_size": cfg["test_size"],
            "normalize": cfg["normalize"],
            "random_state": cfg["random_state"],
        })

        # Train
        model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            random_state=cfg["random_state"],
        )
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Cross-validation metrics
        cv_results = cross_validate_model(model, X_processed, y, cv=5)
        mlflow.log_metrics(cv_results)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        logger.info(f"Completed {cfg['run_name']} | run_id={run_id} | {metrics}")

    return run_id, metrics


def main():
    """Run all experiments and print a summary table."""
    results = []

    for cfg in EXPERIMENT_CONFIGS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting: {cfg['run_name']}")
        logger.info(f"{'='*60}")
        run_id, metrics = run_experiment(cfg)
        results.append({
            "run_name": cfg["run_name"],
            "run_id": run_id,
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "test_size": cfg["test_size"],
            "normalize": cfg["normalize"],
            **metrics,
        })

    # Summary table
    df = pd.DataFrame(results)
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    logger.info(df.to_string(index=False))

    print("\nExperiment Summary:")
    print(df.to_string(index=False))

    return results


if __name__ == "__main__":
    main()
