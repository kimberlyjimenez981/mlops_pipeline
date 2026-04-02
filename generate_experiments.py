"""Generate multiple MLflow experiment runs with different hyperparameters.

Run this script to create evidence of at least 5 MLflow experiment runs:
    python generate_experiments.py
"""

import logging

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.preprocessing import load_data, preprocess_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Six experiment configurations with different hyperparameters
EXPERIMENT_CONFIGS = [
    {
        "n_estimators": 10,
        "max_depth": 3,
        "min_samples_split": 2,
        "random_state": 42,
    },
    {
        "n_estimators": 50,
        "max_depth": 5,
        "min_samples_split": 2,
        "random_state": 42,
    },
    {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "random_state": 42,
    },
    {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 5,
        "random_state": 42,
    },
    {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 10,
        "random_state": 42,
    },
    {
        "n_estimators": 150,
        "max_depth": 8,
        "min_samples_split": 3,
        "random_state": 0,
    },
]


def run_experiment(experiment_name, model_params, test_size=0.2, split_random_state=42):
    """Run a single MLflow experiment run.

    Args:
        experiment_name: MLflow experiment name
        model_params: RandomForest hyperparameters
        test_size: Fraction of data used for testing
        split_random_state: Random seed for the train/test split

    Returns:
        run_id: MLflow run ID
        metrics: Dictionary of evaluation metrics
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Load and preprocess data
        X, y = load_data()
        X_processed, y, _ = preprocess_data(X, y, normalize=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=split_random_state
        )

        # Log parameters (split_random_state logged separately to avoid collision
        # with the model's random_state parameter)
        mlflow.log_params({**model_params, "test_size": test_size, "split_random_state": split_random_state})

        # Train
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run {run_id}: accuracy={metrics['accuracy']:.4f}")

        return run_id, metrics


def main():
    """Run all experiment configurations and log results to MLflow."""
    experiment_name = "iris_classification"
    logger.info(f"Running {len(EXPERIMENT_CONFIGS)} experiments for '{experiment_name}'...\n")

    results = []
    for i, params in enumerate(EXPERIMENT_CONFIGS, 1):
        logger.info(f"Experiment {i}/{len(EXPERIMENT_CONFIGS)}: {params}")
        run_id, metrics = run_experiment(experiment_name, params)
        results.append({"run_id": run_id, "params": params, "metrics": metrics})

    logger.info(f"\nCompleted {len(results)} experiment runs.")
    logger.info("Run 'python compare_experiments.py' to compare results.")
    return results


if __name__ == "__main__":
    main()
