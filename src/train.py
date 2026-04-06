"""Training script with MLflow integration."""

import os
import yaml
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import load_data, preprocess_data
from src.evaluation import evaluate_model, validate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def train_model(X_train, y_train, model_params):
    """Train Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_params: Model hyperparameters
    
    Returns:
        model: Trained model
    """
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    return model


def main():
    """Main training pipeline."""
    # Load configuration
    config = load_config()
    
    # Set MLflow experiment
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run():
        try:
            # Load data
            logger.info("Loading data...")
            X, y = load_data(
                source=config["dataset"]["source"],
                name=config["dataset"]["name"]
            )
            
            # Preprocess
            logger.info("Preprocessing data...")
            X_processed, y, scaler = preprocess_data(
                X, y,
                normalize=config["preprocessing"]["normalize"],
                handle_missing=config["preprocessing"]["handle_missing"]
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y,
                test_size=config["training"]["test_size"],
                random_state=config["training"]["random_state"]
            )
            
            # Log parameters
            mlflow.log_params({
                **config["training"],
                **config["model"],
                **config["preprocessing"]
            })
            
            # Train model
            logger.info("Training model...")
            model_params = {k: v for k, v in config["model"].items() if k != "type"}
            model = train_model(X_train, y_train, model_params)
            
            # Evaluate
            logger.info("Evaluating model...")
            metrics = validate_model(model, X_test, y_test)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Save model
            model_path = "models/model.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            mlflow.sklearn.log_model(model, "model")
            
            logger.info("Training pipeline completed successfully")
            logger.info(f"Run ID: {mlflow.active_run().info.run_id}")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise


if __name__ == "__main__":
    main()
