"""Data preprocessing functions for ML pipeline."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


def load_data(source="sklearn", name="iris"):
    """Load dataset from sklearn.
    
    Args:
        source: Data source ("sklearn")
        name: Dataset name ("iris")
    
    Returns:
        X, y: Features and target
    """
    if source == "sklearn" and name == "iris":
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name="target")
        logger.info(f"Loaded {name} dataset: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    else:
        raise ValueError(f"Unknown dataset: {source}/{name}")


def handle_missing_values(X, strategy="mean"):
    """Handle missing values in features.
    
    Args:
        X: Input features DataFrame
        strategy: Imputation strategy ("mean", "median", "most_frequent")
    
    Returns:
        X_imputed: DataFrame with missing values handled
    """
    n_missing = X.isnull().sum().sum()
    if n_missing > 0:
        logger.info(f"Found {n_missing} missing values, using {strategy} strategy")
        imputer = SimpleImputer(strategy=strategy)
        X_filled = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns
        )
        return X_filled
    return X


def normalize_features(X, fit_on_X=None):
    """Normalize features using StandardScaler.
    
    Args:
        X: Input features DataFrame
        fit_on_X: DataFrame to fit scaler on (if None, fit on X)
    
    Returns:
        X_normalized: Normalized features DataFrame
        scaler: Fitted scaler object
    """
    scaler = StandardScaler()
    if fit_on_X is not None:
        scaler.fit(fit_on_X)
    else:
        scaler.fit(X)
    
    X_normalized = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns
    )
    logger.info("Features normalized using StandardScaler")
    return X_normalized, scaler


def validate_features(X):
    """Validate feature data quality.
    
    Args:
        X: Input features DataFrame
    
    Raises:
        ValueError: If validation fails
    """
    if X.empty:
        raise ValueError("Features DataFrame is empty")
    
    if X.isnull().any().any():
        raise ValueError("Features contain null values after processing")
    
    if not np.all(np.isfinite(X.values)):
        raise ValueError("Features contain non-finite values")
    
    logger.info("Feature validation passed")


def validate_target(y):
    """Validate target data quality.
    
    Args:
        y: Target Series
    
    Raises:
        ValueError: If validation fails
    """
    if y.empty:
        raise ValueError("Target Series is empty")
    
    if y.isnull().any():
        raise ValueError("Target contains null values")
    
    if len(y.unique()) < 2:
        raise ValueError("Target must have at least 2 classes")
    
    logger.info(f"Target validation passed: {len(y.unique())} classes")


def preprocess_data(X, y=None, normalize=True, handle_missing="mean", fit_on_X=None):
    """Complete preprocessing pipeline.
    
    Args:
        X: Input features DataFrame
        y: Target Series (optional)
        normalize: Whether to normalize features
        handle_missing: Missing value strategy
        fit_on_X: DataFrame to fit scaler on (for test set)
    
    Returns:
        X_processed: Processed features
        y: Target (if provided)
        scaler: Fitted scaler (if normalize=True)
    """
    logger.info("Starting preprocessing pipeline")
    
    # Handle missing values
    X_processed = handle_missing_values(X, strategy=handle_missing)
    
    # Normalize
    scaler = None
    if normalize:
        X_processed, scaler = normalize_features(X_processed, fit_on_X=fit_on_X)
    
    # Validate
    validate_features(X_processed)
    if y is not None:
        validate_target(y)
    
    logger.info("Preprocessing pipeline completed")
    
    if normalize:
        return X_processed, y, scaler
    else:
        return X_processed, y
