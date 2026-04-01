"""Unit tests for preprocessing functions."""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import (
    handle_missing_values,
    normalize_features,
    validate_features,
    validate_target,
    preprocess_data
)


class TestHandleMissingValues:
    """Tests for missing value handling."""
    
    def test_no_missing_values(self):
        """Test with no missing values."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = handle_missing_values(X)
        assert result.equals(X)
    
    def test_mean_imputation(self):
        """Test mean imputation strategy."""
        X = pd.DataFrame({"a": [1, 2, np.nan], "b": [4, 5, 6]})
        result = handle_missing_values(X, strategy="mean")
        assert not result.isnull().any().any()
        assert result.loc[2, "a"] == pytest.approx(1.5)
    
    def test_median_imputation(self):
        """Test median imputation strategy."""
        X = pd.DataFrame({"a": [1, 2, np.nan], "b": [4, 5, 6]})
        result = handle_missing_values(X, strategy="median")
        assert not result.isnull().any().any()
        assert result.loc[2, "a"] == pytest.approx(1.5)
    
    def test_preserves_shape(self):
        """Test that output shape is preserved."""
        X = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan]})
        result = handle_missing_values(X)
        assert result.shape == X.shape


class TestNormalizeFeatures:
    """Tests for feature normalization."""
    
    def test_zero_mean_unit_variance(self):
        """Test that normalized features have zero mean and unit variance."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        X_normalized, scaler = normalize_features(X)
        
        assert np.allclose(X_normalized.mean(), 0, atol=1e-10)
        assert np.allclose(X_normalized.std(), 1, atol=1e-10)
    
    def test_fit_on_different_data(self):
        """Test fitting scaler on different data."""
        X_train = pd.DataFrame({"a": [0, 1, 2], "b": [0, 1, 2]})
        X_test = pd.DataFrame({"a": [3, 4], "b": [3, 4]})
        
        X_test_norm, scaler = normalize_features(X_test, fit_on_X=X_train)
        
        # Should use train statistics
        assert X_test_norm.values[0, 0] == pytest.approx(1.5)
    
    def test_preserves_shape(self):
        """Test that output shape is preserved."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        X_normalized, _ = normalize_features(X)
        assert X_normalized.shape == X.shape


class TestValidateFeatures:
    """Tests for feature validation."""
    
    def test_valid_features(self):
        """Test with valid features."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # Should not raise
        validate_features(X)
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        X = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            validate_features(X)
    
    def test_null_values(self):
        """Test with null values."""
        X = pd.DataFrame({"a": [1, np.nan, 3]})
        with pytest.raises(ValueError, match="null"):
            validate_features(X)
    
    def test_non_finite_values(self):
        """Test with non-finite values."""
        X = pd.DataFrame({"a": [1, np.inf, 3]})
        with pytest.raises(ValueError, match="non-finite"):
            validate_features(X)


class TestValidateTarget:
    """Tests for target validation."""
    
    def test_valid_target(self):
        """Test with valid target."""
        y = pd.Series([0, 1, 0, 1])
        # Should not raise
        validate_target(y)
    
    def test_empty_series(self):
        """Test with empty Series."""
        y = pd.Series(dtype=int)
        with pytest.raises(ValueError, match="empty"):
            validate_target(y)
    
    def test_null_values(self):
        """Test with null values."""
        y = pd.Series([0, 1, np.nan])
        with pytest.raises(ValueError, match="null"):
            validate_target(y)
    
    def test_single_class(self):
        """Test with single class."""
        y = pd.Series([0, 0, 0])
        with pytest.raises(ValueError, match="at least 2"):
            validate_target(y)


class TestPreprocessData:
    """Tests for complete preprocessing pipeline."""
    
    def test_full_pipeline_with_normalization(self):
        """Test complete pipeline with normalization."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        
        X_proc, y_proc, scaler = preprocess_data(X, y, normalize=True)
        
        assert X_proc.shape == X.shape
        assert len(y_proc) == len(y)
        assert scaler is not None
    
    def test_full_pipeline_without_normalization(self):
        """Test complete pipeline without normalization."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        y = pd.Series([0, 1, 0])
        
        X_proc, y_proc = preprocess_data(X, y, normalize=False)
        
        assert X_proc.shape == X.shape
        assert len(y_proc) == len(y)
    
    def test_fit_on_different_data(self):
        """Test fitting on train data and transforming test data."""
        X_train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        X_test = pd.DataFrame({"a": [2, 3], "b": [5, 6]})
        y_train = pd.Series([0, 1, 0])
        y_test = pd.Series([1, 0])
        
        X_train_proc, y_train_proc, scaler = preprocess_data(
            X_train, y_train, normalize=True
        )
        X_test_proc, y_test_proc = preprocess_data(
            X_test, y_test, normalize=True, fit_on_X=X_train
        )
        
        assert X_train_proc.shape == X_train.shape
        assert X_test_proc.shape == X_test.shape
