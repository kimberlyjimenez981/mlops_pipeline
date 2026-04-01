"""Data validation tests on the actual dataset."""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import load_data, validate_features, validate_target


class TestDatasetIntegrity:
    """Tests for dataset integrity."""
    
    def test_dataset_loads(self):
        """Test that iris dataset loads successfully."""
        X, y = load_data()
        assert X is not None
        assert y is not None
    
    def test_dataset_shape(self):
        """Test dataset has correct shape."""
        X, y = load_data()
        assert X.shape[0] == 150  # 150 iris samples
        assert X.shape[1] == 4    # 4 features
        assert len(y) == 150
    
    def test_feature_names(self):
        """Test feature names are present."""
        X, y = load_data()
        assert len(X.columns) == 4
        assert "sepal length" in X.columns[0].lower()


class TestFeatureQuality:
    """Tests for feature quality."""
    
    def test_no_missing_features(self):
        """Test that features have no missing values."""
        X, _ = load_data()
        assert not X.isnull().any().any()
    
    def test_features_are_numeric(self):
        """Test that all features are numeric."""
        X, _ = load_data()
        assert all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns)
    
    def test_features_in_valid_range(self):
        """Test that features are in valid range."""
        X, _ = load_data()
        assert (X >= 0).all().all()  # All features should be non-negative
        assert (X <= 10).all().all() # All features should be <= 10 (reasonable for iris)
    
    def test_feature_validation_passes(self):
        """Test that features pass validation."""
        X, _ = load_data()
        validate_features(X)  # Should not raise
    
    def test_features_finite(self):
        """Test that all feature values are finite."""
        X, _ = load_data()
        assert np.all(np.isfinite(X.values))


class TestTargetQuality:
    """Tests for target quality."""
    
    def test_no_missing_target(self):
        """Test that target has no missing values."""
        _, y = load_data()
        assert not y.isnull().any()
    
    def test_target_has_multiple_classes(self):
        """Test that target has multiple classes."""
        _, y = load_data()
        assert len(y.unique()) == 3  # 3 iris species
    
    def test_target_classes_balanced(self):
        """Test that target classes are relatively balanced."""
        _, y = load_data()
        class_counts = y.value_counts()
        # All classes should have 50 samples
        assert (class_counts == 50).all()
    
    def test_target_validation_passes(self):
        """Test that target passes validation."""
        _, y = load_data()
        validate_target(y)  # Should not raise
    
    def test_target_values_valid(self):
        """Test that target values are valid class indices."""
        _, y = load_data()
        assert set(y.unique()) == {0, 1, 2}


class TestDataConsistency:
    """Tests for data consistency."""
    
    def test_features_target_aligned(self):
        """Test that features and target are aligned."""
        X, y = load_data()
        assert len(X) == len(y)
    
    def test_consistent_feature_count(self):
        """Test that all samples have same feature count."""
        X, _ = load_data()
        assert len(set(X.shape[1] for _ in X.values)) == 1
    
    def test_reproducible_load(self):
        """Test that dataset load is reproducible."""
        X1, y1 = load_data()
        X2, y2 = load_data()
        
        assert X1.equals(X2)
        assert y1.equals(y2)
