"""Model validation tests."""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import load_data, preprocess_data


class TestModelOutput:
    """Tests for model output shape and format."""
    
    def test_prediction_shape(self):
        """Test that predictions have correct shape."""
        X, y = load_data()
        X_processed, y = preprocess_data(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert predictions.shape == (len(X_test),)
    
    def test_prediction_probabilities_shape(self):
        """Test that probability predictions have correct shape."""
        X, y = load_data()
        X_processed, y = preprocess_data(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)  # 3 classes
    
    def test_prediction_classes_valid(self):
        """Test that predictions are valid class indices."""
        X, y = load_data()
        X_processed, y = preprocess_data(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        assert set(predictions).issubset({0, 1, 2})
    
    def test_probability_sum_to_one(self):
        """Test that probabilities sum to one."""
        X, y = load_data()
        X_processed, y = preprocess_data(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        proba = model.predict_proba(X_test)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestModelPerformance:
    """Tests for minimum model performance."""
    
    def test_minimum_accuracy(self):
        """Test that model achieves minimum accuracy."""
        X, y = load_data()
        X_processed, y = preprocess_data(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        assert accuracy >= 0.8  # At least 80% accuracy
    
    def test_better_than_baseline(self):
        """Test that model is better than random baseline."""
        X, y = load_data()
        X_processed, y = preprocess_data(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        random_baseline = 1.0 / len(y.unique())  # Probability of random correct guess
        assert accuracy > random_baseline * 2  # Should be much better than random
    
    def test_generalizes_on_train_data(self):
        """Test that model generalizes on training data."""
        X, y = load_data()
        X_processed, y = preprocess_data(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        # Should not overfit too much
        assert train_accuracy - test_accuracy < 0.15
    
    def test_consistent_predictions(self):
        """Test that model makes consistent predictions."""
        X, y = load_data()
        X_processed, y = preprocess_data(X, y)
        
        X_train, X_test, _, _ = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y)
        
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)
        
        assert np.array_equal(pred1, pred2)
