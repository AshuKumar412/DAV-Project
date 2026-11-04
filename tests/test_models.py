"""
Tests for model training and evaluation modules.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from train_models import ModelTrainer
from evaluation import ModelEvaluator
from data_preprocessing import DataPreprocessor


@pytest.fixture
def training_data():
    """Create sample training data."""
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 5)
    # Make labels somewhat predictable from features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    # Add some noise
    noise_idx = np.random.choice(n, size=30, replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    return X, y


@pytest.fixture
def multiclass_data():
    """Create sample multiclass training data."""
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 5)
    y = np.zeros(n, dtype=int)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 1] > 0.5] = 2
    return X, y


@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for models."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestModelTrainer:
    """Test suite for ModelTrainer class."""

    def test_get_available_models(self, temp_models_dir):
        """Test that all expected models are available."""
        trainer = ModelTrainer(models_dir=temp_models_dir)
        models = trainer.get_available_models()
        assert 'Random Forest' in models
        assert 'Decision Tree' in models
        assert 'Logistic Regression' in models
        assert 'XGBoost' in models

    def test_train_model(self, training_data, temp_models_dir):
        """Test training a single model."""
        X, y = training_data
        trainer = ModelTrainer(models_dir=temp_models_dir)
        model, train_time = trainer.train_model('Random Forest', X, y)
        assert model is not None
        assert train_time > 0
        # Check model can predict
        predictions = model.predict(X[:5])
        assert len(predictions) == 5

    def test_train_all_models(self, training_data, temp_models_dir):
        """Test training all models."""
        X, y = training_data
        trainer = ModelTrainer(models_dir=temp_models_dir)
        results = trainer.train_all_models(X, y)
        assert len(results) == 4
        for name, result in results.items():
            assert 'model' in result
            assert 'time' in result

    def test_save_and_load_model(self, training_data, temp_models_dir):
        """Test saving and loading a model."""
        X, y = training_data
        trainer = ModelTrainer(models_dir=temp_models_dir)
        model, _ = trainer.train_model('Random Forest', X, y)

        # Save
        trainer.save_model(model, 'Random Forest', feature_names=['f1', 'f2', 'f3', 'f4', 'f5'])

        # Load
        artifacts = trainer.load_model('Random Forest')
        assert artifacts is not None
        assert 'model' in artifacts
        assert 'model_name' in artifacts

        # Check loaded model can predict
        predictions = artifacts['model'].predict(X[:5])
        assert len(predictions) == 5

    def test_list_saved_models(self, training_data, temp_models_dir):
        """Test listing saved models."""
        X, y = training_data
        trainer = ModelTrainer(models_dir=temp_models_dir)

        # Train and save
        model, _ = trainer.train_model('Random Forest', X, y)
        trainer.save_model(model, 'Random Forest')

        saved = trainer.list_saved_models()
        assert len(saved) >= 1

    def test_multiclass_training(self, multiclass_data, temp_models_dir):
        """Test training with multiclass data."""
        X, y = multiclass_data
        trainer = ModelTrainer(models_dir=temp_models_dir)

        for name in ['Random Forest', 'Decision Tree', 'XGBoost']:
            model, _ = trainer.train_model(name, X, y)
            predictions = model.predict(X[:10])
            assert len(predictions) == 10


class TestModelEvaluator:
    """Test suite for ModelEvaluator class."""

    def test_evaluate_model(self, training_data, temp_models_dir):
        """Test model evaluation."""
        X, y = training_data
        trainer = ModelTrainer(models_dir=temp_models_dir)
        model, _ = trainer.train_model('Random Forest', X, y)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(model, X, y, 'Random Forest')

        assert 'accuracy' in metrics
        assert 'precision_weighted' in metrics
        assert 'recall_weighted' in metrics
        assert 'f1_weighted' in metrics
        assert 'confusion_matrix' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_compare_models(self, training_data, temp_models_dir):
        """Test model comparison."""
        X, y = training_data
        trainer = ModelTrainer(models_dir=temp_models_dir)
        evaluator = ModelEvaluator()

        results = {}
        for name in ['Random Forest', 'Decision Tree']:
            model, _ = trainer.train_model(name, X, y)
            results[name] = evaluator.evaluate_model(model, X, y, name)

        comparison = evaluator.compare_models(results)
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2

    def test_feature_importance(self, training_data, temp_models_dir):
        """Test feature importance extraction."""
        X, y = training_data
        trainer = ModelTrainer(models_dir=temp_models_dir)
        model, _ = trainer.train_model('Random Forest', X, y)

        evaluator = ModelEvaluator()
        importance = evaluator.get_feature_importance(
            model, [f'feature_{i}' for i in range(5)]
        )
        assert importance is not None
        assert len(importance) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
