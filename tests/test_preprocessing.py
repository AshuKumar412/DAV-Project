"""
Tests for data preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from data_preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample network traffic data for testing."""
    np.random.seed(42)
    n = 200
    data = {
        'Flow Duration': np.random.randint(0, 1000000, n),
        'Total Fwd Packets': np.random.randint(1, 100, n),
        'Total Backward Packets': np.random.randint(0, 80, n),
        'Flow Bytes/s': np.random.uniform(0, 1e6, n),
        'Flow Packets/s': np.random.uniform(0, 1e4, n),
        'Packet Length Mean': np.random.uniform(0, 1500, n),
        'Destination Port': np.random.choice([80, 443, 22, 53, 8080], n),
        'Protocol': np.random.choice([6, 17, 1], n),
        'Label': np.random.choice(['BENIGN', 'DoS', 'PortScan'], n, p=[0.7, 0.2, 0.1])
    }
    df = pd.DataFrame(data)
    # Add some missing values
    df.loc[0, 'Flow Bytes/s'] = np.nan
    df.loc[1, 'Flow Bytes/s'] = np.inf
    return df


@pytest.fixture
def preprocessor():
    """Create a DataPreprocessor instance."""
    return DataPreprocessor(max_samples=100)


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

    def test_identify_target_column(self, preprocessor, sample_data):
        """Test that target column is correctly identified."""
        target = preprocessor.identify_target_column(sample_data)
        assert target == 'Label'

    def test_identify_target_column_missing(self, preprocessor):
        """Test behavior when no target column exists."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        target = preprocessor.identify_target_column(df)
        assert target is None

    def test_clean_data(self, preprocessor, sample_data):
        """Test data cleaning handles missing and infinite values."""
        cleaned = preprocessor.clean_data(sample_data)
        assert not cleaned.isnull().any().any(), "Cleaned data should have no NaN values"
        assert not np.isinf(cleaned.select_dtypes(include=[np.number]).values).any(), \
            "Cleaned data should have no infinite values"

    def test_clean_data_removes_duplicates(self, preprocessor):
        """Test that duplicates are removed."""
        df = pd.DataFrame({
            'A': [1, 1, 2, 3],
            'B': [4, 4, 5, 6],
            'Label': ['BENIGN', 'BENIGN', 'DoS', 'DoS']
        })
        cleaned = preprocessor.clean_data(df)
        assert len(cleaned) == 3

    def test_encode_labels(self, preprocessor, sample_data):
        """Test label encoding."""
        y = sample_data['Label']
        encoded, encoder = preprocessor.encode_labels(y)
        assert len(encoded) == len(y)
        assert set(encoded) == set(range(len(y.unique())))
        # Check inverse transform works
        decoded = encoder.inverse_transform(encoded)
        np.testing.assert_array_equal(decoded, y.values)

    def test_identify_features(self, preprocessor, sample_data):
        """Test feature identification."""
        features = preprocessor.identify_features(sample_data, 'Label')
        assert 'Label' not in features
        assert 'Flow Duration' in features
        assert len(features) > 0

    def test_prepare_data(self, preprocessor, sample_data):
        """Test full data preparation pipeline."""
        result = preprocessor.prepare_data(sample_data, 'Label')
        X_train, X_test, y_train, y_test, pipeline, label_encoder, feature_names = result

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        assert pipeline is not None
        assert label_encoder is not None
        assert len(feature_names) > 0

    def test_prepare_data_auto_detect_target(self, preprocessor, sample_data):
        """Test that prepare_data can auto-detect target column."""
        result = preprocessor.prepare_data(sample_data)
        assert result is not None

    def test_max_samples(self, sample_data):
        """Test that max_samples limits the data."""
        preprocessor = DataPreprocessor(max_samples=50)
        result = preprocessor.prepare_data(sample_data)
        X_train, X_test, _, _, _, _, _ = result
        assert len(X_train) + len(X_test) <= 50

    def test_transform_new_data(self, preprocessor, sample_data):
        """Test transforming new data with saved pipeline."""
        result = preprocessor.prepare_data(sample_data, 'Label')
        X_train, X_test, y_train, y_test, pipeline, label_encoder, feature_names = result

        # Create new data without label
        new_data = sample_data.drop(columns=['Label']).head(10)
        transformed = preprocessor.transform_new_data(new_data, pipeline, feature_names)
        assert transformed.shape[0] == 10
        assert transformed.shape[1] == len(feature_names)

    def test_transform_new_data_missing_columns(self, preprocessor, sample_data):
        """Test that missing columns are handled gracefully."""
        result = preprocessor.prepare_data(sample_data, 'Label')
        _, _, _, _, pipeline, _, feature_names = result

        # Create data with missing columns
        new_data = pd.DataFrame({
            'Flow Duration': [100, 200],
            'Protocol': [6, 17]
        })
        transformed = preprocessor.transform_new_data(new_data, pipeline, feature_names)
        assert transformed.shape[0] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
