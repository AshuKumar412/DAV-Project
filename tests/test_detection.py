"""
Tests for intrusion detection and anomaly detection modules.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from anomaly_detection import AnomalyDetector


@pytest.fixture
def sample_numeric_data():
    """Create sample numeric data for anomaly detection."""
    np.random.seed(42)
    n = 200
    data = {
        'Flow Duration': np.random.randint(0, 1000000, n),
        'Total Fwd Packets': np.random.randint(1, 100, n),
        'Total Backward Packets': np.random.randint(0, 80, n),
        'Flow Bytes/s': np.random.uniform(0, 1e6, n),
        'Packet Length Mean': np.random.uniform(0, 1500, n),
        'Destination Port': np.random.choice([80, 443, 22, 53], n),
        'Protocol': np.random.choice([6, 17, 1], n),
    }
    return pd.DataFrame(data)


class TestAnomalyDetector:
    """Test suite for AnomalyDetector class."""

    def test_init(self):
        """Test AnomalyDetector initialization."""
        detector = AnomalyDetector(contamination=0.1)
        assert detector.contamination == 0.1

    def test_fit_predict(self, sample_numeric_data):
        """Test fit_predict returns expected columns."""
        detector = AnomalyDetector(contamination=0.1)
        result = detector.fit_predict(sample_numeric_data)
        assert 'anomaly_score' in result.columns
        assert 'is_anomaly' in result.columns
        assert len(result) == len(sample_numeric_data)

    def test_anomaly_count(self, sample_numeric_data):
        """Test that anomaly count is reasonable."""
        detector = AnomalyDetector(contamination=0.1)
        result = detector.fit_predict(sample_numeric_data)
        anomaly_count = result['is_anomaly'].sum()
        # With contamination=0.1, expect roughly 10% anomalies
        assert anomaly_count > 0
        assert anomaly_count < len(sample_numeric_data)

    def test_get_anomaly_summary(self, sample_numeric_data):
        """Test anomaly summary generation."""
        detector = AnomalyDetector(contamination=0.1)
        detector.fit_predict(sample_numeric_data)
        summary = detector.get_anomaly_summary()
        assert 'total' in summary
        assert 'normal_count' in summary
        assert 'anomaly_count' in summary
        assert 'anomaly_percentage' in summary
        assert summary['total'] == len(sample_numeric_data)

    def test_get_anomalous_records(self, sample_numeric_data):
        """Test filtering anomalous records."""
        detector = AnomalyDetector(contamination=0.1)
        result = detector.fit_predict(sample_numeric_data)
        anomalous = detector.get_anomalous_records(result)
        assert all(anomalous['is_anomaly'] == True)

    def test_get_normal_records(self, sample_numeric_data):
        """Test filtering normal records."""
        detector = AnomalyDetector(contamination=0.1)
        result = detector.fit_predict(sample_numeric_data)
        normal = detector.get_normal_records(result)
        assert all(normal['is_anomaly'] == False)

    def test_contamination_range(self, sample_numeric_data):
        """Test different contamination values."""
        for contamination in [0.01, 0.05, 0.1, 0.2, 0.3]:
            detector = AnomalyDetector(contamination=contamination)
            result = detector.fit_predict(sample_numeric_data)
            assert 'is_anomaly' in result.columns

    def test_custom_feature_names(self, sample_numeric_data):
        """Test with specific feature names."""
        detector = AnomalyDetector(contamination=0.1)
        features = ['Flow Duration', 'Total Fwd Packets', 'Packet Length Mean']
        result = detector.fit_predict(sample_numeric_data, feature_names=features)
        assert len(result) == len(sample_numeric_data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
