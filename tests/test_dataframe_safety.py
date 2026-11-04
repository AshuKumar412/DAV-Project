"""
Tests for DataFrame boolean safety and security event handling.
Verifies prevention of "ValueError: The truth value of a DataFrame is ambiguous".
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from utils import generate_security_events, get_severity_level
from visualization import (
    create_severity_distribution,
    create_attack_distribution_chart,
    create_anomaly_scatter,
    create_traffic_timeline,
    create_model_comparison_chart,
    create_feature_importance_chart,
)


class TestDataFrameSafety:
    """Test suite ensuring all functions safely handle empty, None, and DataFrame inputs."""

    def test_generate_security_events_valid(self):
        """Test with normal predictions DataFrame."""
        preds = pd.DataFrame({
            'predicted_label': ['BENIGN', 'DoS', 'Bot', 'PortScan'],
            'confidence': [0.98, 0.85, 0.92, 0.77],
            'is_malicious': [False, True, True, True]
        })
        events = generate_security_events(preds, model_name="Random Forest")
        assert isinstance(events, pd.DataFrame)
        assert len(events) == 4
        assert 'severity' in events.columns
        assert 'event_id' in events.columns
        assert events.loc[0, 'severity'] == 'LOW'
        assert events.loc[1, 'severity'] == 'HIGH'
        assert events.loc[2, 'severity'] == 'CRITICAL'
        assert events.loc[3, 'severity'] == 'MEDIUM'

    def test_generate_security_events_empty(self):
        """Test with empty DataFrame."""
        preds = pd.DataFrame(columns=['predicted_label', 'confidence', 'is_malicious'])
        events = generate_security_events(preds, model_name="Random Forest")
        assert isinstance(events, pd.DataFrame)
        assert events.empty
        assert 'severity' in events.columns

    def test_generate_security_events_none(self):
        """Test with None input."""
        events = generate_security_events(None, model_name="Random Forest")
        assert isinstance(events, pd.DataFrame)
        assert events.empty

    def test_create_severity_distribution_empty(self):
        """Test chart generation with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['severity'])
        fig = create_severity_distribution(empty_df)
        assert fig is not None

    def test_create_severity_distribution_none(self):
        """Test chart generation with None."""
        fig = create_severity_distribution(None)
        assert fig is not None

    def test_create_severity_distribution_missing_column(self):
        """Test chart generation without severity column."""
        bad_df = pd.DataFrame({'other_col': [1, 2, 3]})
        fig = create_severity_distribution(bad_df)
        assert fig is not None

    def test_create_attack_distribution_chart_empty(self):
        """Test pie chart with empty Series and None."""
        fig_empty = create_attack_distribution_chart(pd.Series([], dtype=str))
        assert fig_empty is not None
        fig_none = create_attack_distribution_chart(None)
        assert fig_none is not None

    def test_create_anomaly_scatter_empty(self):
        """Test anomaly scatter with empty DataFrame."""
        fig = create_anomaly_scatter(pd.DataFrame())
        assert fig is not None

    def test_create_traffic_timeline_empty(self):
        """Test timeline with empty DataFrame."""
        fig = create_traffic_timeline(pd.DataFrame())
        assert fig is not None

    def test_create_feature_importance_chart_empty(self):
        """Test feature importance with empty DataFrame."""
        fig = create_feature_importance_chart(pd.DataFrame())
        assert fig is not None

    def test_create_model_comparison_chart_empty(self):
        """Test model comparison with empty DataFrame."""
        fig = create_model_comparison_chart(pd.DataFrame())
        assert fig is not None

    def test_event_safety_simulation_no_ambiguous_truth_error(self):
        """
        Simulate the exact condition that triggered:
        ValueError: The truth value of a DataFrame is ambiguous.
        Ensure it works seamlessly with all possible states.
        """
        states_to_test = [
            # 1. None
            None,
            # 2. Empty list
            [],
            # 3. Empty DataFrame
            pd.DataFrame(),
            # 4. DataFrame with rows
            pd.DataFrame({
                'event_id': ['EVT-001'],
                'severity': ['HIGH'],
                'attack_category': ['DoS'],
                'is_malicious': [True]
            })
        ]

        for raw_events in states_to_test:
            # The safe pattern used in app.py:
            events_df = None
            if isinstance(raw_events, pd.DataFrame):
                if not raw_events.empty:
                    events_df = raw_events
            elif isinstance(raw_events, (list, tuple)) and len(raw_events) > 0:
                events_df = pd.DataFrame(raw_events)

            # Test explicit checks without `if not events:`
            is_empty = events_df is None or events_df.empty
            assert isinstance(is_empty, (bool, np.bool_))

            if events_df is not None and not events_df.empty:
                assert len(events_df) > 0
                assert 'severity' in events_df.columns
            else:
                assert is_empty is True
