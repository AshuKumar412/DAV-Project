"""
Network Intrusion Detection System (NIDS)
==========================================

A comprehensive machine learning-based network intrusion detection system
that provides data preprocessing, feature engineering, model training,
anomaly detection, evaluation, and visualization capabilities.
"""

__version__ = "1.0.0"
__author__ = "NIDS Team"

from src.utils import (
    set_random_seed,
    get_project_root,
    ensure_directories,
    validate_csv_file,
    format_number,
    get_severity_level,
    generate_security_events,
    safe_divide,
)

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train_models import ModelTrainer
from src.intrusion_detection import IntrusionDetector
from src.anomaly_detection import AnomalyDetector
from src.evaluation import ModelEvaluator

from src.visualization import (
    create_attack_distribution_chart,
    create_traffic_timeline,
    create_correlation_heatmap,
    create_feature_importance_chart,
    create_confusion_matrix_heatmap,
    create_model_comparison_chart,
    create_anomaly_scatter,
    create_protocol_distribution,
    create_port_distribution,
    create_packet_length_distribution,
    create_roc_curve,
    create_severity_distribution,
    create_normal_vs_malicious_chart,
)

__all__ = [
    # Utility functions
    "set_random_seed",
    "get_project_root",
    "ensure_directories",
    "validate_csv_file",
    "format_number",
    "get_severity_level",
    "generate_security_events",
    "safe_divide",
    # Core classes
    "DataPreprocessor",
    "FeatureEngineer",
    "ModelTrainer",
    "IntrusionDetector",
    "AnomalyDetector",
    "ModelEvaluator",
    # Visualization functions
    "create_attack_distribution_chart",
    "create_traffic_timeline",
    "create_correlation_heatmap",
    "create_feature_importance_chart",
    "create_confusion_matrix_heatmap",
    "create_model_comparison_chart",
    "create_anomaly_scatter",
    "create_protocol_distribution",
    "create_port_distribution",
    "create_packet_length_distribution",
    "create_roc_curve",
    "create_severity_distribution",
    "create_normal_vs_malicious_chart",
]
