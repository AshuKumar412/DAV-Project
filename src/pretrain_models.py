"""
Pre-train baseline models on the demo dataset.
==============================================

Trains Random Forest, Decision Tree, Logistic Regression, XGBoost,
and Isolation Forest, and saves artifacts to `models/` directory.
Also generates `reports/evaluation_report.csv`.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data_preprocessing import DataPreprocessor
from train_models import ModelTrainer
from anomaly_detection import AnomalyDetector
from evaluation import ModelEvaluator


def main():
    print("Starting pretraining pipeline...")
    data_path = project_root / 'data' / 'sample_network_traffic.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path.name}")

    preprocessor = DataPreprocessor(max_samples=50000)
    target_col = preprocessor.identify_target_column(df)
    print(f"Target column identified: {target_col}")

    X_train, X_test, y_train, y_test, pipeline, label_encoder, feature_names = preprocessor.prepare_data(
        df, target_col=target_col, test_size=0.25
    )
    print(f"Data split: {len(X_train)} train samples, {len(X_test)} test samples across {len(feature_names)} features.")

    # Train Supervised Models
    trainer = ModelTrainer(models_dir=str(project_root / 'models'))
    evaluator = ModelEvaluator()
    models = trainer.get_available_models()

    evaluation_results = {}

    for name in models.keys():
        print(f"Training {name}...")
        model, train_time = trainer.train_model(name, X_train, y_train)
        saved_path = trainer.save_model(
            model=model,
            model_name=name,
            pipeline=pipeline,
            label_encoder=label_encoder,
            feature_names=feature_names
        )
        print(f"Saved {name} to {saved_path} (Training time: {train_time:.2f}s)")

        metrics = evaluator.evaluate_model(model, X_test, y_test, model_name=name)
        evaluation_results[name] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1 (weighted): {metrics['f1_weighted']:.4f}")

    # Train Anomaly Detector (Isolation Forest)
    print("\nTraining Isolation Forest Anomaly Detector...")
    ad = AnomalyDetector(contamination=0.1, random_state=42)
    ad.fit_predict(df[feature_names], feature_names=feature_names)
    ad_path = project_root / 'models' / 'isolation_forest.pkl'
    ad.save_model(str(ad_path))
    print(f"Saved Isolation Forest to {ad_path}")

    # Generate Evaluation Report
    report_path = project_root / 'reports' / 'evaluation_report.csv'
    report_df = evaluator.generate_evaluation_report(evaluation_results, output_path=str(report_path))
    print(f"\nEvaluation Report generated at {report_path}:")
    print(report_df)
    print("\nPre-training completed successfully!")


if __name__ == '__main__':
    main()
