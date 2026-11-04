"""
Anomaly Detection Module for Network Intrusion Detection System
================================================================

Provides the ``AnomalyDetector`` class that leverages scikit-learn's
Isolation Forest to identify anomalous network-traffic records in an
unsupervised fashion.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """
    Unsupervised anomaly detector for network traffic using Isolation Forest.

    Parameters
    ----------
    contamination : float, default=0.1
        Expected proportion of anomalies in the dataset (passed to
        ``IsolationForest``).
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self, contamination: float = 0.1, random_state: int = 42
    ) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self._model: Optional[IsolationForest] = None
        self._results: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Fit & predict
    # ------------------------------------------------------------------

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fit an Isolation Forest on numeric features and label each record
        as normal or anomalous.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        feature_names : list of str or None
            Specific columns to use.  If ``None``, all numeric columns
            are selected automatically.

        Returns
        -------
        pd.DataFrame
            Copy of *df* augmented with two new columns:

            - ``anomaly_score`` – Isolation Forest decision-function value
              (lower = more anomalous).
            - ``is_anomaly`` – ``True`` where the model labels the record
              as an outlier (prediction == -1).
        """
        result = df.copy()

        # Select features
        if feature_names is not None:
            cols = [c for c in feature_names if c in result.columns]
        else:
            cols = result.select_dtypes(include=[np.number]).columns.tolist()

        X = result[cols].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

        # Fit model
        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )

        predictions = self._model.fit_predict(X)
        scores = self._model.decision_function(X)

        result["anomaly_score"] = scores
        result["is_anomaly"] = predictions == -1

        self._results = result
        return result

    # ------------------------------------------------------------------
    # Summary / filtering
    # ------------------------------------------------------------------

    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Provide a high-level summary of the last ``fit_predict`` run.

        Returns
        -------
        dict
            Keys: ``total``, ``normal_count``, ``anomaly_count``,
            ``anomaly_percentage``.

        Raises
        ------
        RuntimeError
            If ``fit_predict`` has not been called yet.
        """
        if self._results is None:
            raise RuntimeError(
                "No results available. Call fit_predict() first."
            )
        total = len(self._results)
        anomaly_count = int(self._results["is_anomaly"].sum())
        normal_count = total - anomaly_count
        anomaly_pct = (anomaly_count / total * 100) if total > 0 else 0.0
        return {
            "total": total,
            "normal_count": normal_count,
            "anomaly_count": anomaly_count,
            "anomaly_percentage": round(anomaly_pct, 2),
        }

    def get_anomalous_records(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Return only the records flagged as anomalous.

        Parameters
        ----------
        df : pd.DataFrame or None
            DataFrame with ``is_anomaly`` column.  Defaults to the
            internal results from the last ``fit_predict`` call.

        Returns
        -------
        pd.DataFrame
        """
        data = df if df is not None else self._results
        if data is None:
            raise RuntimeError(
                "No results available. Call fit_predict() first."
            )
        return data[data["is_anomaly"]].copy()

    def get_normal_records(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Return only the records classified as normal.

        Parameters
        ----------
        df : pd.DataFrame or None
            DataFrame with ``is_anomaly`` column.  Defaults to the
            internal results from the last ``fit_predict`` call.

        Returns
        -------
        pd.DataFrame
        """
        data = df if df is not None else self._results
        if data is None:
            raise RuntimeError(
                "No results available. Call fit_predict() first."
            )
        return data[~data["is_anomaly"]].copy()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str = "models/isolation_forest.pkl") -> str:
        """
        Save the fitted Isolation Forest model to disk.

        Parameters
        ----------
        path : str
            Destination file path.

        Returns
        -------
        str
            Absolute path of the saved file.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError(
                "No model to save. Call fit_predict() first."
            )
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, save_path)
        return str(save_path.resolve())

    def load_model(self, path: str = "models/isolation_forest.pkl") -> None:
        """
        Load a previously saved Isolation Forest model from disk.

        Parameters
        ----------
        path : str
            Path to the pickle file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self._model = joblib.load(load_path)
