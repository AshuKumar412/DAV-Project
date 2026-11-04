"""
Intrusion Detection Module for Network Intrusion Detection System
==================================================================

Provides the ``IntrusionDetector`` class that wraps a trained model and
its preprocessing artifacts to perform inference on new network-traffic
data, classify flows as benign or malicious, and produce summary reports.
"""

from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


class IntrusionDetector:
    """
    Classify network flows using a previously trained NIDS model.

    Parameters
    ----------
    model_artifacts : dict
        Dictionary produced by ``ModelTrainer.save_model`` / ``load_model``
        containing keys: ``'model'``, ``'pipeline'``, ``'label_encoder'``,
        ``'feature_names'``, ``'model_name'``, ``'timestamp'``.
    """

    def __init__(self, model_artifacts: Dict[str, Any]) -> None:
        self.model = model_artifacts["model"]
        self.pipeline = model_artifacts.get("pipeline")
        self.label_encoder = model_artifacts.get("label_encoder")
        self.feature_names = model_artifacts.get("feature_names")
        self.model_name = model_artifacts.get("model_name", "Unknown")
        self.timestamp = model_artifacts.get("timestamp")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict attack labels for the given dataframe.

        The method preprocesses *df* using the saved pipeline, generates
        integer predictions, decodes them back to string labels, and
        computes per-row confidence (via ``predict_proba`` when available).

        Parameters
        ----------
        df : pd.DataFrame
            Raw network-traffic dataframe with the same features used
            during training.

        Returns
        -------
        pd.DataFrame
            A dataframe with columns:

            - ``predicted_label`` – decoded string label.
            - ``is_malicious`` – ``True`` if the label is not ``BENIGN``.
            - ``confidence`` – maximum class probability (``None`` if the
              model does not support ``predict_proba``).
        """
        X = self._prepare_features(df)

        # Integer predictions
        y_pred = self.model.predict(X)

        # Decode labels
        if self.label_encoder is not None:
            predicted_labels = self.label_encoder.inverse_transform(y_pred)
        else:
            predicted_labels = y_pred.astype(str)

        # Confidence scores
        confidence = self._compute_confidence(X)

        predictions = pd.DataFrame(
            {
                "predicted_label": predicted_labels,
                "is_malicious": [
                    str(lbl).upper() != "BENIGN" for lbl in predicted_labels
                ],
                "confidence": confidence,
            },
            index=df.index[: len(predicted_labels)],
        )
        return predictions

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_attack_distribution(self, predictions: pd.DataFrame) -> pd.Series:
        """
        Count occurrences of each predicted attack category.

        Parameters
        ----------
        predictions : pd.DataFrame
            Output of :meth:`predict`.

        Returns
        -------
        pd.Series
            Value counts of ``predicted_label``, sorted descending.
        """
        return (
            predictions["predicted_label"]
            .value_counts()
            .sort_values(ascending=False)
        )

    def get_suspicious_flows(
        self, predictions: pd.DataFrame, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Return records predicted as malicious alongside their original
        features.

        Parameters
        ----------
        predictions : pd.DataFrame
            Output of :meth:`predict`.
        df : pd.DataFrame
            Original raw dataframe (same rows as *predictions*).

        Returns
        -------
        pd.DataFrame
            Concatenation of original features and prediction columns for
            malicious flows only.
        """
        malicious_mask = predictions["is_malicious"]
        malicious_preds = predictions.loc[malicious_mask]
        malicious_original = df.loc[malicious_mask.index[malicious_mask]]
        return pd.concat(
            [malicious_original.reset_index(drop=True),
             malicious_preds.reset_index(drop=True)],
            axis=1,
        )

    def filter_by_category(
        self, predictions: pd.DataFrame, category: str
    ) -> pd.DataFrame:
        """
        Filter predictions to a single attack category.

        Parameters
        ----------
        predictions : pd.DataFrame
            Output of :meth:`predict`.
        category : str
            Exact label to filter on (case-sensitive).

        Returns
        -------
        pd.DataFrame
            Filtered rows.
        """
        return predictions[predictions["predicted_label"] == category].copy()

    def generate_report(
        self, predictions: pd.DataFrame, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Build a comprehensive intrusion-detection report.

        Parameters
        ----------
        predictions : pd.DataFrame
            Output of :meth:`predict`.
        df : pd.DataFrame
            Original raw dataframe.

        Returns
        -------
        dict
            Keys: ``total_flows``, ``benign_count``, ``malicious_count``,
            ``attack_distribution`` (dict), ``top_suspicious_flows``
            (first 20 rows as list of dicts).
        """
        total = len(predictions)
        malicious_count = int(predictions["is_malicious"].sum())
        benign_count = total - malicious_count
        distribution = self.get_attack_distribution(predictions).to_dict()

        suspicious = self.get_suspicious_flows(predictions, df)
        top_suspicious = suspicious.head(20).to_dict(orient="records")

        return {
            "total_flows": total,
            "benign_count": benign_count,
            "malicious_count": malicious_count,
            "attack_distribution": distribution,
            "top_suspicious_flows": top_suspicious,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Preprocess *df* using the saved pipeline and feature names."""
        df = df.copy()
        df.columns = df.columns.str.strip()

        # Replace infinities
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if self.feature_names is not None:
            # Ensure all required columns exist
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            X = df[self.feature_names].copy()
        else:
            X = df.select_dtypes(include=[np.number]).copy()

        X.fillna(0, inplace=True)

        if self.pipeline is not None:
            return self.pipeline.transform(X)
        return X.values

    def _compute_confidence(self, X: np.ndarray) -> list:
        """Return per-row max class probability, or ``None`` list."""
        if hasattr(self.model, "predict_proba"):
            try:
                probas = self.model.predict_proba(X)
                return np.max(probas, axis=1).tolist()
            except Exception:
                return [None] * X.shape[0]
        return [None] * X.shape[0]
