"""
Model Training Module for Network Intrusion Detection System
==============================================================

Provides the ``ModelTrainer`` class for training, saving, loading, and
listing machine-learning classifiers used by the NIDS.  Supported models
include Random Forest, Decision Tree, Logistic Regression, and XGBoost.
"""

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover – graceful fallback
    XGBClassifier = None  # type: ignore[misc,assignment]


class ModelTrainer:
    """
    Train, persist, and manage NIDS classification models.

    Parameters
    ----------
    models_dir : str, default='models'
        Directory where trained model artifacts are stored.
    """

    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Model catalogue
    # ------------------------------------------------------------------

    def get_available_models(self) -> Dict[str, Any]:
        """
        Return a dictionary of model name → unfitted model instance.

        Available models:
        - ``'Random Forest'``
        - ``'Decision Tree'``
        - ``'Logistic Regression'``
        - ``'XGBoost'`` (only if *xgboost* is installed)

        Returns
        -------
        dict
            ``{model_name: model_instance}``.
        """
        models: Dict[str, Any] = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42
            ),
        }
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric="mlogloss",
            )
        return models

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Any, float]:
        """
        Train a single model by name.

        Parameters
        ----------
        model_name : str
            Key from :meth:`get_available_models`.
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training labels (integer-encoded).
        progress_callback : callable or None
            Optional callback invoked with status messages during training
            (e.g., for UI progress bars).

        Returns
        -------
        model : estimator
            Fitted sklearn / xgboost model.
        training_time : float
            Wall-clock training time in seconds.

        Raises
        ------
        ValueError
            If *model_name* is not in the available catalogue.
        """
        models = self.get_available_models()
        if model_name not in models:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(models.keys())}"
            )

        model = models[model_name]

        if progress_callback:
            progress_callback(f"Training {model_name}...")

        start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start

        if progress_callback:
            progress_callback(
                f"{model_name} trained in {training_time:.2f}s"
            )

        return model, training_time

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train every model in the catalogue.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training labels (integer-encoded).
        progress_callback : callable or None
            Optional callback invoked with status messages.

        Returns
        -------
        dict
            ``{model_name: {'model': fitted_model, 'time': seconds}}``.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for name in self.get_available_models():
            model, elapsed = self.train_model(
                name, X_train, y_train, progress_callback
            )
            results[name] = {"model": model, "time": elapsed}
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert a human-readable model name to a file-safe snake_case string."""
        s = name.strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")

    def save_model(
        self,
        model: Any,
        model_name: str,
        pipeline: Optional[Any] = None,
        label_encoder: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> str:
        """
        Serialize a trained model and its associated artifacts to disk.

        The artifacts are stored as a single dictionary in a ``.pkl`` file
        under the ``models/`` directory.

        Parameters
        ----------
        model : estimator
            Fitted model.
        model_name : str
            Human-readable model name (used to derive the filename).
        pipeline : object or None
            Fitted preprocessing pipeline.
        label_encoder : object or None
            Fitted label encoder.
        feature_names : list of str or None
            Feature column names used during training.

        Returns
        -------
        str
            Absolute path of the saved pickle file.
        """
        artifacts = {
            "model": model,
            "pipeline": pipeline,
            "label_encoder": label_encoder,
            "feature_names": feature_names,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
        }
        filename = f"{self._to_snake_case(model_name)}.pkl"
        save_path = self.models_dir / filename
        joblib.dump(artifacts, save_path)
        return str(save_path.resolve())

    def load_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load model artifacts from a ``.pkl`` file.

        Parameters
        ----------
        model_name : str
            Human-readable model name (converted to snake_case filename).

        Returns
        -------
        dict or None
            Artifacts dictionary, or ``None`` if the file does not exist.
        """
        filename = f"{self._to_snake_case(model_name)}.pkl"
        load_path = self.models_dir / filename
        if not load_path.exists():
            return None
        return joblib.load(load_path)

    def list_saved_models(self) -> List[str]:
        """
        List all saved model names (derived from ``.pkl`` filenames).

        Returns
        -------
        list of str
            Model names (snake_case, without extension).
        """
        if not self.models_dir.exists():
            return []
        return [p.stem for p in sorted(self.models_dir.glob("*.pkl"))]
