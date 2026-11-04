"""
Data Preprocessing Module for Network Intrusion Detection System
=================================================================

Provides the ``DataPreprocessor`` class that handles the full data-cleaning
and transformation pipeline: missing-value imputation, duplicate removal,
label encoding, feature scaling, and stratified train/test splitting.
Preprocessing artifacts (pipeline, encoder, feature names) can be persisted
with joblib for later inference.
"""

import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """
    End-to-end data preprocessing for network intrusion detection datasets.

    Parameters
    ----------
    max_samples : int or None, default=None
        If set, randomly sample this many rows from the dataset before
        processing.  Useful for quick prototyping on large files.
    """

    # Known label column names in common IDS datasets.
    _LABEL_CANDIDATES = [
        "Label",
        "label",
        "Attack",
        "attack",
        "attack_cat",
        "class",
        "Class",
        "target",
    ]

    def __init__(self, max_samples: Optional[int] = None) -> None:
        self.max_samples = max_samples
        self._label_encoder: Optional[LabelEncoder] = None
        self._label_mapping: Optional[Dict[int, str]] = None
        self._pipeline: Optional[Pipeline] = None
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Target / feature identification
    # ------------------------------------------------------------------

    def identify_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Auto-detect the label / target column in *df*.

        Searches for known label column names (case-sensitive) and returns
        the first match.  If none is found, returns ``None``.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        str or None
            Column name of the detected target, or ``None``.
        """
        for candidate in self._LABEL_CANDIDATES:
            if candidate in df.columns:
                return candidate
        # Also check stripped column names
        stripped = {col.strip(): col for col in df.columns}
        for candidate in self._LABEL_CANDIDATES:
            if candidate in stripped:
                return stripped[candidate]
        return None

    def identify_features(
        self, df: pd.DataFrame, target_col: str
    ) -> List[str]:
        """
        Identify numeric feature columns, excluding the target column.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe (should already be cleaned).
        target_col : str
            Name of the target / label column to exclude.

        Returns
        -------
        list of str
            Sorted list of numeric feature column names.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [c for c in numeric_cols if c != target_col]
        return features

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataframe in-place (returns a copy):

        1. Strip whitespace from column names.
        2. Replace ``inf`` / ``-inf`` with ``NaN``.
        3. Fill numeric NaN with column median.
        4. Fill categorical NaN with column mode.
        5. Remove duplicate rows.

        Column names are kept in their original form for display; an
        internal normalised version (lowered, underscored) is *not*
        applied to the DataFrame itself so downstream code sees the
        original headers.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe.
        """
        df = df.copy()

        # Strip leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()

        # Replace infinities
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # Fill categorical columns with mode
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])

        # Remove duplicates
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    # ------------------------------------------------------------------
    # Label encoding
    # ------------------------------------------------------------------

    def encode_labels(
        self, y: pd.Series
    ) -> Tuple[np.ndarray, LabelEncoder]:
        """
        Encode string labels to integers using ``LabelEncoder``.

        Parameters
        ----------
        y : pd.Series
            Label series (string or mixed type).

        Returns
        -------
        encoded : np.ndarray
            Integer-encoded labels.
        encoder : sklearn.preprocessing.LabelEncoder
            Fitted encoder instance (stored internally as well).
        """
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(y.astype(str))
        self._label_encoder = encoder
        self._label_mapping = dict(
            zip(encoder.transform(encoder.classes_), encoder.classes_)
        )
        return encoded, encoder

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """
        Build an sklearn ``Pipeline`` with ``StandardScaler`` for all
        numeric features present in *X*.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (numeric columns only).

        Returns
        -------
        sklearn.pipeline.Pipeline
            Unfitted pipeline ready for ``fit`` / ``fit_transform``.
        """
        pipeline = Pipeline([("scaler", StandardScaler())])
        self._pipeline = pipeline
        return pipeline

    # ------------------------------------------------------------------
    # Main preparation entry-point
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        test_size: float = 0.2,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Pipeline,
        LabelEncoder,
        List[str],
    ]:
        """
        Full preprocessing pipeline: clean → identify features → encode
        labels → build pipeline → stratified train/test split.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe.
        target_col : str or None, default=None
            Name of the target column.  If ``None``, the method will
            attempt to auto-detect it.
        test_size : float, default=0.2
            Fraction of data reserved for testing.

        Returns
        -------
        X_train : np.ndarray
        X_test : np.ndarray
        y_train : np.ndarray
        y_test : np.ndarray
        pipeline : sklearn.pipeline.Pipeline
            Fitted scaler pipeline.
        label_encoder : sklearn.preprocessing.LabelEncoder
            Fitted label encoder.
        feature_names : list of str

        Raises
        ------
        ValueError
            If the target column cannot be found.
        """
        # Optional down-sampling
        if self.max_samples is not None and len(df) > self.max_samples:
            df = df.sample(n=self.max_samples, random_state=42)

        # Clean
        df = self.clean_data(df)

        # Identify target
        if target_col is None:
            target_col = self.identify_target_column(df)
        if target_col is None:
            raise ValueError(
                "Could not auto-detect the target column. "
                "Please specify it explicitly."
            )

        # Identify features
        feature_names = self.identify_features(df, target_col)
        self._feature_names = feature_names

        X = df[feature_names].copy()
        y = df[target_col].copy()

        # Encode labels
        y_encoded, label_encoder = self.encode_labels(y)

        # Build and fit pipeline
        pipeline = self.build_pipeline(X)

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=42,
            stratify=y_encoded,
        )

        # Fit pipeline on training data, transform both sets
        X_train = pipeline.fit_transform(X_train)
        X_test = pipeline.transform(X_test)

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            pipeline,
            label_encoder,
            feature_names,
        )

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    def transform_new_data(
        self,
        df: pd.DataFrame,
        pipeline: Pipeline,
        feature_names: List[str],
    ) -> np.ndarray:
        """
        Apply a previously fitted pipeline to new data for inference.

        Missing feature columns are filled with 0 so that the
        transformation shape matches the training data.

        Parameters
        ----------
        df : pd.DataFrame
            New (unseen) dataframe.
        pipeline : sklearn.pipeline.Pipeline
            Fitted preprocessing pipeline.
        feature_names : list of str
            Feature column names expected by the pipeline.

        Returns
        -------
        np.ndarray
            Transformed feature matrix.
        """
        df = df.copy()
        df.columns = df.columns.str.strip()

        # Replace infinities and fill NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Ensure all required columns exist
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        X = df[feature_names].copy()

        # Fill remaining NaN with 0
        X.fillna(0, inplace=True)

        return pipeline.transform(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_artifacts(self, path: str = "models/preprocessing_artifacts.pkl") -> str:
        """
        Save preprocessing artifacts (pipeline, label encoder, feature
        names, label mapping) to disk with joblib.

        Parameters
        ----------
        path : str
            Destination file path.

        Returns
        -------
        str
            Absolute path of the saved file.
        """
        artifacts = {
            "pipeline": self._pipeline,
            "label_encoder": self._label_encoder,
            "label_mapping": self._label_mapping,
            "feature_names": self._feature_names,
        }
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts, save_path)
        return str(save_path.resolve())

    @staticmethod
    def load_artifacts(path: str = "models/preprocessing_artifacts.pkl") -> Dict[str, Any]:
        """
        Load previously saved preprocessing artifacts.

        Parameters
        ----------
        path : str
            Path to the artifacts pickle file.

        Returns
        -------
        dict
            Dictionary containing 'pipeline', 'label_encoder',
            'label_mapping', and 'feature_names'.
        """
        return joblib.load(path)
