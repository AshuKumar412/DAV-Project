"""
Feature Engineering Module for Network Intrusion Detection System
==================================================================

Provides the ``FeatureEngineer`` class for analysing network-traffic
features: identifying known IDS columns, computing correlations and
statistics, extracting model-based feature importances, detecting
suspicious traffic patterns, and preparing histogram data.
"""

import re
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd


# Well-known IDS feature names (canonical form).
_KNOWN_NETWORK_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Packet Length Mean",
    "Packet Length Max",
    "Packet Length Min",
    "SYN Flag Count",
    "ACK Flag Count",
    "Destination Port",
    "Protocol",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow IAT Mean",
]

# Commonly used network ports.
_COMMON_PORTS = {
    20, 21, 22, 23, 25, 53, 67, 68, 69, 80, 110, 119, 123,
    135, 137, 138, 139, 143, 161, 162, 179, 194, 389, 443,
    445, 465, 514, 515, 587, 631, 636, 993, 995, 1080, 1433,
    1434, 1521, 1723, 2049, 3306, 3389, 5432, 5900, 5901,
    6379, 8000, 8080, 8443, 8888, 9090, 9200, 27017,
}


def _normalise(name: str) -> str:
    """Lower-case, collapse whitespace / underscores / slashes."""
    return re.sub(r"[\s_/]+", "", name).lower()


class FeatureEngineer:
    """
    Network-traffic feature analysis toolkit.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyse.
    target_col : str or None, default=None
        Name of the label / target column (excluded from feature analysis).
    """

    def __init__(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> None:
        self.df = df.copy()
        self.target_col = target_col
        self._numeric_cols = self.df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        if self.target_col and self.target_col in self._numeric_cols:
            self._numeric_cols.remove(self.target_col)

    # ------------------------------------------------------------------
    # Known IDS features
    # ------------------------------------------------------------------

    def get_available_network_features(self) -> List[str]:
        """
        Return the subset of well-known IDS feature names that exist in
        the dataframe, using **case-insensitive** and
        **whitespace/underscore-agnostic** matching.

        Returns
        -------
        list of str
            Matched column names *as they appear in the dataframe*.
        """
        norm_to_col = {_normalise(c): c for c in self.df.columns}
        matched: List[str] = []
        for canonical in _KNOWN_NETWORK_FEATURES:
            norm_key = _normalise(canonical)
            if norm_key in norm_to_col:
                matched.append(norm_to_col[norm_key])
        return matched

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def compute_correlation_matrix(
        self, features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compute the Pearson correlation matrix for numeric features.

        Parameters
        ----------
        features : list of str or None
            Columns to include.  If ``None``, all numeric columns are used.

        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        cols = features if features is not None else self._numeric_cols
        cols = [c for c in cols if c in self.df.columns]
        return self.df[cols].corr()

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def compute_feature_importance(
        self, model: Any, feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Extract feature importances from a tree-based model.

        Parameters
        ----------
        model : estimator
            A fitted model with a ``feature_importances_`` attribute.
        feature_names : list of str
            Feature column names aligned with the model's input.

        Returns
        -------
        pd.DataFrame
            Two-column DataFrame (``feature``, ``importance``) sorted
            descending by importance.

        Raises
        ------
        AttributeError
            If the model does not expose ``feature_importances_``.
        """
        importances = model.feature_importances_
        imp_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        imp_df.sort_values("importance", ascending=False, inplace=True)
        imp_df.reset_index(drop=True, inplace=True)
        return imp_df

    # ------------------------------------------------------------------
    # Descriptive statistics
    # ------------------------------------------------------------------

    def compute_statistics(self) -> pd.DataFrame:
        """
        Return descriptive statistics for all numeric features.

        Returns
        -------
        pd.DataFrame
            Output of ``DataFrame.describe().T`` augmented with
            ``missing_count`` and ``missing_pct`` columns.
        """
        stats = self.df[self._numeric_cols].describe().T
        stats["missing_count"] = self.df[self._numeric_cols].isna().sum()
        stats["missing_pct"] = (
            stats["missing_count"] / len(self.df) * 100
        ).round(2)
        return stats

    # ------------------------------------------------------------------
    # Suspicious-pattern detection
    # ------------------------------------------------------------------

    def identify_suspicious_patterns(
        self, df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Flag records with unusual / suspicious network-traffic values.

        Checks performed (when the relevant columns exist):
        1. **High packet count** – ``Total Fwd Packets`` or
           ``Total Backward Packets`` above the 99th percentile.
        2. **Unusual port** – ``Destination Port`` not in the common-ports
           set.
        3. **Very high flow duration** – ``Flow Duration`` above the 99th
           percentile.
        4. **Zero-length packets** – ``Packet Length Mean`` equal to 0.

        Parameters
        ----------
        df : pd.DataFrame or None
            DataFrame to scan.  Defaults to ``self.df``.

        Returns
        -------
        pd.DataFrame
            Subset of suspicious records with an added ``reason`` column
            listing why the record was flagged (semicolon-separated if
            multiple reasons).
        """
        if df is None:
            df = self.df

        norm_to_col = {_normalise(c): c for c in df.columns}
        reasons: Dict[int, List[str]] = {}

        def _get(canonical: str) -> Optional[str]:
            return norm_to_col.get(_normalise(canonical))

        # 1. High packet counts
        for feat in ["Total Fwd Packets", "Total Backward Packets"]:
            col = _get(feat)
            if col and col in df.columns and np.issubdtype(df[col].dtype, np.number):
                threshold = df[col].quantile(0.99)
                idxs = df.index[df[col] > threshold].tolist()
                for i in idxs:
                    reasons.setdefault(i, []).append(f"High {feat}")

        # 2. Unusual ports
        port_col = _get("Destination Port")
        if port_col and port_col in df.columns:
            try:
                unusual = df.index[
                    ~df[port_col].astype(float).isin(_COMMON_PORTS)
                ].tolist()
                for i in unusual:
                    reasons.setdefault(i, []).append("Unusual port")
            except (ValueError, TypeError):
                pass

        # 3. Very high flow duration
        dur_col = _get("Flow Duration")
        if dur_col and dur_col in df.columns and np.issubdtype(df[dur_col].dtype, np.number):
            threshold = df[dur_col].quantile(0.99)
            idxs = df.index[df[dur_col] > threshold].tolist()
            for i in idxs:
                reasons.setdefault(i, []).append("Very high flow duration")

        # 4. Zero-length packets
        pkt_col = _get("Packet Length Mean")
        if pkt_col and pkt_col in df.columns and np.issubdtype(df[pkt_col].dtype, np.number):
            idxs = df.index[df[pkt_col] == 0].tolist()
            for i in idxs:
                reasons.setdefault(i, []).append("Zero-length packets")

        if not reasons:
            return pd.DataFrame(columns=list(df.columns) + ["reason"])

        suspicious = df.loc[list(reasons.keys())].copy()
        suspicious["reason"] = [
            "; ".join(reasons[i]) for i in suspicious.index
        ]
        return suspicious

    # ------------------------------------------------------------------
    # Histogram / distribution data
    # ------------------------------------------------------------------

    def get_feature_distributions(
        self,
        features: Optional[List[str]] = None,
        bins: int = 50,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute histogram data for selected features.

        Parameters
        ----------
        features : list of str or None
            Features to compute histograms for.  If ``None``, all numeric
            features are used.
        bins : int, default=50
            Number of histogram bins.

        Returns
        -------
        dict
            ``{feature_name: {'counts': list, 'bin_edges': list}}``.
        """
        cols = features if features is not None else self._numeric_cols
        distributions: Dict[str, Dict[str, Any]] = {}
        for col in cols:
            if col not in self.df.columns:
                continue
            data = self.df[col].dropna()
            if data.empty:
                continue
            counts, bin_edges = np.histogram(data, bins=bins)
            distributions[col] = {
                "counts": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
            }
        return distributions
