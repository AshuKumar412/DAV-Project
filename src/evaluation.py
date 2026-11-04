"""
Model Evaluation Module for Network Intrusion Detection System
================================================================

Provides the ``ModelEvaluator`` class for computing classification
metrics, comparing multiple models, generating ROC-AUC curves, and
producing evaluation reports.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


class ModelEvaluator:
    """
    Evaluate and compare classification models for the NIDS.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Single-model evaluation
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        """
        Compute a comprehensive suite of classification metrics.

        Parameters
        ----------
        model : estimator
            Fitted sklearn / xgboost classifier.
        X_test : np.ndarray
            Test feature matrix.
        y_test : np.ndarray
            True integer-encoded labels.
        model_name : str, default='Model'
            Human-readable name stored in the result dict.

        Returns
        -------
        dict
            Keys: ``model_name``, ``accuracy``, ``precision_macro``,
            ``precision_weighted``, ``recall_macro``, ``recall_weighted``,
            ``f1_macro``, ``f1_weighted``, ``confusion_matrix``,
            ``classification_report``, ``false_positive_rate``,
            ``false_negative_rate``.
        """
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        prec_macro = precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        prec_weighted = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        rec_macro = recall_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        rec_weighted = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        f1_mac = f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        f1_wt = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )

        fpr_dict, fnr_dict = self.compute_fpr_fnr(y_test, y_pred)

        return {
            "model_name": model_name,
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_macro": round(rec_macro, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_macro": round(f1_mac, 4),
            "f1_weighted": round(f1_wt, 4),
            "confusion_matrix": cm,
            "classification_report": cr,
            "false_positive_rate": fpr_dict,
            "false_negative_rate": fnr_dict,
        }

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self, results_dict: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Build a comparison table from multiple evaluation results.

        Parameters
        ----------
        results_dict : dict
            ``{model_name: metrics_dict}`` as returned by
            :meth:`evaluate_model`.

        Returns
        -------
        pd.DataFrame
            One row per model with key scalar metrics.
        """
        rows = []
        for name, metrics in results_dict.items():
            rows.append(
                {
                    "Model": name,
                    "Accuracy": metrics.get("accuracy"),
                    "Precision (Macro)": metrics.get("precision_macro"),
                    "Precision (Weighted)": metrics.get("precision_weighted"),
                    "Recall (Macro)": metrics.get("recall_macro"),
                    "Recall (Weighted)": metrics.get("recall_weighted"),
                    "F1 (Macro)": metrics.get("f1_macro"),
                    "F1 (Weighted)": metrics.get("f1_weighted"),
                }
            )
        return pd.DataFrame(rows).set_index("Model")

    # ------------------------------------------------------------------
    # ROC-AUC
    # ------------------------------------------------------------------

    def compute_roc_auc(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute ROC-AUC, handling both binary and multiclass cases.

        For multiclass problems the One-vs-Rest (OVR) strategy is used.

        Parameters
        ----------
        model : estimator
            Fitted classifier.
        X_test : np.ndarray
            Test feature matrix.
        y_test : np.ndarray
            True integer-encoded labels.

        Returns
        -------
        dict or None
            For **binary** classification:
                ``{'fpr': array, 'tpr': array, 'auc': float}``
            For **multiclass**:
                ``{'fpr': {cls: array}, 'tpr': {cls: array},
                  'auc': {cls: float}, 'classes': list}``
            Returns ``None`` if the model does not support
            ``predict_proba``.
        """
        if not hasattr(model, "predict_proba"):
            return None

        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            return None

        classes = np.unique(y_test)
        n_classes = len(classes)

        if n_classes == 2:
            # Binary case
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            auc_val = roc_auc_score(y_test, y_prob[:, 1])
            return {"fpr": fpr, "tpr": tpr, "auc": round(auc_val, 4)}

        # Multiclass OVR
        y_bin = label_binarize(y_test, classes=classes)
        fpr_dict: Dict[Any, np.ndarray] = {}
        tpr_dict: Dict[Any, np.ndarray] = {}
        auc_dict: Dict[Any, float] = {}

        for i, cls in enumerate(classes):
            if i < y_prob.shape[1]:
                fpr_dict[int(cls)], tpr_dict[int(cls)], _ = roc_curve(
                    y_bin[:, i], y_prob[:, i]
                )
                try:
                    auc_dict[int(cls)] = round(
                        roc_auc_score(y_bin[:, i], y_prob[:, i]), 4
                    )
                except ValueError:
                    auc_dict[int(cls)] = 0.0

        return {
            "fpr": fpr_dict,
            "tpr": tpr_dict,
            "auc": auc_dict,
            "classes": classes.tolist(),
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(
        self, model: Any, feature_names: List[str]
    ) -> Optional[pd.DataFrame]:
        """
        Extract feature importances from tree-based models.

        Parameters
        ----------
        model : estimator
            Fitted model.
        feature_names : list of str
            Feature column names.

        Returns
        -------
        pd.DataFrame or None
            Two-column DataFrame (``feature``, ``importance``) sorted
            descending, or ``None`` if the model does not expose
            ``feature_importances_``.
        """
        if not hasattr(model, "feature_importances_"):
            return None
        importances = model.feature_importances_
        imp_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        imp_df.sort_values("importance", ascending=False, inplace=True)
        imp_df.reset_index(drop=True, inplace=True)
        return imp_df

    # ------------------------------------------------------------------
    # FPR / FNR per class
    # ------------------------------------------------------------------

    def compute_fpr_fnr(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple:
        """
        Compute per-class false-positive rate and false-negative rate.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.

        Returns
        -------
        fpr_dict : dict
            ``{class_label: FPR}``
        fnr_dict : dict
            ``{class_label: FNR}``
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        fpr_dict: Dict[Any, float] = {}
        fnr_dict: Dict[Any, float] = {}

        for cls in classes:
            # True positives, false positives, false negatives, true negatives
            tp = int(np.sum((y_pred == cls) & (y_true == cls)))
            fp = int(np.sum((y_pred == cls) & (y_true != cls)))
            fn = int(np.sum((y_pred != cls) & (y_true == cls)))
            tn = int(np.sum((y_pred != cls) & (y_true != cls)))

            fpr_dict[int(cls)] = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
            fnr_dict[int(cls)] = round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0

        return fpr_dict, fnr_dict

    # ------------------------------------------------------------------
    # Evaluation report
    # ------------------------------------------------------------------

    def generate_evaluation_report(
        self, all_results: Dict[str, Dict[str, Any]], output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create a comprehensive evaluation DataFrame and save it to
        ``reports/evaluation_report.csv`` or custom path.

        Parameters
        ----------
        all_results : dict
            ``{model_name: metrics_dict}``.
        output_path : str, optional
            Custom CSV output destination.

        Returns
        -------
        pd.DataFrame
            Comparison table written to disk.
        """
        comparison = self.compare_models(all_results)

        if output_path:
            report_path = Path(output_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            reports_dir = Path("reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = reports_dir / "evaluation_report.csv"

        comparison.to_csv(report_path)
        return comparison
