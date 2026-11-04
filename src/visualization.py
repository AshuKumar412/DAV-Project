"""
Visualization Module for Network Intrusion Detection System
=============================================================

All functions return **Plotly** figure objects styled with a dark
transparent theme suitable for embedding in Streamlit dashboards.

Common style:
- ``template='plotly_dark'``
- ``paper_bgcolor='rgba(0,0,0,0)'``, ``plot_bgcolor='rgba(0,0,0,0)'``
- White text and low-opacity gridlines
- Height ≈ 400–500 px
"""

from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ======================================================================
# Shared dark-theme layout defaults
# ======================================================================

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    height=450,
    margin=dict(l=60, r=30, t=50, b=50),
)

_GRID_STYLE = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.1)",
    zeroline=False,
)

# Colour palettes
_ATTACK_COLORS = {
    "BENIGN": "#2ecc71",
    "Benign": "#2ecc71",
    "DoS": "#e74c3c",
    "DDoS": "#c0392b",
    "DoS Hulk": "#e74c3c",
    "DoS GoldenEye": "#e74c3c",
    "DoS slowloris": "#e74c3c",
    "DoS Slowhttptest": "#e74c3c",
    "PortScan": "#e67e22",
    "Port Scan": "#e67e22",
    "Bot": "#9b59b6",
    "Brute Force": "#f1c40f",
    "BruteForce": "#f1c40f",
    "Web Attack": "#f1c40f",
    "Infiltration": "#f1c40f",
}

_SEVERITY_COLORS = {
    "LOW": "#2ecc71",
    "MEDIUM": "#f1c40f",
    "HIGH": "#e67e22",
    "CRITICAL": "#e74c3c",
}


def _apply_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply shared dark-theme layout to a figure."""
    fig.update_layout(title=title, **_LAYOUT_DEFAULTS)
    fig.update_xaxes(**_GRID_STYLE)
    fig.update_yaxes(**_GRID_STYLE)
    return fig


def _get_attack_color(label: str) -> str:
    """Look up colour for an attack label, with a fallback."""
    return _ATTACK_COLORS.get(label, "#3498db")


# ======================================================================
# Chart functions
# ======================================================================


def create_attack_distribution_chart(labels_series: pd.Series) -> go.Figure:
    """
    Pie chart of attack categories.

    Parameters
    ----------
    labels_series : pd.Series
        Series of string labels (one per flow).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if labels_series is None or len(labels_series) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No attack data available", showarrow=False, font=dict(color="#8892b0", size=14))
        return _apply_layout(fig, "Attack Category Distribution")

    counts = labels_series.value_counts()
    colors = [_get_attack_color(lbl) for lbl in counts.index]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts.index,
                values=counts.values,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hole=0.3,
            )
        ]
    )
    return _apply_layout(fig, "Attack Category Distribution")


def create_traffic_timeline(
    df: pd.DataFrame, time_col: Optional[str] = None
) -> go.Figure:
    """
    Line chart of traffic volume over time.

    If *time_col* is ``None`` or the column does not exist, a synthetic
    index-based timeline is created.

    Parameters
    ----------
    df : pd.DataFrame
    time_col : str or None

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if time_col and time_col in df.columns:
        x = pd.to_datetime(df[time_col], errors="coerce")
    else:
        x = pd.RangeIndex(len(df))

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=list(range(len(df))),
                mode="lines",
                line=dict(color="#3498db", width=1),
                name="Flow Count",
            )
        ]
    )
    fig.update_yaxes(title_text="Cumulative Flows")
    fig.update_xaxes(title_text="Time" if time_col else "Flow Index")
    return _apply_layout(fig, "Network Traffic Timeline")


def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Interactive Plotly heatmap of a correlation matrix.

    Parameters
    ----------
    corr_matrix : pd.DataFrame

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )
    return _apply_layout(fig, "Feature Correlation Matrix")


def create_feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of top-15 most important features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Must contain ``feature`` and ``importance`` columns.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if (
        importance_df is None
        or not isinstance(importance_df, pd.DataFrame)
        or importance_df.empty
        or "importance" not in importance_df.columns
        or "feature" not in importance_df.columns
    ):
        fig = go.Figure()
        fig.add_annotation(text="No feature importance data available", showarrow=False, font=dict(color="#8892b0", size=14))
        return _apply_layout(fig, "Top 15 Feature Importances")

    top = importance_df.head(15).sort_values("importance", ascending=True)
    fig = go.Figure(
        data=[
            go.Bar(
                x=top["importance"],
                y=top["feature"],
                orientation="h",
                marker=dict(
                    color=top["importance"],
                    colorscale="Viridis",
                ),
            )
        ]
    )
    fig.update_xaxes(title_text="Importance")
    fig.update_yaxes(title_text="Feature")
    return _apply_layout(fig, "Top 15 Feature Importances")


def create_confusion_matrix_heatmap(
    cm: np.ndarray, labels: List[str]
) -> go.Figure:
    """
    Annotated heatmap for a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (n_classes × n_classes).
    labels : list of str
        Class label strings.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Build annotation text
    text = [[str(val) for val in row] for row in cm]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_xaxes(title_text="Predicted Label")
    fig.update_yaxes(title_text="True Label")
    return _apply_layout(fig, "Confusion Matrix")


def create_model_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing models on key metrics.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Indexed by model name with columns such as ``Accuracy``,
        ``Precision (Macro)``, ``Recall (Macro)``, ``F1 (Macro)``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if comparison_df is None or not isinstance(comparison_df, pd.DataFrame) or comparison_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No model comparison metrics available", showarrow=False, font=dict(color="#8892b0", size=14))
        return _apply_layout(fig, "Model Comparison")

    metrics = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"]
    available = [m for m in metrics if m in comparison_df.columns]

    # Fallback: try weighted variants
    if not available:
        metrics_alt = [
            "Accuracy",
            "Precision (Weighted)",
            "Recall (Weighted)",
            "F1 (Weighted)",
        ]
        available = [m for m in metrics_alt if m in comparison_df.columns]

    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6"]

    fig = go.Figure()
    for i, metric in enumerate(available):
        fig.add_trace(
            go.Bar(
                name=metric,
                x=comparison_df.index.tolist(),
                y=comparison_df[metric].tolist(),
                marker_color=colors[i % len(colors)],
            )
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Model")
    fig.update_yaxes(title_text="Score", range=[0, 1])
    return _apply_layout(fig, "Model Comparison")


def create_anomaly_scatter(
    df: pd.DataFrame,
    score_col: str = "anomaly_score",
    label_col: str = "is_anomaly",
) -> go.Figure:
    """
    Scatter plot coloured by anomaly status.

    Uses the first two numeric feature columns as x/y axes if available,
    otherwise falls back to the dataframe index.

    Parameters
    ----------
    df : pd.DataFrame
    score_col : str
    label_col : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or label_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No anomaly data available", showarrow=False, font=dict(color="#8892b0", size=14))
        return _apply_layout(fig, "Anomaly Scatter Plot")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude score/label helper columns for axis selection
    axis_candidates = [
        c for c in numeric_cols if c not in (score_col, label_col)
    ]

    x_col = axis_candidates[0] if len(axis_candidates) >= 1 else df.index
    y_col = axis_candidates[1] if len(axis_candidates) >= 2 else score_col

    x_data = df[x_col] if isinstance(x_col, str) else x_col
    y_data = df[y_col]

    colors = df[label_col].map({True: "#e74c3c", False: "#2ecc71"})

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="markers",
                marker=dict(
                    color=colors,
                    size=4,
                    opacity=0.6,
                ),
                text=df[score_col].round(3).astype(str),
                hovertemplate="Score: %{text}<extra></extra>",
            )
        ]
    )
    x_label = x_col if isinstance(x_col, str) else "Index"
    y_label = y_col if isinstance(y_col, str) else "Score"
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    return _apply_layout(fig, "Anomaly Detection Results")


def create_protocol_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of protocol distribution.

    Looks for a ``Protocol`` column (case-insensitive).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    plotly.graph_objects.Figure
    """
    col = _find_column(df, "Protocol")
    if col is None:
        fig = go.Figure()
        fig.add_annotation(text="No Protocol column found", showarrow=False)
        return _apply_layout(fig, "Protocol Distribution")

    counts = df[col].value_counts()
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.astype(str),
                y=counts.values,
                marker_color="#3498db",
            )
        ]
    )
    fig.update_xaxes(title_text="Protocol")
    fig.update_yaxes(title_text="Count")
    return _apply_layout(fig, "Protocol Distribution")


def create_port_distribution(
    df: pd.DataFrame, top_n: int = 20
) -> go.Figure:
    """
    Bar chart of the top *top_n* destination ports.

    Parameters
    ----------
    df : pd.DataFrame
    top_n : int, default=20

    Returns
    -------
    plotly.graph_objects.Figure
    """
    col = _find_column(df, "Destination Port")
    if col is None:
        fig = go.Figure()
        fig.add_annotation(
            text="No Destination Port column found", showarrow=False
        )
        return _apply_layout(fig, "Top Destination Ports")

    counts = df[col].value_counts().head(top_n)
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.astype(str),
                y=counts.values,
                marker_color="#e67e22",
            )
        ]
    )
    fig.update_xaxes(title_text="Destination Port")
    fig.update_yaxes(title_text="Count")
    return _apply_layout(fig, f"Top {top_n} Destination Ports")


def create_packet_length_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of packet lengths.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    plotly.graph_objects.Figure
    """
    col = _find_column(df, "Packet Length Mean")
    if col is None:
        fig = go.Figure()
        fig.add_annotation(
            text="No Packet Length Mean column found", showarrow=False
        )
        return _apply_layout(fig, "Packet Length Distribution")

    fig = go.Figure(
        data=[
            go.Histogram(
                x=df[col].dropna(),
                nbinsx=50,
                marker_color="#9b59b6",
            )
        ]
    )
    fig.update_xaxes(title_text="Packet Length Mean")
    fig.update_yaxes(title_text="Frequency")
    return _apply_layout(fig, "Packet Length Distribution")


def create_roc_curve(roc_data: Dict[str, Any]) -> go.Figure:
    """
    Plot ROC curve(s).

    Accepts the output of ``ModelEvaluator.compute_roc_auc``.

    Parameters
    ----------
    roc_data : dict
        Binary: ``{'fpr': array, 'tpr': array, 'auc': float}``
        Multiclass: ``{'fpr': {cls: array}, 'tpr': {cls: array},
        'auc': {cls: float}, 'classes': list}``

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    if "classes" in roc_data:
        # Multiclass
        colors = px.colors.qualitative.Set2
        for i, cls in enumerate(roc_data["classes"]):
            cls_key = int(cls)
            fpr = roc_data["fpr"].get(cls_key, [])
            tpr = roc_data["tpr"].get(cls_key, [])
            auc_val = roc_data["auc"].get(cls_key, 0)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"Class {cls} (AUC={auc_val:.3f})",
                    line=dict(color=colors[i % len(colors)]),
                )
            )
    else:
        # Binary
        fig.add_trace(
            go.Scatter(
                x=roc_data["fpr"],
                y=roc_data["tpr"],
                mode="lines",
                name=f"ROC (AUC={roc_data['auc']:.3f})",
                line=dict(color="#3498db"),
            )
        )

    # Diagonal reference
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
        )
    )
    fig.update_xaxes(title_text="False Positive Rate")
    fig.update_yaxes(title_text="True Positive Rate")
    return _apply_layout(fig, "ROC Curve")


def create_severity_distribution(events_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of severity levels with colour coding.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must contain a ``severity`` column.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    if events_df is None or not isinstance(events_df, pd.DataFrame) or events_df.empty or "severity" not in events_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No security events recorded", showarrow=False, font=dict(color="#8892b0", size=14))
        return _apply_layout(fig, "Security Event Severity Distribution")

    counts = events_df["severity"].value_counts()
    # Reindex to ensure consistent ordering
    counts = counts.reindex(order).fillna(0).astype(int)
    colors = [_SEVERITY_COLORS.get(s, "#3498db") for s in counts.index]

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color=colors,
            )
        ]
    )
    fig.update_xaxes(title_text="Severity Level")
    fig.update_yaxes(title_text="Event Count")
    return _apply_layout(fig, "Security Event Severity Distribution")


def create_normal_vs_malicious_chart(
    benign_count: int, malicious_count: int
) -> go.Figure:
    """
    Donut chart comparing normal vs malicious traffic.

    Parameters
    ----------
    benign_count : int
    malicious_count : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Normal (Benign)", "Malicious"],
                values=[benign_count, malicious_count],
                marker=dict(colors=["#2ecc71", "#e74c3c"]),
                hole=0.5,
                textinfo="label+percent+value",
            )
        ]
    )
    return _apply_layout(fig, "Normal vs Malicious Traffic")


# ======================================================================
# Internal helpers
# ======================================================================


def _find_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    """Case-insensitive, whitespace-agnostic column lookup."""
    import re as _re

    norm = _re.sub(r"[\s_/]+", "", canonical).lower()
    for col in df.columns:
        if _re.sub(r"[\s_/]+", "", col).lower() == norm:
            return col
    return None
