"""
Utility Functions for Network Intrusion Detection System
=========================================================

Provides common helper functions used across the NIDS application,
including seed management, directory creation, file validation,
number formatting, severity classification, and security event generation.
"""

import os
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Union

import numpy as np
import pandas as pd


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy and Python's random module.

    Parameters
    ----------
    seed : int, default=42
        The seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)


def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    Works on any OS by resolving relative to this file's location.
    The project root is defined as two levels up from this source file
    (i.e., src/utils.py → project_root/).

    Returns
    -------
    pathlib.Path
        Absolute path to the project root directory.
    """
    return Path(__file__).resolve().parent.parent


def ensure_directories() -> dict:
    """
    Create all required project directories if they do not already exist.

    Creates the following directory structure under the project root:
        - data/
        - data/raw/
        - data/processed/
        - models/
        - reports/

    Returns
    -------
    dict
        Mapping of directory name to its absolute Path object.
    """
    root = get_project_root()
    directories = {
        "data": root / "data",
        "data_raw": root / "data" / "raw",
        "data_processed": root / "data" / "processed",
        "models": root / "models",
        "reports": root / "reports",
    }
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return directories


def validate_csv_file(file, max_size_mb: float = 200) -> dict:
    """
    Validate an uploaded CSV file for size and format constraints.

    Parameters
    ----------
    file : file-like object or str or pathlib.Path
        The CSV file to validate. Accepts a file path (str/Path) or a
        file-like object with `name` and `size` attributes (e.g., Streamlit
        UploadedFile).
    max_size_mb : float, default=200
        Maximum allowed file size in megabytes.

    Returns
    -------
    dict
        Validation result with keys:
        - 'valid' (bool): Whether the file passed all checks.
        - 'message' (str): Human-readable status or error description.
        - 'size_mb' (float or None): File size in MB if determinable.
    """
    max_size_bytes = max_size_mb * 1024 * 1024

    # Handle file path (string or Path)
    if isinstance(file, (str, Path)):
        file_path = Path(file)
        if not file_path.exists():
            return {"valid": False, "message": "File does not exist.", "size_mb": None}
        if not file_path.suffix.lower() == ".csv":
            return {
                "valid": False,
                "message": "File must be a CSV file (.csv extension).",
                "size_mb": None,
            }
        file_size = file_path.stat().st_size
        if file_size > max_size_bytes:
            size_mb = file_size / (1024 * 1024)
            return {
                "valid": False,
                "message": (
                    f"File size ({size_mb:.1f} MB) exceeds the maximum "
                    f"allowed size ({max_size_mb} MB)."
                ),
                "size_mb": size_mb,
            }
        # Try reading a few rows to confirm it is valid CSV
        try:
            pd.read_csv(file_path, nrows=5)
        except Exception as exc:
            return {
                "valid": False,
                "message": f"File is not a valid CSV: {exc}",
                "size_mb": file_size / (1024 * 1024),
            }
        return {
            "valid": True,
            "message": "File is valid.",
            "size_mb": file_size / (1024 * 1024),
        }

    # Handle file-like object (e.g., Streamlit UploadedFile)
    file_name = getattr(file, "name", "")
    if not file_name.lower().endswith(".csv"):
        return {
            "valid": False,
            "message": "File must be a CSV file (.csv extension).",
            "size_mb": None,
        }

    file_size = getattr(file, "size", None)
    if file_size is not None:
        if file_size > max_size_bytes:
            size_mb = file_size / (1024 * 1024)
            return {
                "valid": False,
                "message": (
                    f"File size ({size_mb:.1f} MB) exceeds the maximum "
                    f"allowed size ({max_size_mb} MB)."
                ),
                "size_mb": size_mb,
            }
        size_mb = file_size / (1024 * 1024)
    else:
        size_mb = None

    # Try reading a few rows
    try:
        current_pos = file.tell() if hasattr(file, "tell") else 0
        pd.read_csv(file, nrows=5)
        if hasattr(file, "seek"):
            file.seek(current_pos)
    except Exception as exc:
        return {
            "valid": False,
            "message": f"File is not a valid CSV: {exc}",
            "size_mb": size_mb,
        }

    return {"valid": True, "message": "File is valid.", "size_mb": size_mb}


def format_number(n: Union[int, float]) -> str:
    """
    Format a large number with K/M suffixes for readability.

    Parameters
    ----------
    n : int or float
        The number to format.

    Returns
    -------
    str
        Formatted string (e.g., '1.5K', '2.3M', '42').
    """
    if n is None:
        return "0"
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    if n == int(n):
        return str(int(n))
    return f"{n:.2f}"


# ---------------------------------------------------------------------------
# Severity classification rules
# ---------------------------------------------------------------------------
_SEVERITY_MAP = {
    "BENIGN": "LOW",
    "Benign": "LOW",
    "benign": "LOW",
    "PortScan": "MEDIUM",
    "Port Scan": "MEDIUM",
    "portscan": "MEDIUM",
    "DoS": "HIGH",
    "DDoS": "HIGH",
    "DoS Hulk": "HIGH",
    "DoS GoldenEye": "HIGH",
    "DoS slowloris": "HIGH",
    "DoS Slowhttptest": "HIGH",
    "DDoS-HOIC": "HIGH",
    "DDoS-LOIC-HTTP": "HIGH",
    "DDoS-LOIC-UDP": "HIGH",
    "Brute Force": "HIGH",
    "BruteForce": "HIGH",
    "FTP-Patator": "HIGH",
    "SSH-Patator": "HIGH",
    "Bot": "CRITICAL",
    "Web Attack": "CRITICAL",
    "Web Attack - Brute Force": "CRITICAL",
    "Web Attack - XSS": "CRITICAL",
    "Web Attack - SQL Injection": "CRITICAL",
    "WebAttack": "CRITICAL",
    "Infiltration": "CRITICAL",
    "Infilteration": "CRITICAL",
    "Heartbleed": "CRITICAL",
}

_SEVERITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _reduce_severity(severity: str) -> str:
    """Reduce severity by one level (minimum LOW)."""
    idx = _SEVERITY_ORDER.index(severity)
    return _SEVERITY_ORDER[max(0, idx - 1)]


def get_severity_level(attack_type: str, confidence: float) -> str:
    """
    Determine the severity level for a given attack type and confidence.

    Rules
    -----
    - BENIGN → LOW
    - PortScan → MEDIUM
    - DoS / DDoS / BruteForce → HIGH
    - Bot / WebAttack / Infiltration → CRITICAL
    - If confidence < 0.5, severity is reduced by one level.

    Parameters
    ----------
    attack_type : str
        The predicted attack category label.
    confidence : float
        Classifier confidence / probability for the prediction.

    Returns
    -------
    str
        One of 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'.
    """
    # Try exact match first
    severity = _SEVERITY_MAP.get(attack_type)

    # Fuzzy match: check if any key substring appears in attack_type
    if severity is None:
        attack_lower = attack_type.lower()
        if "benign" in attack_lower:
            severity = "LOW"
        elif "portscan" in attack_lower or "port scan" in attack_lower:
            severity = "MEDIUM"
        elif "dos" in attack_lower or "ddos" in attack_lower:
            severity = "HIGH"
        elif "brute" in attack_lower or "patator" in attack_lower:
            severity = "HIGH"
        elif "bot" in attack_lower:
            severity = "CRITICAL"
        elif "web" in attack_lower:
            severity = "CRITICAL"
        elif "infiltr" in attack_lower:
            severity = "CRITICAL"
        else:
            severity = "MEDIUM"  # default for unknown attack types

    if confidence is not None and confidence < 0.5:
        severity = _reduce_severity(severity)

    return severity


def generate_security_events(
    predictions_df: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """
    Create a security event log from model predictions.

    Each row in the predictions DataFrame becomes an event with a timestamp,
    severity level, attack category, and the model that produced the prediction.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with at least 'predicted_label' column and optionally
        'confidence' column.
    model_name : str
        Name of the model that generated the predictions.

    Returns
    -------
    pd.DataFrame
        Event log with columns: event_id, timestamp, attack_category,
        severity, confidence, model_used, is_malicious.
    """
    event_columns = [
        "event_id",
        "timestamp",
        "attack_category",
        "severity",
        "confidence",
        "model_used",
        "is_malicious",
    ]

    if predictions_df is None or not isinstance(predictions_df, pd.DataFrame) or predictions_df.empty:
        return pd.DataFrame(columns=event_columns)

    events = []
    base_time = datetime.now()

    for idx, row in predictions_df.iterrows():
        label = row.get("predicted_label", "UNKNOWN")
        confidence = row.get("confidence", None)
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = None

        severity = get_severity_level(label, confidence)
        is_malicious = str(label).upper() != "BENIGN"

        events.append(
            {
                "event_id": f"EVT-{idx:06d}",
                "timestamp": base_time + timedelta(seconds=idx),
                "attack_category": label,
                "severity": severity,
                "confidence": confidence,
                "model_used": model_name,
                "is_malicious": is_malicious,
            }
        )

    return pd.DataFrame(events)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Perform division with zero-denominator handling.

    Parameters
    ----------
    a : float
        Numerator.
    b : float
        Denominator.
    default : float, default=0.0
        Value to return when denominator is zero.

    Returns
    -------
    float
        Result of a / b, or *default* if b is zero.
    """
    if b == 0:
        return default
    return a / b
