import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
VALID_CLASSES = {"setosa", "versicolor", "virginica"}


def load_data():
    raw = load_iris(as_frame=True)
    frame = raw.data.copy()
    frame.columns = FEATURES
    frame[TARGET] = raw.target.map(dict(enumerate(raw.target_names)))
    return frame


def validate_data(frame):
    """Return transparent quality metrics for an Iris-shaped dataframe."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if not frame.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")

    required = FEATURES + [TARGET]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    selected = frame[required]
    numeric = frame[FEATURES]
    numeric_features = all(
        pd.api.types.is_numeric_dtype(numeric[column]) for column in FEATURES
    )
    if numeric_features:
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
        finite_mask = np.isfinite(values)
        non_finite_values = int((~finite_mask).sum())
        non_positive_values = int((finite_mask & (values <= 0)).sum())
    else:
        non_finite_values = None
        non_positive_values = None

    labels = frame[TARGET].astype("string").str.strip().str.lower()
    invalid_classes = int((labels.isna() | ~labels.isin(VALID_CLASSES)).sum())
    observed_classes = set(labels.dropna().unique()) & VALID_CLASSES
    missing_classes = sorted(VALID_CLASSES - observed_classes)
    missing_values = int(selected.isna().sum().sum())
    valid = bool(
        len(frame) > 0
        and missing_values == 0
        and numeric_features
        and non_finite_values == 0
        and non_positive_values == 0
        and invalid_classes == 0
        and not missing_classes
    )

    return {
        "rows": int(len(frame)),
        "missing_values": missing_values,
        "duplicate_rows": int(selected.duplicated().sum()),
        "classes": sorted(frame[TARGET].dropna().unique().tolist()),
        "numeric_features": bool(numeric_features),
        "non_finite_values": non_finite_values,
        "non_positive_values": non_positive_values,
        "invalid_classes": invalid_classes,
        "missing_classes": missing_classes,
        "valid": valid,
    }
