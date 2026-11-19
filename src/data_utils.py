import pandas as pd
from sklearn.datasets import load_iris

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"

def load_data():
    raw = load_iris(as_frame=True)
    frame = raw.data.copy()
    frame.columns = FEATURES
    frame[TARGET] = raw.target.map(dict(enumerate(raw.target_names)))
    return frame

def validate_data(frame):
    """Return a transparent quality report for the Iris modelling columns."""
    required = FEATURES + [TARGET]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    selected = frame[required]
    x = frame[FEATURES]
    numeric_features = all(
        pd.api.types.is_numeric_dtype(x[column]) for column in FEATURES
    )
    non_finite_values = 0
    if numeric_features:
        import numpy as np
        non_finite_values = int(
            (~np.isfinite(x.to_numpy(dtype=float))).sum()
        )

    missing_values = int(selected.isna().sum().sum())
    duplicate_rows = int(selected.duplicated().sum())
    classes = sorted(frame[TARGET].dropna().unique().tolist())
    valid = bool(
        len(frame) > 0
        and missing_values == 0
        and non_finite_values == 0
        and numeric_features
        and frame[TARGET].notna().all()
        and len(classes) >= 2
    )
    return {
        "rows": int(len(frame)),
        "missing_values": missing_values,
        "non_finite_feature_values": non_finite_values,
        "duplicate_rows": duplicate_rows,
        "classes": classes,
        "numeric_features": bool(numeric_features),
        "valid": valid,
    }
