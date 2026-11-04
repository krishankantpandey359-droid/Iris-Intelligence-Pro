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
    required = FEATURES + [TARGET]
    missing_columns = [c for c in required if c not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")
    x = frame[FEATURES]
    return {
        "rows": int(len(frame)),
        "missing_values": int(frame[required].isna().sum().sum()),
        "duplicate_rows": int(frame[required].duplicated().sum()),
        "classes": sorted(frame[TARGET].dropna().unique().tolist()),
        "numeric_features": bool(all(pd.api.types.is_numeric_dtype(x[c]) for c in FEATURES)),
        "valid": bool(len(frame) > 0 and frame[TARGET].notna().all()
                      and x.notna().all().all()
                      and all(pd.api.types.is_numeric_dtype(x[c]) for c in FEATURES))
    }
