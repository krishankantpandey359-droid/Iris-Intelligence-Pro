from pathlib import Path
import hashlib
import io

import numpy as np
import pandas as pd

from data_utils import FEATURES


def upload_signature(uploaded_file):
    """Return a stable signature for the currently uploaded batch file."""
    if uploaded_file is None:
        return None
    payload = uploaded_file.getvalue()
    return hashlib.sha256(payload).hexdigest()


def validate_batch_csv(payload):
    """Validate a batch CSV and return its numeric feature matrix."""
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise ValueError(f"Unable to read CSV: {exc}") from exc

    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("The uploaded CSV is empty.")

    values = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    if values.isnull().any().any():
        raise ValueError("Some feature values are missing or non-numeric.")
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("The file contains infinite or invalid values.")
    if (array <= 0).any():
        raise ValueError("All flower measurements must be positive.")
    return values
