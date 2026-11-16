import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from batch_utils import upload_signature, validate_batch_csv
from data_utils import FEATURES


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8")


def valid_frame():
    return pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=FEATURES)


def test_valid_batch_csv_returns_numeric_features():
    result = validate_batch_csv(csv_bytes(valid_frame()))
    assert list(result.columns) == FEATURES
    assert result.iloc[0].tolist() == [5.1, 3.5, 1.4, 0.2]


def test_batch_csv_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_batch_csv(csv_bytes(pd.DataFrame({"sepal_length": [5.1]})))


def test_batch_csv_rejects_non_numeric_values():
    frame = valid_frame().astype(object)
    frame.loc[0, "petal_width"] = "bad"
    with pytest.raises(ValueError, match="missing or non-numeric"):
        validate_batch_csv(csv_bytes(frame))


def test_batch_csv_rejects_non_positive_values():
    frame = valid_frame()
    frame.loc[0, "sepal_length"] = 0
    with pytest.raises(ValueError, match="must be positive"):
        validate_batch_csv(csv_bytes(frame))


def test_upload_signature_changes_with_file_content():
    first = io.BytesIO(b"first csv")
    second = io.BytesIO(b"second csv")
    assert upload_signature(first) != upload_signature(second)
