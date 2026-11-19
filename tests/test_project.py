import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_utils import load_data, validate_data, FEATURES, TARGET

def test_iris_dataset_schema():
    df = load_data()
    assert df.shape == (150, 5)
    assert set(FEATURES + [TARGET]).issubset(df.columns)

def test_data_quality():
    result = validate_data(load_data())
    assert result["valid"]
    assert result["missing_values"] == 0
    assert len(result["classes"]) == 3

def test_class_balance():
    counts = load_data()[TARGET].value_counts()
    assert counts.min() == counts.max() == 50



def test_data_quality_rejects_non_finite_feature_values():
    import numpy as np
    import pandas as pd
    frame = load_data()
    frame.loc[0, FEATURES[0]] = np.inf
    result = validate_data(frame)
    assert not result["valid"]
    assert result["non_finite_feature_values"] == 1


def test_data_quality_rejects_non_numeric_features():
    frame = load_data()
    frame[FEATURES[0]] = frame[FEATURES[0]].astype(object)
    frame.loc[0, FEATURES[0]] = "unknown"
    result = validate_data(frame)
    assert not result["valid"]
    assert result["numeric_features"] is False


def test_data_quality_rejects_missing_target_and_single_class():
    frame = load_data()
    frame.loc[0, TARGET] = None
    assert not validate_data(frame)["valid"]

    single_class = load_data()
    single_class[TARGET] = single_class[TARGET].iloc[0]
    assert not validate_data(single_class)["valid"]
