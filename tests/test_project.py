import sys
import pytest
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



def test_data_quality_rejects_non_finite_measurements():
    df = load_data()
    df.loc[0, FEATURES[0]] = float("inf")
    result = validate_data(df)
    assert result["non_finite_values"] == 1
    assert not result["valid"]


def test_data_quality_rejects_non_positive_measurements():
    df = load_data()
    df.loc[0, FEATURES[0]] = 0
    result = validate_data(df)
    assert result["non_positive_values"] == 1
    assert not result["valid"]


def test_data_quality_rejects_unknown_species_label():
    df = load_data()
    df.loc[0, TARGET] = "unknown-species"
    result = validate_data(df)
    assert result["invalid_classes"] == 1
    assert not result["valid"]


def test_data_quality_rejects_duplicate_column_names():
    df = load_data()
    df.columns = [FEATURES[0], FEATURES[0], FEATURES[2], FEATURES[3], TARGET]
    with pytest.raises(ValueError, match="duplicate column names"):
        validate_data(df)


def test_data_quality_rejects_non_dataframe_input():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate_data(None)

def test_data_quality_rejects_missing_target():
    df = load_data()
    df.loc[0, TARGET] = None
    result = validate_data(df)
    assert result["missing_values"] == 1
    assert result["invalid_classes"] == 1
    assert not result["valid"]


def test_data_quality_rejects_single_class_target():
    df = load_data()
    df[TARGET] = df[TARGET].iloc[0]
    result = validate_data(df)
    assert not result["valid"]

def test_data_quality_rejects_non_numeric_features():
    df = load_data()
    df[FEATURES[0]] = df[FEATURES[0]].astype(object)
    df.loc[0, FEATURES[0]] = "not-a-number"
    result = validate_data(df)
    assert not result["valid"]
    assert result["numeric_features"] is False
