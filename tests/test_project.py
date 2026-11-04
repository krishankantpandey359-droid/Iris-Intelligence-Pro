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
