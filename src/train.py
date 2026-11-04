
"""
Iris Intelligence Pro
Model training, hyperparameter tuning and evaluation.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_data, FEATURES, TARGET


RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


def get_candidates():
    """Return candidate models and their hyperparameter grids."""

    return {
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000))
            ]),
            {
                "model__C": [0.1, 1, 10, 100]
            }
        ),

        "K-Nearest Neighbors": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier())
            ]),
            {
                "model__n_neighbors": [3, 5, 7, 9],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2]
            }
        ),

        "Support Vector Machine": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, random_state=RANDOM_STATE))
            ]),
            {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"]
            }
        ),

        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 3, 5, 8],
                "min_samples_leaf": [1, 2, 3]
            }
        ),

        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [1, 2, 3]
            }
        )
    }


def save_roc_data(model, X_test, y_test, output_dir):
    """Calculate and save one-vs-rest ROC curve data."""

    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(X_test)
    classes = model.classes_

    binary_y = label_binarize(y_test, classes=classes)
    roc_rows = []

    for index, class_name in enumerate(classes):
        fpr, tpr, thresholds = roc_curve(
            binary_y[:, index],
            probabilities[:, index]
        )

        for fpr_value, tpr_value, threshold in zip(
            fpr, tpr, thresholds
        ):
            roc_rows.append({
                "class": str(class_name),
                "false_positive_rate": float(fpr_value),
                "true_positive_rate": float(tpr_value),
                "threshold": float(threshold)
            })

    pd.DataFrame(roc_rows).to_csv(
        output_dir / "roc_curve_data.csv",
        index=False
    )

    return float(
        roc_auc_score(
            y_test,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="macro"
        )
    )


def train_and_evaluate():
    """Train, compare and evaluate five classification models."""

    print("=" * 65)
    print("IRIS INTELLIGENCE PRO | MODEL TRAINING")
    print("=" * 65)

    # 1. Load and validate data
    df = load_data()

    if df.empty:
        raise ValueError("The Iris dataset is empty.")

    if df[FEATURES].isnull().any().any():
        raise ValueError("Missing values found in input features.")

    if df[TARGET].isnull().any():
        raise ValueError("Missing target values found.")

    X = df[FEATURES]
    y = df[TARGET]

    # 2. Create a stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 3. Cross-validation strategy
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    candidates = get_candidates()

    fitted_models = {}
    comparison_rows = []

    # 4. Train and tune every classifier
    for model_name, (estimator, parameter_grid) in candidates.items():

        print(f"\nTraining: {model_name}")

        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True
        )

        search.fit(X_train, y_train)

        fitted_models[model_name] = search.best_estimator_

        comparison_rows.append({
            "model": model_name,
            "cv_accuracy": float(search.best_score_),
            "cv_std": float(
                search.cv_results_["std_test_score"][
                    search.best_index_
                ]
            ),
            "best_parameters": json.dumps(
                search.best_params_,
                sort_keys=True
            )
        })

        print(f"Best CV accuracy: {search.best_score_:.4f}")

    # 5. Select the model using CV accuracy ONLY.
    # Test-set metrics are not used for model selection.
    results = pd.DataFrame(comparison_rows)

    results = results.sort_values(
        "cv_accuracy",
        ascending=False,
        kind="stable"
    ).reset_index(drop=True)

    best_name = results.loc[0, "model"]
    best_model = fitted_models[best_name]

    print(f"\nSelected model: {best_name}")

    # 6. Evaluate the selected model on the held-out test set
    y_pred = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    results["test_accuracy"] = np.nan
    results["macro_precision"] = np.nan
    results["macro_recall"] = np.nan
    results["macro_f1"] = np.nan

    selected_index = results.index[0]

    results.loc[selected_index, "test_accuracy"] = test_accuracy
    results.loc[selected_index, "macro_precision"] = macro_precision
    results.loc[selected_index, "macro_recall"] = macro_recall
    results.loc[selected_index, "macro_f1"] = macro_f1

    # 7. Create output directories
    model_dir = ROOT / "models"
    output_dir = ROOT / "outputs"

    model_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # 8. Save selected model and comparison
    joblib.dump(
        best_model,
        model_dir / "best_iris_classifier.joblib"
    )

    results.to_csv(
        output_dir / "model_comparison.csv",
        index=False
    )

    # 9. Save classification report
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    (output_dir / "classification_report.txt").write_text(
        report,
        encoding="utf-8"
    )

    # 10. Save confusion matrix
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=best_model.classes_
    )

    pd.DataFrame(
        matrix,
        index=best_model.classes_,
        columns=best_model.classes_
    ).to_csv(output_dir / "confusion_matrix.csv")

    # 11. Save individual test predictions
    predictions = X_test.copy()
    predictions["actual_species"] = y_test
    predictions["predicted_species"] = y_pred
    predictions["correct_prediction"] = (
        predictions["actual_species"]
        == predictions["predicted_species"]
    )

    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(X_test)

        predictions["prediction_confidence"] = (
            probabilities.max(axis=1)
        )

    predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False
    )

    # 12. Save ROC curve data and AUC
    roc_auc = save_roc_data(
        best_model,
        X_test,
        y_test,
        output_dir
    )

    # 13. Calculate permutation feature importance.
    # This is post-hoc interpretation, not model selection.
    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="accuracy",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std
    }).sort_values(
        "importance_mean",
        ascending=False
    )

    importance_df.to_csv(
        output_dir / "feature_importance.csv",
        index=False
    )

    # 14. Save run metadata
    metadata = {
        "project": "Iris Intelligence Pro",
        "selected_model": best_name,
        "selection_metric": "cross_validation_accuracy",
        "random_state": RANDOM_STATE,
        "test_fraction": TEST_SIZE,
        "cross_validation_folds": CV_FOLDS,
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": list(FEATURES),
        "classes": [str(item) for item in best_model.classes_],
        "test_accuracy": float(test_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "macro_roc_auc_ovr": (
            float(roc_auc) if roc_auc is not None else None
        )
    }

    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    # 15. Print summary
    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)

    print(f"Selected model       : {best_name}")
    print(f"CV accuracy          : {results.loc[0, 'cv_accuracy']:.4f}")
    print(f"Test accuracy        : {test_accuracy:.4f}")
    print(f"Macro precision      : {macro_precision:.4f}")
    print(f"Macro recall         : {macro_recall:.4f}")
    print(f"Macro F1             : {macro_f1:.4f}")

    if roc_auc is not None:
        print(f"Macro ROC-AUC (OvR)  : {roc_auc:.4f}")

    print("\nModel comparison:")
    print(results.to_string(index=False))

    print("\nOutput files saved in:", output_dir)

    return results, metadata, best_model


if __name__ == "__main__":
    train_and_evaluate()
