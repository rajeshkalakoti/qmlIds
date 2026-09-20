"""Train and compare classical models for binary intrusion classification."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def run_statistical_models(dataset: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Train Random Forest, KNN, and SVM on the same split and return metrics."""
    dataset_config = config["dataset"]
    training_config = config["training"]
    model_config = config["models"]
    target_column = dataset_config["target_column"]
    label_mapping = dataset_config["label_mapping"]
    random_state = training_config["random_state"]
    max_samples = training_config["max_samples"]
    total_rows = len(dataset)

    if target_column not in dataset.columns:
        raise ValueError(f"Target column {target_column!r} was not found in the dataset.")
    if (
        not isinstance(label_mapping, dict)
        or len(label_mapping) != 2
        or any(type(value) is not int for value in label_mapping.values())
        or set(label_mapping.values()) != {0, 1}
    ):
        raise ValueError("dataset.label_mapping must map two classes to 0 and 1.")

    y = dataset[target_column].map(label_mapping)
    if y.isna().any():
        unknown = dataset.loc[y.isna(), target_column].drop_duplicates().tolist()
        raise ValueError(f"Missing or unmapped target labels: {unknown!r}")
    y = y.astype("int8")
    if (y.value_counts().reindex([0, 1], fill_value=0) < 2).any():
        raise ValueError("The dataset must contain at least two rows of each class.")

    if max_samples is not None:
        if type(max_samples) is not int or max_samples < 4:
            raise ValueError("training.max_samples must be null or an integer of at least 4.")
        if max_samples < total_rows:
            # Sample indices so the unused portion of the large DataFrame is not copied.
            selected, _ = train_test_split(
                np.arange(total_rows),
                train_size=max_samples,
                stratify=y,
                random_state=random_state,
            )
            dataset = dataset.iloc[selected]
            y = y.iloc[selected]

    if (y.value_counts().reindex([0, 1], fill_value=0) < 2).any():
        raise ValueError("Increase training.max_samples to include both classes in the split.")

    # Remove all label columns before building features to prevent target leakage.
    excluded_columns = list(dict.fromkeys([target_column, *dataset_config["drop_columns"]]))
    X = dataset.drop(columns=excluded_columns).apply(pd.to_numeric, errors="raise")
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after removing the label columns.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training_config["test_size"],
        stratify=y,
        random_state=random_state,
    )
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("Both classes must be present in training and testing; increase max_samples.")
    if model_config["knn"]["n_neighbors"] > len(X_train):
        raise ValueError("KNN n_neighbors cannot exceed the number of training rows.")

    # Each pipeline learns its imputation/scaling from the training data only.
    models = {
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                **{**model_config["random_forest"], "random_state": random_state}
            ),
        ),
        "KNN": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            KNeighborsClassifier(**model_config["knn"]),
        ),
        "SVM": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            SVC(**{**model_config["svm"], "random_state": random_state}),
        ),
    }

    class_names = [name for name, value in sorted(label_mapping.items(), key=lambda item: item[1])]
    print(f"Using {len(X):,} of {total_rows:,} rows and {X.shape[1]} features.", flush=True)
    print(f"Training rows: {len(X_train):,}; test rows: {len(X_test):,}")
    for name, label in sorted(label_mapping.items(), key=lambda item: item[1]):
        print(f"  {name} ({label}): train={(y_train == label).sum():,}, test={(y_test == label).sum():,}")

    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        started = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - started
        predictions = model.predict(X_test)

        # SVM decision scores support ROC AUC without probability calibration.
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)

        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, scores),
                "fit_seconds": fit_seconds,
            }
        )
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
        )
        print(f"Confusion matrix: rows=actual, columns=predicted; order={class_names}")
        print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    metrics = pd.DataFrame(results)
    print("\nModel comparison (precision, recall, and F1 use Attack=1):")
    print(metrics.round(4).to_string(index=False))
    return metrics
