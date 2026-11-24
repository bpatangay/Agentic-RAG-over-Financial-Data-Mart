from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_curve,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = [
    "annual_inc", "dti", "revol_util", "fico_range_low",
    "loan_amnt", "int_rate", "term", "installment", "grade",
]
MIN_THRESHOLD_VALIDATION_ACCURACY = 0.77


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([
        ("numeric", numeric, [column for column in FEATURES if column != "grade"]),
        ("grade", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), ["grade"]),
    ])


def _positive_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    positive_index = list(model.classes_).index(1) if 1 in model.classes_ else None
    if positive_index is None:
        return np.zeros(len(features))
    return probabilities[:, positive_index]


def _classification_metrics(
    actual: pd.Series, predicted: np.ndarray, probabilities: np.ndarray
) -> dict:
    both_classes = actual.nunique() > 1
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probabilities)) if both_classes else None,
        "pr_auc": float(average_precision_score(actual, probabilities)) if both_classes else None,
        "confusion_matrix_labels_0_1": confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).tolist(),
    }


def _regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def _select_decision_threshold(actual: pd.Series, probabilities: np.ndarray) -> tuple[float, str, dict]:
    candidates = np.unique(
        np.append(probabilities, np.nextafter(probabilities.max(), np.inf))
    )
    eligible = []
    for candidate in candidates:
        predicted = (probabilities >= candidate).astype(int)
        accuracy = accuracy_score(actual, predicted)
        if accuracy >= MIN_THRESHOLD_VALIDATION_ACCURACY:
            eligible.append(
                (
                    recall_score(actual, predicted, zero_division=0),
                    f1_score(actual, predicted, zero_division=0),
                    precision_score(actual, predicted, zero_division=0),
                    accuracy,
                    float(candidate),
                )
            )

    if eligible:
        _, _, _, _, threshold = max(eligible)
        selection = "training_oof_max_recall_at_77pct_accuracy"
        floor_met = True
    else:
        precisions, recalls, thresholds = precision_recall_curve(actual, probabilities)
        f1_scores = (
            2 * precisions[:-1] * recalls[:-1]
            / (precisions[:-1] + recalls[:-1] + 1e-12)
        )
        threshold = float(thresholds[int(np.argmax(f1_scores))])
        selection = "training_oof_max_f1_accuracy_floor_unmet"
        floor_met = False

    predicted = (probabilities >= threshold).astype(int)
    validation_metrics = {
        "minimum_accuracy": MIN_THRESHOLD_VALIDATION_ACCURACY,
        "accuracy_floor_met": floor_met,
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }
    return threshold, selection, validation_metrics


def _split_indices(frame: pd.DataFrame, labels: pd.Series, seed: int):
    years = pd.to_datetime(frame["issue_date"], errors="coerce").dt.year
    if years.notna().sum() and years.nunique() >= 2:
        test_year = int(years.max())
        train_mask = years < test_year
        test_mask = years == test_year
        if train_mask.any() and test_mask.any():
            return (
                frame.index[train_mask],
                frame.index[test_mask],
                {
                    "strategy": "latest_issue_year_holdout",
                    "training_years": sorted(int(year) for year in years[train_mask].unique()),
                    "test_year": test_year,
                },
            )

    stratify = labels if labels.value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(
        frame.index, test_size=0.2, random_state=seed, stratify=stratify
    )
    return train_idx, test_idx, {"strategy": "stratified_random_fallback"}


def train_models(frame: pd.DataFrame, artifact_dir: str | Path, *, seed: int = 42) -> dict:
    missing = set(FEATURES + ["issue_date", "loan_status_label", "total_rec_prncp"]) - set(frame.columns)
    if missing:
        raise ValueError("Training frame is missing columns: " + ", ".join(sorted(missing)))
    X = frame[FEATURES]
    y = frame["loan_status_label"].astype(int)
    train_idx, test_idx, split_details = _split_indices(frame, y, seed)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    classification = Pipeline([
        ("preprocess", _preprocessor()),
        ("model", DecisionTreeClassifier(max_depth=6, random_state=seed)),
    ])
    class_counts = y_train.value_counts()
    cv_folds = min(3, int(class_counts.min())) if len(class_counts) > 1 else 0
    cv_average_precision = None
    threshold = 0.5
    threshold_selection = "default_0.5_insufficient_data_for_cv"
    threshold_validation = None
    if len(X_train) >= 30 and cv_folds >= 2:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        search = GridSearchCV(
            classification,
            {
                "model__max_depth": [3, 6, 10, None],
                "model__min_samples_leaf": [1, 25, 100],
                "model__class_weight": [None, "balanced"],
            },
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        classification = search.best_estimator_
        best_params = search.best_params_
        cv_average_precision = float(search.best_score_)
        oof_predictions = cross_val_predict(
            classification,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )
        positive_index = list(classification.classes_).index(1)
        oof_probabilities = oof_predictions[:, positive_index]
        threshold, threshold_selection, threshold_validation = _select_decision_threshold(
            y_train, oof_probabilities
        )
    else:
        classification.fit(X_train, y_train)
        best_params = {"model__max_depth": 6}
    probabilities = _positive_probability(classification, X_test)
    classification_predictions = (probabilities >= threshold).astype(int)
    classification_test = _classification_metrics(
        y_test, classification_predictions, probabilities
    )

    classifier_baseline = DummyClassifier(strategy="most_frequent")
    classifier_baseline.fit(X_train, y_train)
    baseline_predictions = classifier_baseline.predict(X_test)
    baseline_probabilities = _positive_probability(classifier_baseline, X_test)
    classification_baseline = _classification_metrics(
        y_test, baseline_predictions, baseline_probabilities
    )

    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(classification, destination / "loan_default_classifier.joblib")

    defaulted = y.eq(1)
    recovery_train_mask = defaulted & frame.index.isin(train_idx)
    recovery_test_mask = defaulted & frame.index.isin(test_idx)
    recovery_features = X.loc[recovery_train_mask]
    recovery_target = pd.to_numeric(
        frame.loc[recovery_train_mask, "total_rec_prncp"], errors="coerce"
    )
    recovery_test_features = X.loc[recovery_test_mask]
    recovery_test_target = pd.to_numeric(
        frame.loc[recovery_test_mask, "total_rec_prncp"], errors="coerce"
    )
    valid_train = recovery_target.notna()
    recovery_features, recovery_target = (
        recovery_features.loc[valid_train],
        recovery_target.loc[valid_train],
    )
    valid_test = recovery_test_target.notna()
    recovery_test_features, recovery_test_target = (
        recovery_test_features.loc[valid_test],
        recovery_test_target.loc[valid_test],
    )
    recovery_metrics = None
    if len(recovery_features) >= 2 and recovery_target.nunique() > 1 and len(recovery_test_target):
        recovery = Pipeline([
            ("preprocess", _preprocessor()),
            ("model", LinearRegression()),
        ])
        recovery.fit(recovery_features, recovery_target)
        recovery_predictions = recovery.predict(recovery_test_features)
        recovery_baseline = DummyRegressor(strategy="mean")
        recovery_baseline.fit(recovery_features, recovery_target)
        baseline_recovery_predictions = recovery_baseline.predict(recovery_test_features)
        recovery_metrics = _regression_metrics(recovery_test_target, recovery_predictions)
        recovery_metrics.update({
            "recovery_mse": float(mean_squared_error(recovery_test_target, recovery_predictions)),
            "recovery_r2": recovery_metrics["r2"],
            "baseline": _regression_metrics(
                recovery_test_target, baseline_recovery_predictions
            ),
            "training_rows": int(len(recovery_features)),
            "test_rows": int(len(recovery_test_target)),
        })
        joblib.dump(recovery, destination / "recovery_linear_regressor.joblib")
    elif len(recovery_features):
        recovery_metrics = {
            "mae": None,
            "rmse": None,
            "r2": None,
            "recovery_mse": None,
            "recovery_r2": None,
            "reason": "Insufficient charged-off training variation or no charged-off holdout rows",
        }
    metrics = {
        "classification_accuracy": classification_test["accuracy"],
        "classification_roc_auc": classification_test["roc_auc"],
        "classification": {
            "model": "DecisionTreeClassifier",
            "positive_label": 1,
            "test": classification_test,
            "baseline": {
                "model": "DummyClassifier(strategy='most_frequent')",
                **classification_baseline,
            },
            "best_params": best_params,
            "cross_validation_average_precision": cv_average_precision,
            "decision_threshold": threshold,
            "threshold_selection": threshold_selection,
            "threshold_validation": threshold_validation,
        },
        "evaluation": {
            **split_details,
            "training_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "selection_metric": "PR-AUC",
            "cross_validation_folds": cv_folds or None,
        },
        "recovery_linear_regression": recovery_metrics,
        "seed": seed,
        "features": FEATURES,
        "recovery_target": "total_rec_prncp for Charged Off loans",
        "recovery_model": "LinearRegression",
        "recovery_baseline": "DummyRegressor(strategy='mean')",
        "classification_target": "loan_status_label (Fully Paid=0, Charged Off=1)",
        "model_types": ["DecisionTreeClassifier", "LinearRegression"],
    }
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

