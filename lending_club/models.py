from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = [
    "annual_inc", "dti", "revol_util", "fico_range_low",
    "loan_amnt", "int_rate", "term", "installment", "grade",
]


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([
        ("numeric", numeric, [column for column in FEATURES if column != "grade"]),
        ("grade", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), ["grade"]),
    ])


def train_models(frame: pd.DataFrame, artifact_dir: str | Path, *, seed: int = 42) -> dict:
    missing = set(FEATURES + ["loan_status_label", "total_rec_prncp"]) - set(frame.columns)
    if missing:
        raise ValueError("Training frame is missing columns: " + ", ".join(sorted(missing)))
    X = frame[FEATURES]
    y = frame["loan_status_label"].astype(int)
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify
    )
    classification = Pipeline([
        ("preprocess", _preprocessor()),
        ("model", DecisionTreeClassifier(max_depth=6, random_state=seed)),
    ])
    classification.fit(X_train, y_train)
    probabilities = classification.predict_proba(X_test)[:, 1]
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(classification, destination / "loan_default_classifier.joblib")
    defaulted = frame["loan_status_label"].eq(1)
    recovery_features = frame.loc[defaulted, FEATURES]
    recovery_target = frame.loc[defaulted, "total_rec_prncp"]
    recovery_metrics = None
    if len(recovery_features) >= 10 and recovery_target.nunique() > 1:
        rx_train, rx_test, ry_train, ry_test = train_test_split(
            recovery_features, recovery_target, test_size=0.2, random_state=seed
        )
        recovery = Pipeline([
            ("preprocess", _preprocessor()),
            ("model", LinearRegression()),
        ])
        recovery.fit(rx_train, ry_train)
        predictions = recovery.predict(rx_test)
        recovery_metrics = {
            "recovery_mse": float(mean_squared_error(ry_test, predictions)),
            "recovery_r2": float(recovery.score(rx_test, ry_test)),
        }
        joblib.dump(recovery, destination / "recovery_linear_regressor.joblib")
    elif len(recovery_features):
        recovery_metrics = {"recovery_mse": None, "recovery_r2": None, "reason": "Insufficient charged-off variation"}
    metrics = {
        "classification_accuracy": float(accuracy_score(y_test, classification.predict(X_test))),
        "classification_roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() > 1 else None,
        "recovery_linear_regression": recovery_metrics,
        "seed": seed,
        "features": FEATURES,
        "classification_target": "loan_status_label (Fully Paid=0, Charged Off=1)",
        "recovery_target": "total_rec_prncp for Charged Off loans",
        "model_types": ["DecisionTreeClassifier", "LinearRegression"],
    }
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

