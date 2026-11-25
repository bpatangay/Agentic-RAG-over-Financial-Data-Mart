from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
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
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier, export_text

FEATURES = [
    "annual_inc", "dti", "revol_util", "fico_range_low",
    "loan_amnt", "int_rate", "term", "installment", "grade",
]
ENGINEERED_NUMERIC = [
    "annual_inc", "dti", "revol_util", "fico_range_low",
    "loan_amnt", "int_rate", "term", "installment",
    "log_annual_inc", "log_loan_amnt", "installment_to_income",
    "loan_to_income", "grade_ordinal", "dti_x_int_rate", "is_60_month",
    "total_scheduled_pay", "total_interest_cost", "interest_to_loan",
    "fico_x_int_rate", "revol_x_dti", "monthly_disposable",
]
SEASONING_NUMERIC = ENGINEERED_NUMERIC + ["issue_year", "issue_month", "months_to_cutoff", "log_months"]
GRADE_ORDER = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0, "G": 7.0}
MIN_THRESHOLD_VALIDATION_ACCURACY = 0.77


def _add_issue_time_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    annual_inc = pd.to_numeric(df["annual_inc"], errors="coerce").clip(lower=1.0)
    installment = pd.to_numeric(df["installment"], errors="coerce").clip(lower=0.0)
    loan_amnt = pd.to_numeric(df["loan_amnt"], errors="coerce").clip(lower=1.0)
    dti = pd.to_numeric(df["dti"], errors="coerce").fillna(0.0)
    int_rate = pd.to_numeric(df["int_rate"], errors="coerce").fillna(0.0)
    revol_util = pd.to_numeric(df["revol_util"], errors="coerce").fillna(0.0)
    fico = pd.to_numeric(df["fico_range_low"], errors="coerce").fillna(680.0)
    term = pd.to_numeric(df["term"], errors="coerce").fillna(36.0)

    df["log_annual_inc"] = np.log1p(annual_inc)
    df["log_loan_amnt"] = np.log1p(loan_amnt)
    df["installment_to_income"] = installment / ((annual_inc / 12.0) + 1.0)
    df["loan_to_income"] = loan_amnt / annual_inc
    df["grade_ordinal"] = df["grade"].astype(str).str.strip().str.upper().map(GRADE_ORDER).fillna(4.0)
    df["dti_x_int_rate"] = dti * int_rate
    df["is_60_month"] = (term >= 60.0).astype(float)
    df["total_scheduled_pay"] = installment * term
    df["total_interest_cost"] = (installment * term - loan_amnt).clip(lower=0.0)
    df["interest_to_loan"] = df["total_interest_cost"] / loan_amnt
    df["fico_x_int_rate"] = fico * int_rate
    df["revol_x_dti"] = revol_util * dti
    df["monthly_disposable"] = (annual_inc / 12.0) - installment
    return df


def _add_seasoning_features(frame: pd.DataFrame, cutoff_year: int = 2018, cutoff_month: int = 12) -> pd.DataFrame:
    df = _add_issue_time_ratios(frame)
    if "issue_date" in df.columns:
        dt = pd.to_datetime(df["issue_date"], errors="coerce")
        yr = dt.dt.year.fillna(cutoff_year).astype(float)
        mo = dt.dt.month.fillna(cutoff_month).astype(float)
    else:
        yr = pd.Series(float(cutoff_year), index=df.index)
        mo = pd.Series(float(cutoff_month), index=df.index)
    months = ((cutoff_year - yr) * 12.0 + (cutoff_month - mo) + 1.0).clip(lower=1.0)
    df["issue_year"] = yr
    df["issue_month"] = mo
    df["months_to_cutoff"] = months
    df["log_months"] = np.log1p(months)
    return df


def _preprocessor(*, scale_numeric: bool = True, include_seasoning: bool = False) -> Pipeline:
    cols = SEASONING_NUMERIC if include_seasoning else ENGINEERED_NUMERIC
    fn = _add_seasoning_features if include_seasoning else _add_issue_time_ratios
    num_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(num_steps)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer([
        ("numeric", numeric, cols),
        ("grade", categorical, ["grade"]),
    ])
    return Pipeline([
        ("ratios", FunctionTransformer(fn, validate=False)),
        ("columns", ct),
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
    mse = float(mean_squared_error(actual, predicted))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": mse,
        "rmse": float(mse ** 0.5),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else None,
    }


def _select_max_f1_threshold(actual: pd.Series, probabilities: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(actual, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = (
        2 * precisions[:-1] * recalls[:-1]
        / (precisions[:-1] + recalls[:-1] + 1e-12)
    )
    return float(thresholds[int(np.argmax(f1_scores))])


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
        threshold = _select_max_f1_threshold(actual, probabilities)
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


def _extract_tree_rules(classification: Pipeline) -> str:
    preprocess = classification.named_steps["preprocess"]
    ct = preprocess.named_steps["columns"]
    grade_encoder = ct.named_transformers_["grade"].named_steps["encode"]
    grade_names = [f"grade_{cat}" for cat in grade_encoder.categories_[0]]
    feature_names = ENGINEERED_NUMERIC + grade_names
    tree_model = classification.named_steps["model"]
    return export_text(tree_model, feature_names=feature_names, max_depth=4)


def _evaluate_recovery_models(
    X_train: pd.DataFrame,
    y_train_dollars: pd.Series,
    X_test: pd.DataFrame,
    y_test_dollars: pd.Series,
    seed: int,
    *,
    include_poly_and_hgb: bool = False,
) -> tuple[dict, Pipeline]:
    train_amnt = X_train["loan_amnt"].clip(lower=1.0)
    test_amnt = X_test["loan_amnt"].clip(lower=1.0)
    train_rate = (y_train_dollars / train_amnt).clip(0.0, 1.0)
    actual_lgd = 1.0 - (y_test_dollars / test_amnt).clip(0.0, 1.0)

    # 1) Seasoning-aware Bounded Recovery-Rate Ridge (linear model in feature space)
    ridge_rate = Pipeline([
        ("preprocess", _preprocessor(scale_numeric=True, include_seasoning=True)),
        ("model", Ridge(alpha=10.0, random_state=seed)),
    ])
    ridge_rate.fit(X_train, train_rate)
    pred_rate = np.clip(ridge_rate.predict(X_test), 0.0, 1.0)
    ridge_pred = pred_rate * test_amnt.to_numpy()
    ridge_lgd_pred = 1.0 - pred_rate

    # 2) Seasoning-aware OLS LinearRegression on bounded recovery rate
    ols_rate = Pipeline([
        ("preprocess", _preprocessor(scale_numeric=True, include_seasoning=True)),
        ("model", LinearRegression()),
    ])
    ols_rate.fit(X_train, train_rate)
    ols_pred_rate = np.clip(ols_rate.predict(X_test), 0.0, 1.0)
    ols_pred = ols_pred_rate * test_amnt.to_numpy()
    ols_lgd_pred = 1.0 - ols_pred_rate

    # 3) Training-mean baseline
    baseline_reg = DummyRegressor(strategy="mean")
    baseline_reg.fit(X_train, y_train_dollars)
    baseline_pred = baseline_reg.predict(X_test)

    ridge_metrics = _regression_metrics(y_test_dollars, ridge_pred)
    ols_metrics = _regression_metrics(y_test_dollars, ols_pred)
    base_metrics = _regression_metrics(y_test_dollars, baseline_pred)

    result = {
        **ridge_metrics,
        "recovery_mse": ridge_metrics["mse"],
        "recovery_r2": ridge_metrics["r2"],
        "lgd_mae": float(mean_absolute_error(actual_lgd, ridge_lgd_pred)),
        "mean_actual_lgd": float(np.mean(actual_lgd)),
        "mean_predicted_lgd": float(np.mean(ridge_lgd_pred)),
        "ols_linear_regression": {
            **ols_metrics,
            "lgd_mae": float(mean_absolute_error(actual_lgd, ols_lgd_pred)),
            "mean_predicted_lgd": float(np.mean(ols_lgd_pred)),
        },
        "baseline": base_metrics,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    if include_poly_and_hgb and len(X_train) >= 50:
        poly_ridge = Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True, include_seasoning=True)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", Ridge(alpha=100.0, random_state=seed)),
        ])
        poly_ridge.fit(X_train, train_rate)
        pr_poly = np.clip(poly_ridge.predict(X_test), 0.0, 1.0)
        poly_pred = pr_poly * test_amnt.to_numpy()
        result["poly2_ridge_regression"] = {
            **_regression_metrics(y_test_dollars, poly_pred),
            "lgd_mae": float(mean_absolute_error(actual_lgd, 1.0 - pr_poly)),
            "mean_predicted_lgd": float(np.mean(1.0 - pr_poly)),
        }

        hgb_reg = Pipeline([
            ("preprocess", _preprocessor(scale_numeric=False, include_seasoning=True)),
            ("model", HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.05, max_depth=6, min_samples_leaf=50, random_state=seed
            )),
        ])
        hgb_reg.fit(X_train, train_rate)
        pr_hgb = np.clip(hgb_reg.predict(X_test), 0.0, 1.0)
        hgb_pred = pr_hgb * test_amnt.to_numpy()
        result["hist_gradient_boosting_regressor"] = {
            **_regression_metrics(y_test_dollars, hgb_pred),
            "lgd_mae": float(mean_absolute_error(actual_lgd, 1.0 - pr_hgb)),
            "mean_predicted_lgd": float(np.mean(1.0 - pr_hgb)),
        }

    return result, ridge_rate


def train_models(frame: pd.DataFrame, artifact_dir: str | Path, *, seed: int = 42) -> dict:
    missing = set(FEATURES + ["issue_date", "loan_status_label", "total_rec_prncp"]) - set(frame.columns)
    if missing:
        raise ValueError("Training frame is missing columns: " + ", ".join(sorted(missing)))
    X = frame[FEATURES]
    X_with_date = frame[FEATURES + ["issue_date"]]
    y = frame["loan_status_label"].astype(int)
    train_idx, test_idx, split_details = _split_indices(frame, y, seed)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # Unscaled numeric features for DecisionTreeClassifier so extracted rules show real financial units
    classification = Pipeline([
        ("preprocess", _preprocessor(scale_numeric=False)),
        ("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=100, random_state=seed)),
    ])
    class_counts = y_train.value_counts()
    cv_folds = min(3, int(class_counts.min())) if len(class_counts) > 1 else 0
    cv_average_precision = None
    threshold = 0.5
    f1_threshold = 0.5
    threshold_selection = "default_0.5_insufficient_data_for_cv"
    threshold_validation = None

    if len(X_train) >= 30 and cv_folds >= 2:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        search = GridSearchCV(
            classification,
            {
                "model__criterion": ["gini", "log_loss"],
                "model__max_depth": [6, 8],
                "model__min_samples_leaf": [100, 250],
            },
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
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
            n_jobs=-1,
        )
        positive_index = list(classification.classes_).index(1)
        oof_probabilities = oof_predictions[:, positive_index]
        threshold, threshold_selection, threshold_validation = _select_decision_threshold(
            y_train, oof_probabilities
        )
        f1_threshold = _select_max_f1_threshold(y_train, oof_probabilities)
    else:
        classification.fit(X_train, y_train)
        best_params = {"model__max_depth": 6}

    probabilities = _positive_probability(classification, X_test)
    classification_predictions = (probabilities >= threshold).astype(int)
    classification_test = _classification_metrics(
        y_test, classification_predictions, probabilities
    )
    classification_test_f1_mode = _classification_metrics(
        y_test, (probabilities >= f1_threshold).astype(int), probabilities
    )

    # Baselines & Ensemble Benchmarks
    classifier_baseline = DummyClassifier(strategy="most_frequent")
    classifier_baseline.fit(X_train, y_train)
    classification_baseline = _classification_metrics(
        y_test, classifier_baseline.predict(X_test), _positive_probability(classifier_baseline, X_test)
    )

    logistic_baseline_metrics = None
    hgb_classifier_metrics = None
    rf_classifier_metrics = None
    if y_train.nunique() > 1:
        logistic_pipe = Pipeline([
            ("preprocess", _preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ])
        logistic_pipe.fit(X_train, y_train)
        log_probs = _positive_probability(logistic_pipe, X_test)
        logistic_baseline_metrics = _classification_metrics(
            y_test, (log_probs >= threshold).astype(int), log_probs
        )

        if len(X_train) >= 100:
            hgb_pipe = Pipeline([
                ("preprocess", _preprocessor(scale_numeric=False)),
                ("model", HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.05, max_depth=5, min_samples_leaf=100, random_state=seed
                )),
            ])
            hgb_pipe.fit(X_train, y_train)
            hgb_probs = _positive_probability(hgb_pipe, X_test)
            hgb_classifier_metrics = {
                "high_accuracy_mode": _classification_metrics(
                    y_test, (hgb_probs >= threshold).astype(int), hgb_probs
                ),
                "risk_screening_f1_mode": _classification_metrics(
                    y_test, (hgb_probs >= f1_threshold).astype(int), hgb_probs
                ),
            }

            rf_pipe = Pipeline([
                ("preprocess", _preprocessor(scale_numeric=False)),
                ("model", RandomForestClassifier(
                    n_estimators=120, max_depth=10, min_samples_leaf=50, random_state=seed, n_jobs=-1
                )),
            ])
            rf_pipe.fit(X_train, y_train)
            rf_probs = _positive_probability(rf_pipe, X_test)
            rf_classifier_metrics = {
                "high_accuracy_mode": _classification_metrics(
                    y_test, (rf_probs >= threshold).astype(int), rf_probs
                ),
                "risk_screening_f1_mode": _classification_metrics(
                    y_test, (rf_probs >= f1_threshold).astype(int), rf_probs
                ),
            }

    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(classification, destination / "loan_default_classifier.joblib")

    tree_rules = _extract_tree_rules(classification)
    (destination / "decision_tree_rules.txt").write_text(tree_rules)

    # Seasoning-Aware Recovery Rate + OLS/Poly2-Ridge/HistGBR & LGD on Charged Off Loans
    defaulted = y.eq(1)
    recovery_train_mask = defaulted & frame.index.isin(train_idx)
    recovery_test_mask = defaulted & frame.index.isin(test_idx)
    recovery_features = X_with_date.loc[recovery_train_mask]
    recovery_target = pd.to_numeric(
        frame.loc[recovery_train_mask, "total_rec_prncp"], errors="coerce"
    )
    recovery_test_features = X_with_date.loc[recovery_test_mask]
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
    recovery_cohort_matched = None
    if len(recovery_features) >= 2 and recovery_target.nunique() > 1 and len(recovery_test_target):
        recovery_metrics, recovery_model = _evaluate_recovery_models(
            recovery_features, recovery_target, recovery_test_features, recovery_test_target, seed,
            include_poly_and_hgb=False,
        )
        joblib.dump(recovery_model, destination / "recovery_linear_regressor.joblib")

        # Seasoning-matched 80/20 split across all charged-off loans
        all_co = frame.loc[defaulted].copy()
        all_co_target = pd.to_numeric(all_co["total_rec_prncp"], errors="coerce")
        valid_all = all_co_target.notna()
        all_co = all_co.loc[valid_all]
        all_co_target = all_co_target.loc[valid_all]
        if len(all_co) >= 10:
            co_years = pd.to_datetime(all_co["issue_date"], errors="coerce").dt.year.fillna(0).astype(int)
            test_n = max(1, int(len(all_co) * 0.2))
            strat_col = (
                co_years
                if co_years.value_counts().min() >= 2 and test_n >= co_years.nunique()
                else None
            )
            c_tr, c_te = train_test_split(
                all_co.index, test_size=0.2, random_state=seed, stratify=strat_col
            )
            recovery_cohort_matched, _ = _evaluate_recovery_models(
                all_co.loc[c_tr, FEATURES + ["issue_date"]],
                all_co_target.loc[c_tr],
                all_co.loc[c_te, FEATURES + ["issue_date"]],
                all_co_target.loc[c_te],
                seed,
                include_poly_and_hgb=True,
            )
    elif len(recovery_features):
        recovery_metrics = {
            "mae": None,
            "mse": None,
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
            "decision_threshold": threshold,
            "risk_screening_f1_threshold": f1_threshold,
            "test": classification_test,
            "test_risk_screening_f1_mode": classification_test_f1_mode,
            "baseline": {
                "model": "DummyClassifier(strategy='most_frequent')",
                **classification_baseline,
            },
            "logistic_regression_baseline": logistic_baseline_metrics,
            "hist_gradient_boosting_benchmark": hgb_classifier_metrics,
            "random_forest_benchmark": rf_classifier_metrics,
            "best_params": best_params,
            "cross_validation_average_precision": cv_average_precision,
            "threshold_selection": threshold_selection,
            "threshold_validation": threshold_validation,
            "decision_tree_rules": tree_rules,
        },
        "evaluation": {
            **split_details,
            "training_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "selection_metric": "PR-AUC",
            "cross_validation_folds": cv_folds or None,
        },
        "recovery_linear_regression": recovery_metrics,
        "recovery_cohort_matched_80_20": recovery_cohort_matched,
        "seed": seed,
        "features": FEATURES,
        "engineered_features": ENGINEERED_NUMERIC,
        "recovery_target": "Seasoning-aware bounded recovery rate (total_rec_prncp / loan_amnt) reconstructed to dollars for Charged Off loans",
        "recovery_model": "Seasoning-Aware Ridge(alpha=10.0) + OLS LinearRegression + Poly2-Ridge + HistGBR",
        "recovery_baseline": "DummyRegressor(strategy='mean')",
        "classification_target": "loan_status_label (Fully Paid=0, Charged Off=1)",
        "model_types": [
            "DecisionTreeClassifier",
            "HistGradientBoostingClassifier",
            "RandomForestClassifier",
            "LogisticRegression",
            "LinearRegression",
            "Ridge",
            "HistGradientBoostingRegressor",
        ],
    }
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
