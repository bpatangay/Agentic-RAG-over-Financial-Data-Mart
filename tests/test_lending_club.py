from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from lending_club.api import app
import lending_club.api as api_module
from lending_club.cli import main, run_benchmark
from lending_club.data import prepare_lending_data
from lending_club.etl import load_lending_data
from lending_club.models import _select_decision_threshold, train_models
from lending_club.rag import (
    SUPPORTED,
    answer_question,
    clear_cache,
    generate_sql,
    validate_readonly_sql,
)


def raw_fixture(rows=12):
    statuses = ["Fully Paid", "Charged Off", "Current"]
    return pd.DataFrame(
        {
            "issue_d": [f"Jan-{15 + (i % 4):02d}" for i in range(rows)],
            "annual_inc": [50000 + i * 1000 for i in range(rows)],
            "dti": ["10.0%" for _ in range(rows)],
            "revol_util": ["25.0%" for _ in range(rows)],
            "fico_range_low": [680 + i % 4 for i in range(rows)],
            "loan_amnt": [10000 + i * 100 for i in range(rows)],
            "int_rate": ["10.5%" for _ in range(rows)],
            "term": [" 36 months" for _ in range(rows)],
            "installment": [320 + i for i in range(rows)],
            "loan_status": [statuses[i % 3] for i in range(rows)],
            "grade": ["A" if i % 3 == 0 else "B" if i % 3 == 1 else "C" for i in range(rows)],
            "total_rec_prncp": [8000 + i * 50 for i in range(rows)],
        }
    )


def prepared_fixture(tmp_path, rows=12):
    raw = tmp_path / "loans.csv"
    raw_fixture(rows).to_csv(raw, index=False)
    return prepare_lending_data(raw, limit=100_000).frame


def test_prepare_filters_dates_statuses_and_normalizes(tmp_path):
    frame = prepared_fixture(tmp_path)
    assert len(frame) == 8
    assert set(frame["loan_status"]) == {"Fully Paid", "Charged Off"}
    assert frame["dti"].iloc[0] == 10.0
    assert frame["term"].iloc[0] == 36.0
    assert set(frame["loan_status_label"]) == {0, 1}
    assert set(frame["grade"]).issubset(set("ABCDEFG"))
    assert "total_rec_prncp" in frame.columns


def test_prepare_reports_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"issue_d": ["Jan-15"]}).to_csv(path, index=False)
    try:
        prepare_lending_data(path)
        assert False, "expected missing-column error"
    except ValueError as exc:
        assert "required columns" in str(exc)


def test_etl_and_rag(tmp_path):
    clear_cache()
    database = tmp_path / "loans.db"
    frame = prepared_fixture(tmp_path)
    stats = load_lending_data(frame, database)
    assert stats["row_count_match"] is True
    assert stats["foreign_key_violations"] == 0
    result = answer_question("What is the default rate by year?", str(database))
    assert result["rows"]
    assert "SELECT" in result["sql"]
    assert generate_sql("hello")[0] is None


def test_sql_guardrails_reject_writes_and_unknown_tables():
    for bad_sql in (
        "DROP TABLE fact_loan",
        "DELETE FROM dim_loan",
        "SELECT * FROM fact_loan; DROP TABLE dim_date",
        "SELECT * FROM sqlite_master",
    ):
        try:
            validate_readonly_sql(bad_sql)
            assert False, f"Expected ValueError for {bad_sql}"
        except ValueError:
            pass


def test_all_documented_sample_questions_execute(tmp_path):
    clear_cache()
    database = tmp_path / "loans.db"
    frame = prepared_fixture(tmp_path, rows=30)
    load_lending_data(frame, database)
    for question in SUPPORTED:
        sql, explanation = generate_sql(question)
        assert sql is not None
        result = answer_question(question, str(database), use_cache=False)
        assert result["sql"] == sql
        assert result["answer"] == explanation
        assert isinstance(result["rows"], list)


def test_golden_40_benchmark_executes_cleanly(tmp_path):
    clear_cache()
    database = tmp_path / "loans.db"
    frame = prepared_fixture(tmp_path, rows=60)
    load_lending_data(frame, database)
    bench_file = Path("benchmarks/golden_queries_40.json")
    out_file = tmp_path / "bench_out.json"
    summary = run_benchmark(str(database), bench_file, out_file)
    assert summary["total_queries"] == 40
    assert summary["sql_execution_success_pct"] == 100.0
    assert summary["schema_entity_match_pct"] == 100.0
    assert summary["result_set_exact_match_pct"] == 100.0


def test_models_train_on_small_fixture(tmp_path):
    frame = prepared_fixture(tmp_path, rows=30)
    metrics = train_models(frame, tmp_path / "artifacts")
    assert "classification_accuracy" in metrics
    assert (tmp_path / "artifacts" / "loan_default_classifier.joblib").exists()
    assert (tmp_path / "artifacts" / "decision_tree_rules.txt").exists()
    assert "recovery_linear_regression" in metrics


def test_models_use_latest_year_holdout_and_report_baselines(tmp_path):
    frame = prepared_fixture(tmp_path, rows=120)
    metrics = train_models(frame, tmp_path / "artifacts")

    assert metrics["evaluation"]["strategy"] == "latest_issue_year_holdout"
    assert metrics["evaluation"]["training_years"] == [2015, 2016, 2017]
    assert metrics["evaluation"]["test_year"] == 2018
    assert metrics["classification"]["model"] == "DecisionTreeClassifier"
    assert metrics["classification"]["test"]["confusion_matrix_labels_0_1"]
    assert metrics["classification"]["test"]["pr_auc"] is not None
    assert metrics["classification"]["threshold_selection"] in {
        "training_oof_max_recall_at_77pct_accuracy",
        "training_oof_max_f1_accuracy_floor_unmet",
    }
    assert metrics["classification"]["threshold_validation"] is not None
    assert 0 <= metrics["classification"]["decision_threshold"] <= 1
    assert metrics["classification"]["baseline"]["model"].startswith("DummyClassifier")
    assert metrics["recovery_linear_regression"]["baseline"]["mae"] >= 0


def test_threshold_selection_preserves_accuracy_floor_and_maximizes_recall():
    actual = pd.Series([0] * 8 + [1] * 2)
    probabilities = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]).to_numpy()

    threshold, selection, validation = _select_decision_threshold(actual, probabilities)

    assert threshold == 0.9
    assert selection == "training_oof_max_recall_at_77pct_accuracy"
    assert validation["accuracy_floor_met"] is True
    assert validation["accuracy"] >= 0.77
    assert validation["recall"] == 1.0


def test_api_health_query_rules_and_predict(tmp_path, monkeypatch):
    clear_cache()
    database = tmp_path / "loans.db"
    artifacts = tmp_path / "artifacts"
    frame = prepared_fixture(tmp_path, rows=60)
    load_lending_data(frame, database)
    train_models(frame, artifacts)

    monkeypatch.setattr(api_module, "DATABASE", Path(database))
    monkeypatch.setattr(api_module, "ARTIFACTS", Path(artifacts))
    api_module._load_models()

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert "Complete JSON response" in client.get("/").text
    assert client.get("/rules").status_code == 200
    assert client.get("/metrics").status_code == 200
    response = client.post("/query", json={"question": "hello"})
    assert response.status_code == 200
    assert response.json()["sql"] is None

    pred = client.post(
        "/predict",
        json={
            "annual_inc": 75000,
            "dti": 14.5,
            "revol_util": 35.0,
            "fico_range_low": 700,
            "loan_amnt": 15000,
            "int_rate": 11.5,
            "term": 36,
            "installment": 495.0,
            "grade": "B",
        },
    )
    assert pred.status_code == 200
    body = pred.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["expected_lgd"] is not None


def test_cli_prepare_load_and_train(tmp_path):
    raw = tmp_path / "raw.csv"
    prepared = tmp_path / "prepared.csv"
    database = tmp_path / "loans.db"
    artifacts = tmp_path / "artifacts"
    raw_fixture(30).to_csv(raw, index=False)
    assert main(["prepare", str(raw), str(prepared)]) == 0
    assert main(["load", str(prepared), str(database)]) == 0
    assert main(["train", str(prepared), str(artifacts)]) == 0
    assert database.exists()
    assert (artifacts / "metrics.json").exists()
