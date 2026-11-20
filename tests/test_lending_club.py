from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from lending_club.api import app
import lending_club.api as api_module
from lending_club.data import prepare_lending_data
from lending_club.etl import load_lending_data
from lending_club.models import train_models
from lending_club.rag import SUPPORTED, generate_sql, answer_question
from lending_club.cli import main


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
    database = tmp_path / "loans.db"
    frame = prepared_fixture(tmp_path)
    load_lending_data(frame, database)
    result = answer_question("What is the default rate by year?", str(database))
    assert result["rows"]
    assert "SELECT" in result["sql"]
    assert generate_sql("hello")[0] is None


def test_all_documented_sample_questions_execute(tmp_path):
    database = tmp_path / "loans.db"
    frame = prepared_fixture(tmp_path, rows=30)
    load_lending_data(frame, database)
    for question in SUPPORTED:
        sql, explanation = generate_sql(question)
        assert sql is not None
        result = answer_question(question, str(database))
        assert result["sql"] == sql
        assert result["answer"] == explanation
        assert isinstance(result["rows"], list)


def test_models_train_on_small_fixture(tmp_path):
    frame = prepared_fixture(tmp_path, rows=30)
    metrics = train_models(frame, tmp_path / "artifacts")
    assert "classification_accuracy" in metrics
    assert (tmp_path / "artifacts" / "loan_default_classifier.joblib").exists()
    assert "recovery_linear_regression" in metrics


def test_api_health_and_query(tmp_path, monkeypatch):
    database = tmp_path / "loans.db"
    load_lending_data(prepared_fixture(tmp_path), database)
    monkeypatch.setattr(api_module, "DATABASE", Path(database))
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert "Complete JSON response" in client.get("/").text
    response = client.post("/query", json={"question": "hello"})
    assert response.status_code == 200
    assert response.json()["sql"] is None


def test_cli_prepare_load_and_train(tmp_path, capsys):
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

