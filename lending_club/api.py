from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    import orjson  # noqa: F401
    from fastapi.responses import ORJSONResponse as DefaultJSONResponse
except ImportError:
    DefaultJSONResponse = JSONResponse

from .api_ui import QUERY_UI
from .models import FEATURES
from .rag import answer_question, get_cache_stats

DATABASE = Path(os.getenv("LENDING_CLUB_DB", "data/lending_club.db"))
ARTIFACTS = Path(os.getenv("LENDING_CLUB_ARTIFACTS", "artifacts"))


def _load_models() -> None:
    clf_path = ARTIFACTS / "loan_default_classifier.joblib"
    reg_path = ARTIFACTS / "recovery_linear_regressor.joblib"
    metrics_path = ARTIFACTS / "metrics.json"
    rules_path = ARTIFACTS / "decision_tree_rules.txt"

    app.state.classifier = joblib.load(clf_path) if clf_path.exists() else None
    app.state.regressor = joblib.load(reg_path) if reg_path.exists() else None
    app.state.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    app.state.tree_rules = rules_path.read_text() if rules_path.exists() else None

    threshold = 0.5
    if app.state.metrics:
        threshold = float(
            app.state.metrics.get("classification", {}).get("decision_threshold", 0.5)
        )
    app.state.decision_threshold = threshold


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_models()
    yield


app = FastAPI(
    title="Lending Club Financial Data Mart",
    default_response_class=DefaultJSONResponse,
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.state.classifier = None
app.state.regressor = None
app.state.metrics = None
app.state.tree_rules = None
app.state.decision_threshold = 0.5


class QueryRequest(BaseModel):
    question: str


class PredictRequest(BaseModel):
    annual_inc: float
    dti: float
    revol_util: float
    fico_range_low: float
    loan_amnt: float
    int_rate: float
    term: float
    installment: float
    grade: str


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return QUERY_UI


@app.get("/health")
def health() -> dict:
    db_str = os.getenv("LENDING_CLUB_DB", str(DATABASE))
    ready = db_str.startswith(("postgresql://", "postgres://")) or Path(db_str).exists()
    return {
        "status": "ok" if ready else "database_not_loaded",
        "database": db_str,
        "models_loaded": app.state.classifier is not None,
        "decision_threshold": round(float(app.state.decision_threshold), 6),
        "cache": get_cache_stats(),
    }


@app.get("/metrics")
def get_metrics() -> dict:
    if app.state.metrics is None:
        _load_models()
    if app.state.metrics is None:
        raise HTTPException(status_code=503, detail="Model metrics not found. Run the train command first.")
    return app.state.metrics


@app.get("/rules")
def get_rules() -> dict:
    if app.state.tree_rules is None:
        _load_models()
    if app.state.tree_rules is None:
        raise HTTPException(status_code=503, detail="Decision tree rules not found. Run the train command first.")
    return {
        "model": "DecisionTreeClassifier",
        "decision_threshold": app.state.decision_threshold,
        "rules": app.state.tree_rules,
    }


@app.get("/cache/stats")
def cache_stats() -> dict:
    return get_cache_stats()


@app.post("/query")
def query(request: QueryRequest) -> dict:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    db_str = os.getenv("LENDING_CLUB_DB", str(DATABASE))
    if not db_str.startswith(("postgresql://", "postgres://")) and not Path(db_str).exists():
        raise HTTPException(status_code=503, detail="Data has not been loaded. Run the preparation and load commands first.")
    return answer_question(request.question, db_str, use_olap_cube=True)


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    if app.state.classifier is None:
        _load_models()
    if app.state.classifier is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. Run the train command first.")
    row = pd.DataFrame([{k: getattr(request, k) for k in FEATURES}])
    default_prob = float(app.state.classifier.predict_proba(row)[0][1])
    threshold = float(app.state.decision_threshold)
    expected_recovery = None
    expected_lgd = None
    if app.state.regressor is not None:
        rec_rate = float(np.clip(app.state.regressor.predict(row)[0], 0.0, 1.0))
        expected_recovery = round(rec_rate * request.loan_amnt, 2)
        expected_lgd = round(1.0 - rec_rate, 4)
    return {
        "default_probability": round(default_prob, 4),
        "decision_threshold": round(threshold, 4),
        "predicted_default": int(default_prob >= threshold),
        "expected_recovery_amount": expected_recovery,
        "expected_lgd": expected_lgd,
    }
