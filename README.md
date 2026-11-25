# Agentic RAG over a Lending Club Financial Data Mart

CS 7050 (Data Warehousing and Data Mining) course project implementing an end-to-end financial data mart, interpretable credit-risk & LGD machine-learning pipeline, and schema-grounded Agentic RAG FastAPI service over 100,000 settled Lending Club loans (2015–2018) from [All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club).

## Key Highlights & Verified Results

- **Star-Schema Data Mart (SQLite & PostgreSQL):** `fact_loan` (and `fact_loan_performance` view) linked to `dim_date`, `dim_borrower`, `dim_loan`, `dim_loan_grade`, and `dim_outcome`, with 8 covering OLAP indexes and `ANALYZE`. Loads 100,000 rows in **1.148 s** with **0 FK violations** and exact principal reconciliation ($1,455,527,150.00 funded; $1,200,802,179.03 recovered).
- **Decision Tree Default Classifier + Auditable Risk Rules:** On the untouched 2018 temporal holdout (n=6,358), achieves **0.7999 accuracy**, **0.6889 ROC-AUC**, and **0.2727 PR-AUC** (outperforming Logistic Regression ROC-AUC 0.6863 & PR-AUC 0.2704) while exporting human-auditable financial risk rules (`artifacts/decision_tree_rules.txt`, `GET /rules`).
- **Principal Recovery & LGD Modeling:** Bounded recovery-rate `Ridge(alpha=10.0)` and OLS `LinearRegression` on charged-off loans achieve **R² = 0.3847**, **MAE = $2,133.32**, and **LGD MAE = 0.1444** (predicted LGD 72.52% vs. actual 72.43%) on a seasoning-matched 80/20 split, while documenting the 2018 right-censoring effect on the temporal holdout.
- **Hybrid Agentic RAG + 40-Query Golden Benchmark:** Supports NRP GPU LLM endpoints (`NRP_LLM_BASE_URL` for Llama 3.1-8B / Mistral-7B with 2-step self-correction) plus an offline compositional schema-grounded NL-to-SQL planner, read-only SQL guardrails, thread-local connection pooling, and a bounded LRU+TTL cache. On `benchmarks/golden_queries_40.json` (100K loans):
  - **100.0%** SQL execution success
  - **100.0%** schema entity match
  - **100.0%** exact normalized result-set match
  - **29.65 ms** mean cold latency (p50 22.57 ms, p95 64.02 ms) → **0.004 ms** warm-cache latency (**7,412.8x speedup**)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# 1. Prepare 100K settled 2015–2018 records from manually downloaded Kaggle CSV
python -m lending_club.cli prepare /path/to/accepted_2007_to_2018Q4.csv data/loans.csv

# 2. Load indexed star schema (SQLite path or postgresql:// DSN)
python -m lending_club.cli load data/loans.csv data/lending_club.db

# 3. Train Decision Tree classifier + Recovery/LGD regressors
python -m lending_club.cli train data/loans.csv artifacts/

# 4. Run the 40-query golden NL-to-SQL benchmark
python -m lending_club.cli benchmark \
  data/lending_club.db \
  benchmarks/golden_queries_40.json \
  artifacts/benchmark_40.json

# 5. Start the FastAPI server and Web UI
uvicorn lending_club.api:app --reload
```

## Optional NRP GPU LLM & PostgreSQL Configuration

```bash
export LENDING_CLUB_DB="postgresql://user:pass@localhost:5432/lending_club"
export NRP_LLM_BASE_URL="http://localhost:8001/v1"
export NRP_LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"
export NRP_LLM_API_KEY="EMPTY"
```

## FastAPI Endpoints

- `GET /` — Interactive Web UI (charts, SQL preview, latency/mode badges, Decision Tree risk rules)
- `GET /health` — Database, model, threshold, and cache status
- `POST /query` — Natural-language analytical query → validated SQL, rows, and executive summary
- `POST /predict` — Live default probability, predicted default flag, expected recovery amount, and LGD
- `GET /metrics` — Verified ML evaluation metrics (`artifacts/metrics.json`)
- `GET /rules` — Extracted human-auditable Decision Tree risk rules
- `GET /cache/stats` — LRU cache hit/miss telemetry

## Repository Structure

- `lending_club/` — `data.py` (prep), `etl.py` (SQLite/PostgreSQL star schema), `models.py` (ML & LGD), `rag.py` (Agentic RAG + caching + pooling), `api.py` & `api_ui.py` (FastAPI + UI), `cli.py` (CLI & benchmark)
- `benchmarks/golden_queries_40.json` — 40-query golden NL-to-SQL benchmark with gold reference SQL
- `tests/test_lending_club.py` — 11 automated unit and integration tests
- `docs/report.tex`, `docs/report.md`, `docs/report.pdf` — Final IEEE-format CS 7050 project report
- `docs/proposal_compliance.md` — Proposal compliance matrix
