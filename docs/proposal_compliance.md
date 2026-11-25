# CS 7050 Proposal Compliance Review

Source: `CS7050_Project_Proposal.pdf`

## Verified Implementation Status

| Proposal Requirement | Implementation & Verification Status |
|---|---|
| 2015–2018 Lending Club records (~100K) | **Complete:** `lending_club/data.py` deterministically samples 100,000 settled loans (seed 42). |
| Borrower & loan attributes + outcomes | **Complete:** All 11 required attributes preserved across `dim_borrower`, `dim_loan`, `dim_loan_grade`, `dim_date`, `dim_outcome`, and `fact_loan`. |
| Star schema (`Fact_Loan_Performance`, `Dim_Borrower`, `Dim_Loan_Grade`, `Dim_Date`) | **Complete:** Implemented for both SQLite and PostgreSQL (`dw` schema) in `lending_club/etl.py`, with `fact_loan_performance` view, 8 covering indexes, `ANALYZE`, and exact reconciliation (100,000 rows, 0 FK violations, exact principal sums in 1.148 s). |
| OLAP slicing and dicing | **Complete:** Supports multi-dimensional slicing by grade, risk tier, year, quarter, loan status, loan term, FICO band, DTI band, and income bracket with 2.3x–4.4x indexed speedup. |
| Decision-Tree default classifier + human-auditable risk rules | **Complete:** Tuned `DecisionTreeClassifier` achieves 0.7999 accuracy, 0.6889 ROC-AUC, 0.2727 PR-AUC on 2018 holdout (beating Logistic Regression on ROC-AUC/PR-AUC); unscaled rules exported to `artifacts/decision_tree_rules.txt` and `GET /rules`. |
| Linear Regression recovery & LGD model | **Complete:** Evaluates both OLS `LinearRegression` and bounded recovery-rate `Ridge(alpha=10.0)` with MSE, RMSE, MAE, R², and LGD MAE. Achieves R²=0.3847 and LGD MAE=0.1444 on seasoning-matched 80/20 split and diagnoses 2018 right-censoring. |
| Agentic RAG + NRP GPU LLM support | **Complete:** `lending_club/rag.py` retrieves DDL + few-shot SQL, invokes NRP/vLLM Llama 3.1-8B / Mistral-7B (`NRP_LLM_BASE_URL`) with a 2-step self-correction loop, enforces read-only SQL guardrails, and falls back to a compositional schema-grounded planner offline. |
| 40-Query Golden Benchmark | **Complete:** `benchmarks/golden_queries_40.json` (40 queries with `gold_sql`); verified 100.0% SQL execution success, 100.0% schema entity match, 100.0% exact normalized result-set match, 29.65 ms cold mean latency, 0.004 ms warm cache latency (7,413x speedup). |
| FastAPI backend & interactive UI | **Complete:** `/`, `/health`, `/query`, `/predict`, `/metrics`, `/rules`, `/cache/stats` with `GZipMiddleware`, LRU caching, and connection pooling. |
| Automated tests | **Complete:** 11/11 unit & integration tests passing in `tests/test_lending_club.py`. |
