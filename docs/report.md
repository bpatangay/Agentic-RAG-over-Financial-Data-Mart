# CS 7050 Project Report

**Agentic Retrieval-Augmented Generation (RAG) over a Financial Data Mart**

> Companion Markdown version of the IEEE-style LaTeX report in `docs/report.tex` (compiled to `docs/report.pdf`).

## Title and Author
- **Author:** Bhavani Patangay
- **Institution:** Department of Computer Science, Kennesaw State University
- **Course:** CS 7050 — Data Warehousing and Data Mining (Section 01)
- **Location:** Marietta, U.S.A.
- **Email:** bpatanga@students.kennesaw.edu

## Abstract
This project implements and evaluates an end-to-end analytical platform combining a star-schema financial data mart with a 10,062-cell materialized OLAP cube (dual SQLite and PostgreSQL backends), interpretable and ensemble machine-learning models, and a schema-grounded Agentic RAG service over 100,000 settled Lending Club loans (2015–2018). ETL loads 100,000 records, builds 8 covering indexes, and materializes `agg_loan_cube` in 1.368 s with zero foreign-key violations and exact principal reconciliation ($1,455,527,150.00 funded; $1,200,802,179.03 recovered). On the untouched 2018 temporal holdout (n=6,358), the tuned Decision Tree achieves 0.6895 ROC-AUC, 0.7929 accuracy, 0.2716 recall, and 0.2938 F1 (or 0.6809 recall / 0.3660 F1 in risk-screening mode) with human-auditable financial risk rules, while `HistGradientBoostingClassifier` reaches 0.6972 ROC-AUC, 0.2858 PR-AUC, and 0.8023 accuracy. For charged-off principal recovery and Loss Given Default (LGD), incorporating origination seasoning reduces 2018 holdout MAE by 70.8% ($2,932 → $856) and LGD MAE by 72.4% (0.1829 → 0.0505), while achieving R² = 0.4714 (OLS), R² = 0.4712 (Ridge), and R² = 0.4728 (HistGBR) with LGD MAE 0.1311 on a cohort-matched 80/20 split. Across a 40-query golden NL-to-SQL benchmark, the RAG engine achieves 100.0% SQL execution success, 100.0% schema entity match, and 100.0% exact result-set match at 1.70 ms mean cold OLAP cube latency (13.95x faster than star-schema joins) and 0.002 ms warm-cache latency (11,862x speedup).

## I. Star Schema, Materialized OLAP Cube & ETL Reconciliation
| Reconciliation Check | Verified Result |
|---|---:|
| Source / Target Fact Rows | 100,000 / 100,000 (Exact match) |
| Materialized OLAP Cube Cells (`agg_loan_cube`) | 10,062 |
| Foreign-Key Violations | 0 |
| Total Funded Principal (`loan_amnt`) | $1,455,527,150.00 (Exact match) |
| Total Principal Received (`total_rec_prncp`) | $1,200,802,179.03 (Exact match) |
| End-to-End Load + Cube + Indexes + `ANALYZE` Time | 1.368 s |

## II. Default Classification Results (2018 Temporal Holdout, n=6,358)
| Model / Operating Point | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Decision Tree (High-Accuracy, τ=0.4035) | 0.7929 | 0.3201 | 0.2716 | 0.2938 | 0.6895 | 0.2660 |
| Decision Tree (Risk-Screening, τ=0.2275) | 0.6257 | 0.2503 | **0.6809** | **0.3660** | 0.6895 | 0.2660 |
| Logistic Regression Baseline (τ=0.4035) | 0.7941 | 0.3144 | 0.2517 | 0.2796 | 0.6905 | 0.2727 |
| Random Forest Benchmark (High-Acc / Risk-Screen) | 0.7976 / 0.6203 | 0.3176 / 0.2456 | 0.2398 / 0.6720 | 0.2733 / 0.3597 | 0.6959 | 0.2822 |
| **HistGradientBoosting** (High-Acc / Risk-Screen) | **0.8023** / 0.6255 | **0.3301** / 0.2474 | 0.2389 / 0.6660 | 0.2772 / 0.3608 | **0.6972** | **0.2858** |
| Majority-Class Baseline | 0.8413 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1587 |

## III. Seasoning-Aware Principal Recovery & LGD Regression
| Evaluation Split | Model | MAE ($) | RMSE ($) | MSE | R² | LGD MAE |
|---|---|---:|---:|---:|---:|---:|
| Cohort-Matched 80/20 (n=4,330) | **HistGradientBoostingRegressor** | **1,928.13** | **2,794.97** | **7,811,855** | **0.4728** | **0.1311** |
| Cohort-Matched 80/20 (n=4,330) | OLS LinearRegression (Seasoning-Aware) | 1,948.40 | 2,798.65 | 7,832,459 | 0.4714 | 0.1319 |
| Cohort-Matched 80/20 (n=4,330) | Ridge(α=10) (Seasoning-Aware) | 1,948.68 | 2,799.23 | 7,835,681 | 0.4712 | 0.1319 |
| Cohort-Matched 80/20 (n=4,330) | Poly2-Ridge(α=100) | 1,939.50 | 2,842.94 | 8,082,335 | 0.4546 | 0.1315 |
| Cohort-Matched 80/20 (n=4,330) | Training-Mean Baseline | 2,788.79 | 3,849.54 | 14,818,947 | 0.0000 | — |
| 2018 Holdout (n=1,009) | **Ridge(α=10) (Seasoning-Aware)** | **855.87** | **1,837.64** | **3,376,917** | **-0.0101** | **0.0505** |
| 2018 Holdout (n=1,009) | OLS LinearRegression (Seasoning-Aware) | 857.30 | 1,838.21 | 3,379,027 | -0.0107 | 0.0506 |
| 2018 Holdout (n=1,009) | Training-Mean Baseline | 3,156.40 | 3,464.18 | 12,000,544 | -2.5894 | — |

## IV. 40-Query Golden NL-to-SQL Benchmark Results (100K Loans)
| Benchmark Metric | Measured Result |
|---|---:|
| Total Golden Queries | 40 |
| SQL Syntax Execution Success Rate | **100.0%** |
| Schema Entity Match Accuracy | **100.0%** |
| Star-Schema Exact Result-Set Match Rate | **100.0%** |
| OLAP Cube Exact Result-Set Match Rate | **100.0%** |
| Star-Schema Cold Mean / p50 / p95 Latency | 23.72 / 20.60 / 54.67 ms |
| **OLAP Cube Cold Mean / p50 / p95 Latency** | **1.70 / 1.48 / 2.91 ms** |
| OLAP Cube Cold Speedup Factor | **13.95x** |
| **Warm Cache Mean / p95 Latency** | **0.002 / 0.003 ms** |
| Warm Cache Speedup Factor | **11,861.5x** |
