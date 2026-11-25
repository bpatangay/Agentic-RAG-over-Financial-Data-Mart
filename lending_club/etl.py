from __future__ import annotations

import io
import sqlite3
import time
from pathlib import Path

import pandas as pd

GRADE_RISK_TIERS = {
    "A": "Prime",
    "B": "Near-Prime",
    "C": "Standard",
    "D": "Subprime-Low",
    "E": "Subprime-Mid",
    "F": "High-Risk",
    "G": "Very-High-Risk",
}

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_date (
  date_key INTEGER PRIMARY KEY, issue_date TEXT NOT NULL UNIQUE,
  year INTEGER NOT NULL, month INTEGER NOT NULL, quarter INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS dim_loan_grade (
  grade_key INTEGER PRIMARY KEY AUTOINCREMENT, grade TEXT NOT NULL UNIQUE,
  risk_tier TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dim_loan (
  loan_key INTEGER PRIMARY KEY AUTOINCREMENT, loan_id TEXT NOT NULL UNIQUE,
  loan_amnt REAL NOT NULL, int_rate REAL NOT NULL, term_months REAL NOT NULL,
  installment REAL NOT NULL, grade TEXT
);
CREATE TABLE IF NOT EXISTS dim_borrower (
  borrower_key INTEGER PRIMARY KEY AUTOINCREMENT, loan_id TEXT NOT NULL UNIQUE,
  annual_inc REAL NOT NULL, dti REAL NOT NULL, revol_util REAL NOT NULL,
  fico_range_low REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS dim_outcome (
  outcome_key INTEGER PRIMARY KEY AUTOINCREMENT, loan_status TEXT NOT NULL UNIQUE,
  default_flag INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_loan (
  loan_key INTEGER NOT NULL, date_key INTEGER NOT NULL, borrower_key INTEGER NOT NULL,
  outcome_key INTEGER NOT NULL, grade_key INTEGER,
  total_rec_prncp REAL NOT NULL DEFAULT 0, PRIMARY KEY (loan_key),
  FOREIGN KEY (loan_key) REFERENCES dim_loan(loan_key),
  FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
  FOREIGN KEY (borrower_key) REFERENCES dim_borrower(borrower_key),
  FOREIGN KEY (outcome_key) REFERENCES dim_outcome(outcome_key),
  FOREIGN KEY (grade_key) REFERENCES dim_loan_grade(grade_key)
);
CREATE TABLE IF NOT EXISTS agg_loan_cube (
  year INTEGER NOT NULL,
  quarter INTEGER NOT NULL,
  grade TEXT NOT NULL,
  risk_tier TEXT NOT NULL,
  loan_status TEXT NOT NULL,
  default_flag INTEGER NOT NULL,
  term_months REAL NOT NULL,
  fico_band TEXT NOT NULL,
  dti_band TEXT NOT NULL,
  income_band TEXT NOT NULL,
  loan_count INTEGER NOT NULL,
  sum_loan_amnt REAL NOT NULL,
  sum_int_rate REAL NOT NULL,
  sum_installment REAL NOT NULL,
  sum_annual_inc REAL NOT NULL,
  sum_dti REAL NOT NULL,
  sum_revol_util REAL NOT NULL,
  sum_fico REAL NOT NULL,
  sum_rec_prncp REAL NOT NULL
);
CREATE VIEW IF NOT EXISTS fact_loan_performance AS
SELECT
  f.loan_key, f.date_key, f.borrower_key, f.outcome_key, f.grade_key,
  l.loan_amnt, l.int_rate, l.term_months, l.installment, l.grade,
  o.loan_status, o.default_flag, f.total_rec_prncp
FROM fact_loan f
JOIN dim_loan l ON l.loan_key = f.loan_key
JOIN dim_outcome o ON o.outcome_key = f.outcome_key;

CREATE INDEX IF NOT EXISTS idx_fact_date_outcome ON fact_loan(date_key, outcome_key, total_rec_prncp);
CREATE INDEX IF NOT EXISTS idx_fact_outcome_loan ON fact_loan(outcome_key, loan_key, total_rec_prncp);
CREATE INDEX IF NOT EXISTS idx_fact_borrower_outcome ON fact_loan(borrower_key, outcome_key);
CREATE INDEX IF NOT EXISTS idx_fact_grade_outcome ON fact_loan(grade_key, outcome_key, total_rec_prncp);
CREATE INDEX IF NOT EXISTS idx_dim_loan_grade ON dim_loan(grade, loan_amnt, int_rate, term_months);
CREATE INDEX IF NOT EXISTS idx_dim_loan_term ON dim_loan(term_months, grade);
CREATE INDEX IF NOT EXISTS idx_dim_date_yq ON dim_date(year, quarter);
CREATE INDEX IF NOT EXISTS idx_dim_borrower_metrics ON dim_borrower(fico_range_low, dti, annual_inc);
CREATE INDEX IF NOT EXISTS idx_cube_grade_status ON agg_loan_cube(grade, loan_status, year);
CREATE INDEX IF NOT EXISTS idx_cube_year_qtr ON agg_loan_cube(year, quarter, loan_status);
CREATE INDEX IF NOT EXISTS idx_cube_bands ON agg_loan_cube(fico_band, dti_band, income_band);
"""

POPULATE_SQLITE_CUBE_SQL = """
DELETE FROM agg_loan_cube;
INSERT INTO agg_loan_cube
SELECT
  d.year,
  d.quarter,
  COALESCE(l.grade, 'UNK') AS grade,
  COALESCE(g.risk_tier, 'Other') AS risk_tier,
  o.loan_status,
  o.default_flag,
  l.term_months,
  CASE
    WHEN b.fico_range_low < 660 THEN 'Sub-660'
    WHEN b.fico_range_low < 700 THEN '660-699'
    WHEN b.fico_range_low < 740 THEN '700-739'
    ELSE '740+'
  END AS fico_band,
  CASE
    WHEN b.dti < 10 THEN 'Low (<10)'
    WHEN b.dti < 20 THEN 'Moderate (10-19.9)'
    WHEN b.dti < 30 THEN 'Elevated (20-29.9)'
    ELSE 'High (30+)'
  END AS dti_band,
  CASE
    WHEN b.annual_inc < 50000 THEN '<50K'
    WHEN b.annual_inc < 100000 THEN '50K-99K'
    WHEN b.annual_inc < 150000 THEN '100K-149K'
    ELSE '150K+'
  END AS income_band,
  COUNT(*) AS loan_count,
  SUM(l.loan_amnt) AS sum_loan_amnt,
  SUM(l.int_rate) AS sum_int_rate,
  SUM(l.installment) AS sum_installment,
  SUM(b.annual_inc) AS sum_annual_inc,
  SUM(b.dti) AS sum_dti,
  SUM(b.revol_util) AS sum_revol_util,
  SUM(b.fico_range_low) AS sum_fico,
  SUM(f.total_rec_prncp) AS sum_rec_prncp
FROM fact_loan f
JOIN dim_loan l ON l.loan_key = f.loan_key
JOIN dim_date d ON d.date_key = f.date_key
JOIN dim_borrower b ON b.borrower_key = f.borrower_key
JOIN dim_outcome o ON o.outcome_key = f.outcome_key
LEFT JOIN dim_loan_grade g ON g.grade_key = f.grade_key
GROUP BY d.year, d.quarter, l.grade, g.risk_tier, o.loan_status, o.default_flag, l.term_months, fico_band, dti_band, income_band;
"""

POSTGRES_SCHEMA = """
CREATE SCHEMA IF NOT EXISTS dw;

CREATE TABLE IF NOT EXISTS dw.dim_date (
  date_key INTEGER PRIMARY KEY,
  issue_date DATE NOT NULL UNIQUE,
  year SMALLINT NOT NULL,
  month SMALLINT NOT NULL,
  quarter SMALLINT NOT NULL
);

CREATE TABLE IF NOT EXISTS dw.dim_loan_grade (
  grade_key SERIAL PRIMARY KEY,
  grade CHAR(1) NOT NULL UNIQUE,
  risk_tier VARCHAR(32) NOT NULL
);

CREATE TABLE IF NOT EXISTS dw.dim_loan (
  loan_key BIGSERIAL PRIMARY KEY,
  loan_id VARCHAR(32) NOT NULL UNIQUE,
  loan_amnt NUMERIC(12, 2) NOT NULL,
  int_rate NUMERIC(6, 3) NOT NULL,
  term_months SMALLINT NOT NULL,
  installment NUMERIC(10, 2) NOT NULL,
  grade CHAR(1)
);

CREATE TABLE IF NOT EXISTS dw.dim_borrower (
  borrower_key BIGSERIAL PRIMARY KEY,
  loan_id VARCHAR(32) NOT NULL UNIQUE,
  annual_inc NUMERIC(12, 2) NOT NULL,
  dti NUMERIC(6, 2) NOT NULL,
  revol_util NUMERIC(6, 2) NOT NULL,
  fico_range_low INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS dw.dim_outcome (
  outcome_key SERIAL PRIMARY KEY,
  loan_status VARCHAR(32) NOT NULL UNIQUE,
  default_flag SMALLINT NOT NULL
);

CREATE TABLE IF NOT EXISTS dw.fact_loan (
  loan_key BIGINT PRIMARY KEY REFERENCES dw.dim_loan(loan_key),
  date_key INTEGER NOT NULL REFERENCES dw.dim_date(date_key),
  borrower_key BIGINT NOT NULL REFERENCES dw.dim_borrower(borrower_key),
  outcome_key INTEGER NOT NULL REFERENCES dw.dim_outcome(outcome_key),
  grade_key INTEGER REFERENCES dw.dim_loan_grade(grade_key),
  total_rec_prncp NUMERIC(12, 2) NOT NULL DEFAULT 0
);

CREATE OR REPLACE VIEW dw.fact_loan_performance AS
SELECT
  f.loan_key, f.date_key, f.borrower_key, f.outcome_key, f.grade_key,
  l.loan_amnt, l.int_rate, l.term_months, l.installment, l.grade,
  o.loan_status, o.default_flag, f.total_rec_prncp
FROM dw.fact_loan f
JOIN dw.dim_loan l ON l.loan_key = f.loan_key
JOIN dw.dim_outcome o ON o.outcome_key = f.outcome_key;

CREATE INDEX IF NOT EXISTS idx_pg_fact_date_outcome
  ON dw.fact_loan (date_key, outcome_key) INCLUDE (total_rec_prncp);
CREATE INDEX IF NOT EXISTS idx_pg_fact_borrower ON dw.fact_loan (borrower_key);
CREATE INDEX IF NOT EXISTS idx_pg_fact_grade_outcome
  ON dw.fact_loan (grade_key, outcome_key) INCLUDE (total_rec_prncp);
CREATE INDEX IF NOT EXISTS idx_pg_dim_loan_grade
  ON dw.dim_loan (grade) INCLUDE (loan_amnt, int_rate, term_months);
CREATE INDEX IF NOT EXISTS idx_pg_dim_date_yq ON dw.dim_date (year, quarter);
"""


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(SQLITE_SCHEMA)
    date_cols = {row[1] for row in connection.execute("PRAGMA table_info(dim_date)").fetchall()}
    if "quarter" not in date_cols:
        connection.execute("ALTER TABLE dim_date ADD COLUMN quarter INTEGER NOT NULL DEFAULT 1")
    loan_cols = {row[1] for row in connection.execute("PRAGMA table_info(dim_loan)").fetchall()}
    if "grade" not in loan_cols:
        connection.execute("ALTER TABLE dim_loan ADD COLUMN grade TEXT")
    fact_cols = {row[1] for row in connection.execute("PRAGMA table_info(fact_loan)").fetchall()}
    if "total_rec_prncp" not in fact_cols:
        connection.execute("ALTER TABLE fact_loan ADD COLUMN total_rec_prncp REAL NOT NULL DEFAULT 0")
    if "grade_key" not in fact_cols:
        connection.execute("ALTER TABLE fact_loan ADD COLUMN grade_key INTEGER")


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "loan_id", "issue_date", "annual_inc", "dti", "revol_util", "fico_range_low",
        "loan_amnt", "int_rate", "term", "installment", "loan_status", "loan_status_label",
        "grade", "total_rec_prncp",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Prepared frame is missing columns: " + ", ".join(sorted(missing)))
    df = frame.copy()
    dates = pd.to_datetime(df["issue_date"])
    df["_date_key"] = dates.dt.strftime("%Y%m%d").astype(int)
    df["_issue_date_iso"] = dates.dt.strftime("%Y-%m-%d")
    df["_year"] = dates.dt.year.astype(int)
    df["_month"] = dates.dt.month.astype(int)
    df["_quarter"] = dates.dt.quarter.astype(int)
    df["grade"] = df["grade"].astype("string").str.strip().str.upper()
    return df


def _load_postgres(df: pd.DataFrame, dsn: str) -> dict:
    import psycopg2

    t0 = time.perf_counter()

    def _copy_df(cur, sub_df: pd.DataFrame, table: str, cols: list[str]) -> None:
        buf = io.StringIO()
        sub_df[cols].to_csv(buf, index=False, header=False)
        buf.seek(0)
        cur.copy_expert(
            f"COPY {table} ({', '.join(cols)}) FROM STDIN WITH (FORMAT CSV)", buf
        )

    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(POSTGRES_SCHEMA)
            cur.execute(
                "TRUNCATE dw.fact_loan, dw.dim_loan, dw.dim_borrower, dw.dim_date, "
                "dw.dim_outcome, dw.dim_loan_grade RESTART IDENTITY CASCADE;"
            )

            dates_df = (
                df[["_date_key", "_issue_date_iso", "_year", "_month", "_quarter"]]
                .drop_duplicates(subset=["_date_key"])
                .rename(columns={
                    "_date_key": "date_key",
                    "_issue_date_iso": "issue_date",
                    "_year": "year",
                    "_month": "month",
                    "_quarter": "quarter",
                })
            )
            _copy_df(cur, dates_df, "dw.dim_date", ["date_key", "issue_date", "year", "month", "quarter"])

            grades_df = pd.DataFrame([
                {"grade": g, "risk_tier": GRADE_RISK_TIERS.get(str(g), "Other")}
                for g in sorted(df["grade"].dropna().unique())
            ])
            _copy_df(cur, grades_df, "dw.dim_loan_grade", ["grade", "risk_tier"])

            outcomes_df = (
                df[["loan_status", "loan_status_label"]]
                .drop_duplicates(subset=["loan_status"])
                .rename(columns={"loan_status_label": "default_flag"})
            )
            _copy_df(cur, outcomes_df, "dw.dim_outcome", ["loan_status", "default_flag"])

            loans_df = df[["loan_id", "loan_amnt", "int_rate", "term", "installment", "grade"]].rename(
                columns={"term": "term_months"}
            )
            _copy_df(cur, loans_df, "dw.dim_loan", ["loan_id", "loan_amnt", "int_rate", "term_months", "installment", "grade"])

            borrowers_df = df[["loan_id", "annual_inc", "dti", "revol_util", "fico_range_low"]]
            _copy_df(cur, borrowers_df, "dw.dim_borrower", ["loan_id", "annual_inc", "dti", "revol_util", "fico_range_low"])

            cur.execute("SELECT loan_id, loan_key FROM dw.dim_loan")
            loan_map = dict(cur.fetchall())
            cur.execute("SELECT loan_id, borrower_key FROM dw.dim_borrower")
            borrower_map = dict(cur.fetchall())
            cur.execute("SELECT loan_status, outcome_key FROM dw.dim_outcome")
            outcome_map = dict(cur.fetchall())
            cur.execute("SELECT TRIM(grade), grade_key FROM dw.dim_loan_grade")
            grade_map = dict(cur.fetchall())

            fact_df = pd.DataFrame({
                "loan_key": df["loan_id"].map(loan_map),
                "date_key": df["_date_key"],
                "borrower_key": df["loan_id"].map(borrower_map),
                "outcome_key": df["loan_status"].map(outcome_map),
                "grade_key": df["grade"].map(grade_map),
                "total_rec_prncp": df["total_rec_prncp"],
            })
            _copy_df(
                cur,
                fact_df,
                "dw.fact_loan",
                ["loan_key", "date_key", "borrower_key", "outcome_key", "grade_key", "total_rec_prncp"],
            )
            cur.execute("ANALYZE dw.fact_loan, dw.dim_loan, dw.dim_borrower, dw.dim_date, dw.dim_outcome, dw.dim_loan_grade;")

            cur.execute("SELECT COUNT(*), COALESCE(SUM(total_rec_prncp), 0) FROM dw.fact_loan")
            fact_rows, fact_rec_sum = cur.fetchone()
            cur.execute("SELECT COALESCE(SUM(loan_amnt), 0) FROM dw.dim_loan")
            dim_loan_sum = cur.fetchone()[0]
        conn.commit()

    return {
        "backend": "postgresql",
        "source_rows": int(len(df)),
        "target_fact_rows": int(fact_rows),
        "row_count_match": int(len(df)) == int(fact_rows),
        "source_loan_amnt_sum": round(float(df["loan_amnt"].sum()), 2),
        "target_loan_amnt_sum": round(float(dim_loan_sum), 2),
        "source_rec_prncp_sum": round(float(df["total_rec_prncp"].sum()), 2),
        "target_rec_prncp_sum": round(float(fact_rec_sum), 2),
        "load_time_seconds": round(time.perf_counter() - t0, 3),
    }


def load_lending_data(frame: pd.DataFrame, database: str | Path) -> dict:
    df = _validate_frame(frame)
    target = str(database)
    if target.startswith(("postgresql://", "postgres://")):
        return _load_postgres(df, target)

    t0 = time.perf_counter()
    Path(database).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA foreign_keys=ON;")
        connection.execute("PRAGMA temp_store=MEMORY;")
        initialize_database(connection)

        dates = (
            df[["_date_key", "_issue_date_iso", "_year", "_month", "_quarter"]]
            .drop_duplicates(subset=["_date_key"])
            .itertuples(index=False, name=None)
        )
        connection.executemany(
            "INSERT OR IGNORE INTO dim_date (date_key, issue_date, year, month, quarter) VALUES (?, ?, ?, ?, ?)",
            list(dates),
        )

        grades = [
            (str(g), GRADE_RISK_TIERS.get(str(g), "Other"))
            for g in sorted(df["grade"].dropna().unique())
        ]
        connection.executemany(
            "INSERT OR IGNORE INTO dim_loan_grade (grade, risk_tier) VALUES (?, ?)",
            grades,
        )

        outcomes = (
            df[["loan_status", "loan_status_label"]]
            .drop_duplicates(subset=["loan_status"])
            .itertuples(index=False, name=None)
        )
        connection.executemany(
            "INSERT OR IGNORE INTO dim_outcome (loan_status, default_flag) VALUES (?, ?)",
            [(s, int(lbl)) for s, lbl in outcomes],
        )

        loans = df[["loan_id", "loan_amnt", "int_rate", "term", "installment", "grade"]].itertuples(index=False, name=None)
        connection.executemany(
            "INSERT OR IGNORE INTO dim_loan (loan_id, loan_amnt, int_rate, term_months, installment, grade) VALUES (?, ?, ?, ?, ?, ?)",
            list(loans),
        )

        borrowers = df[["loan_id", "annual_inc", "dti", "revol_util", "fico_range_low"]].itertuples(index=False, name=None)
        connection.executemany(
            "INSERT OR IGNORE INTO dim_borrower (loan_id, annual_inc, dti, revol_util, fico_range_low) VALUES (?, ?, ?, ?, ?)",
            list(borrowers),
        )

        loan_map = dict(connection.execute("SELECT loan_id, loan_key FROM dim_loan").fetchall())
        borrower_map = dict(connection.execute("SELECT loan_id, borrower_key FROM dim_borrower").fetchall())
        outcome_map = dict(connection.execute("SELECT loan_status, outcome_key FROM dim_outcome").fetchall())
        grade_map = dict(connection.execute("SELECT grade, grade_key FROM dim_loan_grade").fetchall())

        facts = [
            (
                loan_map[loan_id],
                int(date_key),
                borrower_map[loan_id],
                outcome_map[loan_status],
                grade_map.get(str(grade)),
                float(total_rec_prncp),
            )
            for loan_id, date_key, loan_status, grade, total_rec_prncp in zip(
                df["loan_id"], df["_date_key"], df["loan_status"], df["grade"], df["total_rec_prncp"]
            )
        ]
        connection.executemany(
            "INSERT OR IGNORE INTO fact_loan (loan_key, date_key, borrower_key, outcome_key, grade_key, total_rec_prncp) VALUES (?, ?, ?, ?, ?, ?)",
            facts,
        )
        # Materialize the OLAP aggregate cube (Kimball Aggregate Navigator pattern)
        connection.executescript(POPULATE_SQLITE_CUBE_SQL)
        connection.execute("ANALYZE;")

        fact_rows, fact_rec_sum = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(total_rec_prncp), 0) FROM fact_loan"
        ).fetchone()
        cube_rows = connection.execute("SELECT COUNT(*) FROM agg_loan_cube").fetchone()[0]
        dim_loan_sum = connection.execute(
            "SELECT COALESCE(SUM(loan_amnt), 0) FROM dim_loan"
        ).fetchone()[0]
        fk_violations = len(connection.execute("PRAGMA foreign_key_check;").fetchall())
        connection.commit()

    return {
        "backend": "sqlite",
        "source_rows": int(len(df)),
        "target_fact_rows": int(fact_rows),
        "olap_cube_cells": int(cube_rows),
        "row_count_match": int(len(df)) == int(fact_rows),
        "foreign_key_violations": int(fk_violations),
        "source_loan_amnt_sum": round(float(df["loan_amnt"].sum()), 2),
        "target_loan_amnt_sum": round(float(dim_loan_sum), 2),
        "source_rec_prncp_sum": round(float(df["total_rec_prncp"].sum()), 2),
        "target_rec_prncp_sum": round(float(fact_rec_sum), 2),
        "load_time_seconds": round(time.perf_counter() - t0, 3),
    }
