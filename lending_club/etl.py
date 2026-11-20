from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_date (
  date_key INTEGER PRIMARY KEY, issue_date TEXT NOT NULL UNIQUE,
  year INTEGER NOT NULL, month INTEGER NOT NULL
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
  outcome_key INTEGER NOT NULL, total_rec_prncp REAL NOT NULL DEFAULT 0, PRIMARY KEY (loan_key),
  FOREIGN KEY (loan_key) REFERENCES dim_loan(loan_key),
  FOREIGN KEY (date_key) REFERENCES dim_date(date_key),
  FOREIGN KEY (borrower_key) REFERENCES dim_borrower(borrower_key),
  FOREIGN KEY (outcome_key) REFERENCES dim_outcome(outcome_key)
);
"""


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(dim_loan)").fetchall()}
    if "grade" not in columns:
        connection.execute("ALTER TABLE dim_loan ADD COLUMN grade TEXT")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(fact_loan)").fetchall()}
    if "total_rec_prncp" not in columns:
        connection.execute("ALTER TABLE fact_loan ADD COLUMN total_rec_prncp REAL NOT NULL DEFAULT 0")


def load_lending_data(frame: pd.DataFrame, database: str | Path) -> None:
    required = {
        "loan_id", "issue_date", "annual_inc", "dti", "revol_util", "fico_range_low",
        "loan_amnt", "int_rate", "term", "installment", "loan_status", "loan_status_label",
        "grade", "total_rec_prncp",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Prepared frame is missing columns: " + ", ".join(sorted(missing)))
    with sqlite3.connect(database) as connection:
        initialize_database(connection)
        for row in frame.itertuples(index=False):
            date = pd.Timestamp(row.issue_date)
            connection.execute(
                "INSERT OR IGNORE INTO dim_date VALUES (?, ?, ?, ?)",
                (int(date.strftime("%Y%m%d")), date.date().isoformat(), date.year, date.month),
            )
            connection.execute(
                "INSERT OR IGNORE INTO dim_outcome VALUES (NULL, ?, ?)",
                (row.loan_status, int(row.loan_status_label)),
            )
            connection.execute(
                "INSERT OR IGNORE INTO dim_loan(loan_id,loan_amnt,int_rate,term_months,installment,grade) "
                "VALUES (?,?,?,?,?,?)",
                (row.loan_id, row.loan_amnt, row.int_rate, row.term, row.installment, row.grade),
            )
            connection.execute(
                "INSERT OR IGNORE INTO dim_borrower(loan_id,annual_inc,dti,revol_util,fico_range_low) "
                "VALUES (?,?,?,?,?)",
                (row.loan_id, row.annual_inc, row.dti, row.revol_util, row.fico_range_low),
            )
        for row in frame.itertuples(index=False):
            loan_key = connection.execute(
                "SELECT loan_key FROM dim_loan WHERE loan_id=?", (row.loan_id,)
            ).fetchone()[0]
            borrower_key = connection.execute(
                "SELECT borrower_key FROM dim_borrower WHERE loan_id=?", (row.loan_id,)
            ).fetchone()[0]
            date_key = int(pd.Timestamp(row.issue_date).strftime("%Y%m%d"))
            outcome_key = connection.execute(
                "SELECT outcome_key FROM dim_outcome WHERE loan_status=?", (row.loan_status,)
            ).fetchone()[0]
            connection.execute(
                "INSERT OR IGNORE INTO fact_loan "
                "(loan_key,date_key,borrower_key,outcome_key,total_rec_prncp) "
                "VALUES (?,?,?,?,?)",
                (loan_key, date_key, borrower_key, outcome_key, row.total_rec_prncp),
            )
        connection.commit()

