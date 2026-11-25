from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any

import httpx

SUPPORTED = (
    "What is the average loan amount by grade?",
    "What is the average interest rate by grade?",
    "What is the average interest rate by loan status?",
    "How many loans were issued by year?",
    "What is the default rate by year?",
    "What is the average annual income by grade and loan status?",
    "What is the average DTI by loan status?",
    "What is the average FICO score by loan status?",
    "What is the total principal received by loan status?",
    "What is the total principal received by grade?",
    "What is the total principal received by year?",
)

GREETING = re.compile(r"(hi|hello|hey|thanks|thank you)[!. ]*", re.I)
FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|attach|detach|pragma|grant|revoke|copy)\b",
    re.I,
)
ALLOWED_TABLES = {
    "fact_loan", "fact_loan_performance", "agg_loan_cube", "dim_loan", "dim_loan_grade",
    "dim_borrower", "dim_date", "dim_outcome",
    "dw.fact_loan", "dw.fact_loan_performance", "dw.agg_loan_cube", "dw.dim_loan", "dw.dim_loan_grade",
    "dw.dim_borrower", "dw.dim_date", "dw.dim_outcome",
}

SCHEMA_DOCS = [
    {
        "table": "fact_loan",
        "ddl": "CREATE TABLE fact_loan (loan_key INTEGER PRIMARY KEY, date_key INTEGER, borrower_key INTEGER, outcome_key INTEGER, grade_key INTEGER, total_rec_prncp REAL);",
        "keywords": {"principal", "recovered", "recovery", "total", "loan", "fact", "amount", "received", "lgd", "loss", "exposure"},
    },
    {
        "table": "dim_loan",
        "ddl": "CREATE TABLE dim_loan (loan_key INTEGER PRIMARY KEY, loan_id TEXT, loan_amnt REAL, int_rate REAL, term_months REAL, installment REAL, grade TEXT);",
        "keywords": {"grade", "loan", "amount", "interest", "rate", "term", "installment", "exposure", "pricing", "36", "60", "month"},
    },
    {
        "table": "dim_loan_grade",
        "ddl": "CREATE TABLE dim_loan_grade (grade_key INTEGER PRIMARY KEY, grade TEXT, risk_tier TEXT);",
        "keywords": {"grade", "tier", "risk", "prime", "subprime"},
    },
    {
        "table": "dim_borrower",
        "ddl": "CREATE TABLE dim_borrower (borrower_key INTEGER PRIMARY KEY, loan_id TEXT, annual_inc REAL, dti REAL, revol_util REAL, fico_range_low REAL);",
        "keywords": {"borrower", "income", "annual", "dti", "debt", "fico", "credit", "score", "utilization", "revol", "band", "bracket"},
    },
    {
        "table": "dim_date",
        "ddl": "CREATE TABLE dim_date (date_key INTEGER PRIMARY KEY, issue_date TEXT, year INTEGER, month INTEGER, quarter INTEGER);",
        "keywords": {"year", "month", "quarter", "quarterly", "date", "trend", "cohort", "issued", "time", "2015", "2016", "2017", "2018"},
    },
    {
        "table": "dim_outcome",
        "ddl": "CREATE TABLE dim_outcome (outcome_key INTEGER PRIMARY KEY, loan_status TEXT, default_flag INTEGER); -- loan_status in ('Fully Paid', 'Charged Off'); default_flag=1 for Charged Off",
        "keywords": {"status", "default", "charge", "charged", "paid", "outcome", "rate", "loss", "lgd"},
    },
]

TEMPLATES = {
    "avg_loan_grade": (
        "SELECT l.grade, AVG(l.loan_amnt) AS average_loan_amount "
        "FROM fact_loan f JOIN dim_loan l ON l.loan_key=f.loan_key "
        "WHERE l.grade IS NOT NULL GROUP BY l.grade ORDER BY l.grade",
        "Average loan amount by credit grade.",
    ),
    "avg_rate_grade": (
        "SELECT l.grade, AVG(l.int_rate) AS average_interest_rate "
        "FROM fact_loan f JOIN dim_loan l ON l.loan_key=f.loan_key "
        "WHERE l.grade IS NOT NULL GROUP BY l.grade ORDER BY l.grade",
        "Average interest rate by credit grade.",
    ),
    "avg_rate_status": (
        "SELECT o.loan_status, AVG(l.int_rate) AS average_interest_rate "
        "FROM fact_loan f JOIN dim_loan l ON l.loan_key=f.loan_key "
        "JOIN dim_outcome o ON o.outcome_key=f.outcome_key "
        "GROUP BY o.loan_status ORDER BY o.loan_status",
        "Average interest rate by loan status.",
    ),
    "avg_rate_grade_status": (
        "SELECT l.grade, o.loan_status, AVG(l.int_rate) AS average_interest_rate "
        "FROM fact_loan f JOIN dim_loan l ON l.loan_key=f.loan_key "
        "JOIN dim_outcome o ON o.outcome_key=f.outcome_key "
        "WHERE l.grade IS NOT NULL "
        "GROUP BY l.grade, o.loan_status ORDER BY l.grade, o.loan_status",
        "Average interest rate by credit grade and loan status.",
    ),
    "count_year": (
        "SELECT d.year, COUNT(*) AS loan_count "
        "FROM fact_loan f JOIN dim_date d ON d.date_key=f.date_key "
        "GROUP BY d.year ORDER BY d.year",
        "Loan count by issue year.",
    ),
    "default_year": (
        "SELECT d.year, ROUND(100.0 * AVG(o.default_flag), 2) AS default_rate_percent "
        "FROM fact_loan f JOIN dim_date d ON d.date_key=f.date_key "
        "JOIN dim_outcome o ON o.outcome_key=f.outcome_key "
        "GROUP BY d.year ORDER BY d.year",
        "Charged-off rate by issue year.",
    ),
    "avg_income_grade_status": (
        "SELECT l.grade, o.loan_status, AVG(b.annual_inc) AS average_annual_income "
        "FROM fact_loan f JOIN dim_loan l ON l.loan_key=f.loan_key "
        "JOIN dim_borrower b ON b.borrower_key=f.borrower_key "
        "JOIN dim_outcome o ON o.outcome_key=f.outcome_key "
        "WHERE l.grade IS NOT NULL "
        "GROUP BY l.grade, o.loan_status ORDER BY l.grade, o.loan_status",
        "Average annual income by grade and loan status.",
    ),
    "avg_income_status": (
        "SELECT o.loan_status, AVG(b.annual_inc) AS average_annual_income "
        "FROM fact_loan f JOIN dim_borrower b ON b.borrower_key=f.borrower_key "
        "JOIN dim_outcome o ON o.outcome_key=f.outcome_key "
        "GROUP BY o.loan_status ORDER BY o.loan_status",
        "Average annual income by loan status.",
    ),
    "avg_dti_status": (
        "SELECT o.loan_status, AVG(b.dti) AS average_dti "
        "FROM fact_loan f JOIN dim_borrower b ON b.borrower_key=f.borrower_key "
        "JOIN dim_outcome o ON o.outcome_key=f.outcome_key "
        "GROUP BY o.loan_status ORDER BY o.loan_status",
        "Average debt-to-income ratio by loan status.",
    ),
    "avg_fico_status": (
        "SELECT o.loan_status, AVG(b.fico_range_low) AS average_fico "
        "FROM fact_loan f JOIN dim_borrower b ON b.borrower_key=f.borrower_key "
        "JOIN dim_outcome o ON o.outcome_key=f.outcome_key "
        "GROUP BY o.loan_status ORDER BY o.loan_status",
        "Average lower-bound FICO score by loan status.",
    ),
    "principal_status": (
        "SELECT o.loan_status, SUM(f.total_rec_prncp) AS total_principal_received "
        "FROM fact_loan f JOIN dim_outcome o ON o.outcome_key=f.outcome_key "
        "GROUP BY o.loan_status ORDER BY o.loan_status",
        "Total principal received by loan status.",
    ),
    "principal_grade": (
        "SELECT l.grade, SUM(f.total_rec_prncp) AS total_principal_received "
        "FROM fact_loan f JOIN dim_loan l ON l.loan_key=f.loan_key "
        "WHERE l.grade IS NOT NULL GROUP BY l.grade ORDER BY l.grade",
        "Total principal received by credit grade.",
    ),
    "principal_year": (
        "SELECT d.year, SUM(f.total_rec_prncp) AS total_principal_received "
        "FROM fact_loan f JOIN dim_date d ON d.date_key=f.date_key "
        "GROUP BY d.year ORDER BY d.year",
        "Total principal received by issue year.",
    ),
}

# Exact-match map for the 13 documented sample questions (preserves backward compatibility)
_EXACT_SAMPLE_MAP = {
    "what is the average loan amount by grade?": "avg_loan_grade",
    "what is the average interest rate by grade?": "avg_rate_grade",
    "what is the average interest rate by grade and loan status?": "avg_rate_grade_status",
    "what is the average interest rate by loan status?": "avg_rate_status",
    "how many loans were issued by year?": "count_year",
    "what is the default rate by year?": "default_year",
    "what is the average annual income by grade and loan status?": "avg_income_grade_status",
    "what is the average annual income by loan status?": "avg_income_status",
    "what is the average dti by loan status?": "avg_dti_status",
    "what is the average fico score by loan status?": "avg_fico_status",
    "what is the total principal received by loan status?": "principal_status",
    "what is the total principal received by grade?": "principal_grade",
    "what is the total principal received by year?": "principal_year",
}

# Bounded LRU + TTL query cache
_CACHE_MAX_SIZE = 512
_CACHE_TTL_SECONDS = 3600.0
_CACHE_LOCK = threading.Lock()
_QUERY_CACHE: OrderedDict[tuple[str, str], tuple[float, dict]] = OrderedDict()
_CACHE_STATS = {"hits": 0, "misses": 0}

# Thread-local read-only SQLite connection pool (keyed by resolved path + mtime)
_SQLITE_LOCAL = threading.local()
_HTTP_CLIENT = httpx.Client(timeout=15.0)


def clear_cache() -> None:
    with _CACHE_LOCK:
        _QUERY_CACHE.clear()
        _CACHE_STATS["hits"] = 0
        _CACHE_STATS["misses"] = 0


def get_cache_stats() -> dict[str, Any]:
    with _CACHE_LOCK:
        total = _CACHE_STATS["hits"] + _CACHE_STATS["misses"]
        return {
            "entries": len(_QUERY_CACHE),
            "max_size": _CACHE_MAX_SIZE,
            "ttl_seconds": _CACHE_TTL_SECONDS,
            "hits": _CACHE_STATS["hits"],
            "misses": _CACHE_STATS["misses"],
            "hit_rate_pct": round(100.0 * _CACHE_STATS["hits"] / total, 2) if total else 0.0,
        }


def _get_sqlite_connection(database: str) -> sqlite3.Connection:
    resolved = str(Path(database).resolve())
    try:
        mtime = Path(resolved).stat().st_mtime
    except OSError:
        mtime = 0.0

    pool: dict[str, tuple[float, sqlite3.Connection]] = getattr(_SQLITE_LOCAL, "pool", None)
    if pool is None:
        pool = {}
        _SQLITE_LOCAL.pool = pool

    cached = pool.get(resolved)
    if cached is not None:
        cached_mtime, conn = cached
        if cached_mtime == mtime:
            return conn
        try:
            conn.close()
        except Exception:
            pass

    conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA mmap_size=268435456;")
    conn.execute("PRAGMA cache_size=-64000;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    pool[resolved] = (mtime, conn)
    return conn


def retrieve_warehouse_context(question: str, top_k_examples: int = 3) -> str:
    tokens = set(re.findall(r"[a-z0-9_]+", question.lower()))
    selected_ddls = [SCHEMA_DOCS[0]["ddl"]]  # Always include central fact_loan
    for doc in SCHEMA_DOCS[1:]:
        if tokens & doc["keywords"]:
            selected_ddls.append(doc["ddl"])
    if len(selected_ddls) == 1:
        selected_ddls = [d["ddl"] for d in SCHEMA_DOCS]

    scored_examples: list[tuple[int, str, str]] = []
    for sql, desc in TEMPLATES.values():
        desc_tokens = set(re.findall(r"[a-z0-9_]+", desc.lower()))
        scored_examples.append((len(tokens & desc_tokens), desc, sql))
    scored_examples.sort(key=lambda item: item[0], reverse=True)
    examples_block = "\n".join(
        f"-- Q: {desc}\n{sql};" for _, desc, sql in scored_examples[:top_k_examples]
    )
    return "DDL:\n" + "\n".join(selected_ddls) + "\n\nApproved Examples:\n" + examples_block


def validate_readonly_sql(sql: str, max_rows: int = 200, enforce_limit: bool = True) -> str:
    cleaned = re.sub(r"^```(?:sql)?|```$", "", sql.strip(), flags=re.I | re.M).strip().rstrip(";")
    if not cleaned or not re.match(r"^(select|with)\b", cleaned, re.I):
        raise ValueError("Only read-only SELECT or WITH queries are permitted.")
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not permitted.")
    if FORBIDDEN_SQL.search(cleaned):
        raise ValueError("Forbidden write or administrative SQL keyword detected.")

    referenced = {
        m.group(1).lower()
        for m in re.finditer(r"\b(?:from|join)\s+([a-zA-Z0-9_.]+)", cleaned, re.I)
    }
    if not referenced or not referenced.issubset(ALLOWED_TABLES):
        raise ValueError(f"Unauthorized table reference: {sorted(referenced - ALLOWED_TABLES)}")

    if enforce_limit and not re.search(r"\blimit\s+\d+\b", cleaned, re.I):
        cleaned = f"{cleaned} LIMIT {max_rows}"
    return cleaned


def _call_nrp_llm(
    question: str,
    context: str,
    use_postgres_schema: bool = False,
    error_feedback: str | None = None,
) -> str | None:
    base_url = os.getenv("NRP_LLM_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return None
    model = os.getenv("NRP_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    api_key = os.getenv("NRP_LLM_API_KEY", "EMPTY")
    schema_hint = (
        "Prefix tables with dw. (e.g., dw.fact_loan)."
        if use_postgres_schema
        else "Use unqualified table names (fact_loan, dim_loan, dim_loan_grade, dim_borrower, dim_date, dim_outcome)."
    )
    system_prompt = (
        "You are an agentic financial data warehouse SQL generator. Return ONLY a single read-only SELECT SQL query "
        f"with no markdown or commentary. {schema_hint}\n\n{context}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    if error_feedback:
        messages.append(
            {
                "role": "user",
                "content": f"Your previous SQL failed validation or execution with error: {error_feedback}. "
                           "Fix the SQL query and return ONLY the corrected SELECT statement.",
            }
        )
    try:
        resp = _HTTP_CLIENT.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "temperature": 0.0,
                "max_tokens": 260,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


# --- Schema-Grounded Compositional NL-to-SQL Planner ---

_FICO_BAND_EXPR = (
    "CASE "
    "WHEN b.fico_range_low < 660 THEN 'Sub-660' "
    "WHEN b.fico_range_low < 700 THEN '660-699' "
    "WHEN b.fico_range_low < 740 THEN '700-739' "
    "ELSE '740+' END"
)
_DTI_BAND_EXPR = (
    "CASE "
    "WHEN b.dti < 10 THEN 'Low (<10)' "
    "WHEN b.dti < 20 THEN 'Moderate (10-19.9)' "
    "WHEN b.dti < 30 THEN 'Elevated (20-29.9)' "
    "ELSE 'High (30+)' END"
)
_INCOME_BAND_EXPR = (
    "CASE "
    "WHEN b.annual_inc < 50000 THEN '<50K' "
    "WHEN b.annual_inc < 100000 THEN '50K-99K' "
    "WHEN b.annual_inc < 150000 THEN '100K-149K' "
    "ELSE '150K+' END"
)


def _compose_analytical_sql(question: str) -> tuple[str | None, str]:
    text = " ".join(question.strip().lower().split())
    if not text or GREETING.fullmatch(text):
        return None, "Please ask a Lending Club question about loan amounts, borrower metrics, rates, recovery, or outcomes."

    if text in _EXACT_SAMPLE_MAP:
        return TEMPLATES[_EXACT_SAMPLE_MAP[text]]

    joins: set[str] = set()
    wheres: list[str] = []
    group_exprs: list[tuple[str, str, str]] = []  # (select_expr, group_by_expr, label)

    # 1. Detect filters
    year_matches = re.findall(r"\b(201[5-8])\b", text)
    if year_matches:
        joins.add("date")
        unique_years = sorted(set(year_matches))
        if len(unique_years) == 1:
            wheres.append(f"d.year = {unique_years[0]}")
        else:
            wheres.append(f"d.year IN ({', '.join(unique_years)})")

    quarter_match = re.search(r"\bq([1-4])\b|\bquarter\s+([1-4])\b", text)
    if quarter_match:
        q_num = quarter_match.group(1) or quarter_match.group(2)
        joins.add("date")
        wheres.append(f"d.quarter = {q_num}")

    grade_filter = re.search(r"\bgrade\s+([a-g])\b", text)
    if grade_filter:
        joins.add("loan")
        wheres.append(f"l.grade = '{grade_filter.group(1).upper()}'")

    term_filter = re.search(r"\b(36|60)[\s-]*month", text)
    if term_filter and not ("36" in text and "60" in text):
        joins.add("loan")
        wheres.append(f"l.term_months = {term_filter.group(1)}")

    # Status filter (only if user is filtering to one status rather than grouping by status or asking for default rate)
    has_both_statuses = ("fully paid" in text and ("charged off" in text or "charged-off" in text))
    asks_default_rate = any(k in text for k in ("default rate", "charge-off rate", "charge off rate", "charged-off rate", "charged off rate"))
    if not has_both_statuses and not asks_default_rate and "by loan status" not in text and "by status" not in text:
        if any(k in text for k in ("charged off", "charged-off", "defaulted loans", "defaulted", "defaults")):
            joins.add("outcome")
            wheres.append("o.loan_status = 'Charged Off'")
        elif "fully paid" in text:
            joins.add("outcome")
            wheres.append("o.loan_status = 'Fully Paid'")

    fico_thresh = re.search(r"fico(?:\s+score)?\s*(?:>=|at least|above|over)\s*(\d{3})", text)
    if fico_thresh:
        joins.add("borrower")
        wheres.append(f"b.fico_range_low >= {fico_thresh.group(1)}")
    fico_below = re.search(r"fico(?:\s+score)?\s*(?:<|below|under)\s*(\d{3})", text)
    if fico_below:
        joins.add("borrower")
        wheres.append(f"b.fico_range_low < {fico_below.group(1)}")

    dti_thresh = re.search(r"dti\s*(?:>|above|over|>=)\s*(\d+(?:\.\d+)?)", text)
    if dti_thresh:
        joins.add("borrower")
        wheres.append(f"b.dti > {dti_thresh.group(1)}")

    # 2. Detect grouping dimensions
    if any(k in text for k in ("fico band", "fico bucket", "fico tier", "fico range", "by fico")):
        joins.add("borrower")
        group_exprs.append((f"{_FICO_BAND_EXPR} AS fico_band", "fico_band", "FICO band"))
    if any(k in text for k in ("dti band", "dti bucket", "dti tier", "by dti")):
        joins.add("borrower")
        group_exprs.append((f"{_DTI_BAND_EXPR} AS dti_band", "dti_band", "DTI band"))
    if any(k in text for k in ("income band", "income bracket", "income tier", "by income")):
        joins.add("borrower")
        group_exprs.append((f"{_INCOME_BAND_EXPR} AS income_band", "income_band", "income bracket"))
    if any(k in text for k in ("risk tier", "by tier")):
        joins.add("grade")
        group_exprs.append(("g.risk_tier", "g.risk_tier", "risk tier"))
    if ("by grade" in text or "by loan grade" in text or "by credit grade" in text
            or "each grade" in text or "per grade" in text or "which grade" in text
            or "which loan grade" in text or "grade and" in text or "and grade" in text
            or "grades" in text):
        joins.add("loan")
        wheres.append("l.grade IS NOT NULL")
        group_exprs.append(("l.grade", "l.grade", "credit grade"))
    if any(k in text for k in ("quarterly", "by quarter", "per quarter", "each quarter", "year and quarter")):
        joins.add("date")
        group_exprs.append(("d.year", "d.year", "issue year"))
        group_exprs.append(("d.quarter", "d.quarter", "quarter"))
    elif any(k in text for k in ("by year", "by issue year", "per year", "each year", "annual trend", "yearly", "which year", "and issue year", "and year")):
        joins.add("date")
        group_exprs.append(("d.year", "d.year", "issue year"))
    if any(k in text for k in ("by term", "per term", "each term", "36-month vs 60-month", "36 vs 60", "loan term")):
        joins.add("loan")
        group_exprs.append(("l.term_months", "l.term_months", "loan term"))
    if any(k in text for k in ("by loan status", "by status", "and loan status", "and status", "fully paid and charged-off", "fully paid vs charged off")):
        joins.add("outcome")
        group_exprs.append(("o.loan_status", "o.loan_status", "loan status"))

    # 3. Detect metric(s)
    metric_sql: str | None = None
    metric_alias: str | None = None
    metric_desc = ""

    if any(k in text for k in ("loss given default", "lgd")):
        joins.add("loan")
        metric_sql = "ROUND(100.0 * (1.0 - SUM(f.total_rec_prncp) / NULLIF(SUM(l.loan_amnt), 0)), 2) AS lgd_percent"
        metric_alias = "lgd_percent"
        metric_desc = "Loss Given Default (LGD %)"
    elif any(k in text for k in ("recovery rate", "principal recovery rate", "recovered percentage", "percent recovered")):
        joins.add("loan")
        metric_sql = "ROUND(100.0 * SUM(f.total_rec_prncp) / NULLIF(SUM(l.loan_amnt), 0), 2) AS recovery_rate_percent"
        metric_alias = "recovery_rate_percent"
        metric_desc = "Principal recovery rate (%)"
    elif any(k in text for k in ("unrecovered", "capital loss", "principal loss", "capital exposure")):
        joins.add("loan")
        metric_sql = "ROUND(SUM(l.loan_amnt - f.total_rec_prncp), 2) AS unrecovered_principal"
        metric_alias = "unrecovered_principal"
        metric_desc = "Unrecovered principal exposure"
    elif asks_default_rate:
        joins.add("outcome")
        metric_sql = "ROUND(100.0 * AVG(o.default_flag), 2) AS default_rate_percent"
        metric_alias = "default_rate_percent"
        metric_desc = "Charged-off default rate (%)"
    elif any(k in text for k in ("principal received", "principal recovered", "principal was recovered", "recovered principal", "total principal")):
        if "average" in text or "avg" in text or "mean" in text:
            metric_sql = "ROUND(AVG(f.total_rec_prncp), 2) AS average_principal_received"
            metric_alias = "average_principal_received"
            metric_desc = "Average principal received"
        else:
            metric_sql = "ROUND(SUM(f.total_rec_prncp), 2) AS total_principal_received"
            metric_alias = "total_principal_received"
            metric_desc = "Total principal received"
    elif any(k in text for k in ("interest rate", "int_rate")):
        joins.add("loan")
        metric_sql = "ROUND(AVG(l.int_rate), 2) AS average_interest_rate"
        metric_alias = "average_interest_rate"
        metric_desc = "Average interest rate"
    elif any(k in text for k in ("loan amount", "funded", "origination volume", "exposure")):
        joins.add("loan")
        if any(k in text for k in ("total", "sum", "volume")):
            metric_sql = "ROUND(SUM(l.loan_amnt), 2) AS total_loan_amount"
            metric_alias = "total_loan_amount"
            metric_desc = "Total loan amount"
        else:
            metric_sql = "ROUND(AVG(l.loan_amnt), 2) AS average_loan_amount"
            metric_alias = "average_loan_amount"
            metric_desc = "Average loan amount"
    elif "installment" in text:
        joins.add("loan")
        metric_sql = "ROUND(AVG(l.installment), 2) AS average_installment"
        metric_alias = "average_installment"
        metric_desc = "Average monthly installment"
    elif any(k in text for k in ("annual income", "income")):
        joins.add("borrower")
        metric_sql = "ROUND(AVG(b.annual_inc), 2) AS average_annual_income"
        metric_alias = "average_annual_income"
        metric_desc = "Average annual income"
    elif any(k in text for k in ("dti", "debt-to-income", "debt to income")):
        joins.add("borrower")
        metric_sql = "ROUND(AVG(b.dti), 2) AS average_dti"
        metric_alias = "average_dti"
        metric_desc = "Average debt-to-income ratio"
    elif any(k in text for k in ("revol", "revolving", "utilization")):
        joins.add("borrower")
        metric_sql = "ROUND(AVG(b.revol_util), 2) AS average_revol_util"
        metric_alias = "average_revol_util"
        metric_desc = "Average revolving utilization"
    elif any(k in text for k in ("fico", "credit score")):
        joins.add("borrower")
        metric_sql = "ROUND(AVG(b.fico_range_low), 2) AS average_fico"
        metric_alias = "average_fico"
        metric_desc = "Average lower-bound FICO score"
    elif any(k in text for k in ("how many", "count", "number of loans", "volume of loans", "trend of", "distribution")):
        metric_sql = "COUNT(*) AS loan_count"
        metric_alias = "loan_count"
        metric_desc = "Loan count"

    if metric_sql is None:
        return None, "Unsupported question. Ask about loan amounts, interest rates, borrower income/DTI/FICO/utilization, default rates, principal recovery, or LGD across grades, years, quarters, terms, or statuses."

    # 4. Assemble SQL
    select_parts = [expr for expr, _, _ in group_exprs] + [metric_sql]
    if metric_alias != "loan_count" and group_exprs and any(k in text for k in ("and count", "with count", "summary", "profile")):
        select_parts.append("COUNT(*) AS loan_count")

    join_clauses = ["FROM fact_loan f"]
    if "loan" in joins:
        join_clauses.append("JOIN dim_loan l ON l.loan_key = f.loan_key")
    if "grade" in joins:
        join_clauses.append("JOIN dim_loan_grade g ON g.grade_key = f.grade_key")
    if "borrower" in joins:
        join_clauses.append("JOIN dim_borrower b ON b.borrower_key = f.borrower_key")
    if "date" in joins:
        join_clauses.append("JOIN dim_date d ON d.date_key = f.date_key")
    if "outcome" in joins:
        join_clauses.append("JOIN dim_outcome o ON o.outcome_key = f.outcome_key")

    sql_parts = [f"SELECT {', '.join(select_parts)}", " ".join(join_clauses)]
    if wheres:
        # Deduplicate while preserving order
        seen_w: list[str] = []
        for w in wheres:
            if w not in seen_w:
                seen_w.append(w)
        sql_parts.append("WHERE " + " AND ".join(seen_w))
    if group_exprs:
        group_cols = [gb for _, gb, _ in group_exprs]
        sql_parts.append("GROUP BY " + ", ".join(group_cols))

        # Ordering: highest/top -> DESC by metric; lowest/bottom -> ASC by metric; else by group cols
        if any(k in text for k in ("highest", "top ", "most", "largest", "worst", "maximum")):
            sql_parts.append(f"ORDER BY {metric_alias} DESC")
        elif any(k in text for k in ("lowest", "bottom ", "least", "smallest", "best", "minimum")):
            sql_parts.append(f"ORDER BY {metric_alias} ASC")
        else:
            sql_parts.append("ORDER BY " + ", ".join(group_cols))

        top_match = re.search(r"\btop\s+(\d+)\b", text)
        if top_match:
            sql_parts.append(f"LIMIT {int(top_match.group(1))}")

    sql = " ".join(sql_parts)
    dim_text = " by " + " and ".join(lbl for _, _, lbl in group_exprs) if group_exprs else ""
    explanation = f"{metric_desc}{dim_text}."
    return sql, explanation


def generate_sql(question: str) -> tuple[str | None, str]:
    return _compose_analytical_sql(question)


def _execute_sql(sql: str, database: str) -> tuple[list[str], list[dict[str, Any]]]:
    if database.startswith(("postgresql://", "postgres://")):
        import psycopg2
        import psycopg2.extras

        pg_sql = re.sub(
            r"\b(?<!dw\.)(fact_loan_performance|fact_loan|dim_loan_grade|dim_loan|dim_borrower|dim_date|dim_outcome)\b",
            r"dw.\1",
            sql,
            flags=re.I,
        )
        with psycopg2.connect(database) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SET LOCAL statement_timeout = '3000ms';")
                cur.execute(pg_sql)
                columns = [desc[0] for desc in (cur.description or [])]
                rows = [dict(r) for r in cur.fetchall()]
        return columns, rows

    connection = _get_sqlite_connection(database)
    cursor = connection.execute(sql)
    columns = [item[0] for item in cursor.description]
    rows = [dict(row) for row in cursor.fetchall()]
    return columns, rows


def _summarize_rows(explanation: str, columns: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"{explanation} (0 matching rows returned)."
    if len(rows) == 1:
        preview = ", ".join(f"{k}={v:,.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in rows[0].items())
        return f"{explanation} Result: {preview}."

    numeric_cols = [
        c for c in columns
        if c not in {"year", "quarter", "month", "term_months"}
        and all(isinstance(r.get(c), (int, float)) for r in rows if r.get(c) is not None)
    ]
    if numeric_cols:
        metric_col = numeric_cols[-1]
        label_cols = [c for c in columns if c != metric_col]
        valid_rows = [r for r in rows if r.get(metric_col) is not None]
        if valid_rows and label_cols:
            top_row = max(valid_rows, key=lambda r: r[metric_col])
            low_row = min(valid_rows, key=lambda r: r[metric_col])
            top_lbl = "/".join(str(top_row[c]) for c in label_cols)
            low_lbl = "/".join(str(low_row[c]) for c in label_cols)
            top_val = f"{top_row[metric_col]:,.2f}" if isinstance(top_row[metric_col], float) else f"{top_row[metric_col]:,}"
            low_val = f"{low_row[metric_col]:,.2f}" if isinstance(low_row[metric_col], float) else f"{low_row[metric_col]:,}"
            return (
                f"{explanation} Returned {len(rows)} rows — highest {metric_col}: "
                f"{top_lbl} ({top_val}); lowest: {low_lbl} ({low_val})."
            )
    return f"{explanation} Returned {len(rows)} aggregated rows across {', '.join(columns)}."


def _compose_cube_sql(question: str) -> str | None:
    """Kimball Aggregate Navigator: rewrite eligible OLAP queries onto materialized agg_loan_cube."""
    text = " ".join(question.strip().lower().split())
    if not text or GREETING.fullmatch(text):
        return None
    if re.search(r"(fico(?:\s+score)?\s*(?:>=|at least|above|over|<|below|under)\s*\d{3})|(dti\s*(?:>|above|over|>=)\s*\d)", text):
        return None

    wheres: list[str] = []
    group_cols: list[str] = []

    year_matches = re.findall(r"\b(201[5-8])\b", text)
    if year_matches:
        unique_years = sorted(set(year_matches))
        wheres.append(f"year = {unique_years[0]}" if len(unique_years) == 1 else f"year IN ({', '.join(unique_years)})")

    quarter_match = re.search(r"\bq([1-4])\b|\bquarter\s+([1-4])\b", text)
    if quarter_match:
        wheres.append(f"quarter = {quarter_match.group(1) or quarter_match.group(2)}")

    grade_filter = re.search(r"\bgrade\s+([a-g])\b", text)
    if grade_filter:
        wheres.append(f"grade = '{grade_filter.group(1).upper()}'")

    term_filter = re.search(r"\b(36|60)[\s-]*month", text)
    if term_filter and not ("36" in text and "60" in text):
        wheres.append(f"term_months = {term_filter.group(1)}")

    has_both_statuses = ("fully paid" in text and ("charged off" in text or "charged-off" in text))
    asks_default_rate = any(k in text for k in ("default rate", "charge-off rate", "charge off rate", "charged-off rate", "charged off rate"))
    if not has_both_statuses and not asks_default_rate and "by loan status" not in text and "by status" not in text:
        if any(k in text for k in ("charged off", "charged-off", "defaulted loans", "defaulted", "defaults")):
            wheres.append("loan_status = 'Charged Off'")
        elif "fully paid" in text:
            wheres.append("loan_status = 'Fully Paid'")

    if any(k in text for k in ("fico band", "fico bucket", "fico tier", "fico range", "by fico")):
        group_cols.append("fico_band")
    if any(k in text for k in ("dti band", "dti bucket", "dti tier", "by dti")):
        group_cols.append("dti_band")
    if any(k in text for k in ("income band", "income bracket", "income tier", "by income")):
        group_cols.append("income_band")
    if any(k in text for k in ("risk tier", "by tier")):
        group_cols.append("risk_tier")
    if ("by grade" in text or "by loan grade" in text or "by credit grade" in text
            or "each grade" in text or "per grade" in text or "which grade" in text
            or "which loan grade" in text or "grade and" in text or "and grade" in text
            or "grades" in text):
        wheres.append("grade != 'UNK'")
        group_cols.append("grade")
    if any(k in text for k in ("quarterly", "by quarter", "per quarter", "each quarter", "year and quarter")):
        group_cols.extend(["year", "quarter"])
    elif any(k in text for k in ("by year", "by issue year", "per year", "each year", "annual trend", "yearly", "which year", "and issue year", "and year")):
        group_cols.append("year")
    if any(k in text for k in ("by term", "per term", "each term", "36-month vs 60-month", "36 vs 60", "loan term")):
        group_cols.append("term_months")
    if any(k in text for k in ("by loan status", "by status", "and loan status", "and status", "fully paid and charged-off", "fully paid vs charged off")):
        group_cols.append("loan_status")

    metric_sql: str | None = None
    metric_alias: str | None = None
    if any(k in text for k in ("loss given default", "lgd")):
        metric_sql = "ROUND(100.0 * (1.0 - SUM(sum_rec_prncp) / NULLIF(SUM(sum_loan_amnt), 0)), 2) AS lgd_percent"
        metric_alias = "lgd_percent"
    elif any(k in text for k in ("recovery rate", "principal recovery rate", "recovered percentage", "percent recovered")):
        metric_sql = "ROUND(100.0 * SUM(sum_rec_prncp) / NULLIF(SUM(sum_loan_amnt), 0), 2) AS recovery_rate_percent"
        metric_alias = "recovery_rate_percent"
    elif any(k in text for k in ("unrecovered", "capital loss", "principal loss", "capital exposure")):
        metric_sql = "ROUND(SUM(sum_loan_amnt - sum_rec_prncp), 2) AS unrecovered_principal"
        metric_alias = "unrecovered_principal"
    elif asks_default_rate:
        metric_sql = "ROUND(100.0 * SUM(default_flag * loan_count) / NULLIF(SUM(loan_count), 0), 2) AS default_rate_percent"
        metric_alias = "default_rate_percent"
    elif any(k in text for k in ("principal received", "principal recovered", "principal was recovered", "recovered principal", "total principal")):
        if "average" in text or "avg" in text or "mean" in text:
            metric_sql = "ROUND(SUM(sum_rec_prncp) / NULLIF(SUM(loan_count), 0), 2) AS average_principal_received"
            metric_alias = "average_principal_received"
        else:
            metric_sql = "ROUND(SUM(sum_rec_prncp), 2) AS total_principal_received"
            metric_alias = "total_principal_received"
    elif any(k in text for k in ("interest rate", "int_rate")):
        metric_sql = "SUM(sum_int_rate) / NULLIF(SUM(loan_count), 0) AS average_interest_rate"
        metric_alias = "average_interest_rate"
    elif any(k in text for k in ("loan amount", "funded", "origination volume", "exposure")):
        if any(k in text for k in ("total", "sum", "volume")):
            metric_sql = "ROUND(SUM(sum_loan_amnt), 2) AS total_loan_amount"
            metric_alias = "total_loan_amount"
        else:
            metric_sql = "SUM(sum_loan_amnt) / NULLIF(SUM(loan_count), 0) AS average_loan_amount"
            metric_alias = "average_loan_amount"
    elif "installment" in text:
        metric_sql = "ROUND(SUM(sum_installment) / NULLIF(SUM(loan_count), 0), 2) AS average_installment"
        metric_alias = "average_installment"
    elif any(k in text for k in ("annual income", "income")):
        metric_sql = "SUM(sum_annual_inc) / NULLIF(SUM(loan_count), 0) AS average_annual_income"
        metric_alias = "average_annual_income"
    elif any(k in text for k in ("dti", "debt-to-income", "debt to income")):
        metric_sql = "SUM(sum_dti) / NULLIF(SUM(loan_count), 0) AS average_dti"
        metric_alias = "average_dti"
    elif any(k in text for k in ("revol", "revolving", "utilization")):
        metric_sql = "ROUND(SUM(sum_revol_util) / NULLIF(SUM(loan_count), 0), 2) AS average_revol_util"
        metric_alias = "average_revol_util"
    elif any(k in text for k in ("fico", "credit score")):
        metric_sql = "SUM(sum_fico) / NULLIF(SUM(loan_count), 0) AS average_fico"
        metric_alias = "average_fico"
    elif any(k in text for k in ("how many", "count", "number of loans", "volume of loans", "trend of", "distribution")):
        metric_sql = "SUM(loan_count) AS loan_count"
        metric_alias = "loan_count"

    if metric_sql is None:
        return None

    select_parts = group_cols + [metric_sql]
    parts = [f"SELECT {', '.join(select_parts)}", "FROM agg_loan_cube"]
    if wheres:
        parts.append("WHERE " + " AND ".join(dict.fromkeys(wheres)))
    if group_cols:
        parts.append("GROUP BY " + ", ".join(group_cols))
        if any(k in text for k in ("highest", "top ", "most", "largest", "worst", "maximum")):
            parts.append(f"ORDER BY {metric_alias} DESC")
        elif any(k in text for k in ("lowest", "bottom ", "least", "smallest", "best", "minimum")):
            parts.append(f"ORDER BY {metric_alias} ASC")
        else:
            parts.append("ORDER BY " + ", ".join(group_cols))
        top_match = re.search(r"\btop\s+(\d+)\b", text)
        if top_match:
            parts.append(f"LIMIT {int(top_match.group(1))}")
    return " ".join(parts)


def answer_question(
    question: str,
    database: str,
    use_cache: bool = True,
    use_olap_cube: bool = False,
) -> dict:
    t0 = time.perf_counter()
    norm_key = (database, " ".join(question.strip().lower().split()), str(use_olap_cube))
    if use_cache:
        with _CACHE_LOCK:
            cached_entry = _QUERY_CACHE.get(norm_key)
            if cached_entry is not None:
                ts, cached = cached_entry
                if (time.perf_counter() - ts) < _CACHE_TTL_SECONDS:
                    _QUERY_CACHE.move_to_end(norm_key)
                    _CACHE_STATS["hits"] += 1
                    return {
                        **cached,
                        "execution_mode": "cache",
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                    }
                del _QUERY_CACHE[norm_key]
            _CACHE_STATS["misses"] += 1

    is_pg = database.startswith(("postgresql://", "postgres://"))
    context = retrieve_warehouse_context(question)

    mode = "schema_planner"
    sql: str | None = None
    explanation = ""
    columns: list[str] = []
    rows: list[dict[str, Any]] = []

    if os.getenv("NRP_LLM_BASE_URL", "").strip():
        feedback: str | None = None
        for _ in range(2):
            llm_raw_sql = _call_nrp_llm(
                question, context, use_postgres_schema=is_pg, error_feedback=feedback
            )
            if not llm_raw_sql:
                break
            try:
                candidate = validate_readonly_sql(llm_raw_sql)
                columns, rows = _execute_sql(candidate, database)
                sql = candidate
                explanation = "Schema-grounded RAG SQL result."
                mode = "rag_llm"
                break
            except Exception as exc:
                feedback = str(exc)
                sql = None

    if sql is None:
        sql, explanation = generate_sql(question)
        if sql is None:
            return {
                "answer": explanation,
                "summary": explanation,
                "sql": None,
                "columns": [],
                "rows": [],
                "execution_mode": "rejected",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
            }
        is_exact_sample = " ".join(question.strip().lower().split()) in _EXACT_SAMPLE_MAP
        sql = validate_readonly_sql(sql, enforce_limit=False)

        cube_sql = _compose_cube_sql(question) if (use_olap_cube and not is_pg) else None
        if cube_sql:
            try:
                columns, rows = _execute_sql(validate_readonly_sql(cube_sql, enforce_limit=False), database)
                mode = "olap_cube"
            except Exception:
                columns, rows = _execute_sql(sql, database)
                mode = "deterministic_template" if is_exact_sample else "schema_planner"
        else:
            mode = "deterministic_template" if is_exact_sample else "schema_planner"
            columns, rows = _execute_sql(sql, database)

    summary = _summarize_rows(explanation, columns, rows)
    result = {
        "answer": explanation if mode == "deterministic_template" else summary,
        "summary": summary,
        "sql": sql,
        "columns": columns,
        "rows": rows,
        "execution_mode": mode,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }
    if use_cache:
        with _CACHE_LOCK:
            _QUERY_CACHE[norm_key] = (time.perf_counter(), result)
            if len(_QUERY_CACHE) > _CACHE_MAX_SIZE:
                _QUERY_CACHE.popitem(last=False)
    return result
