from __future__ import annotations

import re
import sqlite3

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


def generate_sql(question: str) -> tuple[str | None, str]:
    text = question.strip().lower()
    if not text or GREETING.fullmatch(text):
        return None, "Please ask a Lending Club question about loan amounts, borrower metrics, rates, or outcomes."

    if "principal" in text or ("principal received" in text):
        if "grade" in text:
            key = "principal_grade"
        elif "year" in text:
            key = "principal_year"
        else:
            key = "principal_status"
    elif "annual income" in text or "income" in text:
        key = "avg_income_grade_status" if "grade" in text and "status" in text else "avg_income_status"
    elif "fico" in text or "credit score" in text:
        key = "avg_fico_status"
    elif ("dti" in text or "debt-to-income" in text) and "status" in text:
        key = "avg_dti_status"
    elif "default" in text and "year" in text:
        key = "default_year"
    elif "how many" in text or ("count" in text and "year" in text):
        key = "count_year"
    elif ("average" in text or "avg" in text) and "loan amount" in text and "grade" in text:
        key = "avg_loan_grade"
    elif ("interest" in text or "rate" in text):
        if "grade" in text and "status" in text:
            key = "avg_rate_grade_status"
        elif "grade" in text:
            key = "avg_rate_grade"
        else:
            key = "avg_rate_status"
    else:
        return None, "Unsupported question. Choose one of the documented Lending Club sample questions."

    sql, explanation = TEMPLATES[key]
    return sql, explanation


def answer_question(question: str, database: str) -> dict:
    sql, explanation = generate_sql(question)
    if sql is None:
        return {"answer": explanation, "summary": explanation, "sql": None, "columns": [], "rows": []}
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(sql)
        columns = [item[0] for item in cursor.description]
        rows = [dict(row) for row in cursor.fetchall()]
    return {
        "answer": explanation,
        "summary": explanation,
        "sql": sql,
        "columns": columns,
        "rows": rows,
    }
