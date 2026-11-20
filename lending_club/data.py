from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from datetime import datetime

import pandas as pd

REQUIRED_COLUMNS = {
    "issue_date": ("issue_d", "issue_date"),
    "annual_inc": ("annual_inc",),
    "dti": ("dti",),
    "revol_util": ("revol_util",),
    "fico_range_low": ("fico_range_low",),
    "loan_amnt": ("loan_amnt",),
    "int_rate": ("int_rate",),
    "term": ("term",),
    "installment": ("installment",),
    "loan_status": ("loan_status",),
    "grade": ("grade",),
    "total_rec_prncp": ("total_rec_prncp",),
}
DEFAULT_LABELS = {"Fully Paid": 0, "Charged Off": 1}


@dataclass(frozen=True)
class PreparedData:
    frame: pd.DataFrame
    excluded_statuses: tuple[str, ...]


def _find_column(columns: pd.Index, aliases: tuple[str, ...]) -> str | None:
    lower = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in lower:
            return str(lower[alias])
    return None


def _percent(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip().replace("%", "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _term(value: object) -> float | None:
    if pd.isna(value):
        return None
    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else None


def _issue_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if re.fullmatch(r"[A-Za-z]{3}-\d{2}", text):
        return pd.Timestamp(datetime.strptime(text, "%b-%y"))
    return pd.to_datetime(text, errors="coerce")


def prepare_lending_data(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    limit: int = 100_000,
    seed: int = 42,
) -> PreparedData:
    """Read, normalize, filter, and deterministically sample Lending Club loans."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing = [
        logical
        for logical, aliases in REQUIRED_COLUMNS.items()
        if _find_column(frame.columns, aliases) is None
    ]
    if missing:
        raise ValueError(
            "Lending Club CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    rename = {
        _find_column(frame.columns, aliases): logical
        for logical, aliases in REQUIRED_COLUMNS.items()
    }
    data = frame.rename(columns=rename).copy()
    data["issue_date"] = data["issue_date"].map(_issue_date)
    data = data[data["issue_date"].dt.year.between(2015, 2018, inclusive="both")]
    data["loan_status"] = data["loan_status"].astype("string").str.strip()
    accepted = data["loan_status"].isin(DEFAULT_LABELS)
    excluded_statuses = tuple(sorted(data.loc[~accepted, "loan_status"].dropna().unique()))
    data = data[accepted].copy()

    for column in ("dti", "revol_util", "int_rate"):
        data[column] = data[column].map(_percent)
    data["term"] = data["term"].map(_term)
    numeric = ["annual_inc", "fico_range_low", "loan_amnt", "installment", "total_rec_prncp"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["grade"] = data["grade"].astype("string").str.strip().str.upper()

    data = data.dropna(subset=list(REQUIRED_COLUMNS))
    data["loan_status_label"] = data["loan_status"].map(DEFAULT_LABELS).astype("int64")
    data = data.sort_values(["issue_date", "loan_amnt", "loan_status", "annual_inc"]).reset_index(
        drop=True
    )
    if len(data) > limit:
        data = data.sample(n=limit, random_state=seed).sort_index().reset_index(drop=True)
    data["loan_id"] = [f"LC-{i:06d}" for i in range(len(data))]
    columns = [
        "loan_id",
        "issue_date",
        "annual_inc",
        "dti",
        "revol_util",
        "fico_range_low",
        "loan_amnt",
        "int_rate",
        "term",
        "installment",
        "loan_status",
        "loan_status_label",
        "grade",
        "total_rec_prncp",
    ]
    data = data[columns]
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)
    return PreparedData(data, excluded_statuses)

