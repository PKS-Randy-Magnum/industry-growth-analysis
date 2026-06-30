"""Parse BLS CES-style wide-format CSV into tidy monthly tables."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUNE": 6,
    "JUN": 6,
    "JULY": 7,
    "JUL": 7,
    "AUGUST": 8,
    "AUG": 8,
    "SEPT": 9,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

BLOCK_METRICS = {
    0: "employment_thousands",
    1: "avg_hourly_earnings",
    2: "avg_weekly_earnings",
}


def _indent_level(name: str) -> int:
    stripped = name.replace("\xa0", " ")
    return len(name) - len(stripped.lstrip(" "))


def _clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("\xa0", " ").strip())


def _normalize_month(token: str) -> str | None:
    token = token.strip().upper().replace("(P)", "").strip()
    for key in sorted(MONTH_MAP, key=len, reverse=True):
        if token.startswith(key):
            return key
    return None


def _find_blocks(header_row: list[str]) -> list[tuple[int, int, str]]:
    starts = [i for i, value in enumerate(header_row) if value.strip() == "Line"]
    starts.append(len(header_row))
    blocks = []
    for idx, start in enumerate(starts[:-1]):
        end = starts[idx + 1]
        metric = BLOCK_METRICS.get(idx, f"block_{idx}")
        blocks.append((start, end, metric))
    return blocks


def _month_columns(header_row: list[str], period_row: list[str], start: int, end: int) -> list[tuple[int, str]]:
    columns: list[tuple[int, str]] = []
    current_year: str | None = None
    for col in range(start + 2, end):
        year_token = header_row[col].strip() if col < len(header_row) else ""
        if year_token.isdigit():
            current_year = year_token
        month_token = period_row[col].strip() if col < len(period_row) else ""
        month_key = _normalize_month(month_token)
        if current_year and month_key:
            month_num = MONTH_MAP[month_key]
            columns.append((col, f"{current_year}-{month_num:02d}"))
    return columns


def _parse_numeric(value: str) -> float | None:
    value = value.strip().replace(",", "")
    if not value or value.upper() in {"N/A", "NA", "#N/A"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_bls_csv(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (industries, observations) tidy DataFrames."""
    path = Path(path)
    raw = pd.read_csv(path, header=None, dtype=str, encoding="utf-8-sig").fillna("")
    header_row = raw.iloc[0].tolist()
    period_row = raw.iloc[1].tolist()
    blocks = _find_blocks(header_row)

    industries: dict[str, dict] = {}
    records: list[dict] = []

    for row_idx in range(2, len(raw)):
        row = raw.iloc[row_idx].tolist()
        line = row[0].strip() if row else ""
        name = _clean_name(row[1]) if len(row) > 1 else ""

        # Second table in export = precomputed growth rates; skip for level-based pipeline.
        if line == "Line" or "change in" in name.lower():
            break
        if not line and not name:
            continue
        if not line.isdigit():
            continue
        if not name:
            continue

        line_id = int(line)
        industries[line_id] = {
            "line_id": line_id,
            "industry_name": name,
            "indent_level": _indent_level(row[1]),
            "is_private": "government" not in name.lower(),
        }

        for block_idx, (start, end, metric) in enumerate(blocks):
            for col, period in _month_columns(header_row, period_row, start, end):
                value = _parse_numeric(row[col] if col < len(row) else "")
                if value is None:
                    continue
                records.append(
                    {
                        "line_id": line_id,
                        "period_month": period,
                        "metric": metric,
                        "value": value,
                    }
                )

    industries_df = pd.DataFrame(industries.values()).sort_values("line_id")
    observations_df = pd.DataFrame(records)
    return industries_df, observations_df


def bls_quarterly_growth(observations: pd.DataFrame) -> pd.DataFrame:
    """Compute quarterly employment and wage growth from monthly levels."""
    if observations.empty:
        return pd.DataFrame()

    monthly = observations.pivot_table(
        index=["line_id", "period_month"], columns="metric", values="value", aggfunc="first"
    ).reset_index()

    monthly["year"] = monthly["period_month"].str.slice(0, 4).astype(int)
    monthly["month"] = monthly["period_month"].str.slice(5, 7).astype(int)
    monthly["quarter"] = ((monthly["month"] - 1) // 3 + 1).astype(int)

    quarterly = (
        monthly.groupby(["line_id", "year", "quarter"], as_index=False)
        .agg(
            employment_thousands=("employment_thousands", "mean"),
            avg_hourly_earnings=("avg_hourly_earnings", "mean"),
            avg_weekly_earnings=("avg_weekly_earnings", "mean"),
        )
        .sort_values(["line_id", "year", "quarter"])
    )

    quarterly["period"] = quarterly["year"].astype(str) + "-Q" + quarterly["quarter"].astype(str)
    for col in ["employment_thousands", "avg_hourly_earnings", "avg_weekly_earnings"]:
        if col in quarterly.columns:
            quarterly[f"{col}_growth"] = quarterly.groupby("line_id")[col].pct_change()

    return quarterly
