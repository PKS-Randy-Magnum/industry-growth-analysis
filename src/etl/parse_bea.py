"""Parse BEA chain-type quantity/price index wide-format CSV into tidy tables."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BLOCK_METRICS = [
    "quantity_index",
    "price_index",
    "quantity_growth",
    "price_growth",
    "quantity_price_ratio",
]


def _indent_level(name: str) -> int:
    stripped = name.replace("\xa0", " ")
    return len(name) - len(stripped.lstrip(" "))


def _clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("\xa0", " ").strip())


def _find_blocks(header_row: list[str]) -> list[tuple[int, int, str]]:
    starts = [i for i, value in enumerate(header_row) if value.strip() == "Line"]
    starts.append(len(header_row))
    blocks = []
    for idx, start in enumerate(starts[:-1]):
        end = starts[idx + 1]
        metric = BLOCK_METRICS[idx] if idx < len(BLOCK_METRICS) else f"block_{idx}"
        blocks.append((start, end, metric))
    return blocks


def _quarter_columns(header_row: list[str], period_row: list[str], start: int, end: int) -> list[tuple[int, str]]:
    columns: list[tuple[int, str]] = []
    current_year: str | None = None
    for col in range(start + 2, end):
        year_token = header_row[col].strip() if col < len(header_row) else ""
        if year_token.isdigit():
            current_year = year_token
        quarter = period_row[col].strip() if col < len(period_row) else ""
        if current_year and quarter.upper().startswith("Q"):
            quarter_num = quarter[1]
            columns.append((col, f"{current_year}-Q{quarter_num}"))
    return columns


def parse_bea_csv(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        if not line.isdigit():
            continue

        name = _clean_name(row[1]) if len(row) > 1 else ""
        if not name:
            continue

        line_id = int(line)
        industries[line_id] = {
            "line_id": line_id,
            "industry_name": name,
            "indent_level": _indent_level(row[1]),
            "is_private": "government" not in name.lower(),
        }

        for start, end, metric in blocks:
            for col, period in _quarter_columns(header_row, period_row, start, end):
                value = row[col].strip() if col < len(row) else ""
                if not value:
                    continue
                try:
                    numeric = float(value)
                except ValueError:
                    continue
                records.append(
                    {
                        "line_id": line_id,
                        "period": period,
                        "metric": metric,
                        "value": numeric,
                    }
                )

    industries_df = pd.DataFrame(industries.values()).sort_values("line_id")
    observations_df = pd.DataFrame(records)
    return industries_df, observations_df


def compute_bea_growth_from_indexes(observations: pd.DataFrame) -> pd.DataFrame:
    """Derive QoQ growth and Q/P ratio from quantity_index and price_index rows.

    %ΔQ_t = (Q_t - Q_{t-1}) / Q_{t-1}
    %ΔP_t = (P_t - P_{t-1}) / P_{t-1}
    quantity_price_ratio_t = %ΔQ_t / %ΔP_t  (NaN when %ΔP_t is 0)
    First quarter per line_id: growth set to 0 (matches iTable CSV export).
    """
    required = {"line_id", "period", "metric", "value"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"observations missing columns: {sorted(missing)}")

    index_obs = observations[observations["metric"].isin(["quantity_index", "price_index"])].copy()
    if index_obs.empty:
        return observations.copy()

    wide = (
        index_obs.pivot_table(
            index=["line_id", "period"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["line_id", "period"])
    )
    if "quantity_index" not in wide.columns or "price_index" not in wide.columns:
        return index_obs

    derived: list[dict] = []
    for line_id, grp in wide.groupby("line_id", sort=False):
        grp = grp.sort_values("period").reset_index(drop=True)
        q_growth = grp["quantity_index"].astype(float).pct_change()
        p_growth = grp["price_index"].astype(float).pct_change()
        if len(q_growth) > 0:
            q_growth.iloc[0] = 0.0
        if len(p_growth) > 0:
            p_growth.iloc[0] = 0.0
        ratio = q_growth / p_growth.replace(0, pd.NA)

        for i, row in grp.iterrows():
            period = row["period"]
            derived.append(
                {
                    "line_id": int(line_id),
                    "period": period,
                    "metric": "quantity_growth",
                    "value": float(q_growth.iloc[i]),
                }
            )
            derived.append(
                {
                    "line_id": int(line_id),
                    "period": period,
                    "metric": "price_growth",
                    "value": float(p_growth.iloc[i]),
                }
            )
            r = ratio.iloc[i]
            if pd.notna(r):
                derived.append(
                    {
                        "line_id": int(line_id),
                        "period": period,
                        "metric": "quantity_price_ratio",
                        "value": float(r),
                    }
                )

    derived_df = pd.DataFrame(derived)
    return pd.concat([index_obs, derived_df], ignore_index=True)


def bea_growth_panel(observations: pd.DataFrame) -> pd.DataFrame:
    """Wide panel with price and quantity growth by industry-quarter."""
    growth = observations[observations["metric"].isin(["quantity_growth", "price_growth"])]
    panel = (
        growth.pivot_table(index=["line_id", "period"], columns="metric", values="value", aggfunc="first")
        .reset_index()
        .rename(columns={"quantity_growth": "quantity_growth", "price_growth": "price_growth"})
    )
    return panel
