"""Compute endpoint, yearly Q1, and quarterly BEA growth panels from indexes."""

from __future__ import annotations

import pandas as pd

from src.analysis.export_utils import add_qp_sign_case, qp_sign_case


def _index_wide(observations: pd.DataFrame) -> pd.DataFrame:
    idx = observations[observations["metric"].isin(["quantity_index", "price_index"])].copy()
    wide = idx.pivot_table(
        index=["line_id", "period"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    return wide


def _span_growth(qty_start: float, qty_end: float, price_start: float, price_end: float) -> dict:
    q_g = (qty_end - qty_start) / qty_start if qty_start else 0.0
    p_g = (price_end - price_start) / price_start if price_start else 0.0
    qp = q_g / p_g if p_g else 0.0
    return {
        "quantity_growth": q_g,
        "price_growth": p_g,
        "qp_ratio": qp,
        "qp_sign_case": qp_sign_case(q_g, p_g),
    }


def compute_endpoint_growth(
    observations: pd.DataFrame,
    industries: pd.DataFrame,
    period_start: str,
    period_end: str,
) -> pd.DataFrame:
    wide = _index_wide(observations)
    meta = industries[["line_id", "industry_name", "indent_level", "is_private"]].drop_duplicates("line_id")
    records: list[dict] = []

    for line_id, group in wide.groupby("line_id"):
        g = group.sort_values("period")
        start_row = g[g["period"] == period_start]
        end_row = g[g["period"] == period_end]
        if start_row.empty or end_row.empty:
            continue
        s, e = start_row.iloc[0], end_row.iloc[0]
        if pd.isna(s.get("quantity_index")) or pd.isna(e.get("quantity_index")):
            continue
        growth = _span_growth(
            float(s["quantity_index"]),
            float(e["quantity_index"]),
            float(s["price_index"]),
            float(e["price_index"]),
        )
        records.append({"line_id": int(line_id), "period_start": period_start, "period_end": period_end, **growth})

    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records).merge(meta, on="line_id", how="left")
    return out


def compute_yearly_q1_growth(observations: pd.DataFrame, industries: pd.DataFrame) -> pd.DataFrame:
    wide = _index_wide(observations)
    meta = industries[["line_id", "industry_name", "indent_level", "is_private"]].drop_duplicates("line_id")
    q1 = wide[wide["period"].str.endswith("-Q1")].copy()
    q1["year"] = q1["period"].str[:4].astype(int)
    records: list[dict] = []

    for line_id, group in q1.groupby("line_id"):
        g = group.sort_values("year")
        years = g["year"].tolist()
        for i in range(len(years) - 1):
            y0, y1 = years[i], years[i + 1]
            if y1 != y0 + 1:
                continue
            s = g[g["year"] == y0].iloc[0]
            e = g[g["year"] == y1].iloc[0]
            growth = _span_growth(
                float(s["quantity_index"]),
                float(e["quantity_index"]),
                float(s["price_index"]),
                float(e["price_index"]),
            )
            records.append(
                {
                    "line_id": int(line_id),
                    "year": y0,
                    "period_start": f"{y0}-Q1",
                    "period_end": f"{y1}-Q1",
                    **growth,
                }
            )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).merge(meta, on="line_id", how="left")


def quarterly_growth_panel(bea_growth: pd.DataFrame) -> pd.DataFrame:
    return add_qp_sign_case(bea_growth.copy())


def _level_change(start: float, end: float) -> float:
    if pd.isna(start) or pd.isna(end) or start == 0:
        return float("nan")
    return float((end - start) / start)


def compute_bls_endpoint_growth(
    bls_quarterly: pd.DataFrame,
    industries: pd.DataFrame,
    period_start: str,
    period_end: str,
) -> pd.DataFrame:
    meta = industries[["line_id", "industry_name", "indent_level", "is_private"]].drop_duplicates("line_id")
    records: list[dict] = []

    for line_id, group in bls_quarterly.groupby("line_id"):
        g = group.sort_values("period")
        start_row = g[g["period"] == period_start]
        end_row = g[g["period"] == period_end]
        if start_row.empty or end_row.empty:
            continue
        s, e = start_row.iloc[0], end_row.iloc[0]
        records.append(
            {
                "line_id": int(line_id),
                "period_start": period_start,
                "period_end": period_end,
                "employment_thousands_growth": _level_change(
                    s.get("employment_thousands"), e.get("employment_thousands")
                ),
                "avg_hourly_earnings_growth": _level_change(
                    s.get("avg_hourly_earnings"), e.get("avg_hourly_earnings")
                ),
            }
        )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).merge(meta, on="line_id", how="left")


def compute_bls_yearly_q1_growth(bls_quarterly: pd.DataFrame, industries: pd.DataFrame) -> pd.DataFrame:
    meta = industries[["line_id", "industry_name", "indent_level", "is_private"]].drop_duplicates("line_id")
    q1 = bls_quarterly[bls_quarterly["period"].str.endswith("-Q1")].copy()
    q1["year"] = q1["period"].str[:4].astype(int)
    records: list[dict] = []

    for line_id, group in q1.groupby("line_id"):
        g = group.sort_values("year")
        years = g["year"].tolist()
        for i in range(len(years) - 1):
            y0, y1 = years[i], years[i + 1]
            if y1 != y0 + 1:
                continue
            s = g[g["year"] == y0].iloc[0]
            e = g[g["year"] == y1].iloc[0]
            records.append(
                {
                    "line_id": int(line_id),
                    "year": y0,
                    "period_start": f"{y0}-Q1",
                    "period_end": f"{y1}-Q1",
                    "employment_thousands_growth": _level_change(
                        s.get("employment_thousands"), e.get("employment_thousands")
                    ),
                    "avg_hourly_earnings_growth": _level_change(
                        s.get("avg_hourly_earnings"), e.get("avg_hourly_earnings")
                    ),
                }
            )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).merge(meta, on="line_id", how="left")
