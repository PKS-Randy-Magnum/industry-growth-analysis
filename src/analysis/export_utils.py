"""Excel-friendly CSV export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

NUMERIC_COLS_DEFAULT = {
    "line_id",
    "indent_level",
    "value",
    "quantity_growth",
    "price_growth",
    "qp_ratio",
    "quantity_index",
    "price_index",
    "avg_price_growth",
    "avg_quantity_growth",
    "vol_price_growth",
    "vol_quantity_growth",
    "avg_qp_ratio",
    "employment_thousands_growth",
    "avg_hourly_earnings_growth",
    "cluster_id",
    "is_private",
    "year",
}


def qp_sign_case(quantity_growth: float, price_growth: float) -> str:
    if pd.isna(quantity_growth) or pd.isna(price_growth):
        return "zero_component"
    if quantity_growth == 0 or price_growth == 0:
        return "zero_component"
    if quantity_growth > 0 and price_growth > 0:
        return "both_positive"
    if quantity_growth < 0 and price_growth > 0:
        return "neg_num_pos_den"
    if quantity_growth > 0 and price_growth < 0:
        return "pos_num_neg_den"
    if quantity_growth < 0 and price_growth < 0:
        return "both_negative"
    return "zero_component"


QP_SIGN_LABELS: dict[str, str] = {
    "both_positive": "Q↑ P↑",
    "both_negative": "Q↓ P↓",
    "pos_num_neg_den": "Q↑ P↓",
    "neg_num_pos_den": "Q↓ P↑",
    "zero_component": "near zero",
}

QP_SIGN_COLORS: dict[str, str] = {
    "both_positive": "#0072B2",
    "both_negative": "#D55E00",
    "pos_num_neg_den": "#009E73",
    "neg_num_pos_den": "#CC6677",
    "zero_component": "#999999",
}


def qp_sign_label(case: str) -> str:
    return QP_SIGN_LABELS.get(case, str(case))


def qp_sign_color(case: str) -> str:
    return QP_SIGN_COLORS.get(case, "#888888")


def filter_stable_qp_rows(
    df: pd.DataFrame,
    *,
    price_col: str = "price_growth",
    min_abs_price: float = 0.005,
) -> pd.DataFrame:
    """Drop rows with near-zero price growth (unstable Q/P denominator)."""
    if df.empty or price_col not in df.columns:
        return df
    out = df.copy()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    return out[out[price_col].abs() >= min_abs_price]


def add_qp_sign_case(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "quantity_growth" in out.columns and "price_growth" in out.columns:
        out["qp_sign_case"] = [
            qp_sign_case(q, p) for q, p in zip(out["quantity_growth"], out["price_growth"], strict=True)
        ]
    return out


def prepare_for_excel(df: pd.DataFrame, numeric_cols: set[str] | None = None) -> pd.DataFrame:
    numeric_cols = numeric_cols or NUMERIC_COLS_DEFAULT
    out = df.copy()
    for col in out.columns:
        if col in numeric_cols or col.endswith("_growth") or col.endswith("_ratio") or col.endswith("_index"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    float_cols = out.select_dtypes(include=["float"]).columns
    for col in float_cols:
        out[col] = out[col].round(6)
    if "line_id" in out.columns:
        out["line_id"] = pd.to_numeric(out["line_id"], errors="coerce").astype("Int64")
    return out


def write_excel_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = prepare_for_excel(df)
    prepared.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f", **kwargs)
    return path


def pivot_growth_wide(
    df: pd.DataFrame,
    value_col: str,
    index_cols: list[str] | None = None,
) -> pd.DataFrame:
    index_cols = index_cols or ["line_id", "industry_name"]
    wide = df.pivot_table(
        index=index_cols,
        columns="period",
        values=value_col,
        aggfunc="first",
    ).reset_index()
    wide.columns = [str(c) if not isinstance(c, tuple) else c for c in wide.columns]
    return prepare_for_excel(wide)
