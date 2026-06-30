"""Load snapshot data and build horizon panels for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.export_utils import add_qp_sign_case
from src.analysis.growth_horizons import (
    compute_bls_endpoint_growth,
    compute_bls_yearly_q1_growth,
    compute_endpoint_growth,
    compute_yearly_q1_growth,
)
from src.analysis.forecasting import FORECAST_METRICS, build_sector_forecasts, forecast_settings
from src.analysis.industry_filters import load_config, profile_exclude_line_ids

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _apply_profile_excludes(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    exclude_ids = profile_exclude_line_ids(profile)
    if df.empty or not exclude_ids:
        return df
    return df[~df["line_id"].isin(exclude_ids)].copy()


def load_profile_data(profile: str) -> dict[str, pd.DataFrame]:
    snap = SNAPSHOTS_DIR / profile
    return {
        "bea_observations": _read_csv(snap / "bea_observations.csv"),
        "bea_quarterly": _read_csv(snap / "bea_growth_quarterly.csv"),
        "bls_quarterly": _read_csv(snap / "bls_quarterly_growth.csv"),
    }


def available_periods(bea_quarterly: pd.DataFrame) -> list[str]:
    if bea_quarterly.empty or "period" not in bea_quarterly.columns:
        return []
    return sorted(bea_quarterly["period"].unique().tolist())


def filter_quarterly(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty or "period" not in df.columns:
        return df
    return df[(df["period"] >= start) & (df["period"] <= end)].copy()


def _industry_meta(bea_quarterly: pd.DataFrame) -> pd.DataFrame:
    if bea_quarterly.empty or "line_id" not in bea_quarterly.columns:
        return pd.DataFrame()
    cols = [c for c in ["line_id", "industry_name", "indent_level", "is_private"] if c in bea_quarterly.columns]
    return bea_quarterly[cols].drop_duplicates("line_id")


def _attach_industry_names(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    if df.empty or meta.empty or "line_id" not in df.columns:
        return df
    if "industry_name" in df.columns:
        return df
    return df.merge(meta.drop_duplicates("line_id"), on="line_id", how="left")


def build_panels(
    profile: str,
    start: str,
    end: str,
    horizon: str,
) -> dict[str, pd.DataFrame]:
    """Return chart-ready panels for the selected horizon."""
    cfg = load_config()
    raw = load_profile_data(profile)
    bea_q = filter_quarterly(raw["bea_quarterly"], start, end)
    bls_q = filter_quarterly(raw["bls_quarterly"], start, end)
    bea_obs = raw["bea_observations"]

    if "qp_sign_case" not in bea_q.columns and not bea_q.empty:
        bea_q = add_qp_sign_case(bea_q)

    industries = _industry_meta(bea_q)
    bls_q = _attach_industry_names(bls_q, industries)

    effective_end = end
    if not bea_q.empty:
        latest = bea_q["period"].max()
        if latest < end:
            effective_end = latest

    result: dict[str, pd.DataFrame] = {
        "bea_quarterly": bea_q,
        "bls_quarterly": bls_q,
    }

    if not bea_obs.empty and not industries.empty:
        result["bea_endpoint"] = compute_endpoint_growth(bea_obs, industries, start, effective_end)
        result["bls_endpoint"] = compute_bls_endpoint_growth(bls_q, industries, start, effective_end)

    if horizon == "quarterly":
        return result

    if horizon == "endpoint":
        return result

    if horizon == "yearly_q1" and not bea_obs.empty and not industries.empty:
        yearly_bea = compute_yearly_q1_growth(bea_obs, industries)
        yearly_bls = compute_bls_yearly_q1_growth(bls_q, industries)
        if not yearly_bea.empty and "period_end" in yearly_bea.columns:
            yearly_bea = yearly_bea[
                (yearly_bea["period_start"] >= start) & (yearly_bea["period_end"] <= effective_end)
            ]
        if not yearly_bls.empty and "period_end" in yearly_bls.columns:
            yearly_bls = yearly_bls[
                (yearly_bls["period_start"] >= start) & (yearly_bls["period_end"] <= effective_end)
            ]
        result["bea_yearly"] = yearly_bea
        result["bls_yearly"] = yearly_bls
        return result

    return result


METRIC_OPTIONS = {
    "BEA price growth": ("bea", "price_growth"),
    "BEA quantity growth": ("bea", "quantity_growth"),
    "BEA Q/P ratio": ("bea", "qp_ratio"),
    "BLS wage growth": ("bls", "avg_hourly_earnings_growth"),
    "BLS employment growth": ("bls", "employment_thousands_growth"),
}

COMPARISON_PAIRS = {
    "Price vs wage": ("price_growth", "avg_hourly_earnings_growth"),
    "Quantity vs employment": ("quantity_growth", "employment_thousands_growth"),
}

def scatter_subtitle(horizon: str, start: str, end: str) -> str:
    from src.analysis.plot_style import format_date_range
    date_part = format_date_range(start, end)
    if horizon == "quarterly":
        return f"Average quarterly growth · {date_part}"
    if horizon == "yearly_q1":
        return f"Average Q1→Q1 annual growth · {date_part}"
    return f"Total change · {date_part}"


def build_scatter_frame(
    panels: dict,
    horizon: str,
    sectors: list[str],
) -> pd.DataFrame:
    if horizon == "quarterly":
        df = panels.get("bea_quarterly", pd.DataFrame())
        if df.empty:
            return df
        sub = df[df["industry_name"].isin(sectors)]
        return sub.groupby("industry_name", as_index=False)[["price_growth", "quantity_growth", "line_id"]].mean()

    if horizon == "yearly_q1":
        df = panels.get("bea_yearly", pd.DataFrame())
        if df.empty:
            return pd.DataFrame()
        sub = df[df["industry_name"].isin(sectors)]
        return sub.groupby("industry_name", as_index=False)[["price_growth", "quantity_growth", "line_id"]].mean()

    df = panels.get("bea_endpoint", pd.DataFrame())
    if df.empty:
        return df
    return df[df["industry_name"].isin(sectors)].drop_duplicates("line_id")


def build_forecast_panels(
    profile: str,
    start: str,
    end: str,
    metric_label: str,
    sectors: list[str],
) -> pd.DataFrame:
    """Build historical + annual Q1→Q1 SARIMA forecast rows for selected sectors."""
    if metric_label not in FORECAST_METRICS:
        return pd.DataFrame()

    source, value_col = FORECAST_METRICS[metric_label]
    panels = build_panels(profile, start, end, "yearly_q1")
    key = "bea_yearly" if source == "bea" else "bls_yearly"
    history = panels.get(key, pd.DataFrame())
    if history.empty or value_col not in history.columns:
        return pd.DataFrame()

    history = history[history["industry_name"].isin(sectors)].copy()
    if "period_end" not in history.columns:
        return pd.DataFrame()

    history["period"] = history["period_end"]
    history = history[(history["period_start"] >= start) & (history["period_end"] <= end)]

    settings = forecast_settings()
    return build_sector_forecasts(
        history,
        sectors,
        value_col,
        end,
        settings=settings,
    )
