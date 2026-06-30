"""Annual (Q1→Q1) SARIMA forecasts with optional price-growth shock scenarios."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.analysis.industry_filters import load_config

MIN_OBS_ANNUAL = 6

FORECAST_METRICS: dict[str, tuple[str, str]] = {
    "BEA price growth": ("bea", "price_growth"),
    "BEA quantity growth": ("bea", "quantity_growth"),
    "BLS wage growth": ("bls", "avg_hourly_earnings_growth"),
    "BLS employment growth": ("bls", "employment_thousands_growth"),
}

PRICE_SHOCK_SCENARIOS: dict[str, str] = {
    "Low inflation": "low_inflation_pp",
    "High inflation": "high_inflation_pp",
    "Severe inflation": "severe_inflation_pp",
    "Mild tariff shock": "mild_tariff_pp",
    "Severe tariff shock": "severe_tariff_pp",
    "Sector tariff shock": "sector_tariff_pp",
}


def forecast_settings(cfg: dict | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    fc = cfg.get("forecast", {})
    return {
        "frequency": fc.get("frequency", "annual_q1"),
        "through": fc.get("through", "2027-Q1"),
        "confidence_level": float(fc.get("confidence_level", 0.80)),
        "min_obs": int(fc.get("min_obs_annual", MIN_OBS_ANNUAL)),
        "order": tuple(fc.get("sarima_order", [1, 0, 1])),
        "seasonal_order": tuple(fc.get("sarima_seasonal_order", [0, 0, 0, 0])),
        "shock_presets": fc.get(
            "shock_presets",
            {
                "low_inflation_pp": -2,
                "high_inflation_pp": 2,
                "severe_inflation_pp": 5,
                "mild_tariff_pp": 0.5,
                "severe_tariff_pp": 2,
                "sector_tariff_pp": 2,
            },
        ),
    }


def next_quarter(period: str) -> str:
    year = int(period[:4])
    quarter = int(period[-1])
    if quarter == 4:
        return f"{year + 1}-Q1"
    return f"{year}-Q{quarter + 1}"


def future_quarters(after: str, through: str) -> list[str]:
    periods: list[str] = []
    current = next_quarter(after)
    while current <= through:
        periods.append(current)
        current = next_quarter(current)
    return periods


def future_annual_period_ends(after_period_end: str, through: str) -> list[str]:
    """Q1→Q1 annual steps: period labels are period_end (e.g. 2026-Q1)."""
    start_year = int(after_period_end[:4]) + 1
    end_year = int(through[:4])
    return [f"{year}-Q1" for year in range(start_year, end_year + 1)]


def _period_index(periods: pd.Index, frequency: Literal["quarterly", "annual_q1"] = "annual_q1") -> pd.Index:
    if frequency == "annual_q1":
        # Integer year index (period_end year); statsmodels forecasts with RangeIndex steps.
        return pd.Index([int(p[:4]) for p in periods], name="year")
    return pd.PeriodIndex([pd.Period(p, freq="Q") for p in periods], freq="Q")


def fit_sarima_series(
    y: pd.Series,
    steps: int,
    *,
    order: tuple[int, int, int] = (1, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    alpha: float = 0.20,
    min_obs: int = MIN_OBS_ANNUAL,
    frequency: Literal["quarterly", "annual_q1"] = "annual_q1",
) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    """Fit SARIMA and return mean forecast plus prediction-interval bounds."""
    clean = y.dropna()
    if len(clean) < min_obs or steps <= 0:
        return None

    if frequency == "quarterly" and not isinstance(clean.index, pd.PeriodIndex):
        clean = clean.copy()
        clean.index = _period_index(clean.index, "quarterly")
    elif frequency == "annual_q1" and not pd.api.types.is_integer_dtype(clean.index):
        clean = clean.copy()
        clean.index = _period_index(clean.index, "annual_q1")

    if frequency == "annual_q1":
        specs = [
            (order, seasonal_order),
            ((1, 0, 0), (0, 0, 0, 0)),
            ((0, 0, 1), (0, 0, 0, 0)),
        ]
    else:
        specs = [
            (order, seasonal_order),
            ((1, 0, 0), (1, 0, 0, 4)),
            ((1, 0, 1), (0, 0, 0, 0)),
        ]

    for ord_spec, seas_spec in specs:
        try:
            model = SARIMAX(
                clean,
                order=ord_spec,
                seasonal_order=seas_spec,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False)
            frame = result.get_forecast(steps=steps).summary_frame(alpha=alpha)
            lower_col = "mean_ci_lower" if "mean_ci_lower" in frame.columns else "mean ci lower"
            upper_col = "mean_ci_upper" if "mean_ci_upper" in frame.columns else "mean ci upper"
            return frame["mean"], frame[lower_col], frame[upper_col]
        except Exception:
            continue
    return None


def apply_pp_shock(values: pd.Series, pp: float) -> pd.Series:
    """Add percentage-point shock to decimal growth rates (2 pp -> +0.02)."""
    return values + (pp / 100.0)


def scenario_pp(
    scenario: str,
    shock_presets: dict[str, float],
    custom_pp: float = 0.0,
) -> float | None:
    if scenario == "Custom shock":
        return custom_pp if custom_pp != 0 else None
    key = PRICE_SHOCK_SCENARIOS.get(scenario)
    if not key:
        return None
    return float(shock_presets.get(key, 0))


def supports_price_scenarios(metric_label: str) -> bool:
    return metric_label == "BEA price growth"


def _shock_unit_label(settings: dict[str, Any] | None = None) -> str:
    settings = settings or forecast_settings()
    if settings.get("frequency") == "annual_q1":
        return "pp/yr"
    return "pp/qtr"


def scenario_option_label(scenario: str, shock_presets: dict[str, float] | None = None) -> str:
    """Human-readable multiselect label: shock size vs SARIMA baseline (not a separate model)."""
    shock_presets = shock_presets or forecast_settings()["shock_presets"]
    unit = _shock_unit_label()
    labels = {
        "Low inflation": f"Low inflation ({shock_presets['low_inflation_pp']:+.1f} {unit} vs baseline)",
        "High inflation": f"High inflation (+{shock_presets['high_inflation_pp']:.1f} {unit} vs baseline)",
        "Severe inflation": f"Severe inflation (+{shock_presets['severe_inflation_pp']:.1f} {unit} vs baseline)",
        "Mild tariff shock": f"Mild tariff shock (+{shock_presets['mild_tariff_pp']:.1f} {unit} vs baseline)",
        "Severe tariff shock": f"Severe tariff shock (+{shock_presets['severe_tariff_pp']:.1f} {unit} vs baseline)",
        "Sector tariff shock": (
            f"Sector tariff shock (+{shock_presets['sector_tariff_pp']:.1f} {unit} vs baseline, one sector only)"
        ),
        "Custom shock": f"Custom shock (slider value, {unit} vs baseline)",
    }
    return labels.get(scenario, scenario)


def _effective_annual_train_end(history: pd.DataFrame, train_end: str) -> str:
    """Latest Q1 period_end in history that does not exceed train_end."""
    if history.empty:
        return train_end
    q1 = history[history["period"].str.endswith("-Q1", na=False)]
    if q1.empty:
        return train_end
    eligible = q1[q1["period"] <= train_end]
    if eligible.empty:
        return q1["period"].min()
    return eligible["period"].max()


def forecast_one_sector(
    history: pd.DataFrame,
    industry_name: str,
    value_col: str,
    train_end: str,
    *,
    settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return historical + forecast rows for one sector (annual Q1→Q1 by default)."""
    settings = settings or forecast_settings()
    frequency = settings.get("frequency", "annual_q1")
    sub = history[history["industry_name"] == industry_name].sort_values("period")
    if sub.empty or value_col not in sub.columns:
        return pd.DataFrame()

    effective_end = _effective_annual_train_end(sub, train_end) if frequency == "annual_q1" else train_end
    train = sub[sub["period"] <= effective_end].dropna(subset=[value_col])
    if frequency == "annual_q1":
        future = future_annual_period_ends(effective_end, settings["through"])
    else:
        future = future_quarters(effective_end, settings["through"])
    if not future:
        return pd.DataFrame()

    alpha = 1.0 - settings["confidence_level"]
    fit = fit_sarima_series(
        train.set_index("period")[value_col],
        steps=len(future),
        order=settings["order"],
        seasonal_order=settings["seasonal_order"],
        alpha=alpha,
        min_obs=settings["min_obs"],
        frequency=frequency,
    )

    records: list[dict] = []
    for _, row in train.iterrows():
        records.append(
            {
                "period": row["period"],
                "industry_name": industry_name,
                "actual": float(row[value_col]),
                "forecast": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "is_forecast": False,
            }
        )

    if fit is None:
        return pd.DataFrame(records)

    mean, lower, upper = fit
    for period, fval, lo, hi in zip(future, mean, lower, upper, strict=True):
        records.append(
            {
                "period": period,
                "industry_name": industry_name,
                "actual": np.nan,
                "forecast": float(fval),
                "lower": float(lo),
                "upper": float(hi),
                "is_forecast": True,
            }
        )
    return pd.DataFrame(records)


def build_sector_forecasts(
    history: pd.DataFrame,
    sectors: list[str],
    value_col: str,
    train_end: str,
    *,
    settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    frames = [
        forecast_one_sector(history, name, value_col, train_end, settings=settings)
        for name in sectors
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_scenario_column(
    df: pd.DataFrame,
    scenario: str,
    shock_presets: dict[str, float],
    *,
    custom_pp: float = 0.0,
    target_sector: str | None = None,
) -> pd.DataFrame:
    """Add a scenario overlay column (forecast + pp shock on forecast window)."""
    pp = scenario_pp(scenario, shock_presets, custom_pp=custom_pp)
    if pp is None:
        return df

    col = f"scenario_{scenario.lower().replace(' ', '_')}"
    out = df.copy()
    mask = out["is_forecast"] & out["forecast"].notna()
    if scenario == "Sector tariff shock" and target_sector:
        mask &= out["industry_name"] == target_sector
    out[col] = np.nan
    out.loc[mask, col] = apply_pp_shock(out.loc[mask, "forecast"], pp)
    return out
