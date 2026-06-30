"""Tests for SARIMA forecasting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.forecasting import (
    MIN_OBS_ANNUAL,
    add_scenario_column,
    apply_pp_shock,
    build_sector_forecasts,
    fit_sarima_series,
    forecast_settings,
    future_annual_period_ends,
    future_quarters,
    next_quarter,
)


def test_next_quarter_and_future():
    assert next_quarter("2019-Q4") == "2020-Q1"
    assert next_quarter("2020-Q1") == "2020-Q2"
    future = future_quarters("2026-Q1", "2027-Q4")
    assert future[0] == "2026-Q2"
    assert future[-1] == "2027-Q4"
    assert len(future) == 7


def test_future_annual_period_ends():
    future = future_annual_period_ends("2025-Q1", "2027-Q1")
    assert future == ["2026-Q1", "2027-Q1"]


def test_apply_pp_shock():
    s = pd.Series([0.01, 0.02])
    out = apply_pp_shock(s, 2.0)
    assert np.isclose(out.iloc[0], 0.03)
    assert np.isclose(out.iloc[1], 0.04)


def test_fit_sarima_synthetic_annual_series():
    rng = np.random.default_rng(42)
    years = list(range(2010, 2026))
    y = 0.03 + 0.005 * np.sin(2 * np.pi * np.arange(len(years)) / 5) + rng.normal(0, 0.002, len(years))
    periods = [f"{y}-Q1" for y in years]
    series = pd.Series(y, index=periods)

    result = fit_sarima_series(
        series,
        steps=2,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        alpha=0.20,
        min_obs=MIN_OBS_ANNUAL,
        frequency="annual_q1",
    )
    assert result is not None
    mean, lower, upper = result
    assert len(mean) == 2
    assert (lower <= mean).all()
    assert (mean <= upper).all()


def test_build_sector_forecasts_annual_history():
    rows = []
    for y in range(2019, 2026):
        rows.append(
            {
                "period": f"{y + 1}-Q1",
                "industry_name": "Mining",
                "price_growth": 0.02 + 0.001 * (y - 2019),
            }
        )
    history = pd.DataFrame(rows)
    settings = forecast_settings()
    settings["through"] = "2027-Q1"
    out = build_sector_forecasts(history, ["Mining"], "price_growth", "2025-Q1", settings=settings)
    assert not out.empty
    assert out["is_forecast"].any()
    assert out.loc[out["is_forecast"], "period"].max() == "2027-Q1"
    fc = out[out["is_forecast"]]
    assert fc["lower"].notna().any()
    assert (fc["lower"] <= fc["forecast"]).all()
    assert (fc["forecast"] <= fc["upper"]).all()


def test_add_scenario_column_sector_tariff():
    df = pd.DataFrame(
        [
            {"period": "2027-Q1", "industry_name": "A", "forecast": 0.02, "is_forecast": True},
            {"period": "2027-Q1", "industry_name": "B", "forecast": 0.03, "is_forecast": True},
        ]
    )
    presets = {"sector_tariff_pp": 2.0}
    out = add_scenario_column(df, "Sector tariff shock", presets, target_sector="A")
    col = "scenario_sector_tariff_shock"
    assert np.isclose(out.loc[out["industry_name"] == "A", col].iloc[0], 0.04)
    assert pd.isna(out.loc[out["industry_name"] == "B", col].iloc[0])
