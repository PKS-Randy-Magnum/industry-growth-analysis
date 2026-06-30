"""Interactive industry growth & inflation dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.forecasting import (
    PRICE_SHOCK_SCENARIOS,
    add_scenario_column,
    forecast_settings,
    scenario_option_label,
    supports_price_scenarios,
)
from src.analysis.industry_filters import PROFILE_LABELS, load_config
from src.dashboard.chart_rules import (
    HORIZON_LABELS,
    effective_horizon,
    horizon_is_locked,
    locked_horizon_caption,
    should_warn_many_sectors,
    supports_show_events,
    supports_show_labels,
)
from src.dashboard.chart_blurbs import chart_reading_guide
from src.dashboard.charts import chart_for_selection
from src.dashboard.data import available_periods, build_forecast_panels, build_panels, load_profile_data
from src.dashboard.sector_picker import render_sector_picker

st.set_page_config(page_title="Industry Growth & Inflation", layout="wide")
st.title("Industry Growth & Inflation")
st.caption("Explore BEA price/quantity decomposition and BLS labor data across horizons.")

cfg = load_config()
sector_options = cfg["sector_plot"]
default_sectors = list(sector_options)

CHART_TYPES = [
    "BEA ΔQ/ΔP decomposition",
    "Q/P ratio (faceted)",
    "Wage minus price spread",
    "Wage vs price (grouped bars)",
    "Single metric",
    "BEA vs BLS comparison",
    "Scatter (BEA price vs quantity)",
    "SARIMA forecast",
]

FORECAST_METRIC_OPTIONS = [
    "BEA price growth",
    "BEA quantity growth",
    "BLS wage growth",
    "BLS employment growth",
]

with st.sidebar:
    st.header("Controls")
    profile = st.selectbox(
        "Profile",
        options=["excl_trust_funds", "full"],
        format_func=lambda p: PROFILE_LABELS.get(p, p),
        index=0,
    )
    st.caption(
        "Only affects trust funds and economy-wide aggregates. "
        "The 15 sector lines are identical across profiles."
    )

    @st.cache_data
    def cached_periods(profile_name: str) -> list[str]:
        raw = load_profile_data(profile_name)
        return available_periods(raw["bea_quarterly"])

    periods = cached_periods(profile)
    if not periods:
        st.error("No snapshot data found. Run `python run.py` first to build `data/snapshots/`.")
        st.stop()

    start = st.selectbox("Start quarter", periods, index=0)
    end = st.selectbox("End quarter", periods, index=len(periods) - 1)
    if start > end:
        st.warning("Start is after end; swapping.")
        start, end = end, start

    chart_mode = st.selectbox("Chart type", CHART_TYPES)

    horizon_options = ["quarterly", "yearly_q1", "endpoint"]
    locked = horizon_is_locked(chart_mode)
    if locked:
        locked_val = effective_horizon(chart_mode, "quarterly")
        st.radio(
            "Horizon",
            options=horizon_options,
            index=horizon_options.index(locked_val),
            format_func=lambda h: HORIZON_LABELS[h],
            disabled=True,
        )
        caption = locked_horizon_caption(chart_mode)
        if caption:
            st.caption(caption)
        horizon = locked_val
    else:
        horizon = st.radio(
            "Horizon",
            options=horizon_options,
            format_func=lambda h: HORIZON_LABELS[h],
        )

    selected_sectors = render_sector_picker(profile, default_sectors=default_sectors)
    if not selected_sectors:
        st.warning("Select at least one sector.")
        st.stop()

    metric_a = ""
    metric_b = None
    clip_qp = True
    show_confidence_band = True
    show_scenario_overlays = False
    forecast_scenarios: list[str] = []
    custom_shock_pp = 0.0
    tariff_sector: str | None = None

    if chart_mode == "Single metric":
        metric_a = st.selectbox(
            "Metric",
            [
                "BEA price growth",
                "BEA quantity growth",
                "BEA Q/P ratio",
                "BLS wage growth",
                "BLS employment growth",
            ],
        )
        if metric_a == "BEA Q/P ratio":
            clip_qp = st.checkbox("Clip Q/P y-axis (±10)", value=True)
    elif chart_mode == "BEA vs BLS comparison":
        metric_a = "comparison"
        metric_b = st.selectbox("Comparison", ["Price vs wage", "Quantity vs employment"])
    elif chart_mode == "Scatter (BEA price vs quantity)":
        metric_a = "scatter"
    elif chart_mode == "Q/P ratio (faceted)":
        clip_qp = st.checkbox("Clip Q/P y-axis (±10)", value=True)
    elif chart_mode == "SARIMA forecast":
        metric_a = st.selectbox("Forecast metric", FORECAST_METRIC_OPTIONS)
        st.caption(
            f"Training window: {start} → {end} (Q1→Q1 annual steps). "
            f"Forecast through {forecast_settings()['through']}."
        )
        show_confidence_band = st.checkbox(
            "Show confidence band (80% PI)",
            value=True,
            help="Shaded band around the dotted SARIMA baseline (lower/upper statistical bounds).",
        )
        price_scenarios_ok = supports_price_scenarios(metric_a)
        show_scenario_overlays = st.checkbox(
            "Show scenario overlays",
            value=False,
            disabled=not price_scenarios_ok,
            help=(
                "Dashed lines only: each year = SARIMA baseline + a fixed bump (pp). "
                "Not a separate model and not economy-wide CPI."
            ),
        )
        if not price_scenarios_ok:
            show_scenario_overlays = False
        if show_scenario_overlays:
            shock_presets = forecast_settings()["shock_presets"]
            st.caption(
                "**Baseline** = dotted SARIMA forecast. **Shock overlays** add percentage points "
                "(pp) per **year** to that baseline—e.g. High inflation (+2 pp) turns a 3.0% annual forecast into 5.0%."
            )
            scenario_options = list(PRICE_SHOCK_SCENARIOS.keys())
            if len(selected_sectors) != 1:
                scenario_options = [s for s in scenario_options if s != "Sector tariff shock"]
            forecast_scenarios = st.multiselect(
                "Shock scenarios (vs SARIMA baseline)",
                options=scenario_options,
                default=["High inflation"] if "High inflation" in scenario_options else [],
                format_func=lambda s: scenario_option_label(s, shock_presets),
            )
            custom_shock_pp = st.slider(
                "Custom price shock (percentage points per year)",
                min_value=-5.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                help=(
                    "pp = percentage points added to the baseline annual forecast each year "
                    "(+2 pp on a 3.0% forecast → 5.0%). Does not re-fit SARIMA."
                ),
            )
            if custom_shock_pp != 0.0:
                forecast_scenarios = list(forecast_scenarios) + ["Custom shock"]
            if "Sector tariff shock" in forecast_scenarios and len(selected_sectors) == 1:
                tariff_sector = selected_sectors[0]
            elif "Sector tariff shock" in forecast_scenarios:
                tariff_sector = st.selectbox("Sector for tariff shock", selected_sectors)

    plot_sectors = selected_sectors
    if should_warn_many_sectors(chart_mode, len(selected_sectors), metric_a):
        st.info(
            "Many sectors selected — line charts can get crowded. "
            "Try Q/P faceted or fewer sectors for readability."
        )

    events_ok = supports_show_events(chart_mode, metric_a, horizon)
    labels_ok = supports_show_labels(chart_mode, metric_a, horizon)
    default_labels = labels_ok and len(plot_sectors) <= 8

    show_events = st.checkbox(
        "Show event periods",
        value=True,
        disabled=not events_ok,
        help="COVID / recovery / tariff bands on time-series charts only.",
    )
    show_labels = st.checkbox(
        "Show sector labels",
        value=default_labels,
        disabled=not labels_ok,
        help="Point labels on scatter charts only (hover always has full names).",
    )
    if not events_ok:
        show_events = False
    if not labels_ok:
        show_labels = False

    st.divider()
    st.markdown("**Tip:** Run `python run.py --refresh` when new quarters publish.")

chart_horizon = effective_horizon(chart_mode, horizon)


@st.cache_data
def cached_panels(profile_name: str, start_q: str, end_q: str, horizon_name: str) -> dict:
    return build_panels(profile_name, start_q, end_q, horizon_name)


@st.cache_data
def cached_forecast(
    profile_name: str,
    start_q: str,
    end_q: str,
    metric_label: str,
    sectors_key: tuple[str, ...],
) -> pd.DataFrame:
    return build_forecast_panels(profile_name, start_q, end_q, metric_label, list(sectors_key))


panels = cached_panels(profile, start, end, horizon)

forecast_df = pd.DataFrame()
if chart_mode == "SARIMA forecast":
    forecast_df = cached_forecast(profile, start, end, metric_a, tuple(plot_sectors))
    if forecast_df.empty:
        st.warning(
            "Not enough annual Q1→Q1 data to fit SARIMA for this selection "
            f"(need ≥{forecast_settings()['min_obs']} years per sector)."
        )
    else:
        shock_presets = forecast_settings()["shock_presets"]
        for scenario in forecast_scenarios:
            forecast_df = add_scenario_column(
                forecast_df,
                scenario,
                shock_presets,
                custom_pp=custom_shock_pp,
                target_sector=tariff_sector,
            )

fig = chart_for_selection(
    panels,
    chart_horizon,
    chart_mode,
    metric_a,
    metric_b,
    plot_sectors,
    start,
    end,
    clip_qp=clip_qp,
    show_events=show_events,
    show_labels=show_labels,
    forecast_df=forecast_df,
    show_confidence_band=show_confidence_band,
    show_scenario_overlays=show_scenario_overlays,
    forecast_scenarios=forecast_scenarios,
)

PLOTLY_CONFIG = {
    "responsive": True,
    "scrollZoom": True,
    "displaylogo": False,
}

with st.container():
    reading = chart_reading_guide(chart_mode, metric_a, metric_b, chart_horizon)
    if reading:
        st.info(reading)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

is_qp_chart = chart_mode == "Q/P ratio (faceted)" or (
    chart_mode == "Single metric" and metric_a == "BEA Q/P ratio"
)
if is_qp_chart:
    threshold = cfg.get("qp_unstable_price_threshold", 0.005)
    st.caption(f"Ratio is unstable when |ΔP| is near zero (threshold {threshold:g}).")

col_bea, col_bls = st.columns(2)


def _filter_export(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "industry_name" in out.columns:
        out = out[out["industry_name"].isin(selected_sectors)]
    elif "line_id" in out.columns and not panels.get("bea_quarterly", pd.DataFrame()).empty:
        meta = panels["bea_quarterly"][["line_id", "industry_name"]].drop_duplicates("line_id")
        ids = meta.loc[meta["industry_name"].isin(selected_sectors), "line_id"]
        out = out[out["line_id"].isin(ids)]
    if "period" in out.columns:
        out = out[(out["period"] >= start) & (out["period"] <= end)]
    return out


bea_export = _filter_export(panels.get("bea_quarterly", pd.DataFrame()))
bls_export = _filter_export(panels.get("bls_quarterly", pd.DataFrame()))

with col_bea:
    st.download_button(
        "Download BEA CSV",
        bea_export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"bea_{profile}_{start}_{end}.csv",
        mime="text/csv",
        disabled=bea_export.empty,
    )
with col_bls:
    st.download_button(
        "Download BLS CSV",
        bls_export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"bls_{profile}_{start}_{end}.csv",
        mime="text/csv",
        disabled=bls_export.empty,
    )

with st.expander("Data preview (BEA)"):
    preview_key = {"quarterly": "bea_quarterly", "endpoint": "bea_endpoint", "yearly_q1": "bea_yearly"}[
        chart_horizon
    ]
    preview = panels.get(preview_key, panels.get("bea_quarterly", pd.DataFrame()))
    if not preview.empty:
        st.dataframe(preview[preview["industry_name"].isin(selected_sectors)].head(100), use_container_width=True)
    else:
        st.write("No rows for this selection.")
