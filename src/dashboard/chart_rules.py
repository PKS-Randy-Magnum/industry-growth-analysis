"""Chart type horizon compatibility for the Streamlit dashboard."""

from __future__ import annotations

HORIZON_LABELS = {
    "quarterly": "Quarter to quarter",
    "yearly_q1": "Year to year (Q1->Q1)",
    "endpoint": "Endpoint (total change)",
}

LOCKED_HORIZON: dict[str, str | None] = {
    "Wage minus price spread": "endpoint",
    "Wage vs price (grouped bars)": "endpoint",
    "Q/P ratio (faceted)": "quarterly",
    "SARIMA forecast": "yearly_q1",
}

LOCKED_HORIZON_CAPTION: dict[str, str] = {
    "endpoint": "Total change from start to end quarter (horizon fixed).",
    "quarterly": "Quarterly Q/P only - one mini-chart per sector (horizon fixed).",
}

FORECAST_HORIZON_CAPTION = (
    "SARIMA forecasts use Q1→Q1 annual growth through 2027-Q1 (horizon fixed)."
)

MULTI_LINE_CHARTS = frozenset(
    {
        "BEA ΔQ/ΔP decomposition",
        "Single metric",
        "BEA vs BLS comparison",
    }
)

SECTOR_WARN_THRESHOLD = 10


def effective_horizon(chart_mode: str, user_horizon: str) -> str:
    locked = LOCKED_HORIZON.get(chart_mode)
    return locked if locked else user_horizon


def horizon_is_locked(chart_mode: str) -> bool:
    return chart_mode in LOCKED_HORIZON


def locked_horizon_caption(chart_mode: str) -> str | None:
    locked = LOCKED_HORIZON.get(chart_mode)
    if not locked:
        return None
    if chart_mode == "SARIMA forecast":
        return FORECAST_HORIZON_CAPTION
    return LOCKED_HORIZON_CAPTION.get(locked)


def should_warn_many_sectors(chart_mode: str, n_sectors: int, metric_a: str = "") -> bool:
    if n_sectors <= SECTOR_WARN_THRESHOLD:
        return False
    if chart_mode in {"Q/P ratio (faceted)", "Scatter (BEA price vs quantity)"}:
        return False
    if chart_mode in {"Wage minus price spread", "Wage vs price (grouped bars)"}:
        return False
    if chart_mode == "SARIMA forecast" and n_sectors > 5:
        return True
    if chart_mode == "Single metric" and metric_a == "BEA Q/P ratio":
        return True
    return chart_mode in MULTI_LINE_CHARTS


def supports_show_events(chart_mode: str, metric_a: str = "", horizon: str = "quarterly") -> bool:
    """Event bands apply to time-series charts only."""
    if chart_mode in {"Wage minus price spread", "Wage vs price (grouped bars)"}:
        return False
    if chart_mode == "Scatter (BEA price vs quantity)":
        return False
    if chart_mode == "BEA ΔQ/ΔP decomposition" and horizon == "endpoint":
        return False
    if chart_mode == "Single metric" and horizon == "endpoint":
        return False
    if chart_mode == "BEA vs BLS comparison" and horizon == "endpoint":
        return False
    return True


def supports_show_labels(
    chart_mode: str,
    metric_a: str = "",
    horizon: str = "quarterly",
) -> bool:
    """On-chart sector labels apply to scatters only."""
    if chart_mode == "Scatter (BEA price vs quantity)":
        return True
    if chart_mode == "BEA vs BLS comparison" and horizon == "endpoint":
        return True
    return False
