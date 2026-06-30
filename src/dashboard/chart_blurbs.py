"""One-line reading guides for dashboard chart types."""

from __future__ import annotations

HORIZON_READING = {
    "quarterly": "Horizon: quarter-to-quarter — typical short-run volatility.",
    "yearly_q1": "Horizon: Q1→Q1 yearly — one annual step per year in range.",
    "endpoint": "Horizon: endpoint — total % change from start quarter to end quarter (e.g. 0.44 = 44%).",
}

CHART_BLURBS: dict[str, str] = {
    "BEA ΔQ/ΔP decomposition": (
        "Shows whether activity changed because the sector produced more (quantity) "
        "or charged more (price)—the core split between real expansion and industry inflation."
    ),
    "Q/P ratio (faceted)": (
        "Shows whether each quarter was quantity-led or price-led (Q/P = %ΔQ ÷ %ΔP). "
        "Large spikes often mean price barely moved, not that the sector exploded."
    ),
    "Wage minus price spread": (
        "Ranks sectors by wage growth minus output price growth: positive means wages "
        "outpaced BEA prices; negative means prices rose faster than wages."
    ),
    "Wage vs price (grouped bars)": (
        "Compares BLS wage growth to BEA output price growth side by side—did labor "
        "costs and output prices move together or diverge?"
    ),
    "Scatter (BEA price vs quantity)": (
        "Maps each sector on price vs quantity growth: expansion (both up), price-led "
        "inflation, quantity-led growth, or contraction."
    ),
    "SARIMA forecast": (
        "Dotted line = SARIMA baseline on Q1→Q1 annual growth (% change from one Q1 to the next). "
        "Shaded band = 80% prediction interval (statistical uncertainty). "
        "Dashed shock lines = baseline plus percentage points per year—not a re-fit."
    ),
}

SINGLE_METRIC_BLURBS: dict[str, str] = {
    "BEA price growth": "Industry-specific output price inflation (BEA deflator)—not the same as CPI.",
    "BEA quantity growth": "Real output expansion or contraction for the sector.",
    "BEA Q/P ratio": "Ratio of quantity growth to price growth; unstable when |ΔP| is near zero.",
    "BLS wage growth": "Change in average hourly earnings—labor cost to employers.",
    "BLS employment growth": "Hiring vs layoffs; not the same as output growth.",
}

COMPARISON_BLURBS: dict[str, str] = {
    "Price vs wage": "Do output prices and wages move together? Points above the diagonal mean wages grew faster.",
    "Quantity vs employment": "Did output grow without adding jobs (productivity) or did hiring track the boom?",
}


def chart_reading_guide(
    chart_mode: str,
    metric_a: str = "",
    metric_b: str | None = None,
    horizon: str = "quarterly",
) -> str:
    """Return a short how-to-read blurb for the current chart selection."""
    parts: list[str] = []

    if chart_mode in CHART_BLURBS:
        parts.append(CHART_BLURBS[chart_mode])
    elif chart_mode == "Single metric" and metric_a in SINGLE_METRIC_BLURBS:
        parts.append(SINGLE_METRIC_BLURBS[metric_a])
    elif chart_mode == "BEA vs BLS comparison" and metric_b in COMPARISON_BLURBS:
        parts.append(COMPARISON_BLURBS[metric_b])

    if horizon in HORIZON_READING:
        parts.append(HORIZON_READING[horizon])

    parts.append("Tip: hover any point for full sector names; growth rates show as % in the tooltip.")
    return " ".join(parts)
