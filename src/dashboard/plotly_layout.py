"""Plotly layout helpers: responsive margins and event-period overlays."""

from __future__ import annotations

import math

import plotly.graph_objects as go

from src.analysis.industry_filters import load_config

EVENT_COLORS = {
    "covid_shock": ("rgba(255, 204, 204, 0.35)", "COVID shock"),
    "recovery": ("rgba(232, 244, 232, 0.25)", "Recovery"),
    "tariffs": ("rgba(255, 243, 205, 0.3)", "Tariffs"),
}


def legend_layout(n_traces: int, *, base_bottom: int = 80) -> dict:
    n_rows = max(1, math.ceil(n_traces / 4))
    extra = 28 * n_rows
    return {
        "margin": {"b": base_bottom + extra, "t": 80, "l": 50, "r": 30},
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.12 - 0.04 * n_rows,
            "x": 0,
            "font": {"size": 9},
        },
    }


def faceted_height(n_sectors: int, ncols: int = 3, row_h: int = 240) -> int:
    nrows = int(math.ceil(max(1, n_sectors) / ncols))
    return row_h * nrows + 140


def add_event_bands_categorical(fig: go.Figure, periods: list[str], cfg: dict | None = None) -> go.Figure:
    if not periods:
        return fig
    cfg = cfg or load_config()
    sorted_periods = sorted(set(periods))

    for name, spec in cfg.get("event_periods", {}).items():
        start = spec.get("start")
        end = spec.get("end") or sorted_periods[-1]
        if start not in sorted_periods:
            continue
        if end not in sorted_periods:
            end = sorted_periods[-1]
        color, label = EVENT_COLORS.get(name, ("rgba(238, 238, 238, 0.2)", name))
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            layer="below",
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font_size=8,
            annotation_font_color="#888888",
        )
    return fig


def scatter_quadrant_hints(fig: go.Figure) -> go.Figure:
    hints = [
        (0.98, 0.98, "Expansion", "top right"),
        (0.98, 0.02, "Price-led", "bottom right"),
        (0.02, 0.98, "Qty-led", "top left"),
        (0.02, 0.02, "Contraction", "bottom left"),
    ]
    for x, y, text, pos in hints:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font={"size": 9, "color": "rgba(120,120,120,0.55)"},
            xanchor="right" if "right" in pos else "left",
            yanchor="top" if "top" in pos else "bottom",
        )
    return fig


def add_equal_growth_line(fig: go.Figure, x_vals: list[float], y_vals: list[float]) -> go.Figure:
    finite = [(x, y) for x, y in zip(x_vals, y_vals) if x == x and y == y]
    if not finite:
        return fig
    lo = min(min(x for x, _ in finite), min(y for _, y in finite))
    hi = max(max(x for x, _ in finite), max(y for _, y in finite))
    pad = (hi - lo) * 0.05 if hi > lo else 0.05
    lo -= pad
    hi += pad
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Equal growth",
            line={"dash": "dot", "color": "#999999", "width": 1},
            hoverinfo="skip",
        )
    )
    return fig
