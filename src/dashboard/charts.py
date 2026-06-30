"""Plotly chart builders for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis.export_utils import qp_sign_label
from src.analysis.forecasting import forecast_settings, scenario_option_label
from src.analysis.plot_colors import get_sector_color
from src.analysis.plot_style import format_chart_title, short_sector_name
from src.analysis.timeseries_plots import (
    _merge_bea_bls_endpoint,
    _merge_bea_bls_quarterly,
    _merge_bea_bls_yearly,
)
from src.dashboard.data import build_scatter_frame, scatter_subtitle
from src.dashboard.plotly_layout import (
    add_equal_growth_line,
    add_event_bands_categorical,
    faceted_height,
    legend_layout,
    scatter_quadrant_hints,
)

QP_LINE_CLIP = 10.0
_LABEL_OFFSETS = (-40, -20, 20, 40)
_LABEL_PROXIMITY = 0.03


def _sector_trace_color(name: str, idx: int) -> str:
    return get_sector_color(name, idx)


def _title_layout(title: str, subtitle: str) -> dict:
    return {"title": {"text": f"<b>{title}</b><br><sup>{subtitle}</sup>", "x": 0}}


def _periods_from_df(df: pd.DataFrame, x_col: str) -> list[str]:
    if df.empty or x_col not in df.columns:
        return []
    return sorted(df[x_col].dropna().unique().tolist())


def _apply_legend(fig: go.Figure, n_traces: int) -> None:
    fig.update_layout(
        **legend_layout(n_traces),
        autosize=True,
        template="plotly_white",
        hoverlabel={"namelength": -1, "font_size": 12},
    )


def _growth_hover_value(col: str) -> str:
    """Plotly hover format: percent for growth rates, decimal for ratios."""
    if col == "qp_ratio":
        return "%{y:.3f}"
    return "%{y:.1%}"


def _staggered_label_offsets(xs: list[float], ys: list[float]) -> list[int]:
    offsets: list[int] = []
    placed: list[tuple[float, float, int]] = []
    for x, y in zip(xs, ys):
        ay = _LABEL_OFFSETS[len(offsets) % len(_LABEL_OFFSETS)]
        for px, py, pay in placed:
            if abs(x - px) < _LABEL_PROXIMITY and abs(y - py) < _LABEL_PROXIMITY:
                ay = _LABEL_OFFSETS[(len(offsets) + 2) % len(_LABEL_OFFSETS)]
                break
        offsets.append(ay)
        placed.append((x, y, ay))
    return offsets


def line_chart(
    df: pd.DataFrame,
    y_col: str,
    sectors: list[str],
    title: str,
    subtitle: str,
    *,
    x_col: str = "period",
    y_clip: float | None = None,
    show_events: bool = False,
) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig

    for idx, name in enumerate(sectors):
        g = df[df["industry_name"] == name].sort_values(x_col)
        if g.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=g[x_col],
                y=g[y_col],
                mode="lines+markers",
                name=name,
                line={"color": _sector_trace_color(name, idx), "width": 2},
                marker={"size": 5},
                hovertemplate=(
                    f"<b>{name}</b><br>%{{x}}<br>"
                    f"{y_col.replace('_', ' ')}: {_growth_hover_value(y_col)}<extra></extra>"
                ),
            )
        )

    if y_clip is not None:
        fig.update_yaxes(range=[-y_clip, y_clip])
    elif y_col != "qp_ratio":
        fig.update_yaxes(tickformat=".0%")

    fig.update_layout(
        **_title_layout(title, subtitle),
        xaxis_title="",
        yaxis_title=y_col.replace("_", " "),
        hovermode="x unified",
        height=560,
    )
    _apply_legend(fig, len(fig.data))
    fig.add_hline(y=0, line_dash="dash", line_color="#aaaaaa")

    if show_events:
        add_event_bands_categorical(fig, _periods_from_df(df, x_col))
    return fig


def dual_line_chart(
    merged: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_a: str,
    label_b: str,
    sectors: list[str],
    title: str,
    subtitle: str,
    *,
    x_col: str = "period",
    show_events: bool = False,
) -> go.Figure:
    fig = go.Figure()
    if merged.empty:
        return fig

    for idx, name in enumerate(sectors):
        g = merged[merged["industry_name"] == name].sort_values(x_col)
        if g.empty:
            continue
        color = _sector_trace_color(name, idx)
        if col_a in g.columns:
            fig.add_trace(
                go.Scatter(
                    x=g[x_col],
                    y=g[col_a],
                    mode="lines",
                    name=f"{name} ({label_a})",
                    line={"color": color, "width": 2},
                    legendgroup=name,
                    hovertemplate=f"<b>{name}</b><br>%{{x}}<br>{label_a}: %{{y:.1%}}<extra></extra>",
                )
            )
        if col_b in g.columns:
            fig.add_trace(
                go.Scatter(
                    x=g[x_col],
                    y=g[col_b],
                    mode="lines",
                    name=f"{name} ({label_b})",
                    line={"color": color, "width": 2, "dash": "dash"},
                    legendgroup=name,
                    showlegend=True,
                    hovertemplate=f"<b>{name}</b><br>%{{x}}<br>{label_b}: %{{y:.1%}}<extra></extra>",
                )
            )

    fig.update_layout(
        **_title_layout(title, subtitle),
        hovermode="x unified",
        height=580,
    )
    _apply_legend(fig, len(fig.data))
    fig.add_hline(y=0, line_dash="dash", line_color="#aaaaaa")
    fig.update_yaxes(tickformat=".0%")

    if show_events:
        add_event_bands_categorical(fig, _periods_from_df(merged, x_col))
    return fig


def _dedupe_sectors(df: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    plot_df = df[df["industry_name"].isin(sectors)].copy()
    if plot_df.empty:
        return plot_df
    subset = "line_id" if "line_id" in plot_df.columns else "industry_name"
    return plot_df.drop_duplicates(subset)


def scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sectors: list[str],
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    *,
    show_labels: bool = True,
    equal_growth_line: bool = False,
    quadrant_hints: bool = False,
) -> go.Figure:
    fig = go.Figure()
    plot_df = _dedupe_sectors(df, sectors)
    if plot_df.empty:
        return fig

    xs: list[float] = []
    ys: list[float] = []
    names: list[str] = []
    colors: list[str] = []

    for i, (_, row) in enumerate(plot_df.iterrows()):
        name = row["industry_name"]
        x_val = row[x_col]
        y_val = row[y_col]
        if pd.isna(x_val) or pd.isna(y_val):
            continue
        xs.append(float(x_val))
        ys.append(float(y_val))
        names.append(name)
        colors.append(_sector_trace_color(name, i))

    if not xs:
        return fig

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            name="Sectors",
            marker={"size": 10, "color": colors},
            text=names,
            hovertemplate="%{text}<br>"
            + f"{x_label}: %{{x:.1%}}<br>"
            + f"{y_label}: %{{y:.1%}}<extra></extra>",
        )
    )

    if show_labels and len(names) > 10:
        show_labels = False

    if show_labels:
        offsets = _staggered_label_offsets(xs, ys)
        for x, y, name, ay in zip(xs, ys, names, offsets):
            fig.add_annotation(
                x=x,
                y=y,
                text=short_sector_name(name, max_len=28),
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                arrowwidth=1,
                arrowcolor="#888888",
                ax=0,
                ay=ay,
                font={"size": 9},
                bgcolor="rgba(255,255,255,0.7)",
                borderpad=2,
            )

    if equal_growth_line:
        add_equal_growth_line(fig, xs, ys)
    if quadrant_hints:
        scatter_quadrant_hints(fig)

    fig.update_layout(
        **_title_layout(title, subtitle),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=560,
        autosize=True,
        margin={"l": 60, "r": 40, "t": 80, "b": 60},
        template="plotly_white",
        showlegend=equal_growth_line,
        hoverlabel={"namelength": -1, "font_size": 12},
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    fig.add_hline(y=0, line_dash="dash", line_color="#aaaaaa")
    fig.add_vline(x=0, line_dash="dash", line_color="#aaaaaa")
    return fig


def decomposition_chart(
    df: pd.DataFrame,
    sectors: list[str],
    title: str,
    subtitle: str,
    *,
    x_col: str = "period",
    show_events: bool = False,
) -> go.Figure:
    return dual_line_chart(
        df,
        "quantity_growth",
        "price_growth",
        "quantity (%ΔQ)",
        "price (%ΔP)",
        sectors,
        title,
        subtitle,
        x_col=x_col,
        show_events=show_events,
    )


def endpoint_decomposition_chart(
    df: pd.DataFrame,
    sectors: list[str],
    title: str,
    subtitle: str,
) -> go.Figure:
    """Grouped bars for total-change quantity vs price (one row per sector)."""
    fig = go.Figure()
    plot_df = _dedupe_sectors(df, sectors)
    if plot_df.empty:
        return fig

    order = {name: i for i, name in enumerate(sectors)}
    plot_df = plot_df.assign(_order=plot_df["industry_name"].map(order)).sort_values("_order")
    labels = [short_sector_name(n) for n in plot_df["industry_name"]]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=plot_df["quantity_growth"],
            name="quantity (%ΔQ)",
            marker_color="#4477AA",
            customdata=plot_df["industry_name"],
            hovertemplate="<b>%{customdata}</b><br>quantity: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=plot_df["price_growth"],
            name="price (%ΔP)",
            marker_color="#CC6677",
            customdata=plot_df["industry_name"],
            hovertemplate="<b>%{customdata}</b><br>price: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        **_title_layout(title, subtitle),
        barmode="group",
        xaxis_tickangle=-45,
        height=max(560, len(labels) * 32 + 140),
        autosize=True,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25, "x": 0},
        margin={"b": 120, "t": 80},
        hoverlabel={"namelength": -1, "font_size": 12},
    )
    fig.update_yaxes(tickformat=".0%")
    fig.add_hline(y=0, line_dash="dash", line_color="#aaaaaa")
    return fig


def qp_faceted_chart(
    df: pd.DataFrame,
    sectors: list[str],
    title: str,
    subtitle: str,
    *,
    y_clip: float | None = QP_LINE_CLIP,
    show_events: bool = False,
) -> go.Figure:
    n = len(sectors)
    ncols = min(3, max(1, n))
    nrows = int((n + ncols - 1) / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[short_sector_name(s, max_len=22) for s in sectors],
        vertical_spacing=0.12,
    )
    for idx, name in enumerate(sectors):
        r, c = divmod(idx, ncols)
        g = df[df["industry_name"] == name].sort_values("period")
        if g.empty:
            continue
        color = _sector_trace_color(name, idx)
        fig.add_trace(
            go.Scatter(
                x=g["period"],
                y=g["qp_ratio"],
                mode="lines+markers",
                line={"color": color},
                name=name,
                showlegend=False,
                hovertemplate=f"<b>{name}</b><br>%{{x}}<br>Q/P: %{{y:.3f}}<extra></extra>",
            ),
            row=r + 1,
            col=c + 1,
        )
        if y_clip is not None:
            fig.update_yaxes(range=[-y_clip, y_clip], row=r + 1, col=c + 1)

    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_layout(
        **_title_layout(title, subtitle),
        height=faceted_height(n, ncols=ncols),
        margin={"b": 120, "t": 140},
        autosize=True,
        template="plotly_white",
        hoverlabel={"namelength": -1, "font_size": 12},
    )

    if show_events:
        add_event_bands_categorical(fig, _periods_from_df(df, "period"))
    return fig


def spread_chart(
    merged: pd.DataFrame,
    col_a: str,
    col_b: str,
    sectors: list[str],
    title: str,
    subtitle: str,
    *,
    spread_name: str = "Spread",
) -> go.Figure:
    fig = go.Figure()
    plot_df = _dedupe_sectors(merged, sectors).copy()
    plot_df["spread"] = plot_df[col_a] - plot_df[col_b]
    plot_df = plot_df.sort_values("spread")
    labels = [short_sector_name(n, max_len=30) for n in plot_df["industry_name"]]
    colors = ["#009E73" if v >= 0 else "#CC6677" for v in plot_df["spread"]]
    fig.add_trace(
        go.Bar(
            x=plot_df["spread"],
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{customdata}</b><br>spread: %{x:.1%}<extra></extra>",
            customdata=plot_df["industry_name"],
        )
    )
    fig.update_layout(
        **_title_layout(title, subtitle),
        xaxis_title=spread_name,
        height=max(560, len(labels) * 36 + 120),
        autosize=True,
        margin={"l": 60, "r": 40, "t": 80, "b": 60},
        template="plotly_white",
        hoverlabel={"namelength": -1, "font_size": 12},
    )
    fig.update_xaxes(tickformat=".0%")
    fig.add_vline(x=0, line_dash="dash", line_color="#aaaaaa")
    return fig


def grouped_wage_price_chart(
    merged: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_a: str,
    label_b: str,
    sectors: list[str],
    title: str,
    subtitle: str,
) -> go.Figure:
    """Horizontal grouped bars: BEA price vs BLS wage (endpoint totals)."""
    fig = go.Figure()
    plot_df = _dedupe_sectors(merged, sectors).dropna(subset=[col_a, col_b])
    if plot_df.empty:
        return fig

    order = {name: i for i, name in enumerate(sectors)}
    plot_df = plot_df.assign(_order=plot_df["industry_name"].map(order)).sort_values("_order")
    labels = [short_sector_name(n, max_len=30) for n in plot_df["industry_name"]]

    fig.add_trace(
        go.Bar(
            y=labels,
            x=plot_df[col_a],
            name=label_a,
            orientation="h",
            marker_color="#0072B2",
            opacity=0.85,
            customdata=plot_df["industry_name"],
            hovertemplate=f"<b>%{{customdata}}</b><br>{label_a}: %{{x:.1%}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=plot_df[col_b],
            name=label_b,
            orientation="h",
            marker_color="#D55E00",
            opacity=0.85,
            customdata=plot_df["industry_name"],
            hovertemplate=f"<b>%{{customdata}}</b><br>{label_b}: %{{x:.1%}}<extra></extra>",
        )
    )
    fig.update_layout(
        **_title_layout(title, subtitle),
        barmode="group",
        xaxis_title="Growth rate",
        height=max(560, len(labels) * 42 + 120),
        autosize=True,
        margin={"l": 60, "r": 40, "t": 80, "b": 60},
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "x": 0},
        hoverlabel={"namelength": -1, "font_size": 12},
    )
    fig.update_xaxes(tickformat=".0%")
    fig.add_vline(x=0, line_dash="dash", line_color="#aaaaaa")
    return fig


def _scenario_column_name(scenario: str) -> str:
    return f"scenario_{scenario.lower().replace(' ', '_')}"


def forecast_chart(
    df: pd.DataFrame,
    sectors: list[str],
    metric_label: str,
    title: str,
    subtitle: str,
    *,
    train_end: str,
    show_confidence_band: bool = True,
    show_scenario_overlays: bool = False,
    scenarios: list[str] | None = None,
    show_events: bool = False,
) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig

    scenario_palette = ["#E69F00", "#56B4E9", "#F0E442", "#D55E00", "#CC79A7", "#999999"]
    all_periods = sorted(df["period"].unique().tolist())

    for idx, name in enumerate(sectors):
        g = df[df["industry_name"] == name].sort_values("period")
        if g.empty:
            continue
        color = _sector_trace_color(name, idx)
        hist = g[~g["is_forecast"]]
        fcst = g[g["is_forecast"]]

        if not hist.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist["period"],
                    y=hist["actual"],
                    mode="lines+markers",
                    name=f"{name} (actual)",
                    line={"color": color, "width": 2},
                    marker={"size": 5},
                    legendgroup=name,
                    hovertemplate=f"<b>{name}</b><br>%{{x}}<br>actual: %{{y:.1%}}<extra></extra>",
                )
            )

        if not fcst.empty:
            bridge_x = [hist["period"].iloc[-1]] if not hist.empty else []
            bridge_y = [hist["actual"].iloc[-1]] if not hist.empty else []
            x_fc = bridge_x + fcst["period"].tolist()
            y_fc = bridge_y + fcst["forecast"].tolist()

            fig.add_trace(
                go.Scatter(
                    x=x_fc,
                    y=y_fc,
                    mode="lines+markers",
                    name=f"{name} (forecast)",
                    line={"color": color, "width": 2, "dash": "dot"},
                    marker={"size": 4},
                    legendgroup=name,
                    hovertemplate=f"<b>{name}</b><br>%{{x}}<br>forecast: %{{y:.1%}}<extra></extra>",
                )
            )

            if show_confidence_band and fcst["lower"].notna().any() and fcst["upper"].notna().any():
                band_x = bridge_x + fcst["period"].tolist() + fcst["period"].tolist()[::-1]
                band_y = (
                    bridge_y
                    + fcst["upper"].tolist()
                    + fcst["lower"].tolist()[::-1]
                )
                fig.add_trace(
                    go.Scatter(
                        x=band_x,
                        y=band_y,
                        fill="toself",
                        fillcolor=_rgba_from_hex(color, 0.12),
                        line={"color": "rgba(0,0,0,0)"},
                        name=f"{name} (80% PI)",
                        legendgroup=name,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        if show_scenario_overlays and scenarios:
            shock_presets = forecast_settings()["shock_presets"]
            for s_idx, scenario in enumerate(scenarios):
                col = _scenario_column_name(scenario)
                if col not in g.columns:
                    continue
                scen = g[g["is_forecast"] & g[col].notna()]
                if scen.empty:
                    continue
                scen_label = scenario_option_label(scenario, shock_presets)
                bridge_x = [hist["period"].iloc[-1]] if not hist.empty else []
                bridge_y = [hist["actual"].iloc[-1]] if not hist.empty else []
                scen_color = scenario_palette[s_idx % len(scenario_palette)]
                fig.add_trace(
                    go.Scatter(
                        x=bridge_x + scen["period"].tolist(),
                        y=bridge_y + scen[col].tolist(),
                        mode="lines",
                        name=f"{name} — {scen_label}",
                        line={"color": scen_color, "width": 1.8, "dash": "dash"},
                        legendgroup=f"{name}-{scenario}",
                        hovertemplate=f"<b>{name}</b><br>%{{x}}<br>{scen_label}: %{{y:.1%}}<extra></extra>",
                    )
                )

    fig.update_layout(
        **_title_layout(title, subtitle),
        xaxis_title="Q1→Q1 year (period end)",
        yaxis_title=metric_label,
        hovermode="x unified",
        height=600,
        xaxis={"categoryorder": "array", "categoryarray": all_periods},
    )
    _apply_legend(fig, len(fig.data))
    fig.update_yaxes(tickformat=".0%")
    fig.add_hline(y=0, line_dash="dash", line_color="#aaaaaa")
    fig.add_shape(
        type="line",
        x0=train_end,
        x1=train_end,
        y0=0,
        y1=1,
        yref="paper",
        line={"dash": "dot", "color": "#888888", "width": 1},
    )
    fig.add_annotation(
        x=train_end,
        y=1.02,
        yref="paper",
        text="Forecast start",
        showarrow=False,
        font={"size": 10, "color": "#666666"},
    )

    if show_events:
        hist_periods = [p for p in all_periods if p <= train_end]
        add_event_bands_categorical(fig, hist_periods)
    return fig


def _rgba_from_hex(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(128,128,128,{alpha})"
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def build_comparison_frame(
    panels: dict[str, pd.DataFrame],
    horizon: str,
) -> pd.DataFrame:
    if horizon == "quarterly":
        return _merge_bea_bls_quarterly(
            panels.get("bea_quarterly", pd.DataFrame()),
            panels.get("bls_quarterly", pd.DataFrame()),
        )
    if horizon == "endpoint":
        return _merge_bea_bls_endpoint(
            panels.get("bea_endpoint", pd.DataFrame()),
            panels.get("bls_endpoint", pd.DataFrame()),
        )
    if horizon == "yearly_q1":
        return _merge_bea_bls_yearly(
            panels.get("bea_yearly", pd.DataFrame()),
            panels.get("bls_yearly", pd.DataFrame()),
        )
    return pd.DataFrame()


def metric_frame(panels: dict[str, pd.DataFrame], source: str, horizon: str) -> pd.DataFrame:
    if horizon == "quarterly":
        return panels.get("bls_quarterly" if source == "bls" else "bea_quarterly", pd.DataFrame())
    if horizon == "endpoint":
        return panels.get("bls_endpoint" if source == "bls" else "bea_endpoint", pd.DataFrame())
    if horizon == "yearly_q1":
        df = panels.get("bls_yearly" if source == "bls" else "bea_yearly", pd.DataFrame())
        if not df.empty and "period_end" in df.columns:
            return df
    return pd.DataFrame()


def chart_for_selection(
    panels: dict[str, pd.DataFrame],
    horizon: str,
    chart_mode: str,
    metric_a: str,
    metric_b: str | None,
    sectors: list[str],
    start: str,
    end: str,
    *,
    clip_qp: bool = True,
    show_events: bool = True,
    show_labels: bool = True,
    forecast_df: pd.DataFrame | None = None,
    show_confidence_band: bool = True,
    show_scenario_overlays: bool = False,
    forecast_scenarios: list[str] | None = None,
) -> go.Figure:
    if chart_mode == "BEA ΔQ/ΔP decomposition":
        df = metric_frame(panels, "bea", horizon)
        t, s = format_chart_title("Quantity & Price Decomposition", horizon, start, end)
        if horizon == "endpoint":
            return endpoint_decomposition_chart(df, sectors, t, s)
        x_col = "period" if horizon == "quarterly" else "period_end"
        return decomposition_chart(df, sectors, t, s, x_col=x_col, show_events=show_events)

    if chart_mode == "Wage minus price spread":
        merged = build_comparison_frame(panels, "endpoint")
        t, s = format_chart_title("Wage Minus Price Spread", "endpoint", start, end)
        return spread_chart(
            merged,
            "avg_hourly_earnings_growth",
            "price_growth",
            sectors,
            t,
            s,
            spread_name="Wage − price growth",
        )

    if chart_mode == "Wage vs price (grouped bars)":
        merged = build_comparison_frame(panels, "endpoint")
        t, s = format_chart_title("Wage vs Price Growth", "endpoint", start, end)
        return grouped_wage_price_chart(
            merged,
            "price_growth",
            "avg_hourly_earnings_growth",
            "BEA price growth",
            "BLS wage growth",
            sectors,
            t,
            s,
        )

    if chart_mode == "Q/P ratio (faceted)":
        df = metric_frame(panels, "bea", "quarterly")
        t, s = format_chart_title("Q/P Ratio", "quarterly", start, end, scope="by Sector")
        y_clip = QP_LINE_CLIP if clip_qp else None
        return qp_faceted_chart(df, sectors, t, s, y_clip=y_clip, show_events=show_events)

    if chart_mode == "Single metric":
        source, col = {
            "BEA price growth": ("bea", "price_growth"),
            "BEA quantity growth": ("bea", "quantity_growth"),
            "BEA Q/P ratio": ("bea", "qp_ratio"),
            "BLS wage growth": ("bls", "avg_hourly_earnings_growth"),
            "BLS employment growth": ("bls", "employment_thousands_growth"),
        }[metric_a]
        df = metric_frame(panels, source, horizon)
        x_col = "period" if horizon == "quarterly" else "period_end"
        t, s = format_chart_title(metric_a, horizon, start, end)

        if horizon == "endpoint":
            fig = go.Figure()
            plot_df = df[df["industry_name"].isin(sectors)].sort_values(col)
            colors = [_sector_trace_color(n, i) for i, n in enumerate(plot_df["industry_name"])]
            labels = [short_sector_name(n) for n in plot_df["industry_name"]]
            hover_fmt = "%{y:.3f}" if col == "qp_ratio" else "%{y:.1%}"
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=plot_df[col],
                    marker_color=colors,
                    customdata=plot_df["industry_name"],
                    hovertemplate=f"<b>%{{customdata}}</b><br>{metric_a}: {hover_fmt}<extra></extra>",
                )
            )
            fig.update_layout(
                **_title_layout(t, s),
                xaxis_tickangle=-45,
                height=max(560, len(labels) * 32 + 140),
                autosize=True,
                template="plotly_white",
                hoverlabel={"namelength": -1, "font_size": 12},
            )
            if col != "qp_ratio":
                fig.update_yaxes(tickformat=".0%")
            return fig

        y_clip = QP_LINE_CLIP if col == "qp_ratio" and clip_qp else None
        fig = line_chart(df, col, sectors, t, s, x_col=x_col, y_clip=y_clip, show_events=show_events)
        if col == "qp_ratio" and "qp_sign_case" in df.columns:
            for trace, name in zip(fig.data, sectors, strict=False):
                g = df[df["industry_name"] == name]
                if not g.empty and "qp_sign_case" in g.columns:
                    trace.customdata = [qp_sign_label(c) for c in g.sort_values(x_col)["qp_sign_case"]]
                    trace.hovertemplate = (
                        f"<b>{name}</b><br>%{{x}}<br>Q/P: %{{y:.3f}}<br>%{{customdata}}<extra></extra>"
                    )
        return fig

    if chart_mode == "BEA vs BLS comparison" and metric_b:
        merged = build_comparison_frame(panels, horizon)
        pairs = {
            "Price vs wage": (
                "price_growth",
                "avg_hourly_earnings_growth",
                "Price vs Wage Growth",
                "BEA price growth",
                "BLS wage growth",
            ),
            "Quantity vs employment": (
                "quantity_growth",
                "employment_thousands_growth",
                "Quantity vs Employment Growth",
                "BEA quantity growth",
                "BLS employment growth",
            ),
        }
        col_a, col_b, tname, x_label, y_label = pairs[metric_b]
        t, s = format_chart_title(tname, horizon, start, end)
        if horizon == "endpoint":
            return scatter_chart(
                merged,
                col_a,
                col_b,
                sectors,
                t,
                s,
                x_label,
                y_label,
                show_labels=show_labels,
                equal_growth_line=True,
                quadrant_hints=True,
            )
        x_col = "period" if horizon == "quarterly" else "period_end"
        return dual_line_chart(
            merged,
            col_a,
            col_b,
            x_label,
            y_label,
            sectors,
            t,
            s,
            x_col=x_col,
            show_events=show_events,
        )

    if chart_mode == "Scatter (BEA price vs quantity)":
        df = build_scatter_frame(panels, horizon, sectors)
        t, _ = format_chart_title("Price vs Quantity Growth", horizon, start, end)
        s = scatter_subtitle(horizon, start, end)
        return scatter_chart(
            df,
            "price_growth",
            "quantity_growth",
            sectors,
            t,
            s,
            "Price growth",
            "Quantity growth",
            show_labels=show_labels,
            quadrant_hints=True,
        )

    if chart_mode == "SARIMA forecast":
        t, s = format_chart_title(f"SARIMA Forecast — {metric_a}", "yearly_q1", start, end)
        s = f"{s} · projected through {forecast_settings()['through']}"
        return forecast_chart(
            forecast_df if forecast_df is not None else pd.DataFrame(),
            sectors,
            metric_a,
            t,
            s,
            train_end=end,
            show_confidence_band=show_confidence_band,
            show_scenario_overlays=show_scenario_overlays,
            scenarios=forecast_scenarios or [],
            show_events=show_events,
        )

    return go.Figure()
