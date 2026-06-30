"""Time-series, endpoint, ranked Q/P, labor, and BEA–BLS comparison charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.export_utils import (
    filter_stable_qp_rows,
    qp_sign_color,
    qp_sign_label,
)
from src.analysis.industry_filters import load_config, sector_subsector_split
from src.analysis.outlier_labels import annotate_points, flag_scatter_outliers, flag_series_peaks
from src.analysis.plot_colors import get_sector_color
from src.analysis.plot_style import (
    apply_axes_style,
    format_chart_title,
    place_legend_below,
    save_figure,
    set_chart_titles,
    short_sector_name,
)
from src.analysis.plots import _scatter_growth

EVENT_COLORS = {
    "covid_shock": ("#ffcccc", 0.35),
    "recovery": ("#e8f4e8", 0.25),
    "tariffs": ("#fff3cd", 0.3),
}

QP_CLIP = 5.0
QP_LINE_CLIP = 10.0


def _qp_price_threshold(cfg: dict) -> float:
    return float(cfg.get("qp_unstable_price_threshold", 0.005))


def _prepare_qp_ranking(df: pd.DataFrame, cfg: dict, sector_names: list[str] | None = None) -> pd.DataFrame:
    work = df.copy()
    if sector_names and "industry_name" in work.columns:
        work = work[work["industry_name"].isin(sector_names)]
    if "line_id" in work.columns:
        work = work.drop_duplicates("line_id")
    return filter_stable_qp_rows(work, min_abs_price=_qp_price_threshold(cfg))


def _bar_colors_for_qp(plot_df: pd.DataFrame, value_col: str) -> list[str]:
    if "qp_sign_case" in plot_df.columns and value_col == "qp_ratio":
        return [qp_sign_color(c) for c in plot_df["qp_sign_case"]]
    return ["#2ca02c" if v >= 0 else "#d62728" for v in plot_df[value_col]]


def _qp_bar_labels(plot_df: pd.DataFrame, value_col: str) -> list[str]:
    labels = []
    for row in plot_df.itertuples():
        label = str(row.industry_name)[:45]
        if value_col == "qp_ratio" and hasattr(row, "qp_sign_case") and pd.notna(getattr(row, "qp_sign_case", None)):
            label += f" ({qp_sign_label(row.qp_sign_case)})"
        labels.append(label)
    return labels


def _shade_events(ax: plt.Axes, cfg: dict, periods: list[str]) -> None:
    if not periods:
        return
    period_to_x = {p: i for i, p in enumerate(sorted(periods))}
    for _name, spec in cfg.get("event_periods", {}).items():
        start = spec.get("start")
        end = spec.get("end") or max(periods)
        if start not in period_to_x:
            continue
        x0 = period_to_x[start]
        x1 = period_to_x.get(end, len(periods) - 1)
        color, alpha = EVENT_COLORS.get(_name, ("#eeeeee", 0.2))
        ax.axvspan(x0 - 0.5, x1 + 0.5, color=color, alpha=alpha)


def _annotate_series_outliers(
    ax: plt.Axes,
    xs: list[int],
    ys: list[float],
    period_labels: list[str],
    name: str,
    *,
    y_clip: float | None,
) -> None:
    for x, y, period in zip(xs, ys, period_labels, strict=True):
        if not np.isfinite(y):
            continue
        if y_clip is not None and abs(y) <= y_clip:
            continue
        short = short_sector_name(name)
        label = f"{short} ({period}): {y:.1f}"
        if y_clip is not None:
            edge_y = y_clip if y > 0 else -y_clip
            ax.scatter([x], [edge_y], color=get_sector_color(name), s=28, zorder=5, clip_on=False)
            ax.annotate(
                label,
                xy=(x, edge_y),
                xytext=(0, 8 if y > 0 else -8),
                textcoords="offset points",
                fontsize=6.5,
                ha="center",
                color="#333333",
                arrowprops={"arrowstyle": "->", "color": "#888888", "lw": 0.6},
            )
        else:
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.5,
                color="#333333",
                alpha=0.9,
            )


def _line_timeseries(
    data: pd.DataFrame,
    y_col: str,
    industries: list[str],
    title: str,
    y_label: str,
    output_path: Path,
    cfg: dict,
    x_col: str = "period",
    *,
    subtitle: str | None = None,
    footnote: str | None = None,
    y_clip: float | None = None,
    annotate_outliers: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    subset = data[data["industry_name"].isin(industries)].copy()
    periods = sorted(subset[x_col].unique())
    for idx, name in enumerate(industries):
        group = subset[subset["industry_name"] == name]
        if group.empty:
            continue
        g = group.sort_values(x_col)
        xs = [periods.index(p) for p in g[x_col]]
        ys = g[y_col].tolist()
        period_labels = g[x_col].tolist()
        color = get_sector_color(name, idx)
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=name, alpha=0.9, color=color)
        if annotate_outliers:
            _annotate_series_outliers(ax, xs, ys, period_labels, name, y_clip=y_clip)

    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    _shade_events(ax, cfg, periods if x_col == "period" else [])
    if y_clip is not None:
        ax.set_ylim(-y_clip, y_clip)
        if not footnote:
            footnote = f"Y-axis clipped to ±{y_clip:g}; labels show true values for extreme points."
    apply_axes_style(ax, ylabel=y_label)
    set_chart_titles(fig, ax, title, subtitle, footnote=footnote)
    place_legend_below(ax, ncol=3)
    return save_figure(fig, output_path, bottom=0.22)


def _sector_bar_panel(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    output_path: Path,
    sector_names: list[str],
    *,
    subtitle: str | None = None,
    clip: float | None = None,
    xlabel: str = "Growth rate",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df[df["industry_name"].isin(sector_names)].copy()
    plot_df = plot_df.dropna(subset=[value_col])
    plot_df = plot_df[np.isfinite(plot_df[value_col])]
    if plot_df.empty:
        return output_path

    footnote = None
    if clip is not None:
        n_out = int((plot_df[value_col].abs() > clip).sum())
        if n_out:
            footnote = f"{n_out} sector(s) clipped to ±{clip:g} for readability."
        plot_df = plot_df.copy()
        plot_df["_plot_val"] = plot_df[value_col].clip(-clip, clip)
        val_col = "_plot_val"
    else:
        val_col = value_col

    plot_df = plot_df.sort_values(val_col)
    labels = _qp_bar_labels(plot_df, value_col)
    colors = _bar_colors_for_qp(plot_df, value_col)
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.38)))
    ax.barh(labels, plot_df[val_col], color=colors)
    ax.axvline(0, color="#aaaaaa", linewidth=0.8)
    apply_axes_style(ax, xlabel=xlabel)
    set_chart_titles(fig, ax, title, subtitle, footnote=footnote)
    return save_figure(fig, output_path, bottom=0.12)


def _bar_top_bottom(
    ranked: pd.DataFrame,
    title: str,
    output_path: Path,
    cfg: dict,
    n: int = 5,
    value_col: str = "qp_ratio",
    xlabel: str = "Q/P ratio (%ΔQ / %ΔP)",
    *,
    subtitle: str | None = None,
    sector_names: list[str] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = _prepare_qp_ranking(ranked, cfg, sector_names) if value_col == "qp_ratio" else ranked.copy()
    valid = valid.dropna(subset=[value_col])
    valid = valid[np.isfinite(valid[value_col])]
    if valid.empty:
        return output_path
    top = valid.nlargest(n, value_col)
    bottom = valid.nsmallest(n, value_col)
    combined = pd.concat([top, bottom], ignore_index=True)
    colors = _bar_colors_for_qp(combined, value_col)

    fig, ax = plt.subplots(figsize=(10, max(5, len(combined) * 0.45)))
    labels = _qp_bar_labels(combined, value_col)
    ax.barh(labels, combined[value_col], color=colors)
    ax.axvline(0, color="#aaaaaa", linewidth=0.8)
    apply_axes_style(ax, xlabel=xlabel)
    footnote = None
    if value_col == "qp_ratio":
        footnote = "Colors: Q↑P↑ (blue), Q↓P↓ (orange), Q↑P↓ (green), Q↓P↑ (pink). Unstable |ΔP| excluded."
    set_chart_titles(fig, ax, title, subtitle, footnote=footnote)
    ax.invert_yaxis()
    return save_figure(fig, output_path, bottom=0.14)


def _endpoint_scatter_one_dot(
    endpoint: pd.DataFrame,
    title: str,
    output_path: Path,
    x_col: str = "price_growth",
    y_col: str = "quantity_growth",
    x_label: str = "Price growth (%ΔP)",
    y_label: str = "Quantity growth (%ΔQ)",
    *,
    subtitle: str | None = None,
) -> Path:
    one_per = endpoint[["line_id", "industry_name", x_col, y_col]].drop_duplicates("line_id")
    _scatter_growth(
        one_per, x_col, y_col, title, x_label, y_label, output_path, subtitle=subtitle, label_outliers=True
    )
    return output_path


def _comparison_scatter(
    merged: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    output_path: Path,
    x_label: str,
    y_label: str,
    sector_names: list[str],
    *,
    subtitle: str | None = None,
    show_diagonal: bool = False,
) -> Path:
    data = merged[merged["industry_name"].isin(sector_names)].drop_duplicates("line_id")
    _scatter_growth(
        data,
        x_col,
        y_col,
        title,
        x_label,
        y_label,
        output_path,
        subtitle=subtitle,
        label_outliers=True,
        show_diagonal=show_diagonal,
    )
    return output_path


def _dual_metric_timeseries(
    data: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_a: str,
    label_b: str,
    industries: list[str],
    title: str,
    output_path: Path,
    cfg: dict,
    x_col: str = "period",
    *,
    subtitle: str | None = None,
) -> Path:
    """Per sector: solid = col_a, dashed = col_b (same color)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    subset = data[data["industry_name"].isin(industries)].copy()
    periods = sorted(subset[x_col].unique())

    for idx, name in enumerate(industries):
        group = subset[subset["industry_name"] == name]
        if group.empty:
            continue
        g = group.sort_values(x_col)
        xs = [periods.index(p) for p in g[x_col]]
        color = get_sector_color(name, idx)
        if col_a in g.columns:
            ax.plot(
                xs,
                g[col_a],
                color=color,
                linewidth=1.5,
                linestyle="-",
                alpha=0.9,
                label=name,
            )
        if col_b in g.columns:
            ax.plot(xs, g[col_b], color=color, linewidth=1.5, linestyle="--", alpha=0.9)

    if x_col == "period" and not subset.empty:
        period_to_x = {p: i for i, p in enumerate(periods)}
        for col in [c for c in (col_a, col_b) if c in subset.columns]:
            peaks = flag_series_peaks(subset, "industry_name", col, top_n_per_series=1, x_col=x_col)
            for row in peaks.itertuples():
                px = period_to_x.get(getattr(row, x_col))
                py = getattr(row, col)
                if px is None or not np.isfinite(py):
                    continue
                ax.annotate(
                    short_sector_name(str(row.industry_name)),
                    xy=(px, py),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6.5,
                    color="#333333",
                )

    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    if x_col == "period":
        _shade_events(ax, cfg, periods)
    apply_axes_style(ax, ylabel="Growth rate")
    footnote = f"Solid = {label_a}, dashed = {label_b}"
    set_chart_titles(fig, ax, title, subtitle, footnote=footnote)
    place_legend_below(ax, ncol=3)
    return save_figure(fig, output_path, bottom=0.24)


def _bea_decomposition_quarterly(
    panel: pd.DataFrame,
    industries: list[str],
    title: str,
    output_path: Path,
    cfg: dict,
    *,
    subtitle: str | None = None,
) -> Path:
    """Solid = %ΔQ, dashed = %ΔP per sector."""
    return _dual_metric_timeseries(
        panel,
        "quantity_growth",
        "price_growth",
        "quantity (%ΔQ)",
        "price (%ΔP)",
        industries,
        title,
        output_path,
        cfg,
        subtitle=subtitle,
    )


def _qp_faceted_timeseries(
    data: pd.DataFrame,
    industries: list[str],
    title: str,
    output_path: Path,
    cfg: dict,
    *,
    subtitle: str | None = None,
    y_clip: float = QP_LINE_CLIP,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(industries)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.8 * nrows), sharex=True)
    axes_flat = np.array(axes).flatten()

    subset = data[data["industry_name"].isin(industries)]
    periods = sorted(subset["period"].unique()) if not subset.empty else []

    for idx, name in enumerate(industries):
        ax = axes_flat[idx]
        group = subset[subset["industry_name"] == name].sort_values("period")
        color = get_sector_color(name, idx)
        if not group.empty:
            xs = [periods.index(p) for p in group["period"]]
            ys = group["qp_ratio"].tolist()
            ax.plot(xs, ys, color=color, marker="o", markersize=2, linewidth=1.2)
            peaks = flag_series_peaks(group, "industry_name", "qp_ratio", top_n_per_series=1, x_col="period")
            if not peaks.empty:
                for prow in peaks.itertuples():
                    px = periods.index(prow.period)
                    ax.annotate(
                        short_sector_name(name),
                        xy=(px, prow.qp_ratio),
                        xytext=(2, 2),
                        textcoords="offset points",
                        fontsize=5.5,
                        color="#333333",
                    )
        ax.axhline(0, color="#cccccc", linewidth=0.6, linestyle="--")
        ax.set_ylim(-y_clip, y_clip)
        ax.set_title(short_sector_name(name, 28), fontsize=8)
        ax.tick_params(labelsize=6)

    for j in range(len(industries), len(axes_flat)):
        axes_flat[j].set_visible(False)

    if periods:
        step = max(1, len(periods) // 8)
        tick_idx = list(range(0, len(periods), step))
        tick_labels = [periods[i] for i in tick_idx]
        for ax in axes_flat[: len(industries)]:
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=6)

    footnote = f"Each panel: QoQ Q/P ratio (%ΔQ / %ΔP), y-axis clipped to ±{y_clip:g}"
    set_chart_titles(fig, axes_flat[0], title, subtitle, footnote=footnote)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _grouped_endpoint_bars(
    merged: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_a: str,
    label_b: str,
    sector_names: list[str],
    title: str,
    output_path: Path,
    *,
    subtitle: str | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = merged[merged["industry_name"].isin(sector_names)].drop_duplicates("line_id")
    plot_df = plot_df.dropna(subset=[col_a, col_b])
    plot_df = plot_df.sort_values(col_a)
    if plot_df.empty:
        return output_path

    labels = [short_sector_name(n, 30) for n in plot_df["industry_name"]]
    y = np.arange(len(plot_df))
    height = 0.35
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.42)))
    ax.barh(y - height / 2, plot_df[col_a], height=height, label=label_a, color="#0072B2", alpha=0.85)
    ax.barh(y + height / 2, plot_df[col_b], height=height, label=label_b, color="#D55E00", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="#aaaaaa", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=8)
    apply_axes_style(ax, xlabel="Growth rate")
    set_chart_titles(fig, ax, title, subtitle)
    return save_figure(fig, output_path, bottom=0.12)


def _spread_endpoint_bars(
    merged: pd.DataFrame,
    col_a: str,
    col_b: str,
    sector_names: list[str],
    title: str,
    output_path: Path,
    *,
    subtitle: str | None = None,
    spread_label: str = "Spread (A − B)",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = merged[merged["industry_name"].isin(sector_names)].drop_duplicates("line_id").copy()
    plot_df["_spread"] = plot_df[col_a] - plot_df[col_b]
    plot_df = plot_df.dropna(subset=["_spread"]).sort_values("_spread")
    if plot_df.empty:
        return output_path

    labels = [short_sector_name(n, 30) for n in plot_df["industry_name"]]
    colors = ["#009E73" if v >= 0 else "#CC6677" for v in plot_df["_spread"]]
    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.38)))
    ax.barh(labels, plot_df["_spread"], color=colors)
    ax.axvline(0, color="#aaaaaa", linewidth=0.8)
    apply_axes_style(ax, xlabel=spread_label)
    set_chart_titles(fig, ax, title, subtitle)
    return save_figure(fig, output_path, bottom=0.12)


def _merge_bea_bls_quarterly(bea: pd.DataFrame, bls: pd.DataFrame) -> pd.DataFrame:
    b_cols = ["line_id", "period", "quantity_growth", "price_growth", "qp_ratio"]
    l_cols = ["line_id", "period", "employment_thousands_growth", "avg_hourly_earnings_growth"]
    b = bea[[c for c in b_cols if c in bea.columns]]
    bl = bls[[c for c in l_cols if c in bls.columns]]
    merged = b.merge(bl, on=["line_id", "period"], how="inner")
    if "industry_name" in bea.columns:
        names = bea[["line_id", "industry_name"]].drop_duplicates("line_id")
        merged = merged.merge(names, on="line_id", how="left")
    return merged


def _merge_bea_bls_endpoint(bea_ep: pd.DataFrame, bls_ep: pd.DataFrame) -> pd.DataFrame:
    if bea_ep.empty or bls_ep.empty or "line_id" not in bea_ep.columns:
        return pd.DataFrame()
    bls_cols = [
        c
        for c in ["line_id", "employment_thousands_growth", "avg_hourly_earnings_growth"]
        if c in bls_ep.columns
    ]
    if len(bls_cols) < 2:
        return pd.DataFrame()
    merged = bea_ep.merge(bls_ep[bls_cols], on="line_id", how="inner", suffixes=("_bea", "_bls"))
    return merged


def _merge_bea_bls_yearly(bea_yr: pd.DataFrame, bls_yr: pd.DataFrame) -> pd.DataFrame:
    bls_cols = [
        c
        for c in [
            "line_id",
            "year",
            "employment_thousands_growth",
            "avg_hourly_earnings_growth",
        ]
        if c in bls_yr.columns
    ]
    return bea_yr.merge(bls_yr[bls_cols], on=["line_id", "year"], how="inner")


def plot_all_horizons(
    bea_growth: pd.DataFrame,
    bea_industries: pd.DataFrame,
    endpoint: pd.DataFrame,
    yearly: pd.DataFrame,
    panel: pd.DataFrame,
    bls_quarterly: pd.DataFrame,
    bls_industries: pd.DataFrame,
    bls_endpoint: pd.DataFrame,
    bls_yearly: pd.DataFrame,
    figures_base: Path,
    period_start: str,
    period_end: str,
) -> list[Path]:
    cfg = load_config()
    outputs: list[Path] = []
    sector_names = cfg["sector_plot"]

    sectors, subsectors = sector_subsector_split(panel, cfg)

    if "industry_name" in bls_quarterly.columns:
        bls_panel = bls_quarterly.copy()
    else:
        bls_panel = bls_quarterly.merge(
            bls_industries[["line_id", "industry_name"]],
            on="line_id",
            how="left",
        )

    ep_dir = figures_base / "endpoint"
    yr_dir = figures_base / "yearly_q1"
    q_dir = figures_base / "quarterly"
    cmp_dir = figures_base / "comparison"

    ep_title, ep_sub = format_chart_title("", "endpoint", period_start, period_end)
    q_title, q_sub = format_chart_title("", "quarterly", period_start, period_end)
    yr_title, yr_sub = format_chart_title("", "yearly", period_start, period_end)

    # --- endpoint ---
    if not endpoint.empty and "line_id" in endpoint.columns:
        ep_sectors = endpoint[endpoint["line_id"].isin(sectors["line_id"].unique())]
        ep_subsectors = endpoint[endpoint["line_id"].isin(subsectors["line_id"].unique())]

        t, s = format_chart_title("Price vs Quantity Growth", "endpoint", period_start, period_end)
        outputs.append(
            _endpoint_scatter_one_dot(
                ep_sectors,
                t,
                ep_dir / "bea_sector_price_vs_quantity_endpoint.png",
                subtitle=s,
            )
        )
        t, s = format_chart_title("Price vs Quantity Growth", "endpoint", period_start, period_end, scope="by Subsector")
        outputs.append(
            _endpoint_scatter_one_dot(
                ep_subsectors,
                t,
                ep_dir / "bea_subsector_price_vs_quantity_endpoint.png",
                subtitle=s,
            )
        )
        ep_plot = ep_sectors if not ep_sectors.empty else endpoint
        t, s = format_chart_title("Q/P Ratio", "endpoint", period_start, period_end)
        outputs.append(
            _sector_bar_panel(
                ep_plot,
                "qp_ratio",
                t,
                ep_dir / "bea_sector_qp_ratio_endpoint_bars.png",
                sector_names,
                subtitle=s,
                clip=QP_CLIP,
                xlabel="Q/P ratio (%ΔQ / %ΔP)",
            )
        )
        outputs.append(
            _bar_top_bottom(
                ep_plot,
                "Top & Bottom Q/P Ratio",
                ep_dir / "bea_top_bottom_qp_ratio_endpoint.png",
                cfg,
                subtitle=ep_sub,
                sector_names=sector_names,
            )
        )

    if not endpoint.empty and not bls_endpoint.empty:
        ep_cmp = _merge_bea_bls_endpoint(endpoint, bls_endpoint)
        t, s = format_chart_title("Wage vs Price Growth", "endpoint", period_start, period_end)
        outputs.append(
            _grouped_endpoint_bars(
                ep_cmp,
                "avg_hourly_earnings_growth",
                "price_growth",
                "BLS wage",
                "BEA price",
                sector_names,
                t,
                ep_dir / "bea_bls_wage_vs_price_endpoint_grouped.png",
                subtitle=s,
            )
        )
        t, s = format_chart_title("Wage Minus Price Spread", "endpoint", period_start, period_end)
        outputs.append(
            _spread_endpoint_bars(
                ep_cmp,
                "avg_hourly_earnings_growth",
                "price_growth",
                sector_names,
                t,
                ep_dir / "bea_bls_wage_minus_price_endpoint.png",
                subtitle=s,
                spread_label="Wage growth − price growth",
            )
        )
        t, s = format_chart_title("Quantity vs Employment Growth", "endpoint", period_start, period_end)
        outputs.append(
            _grouped_endpoint_bars(
                ep_cmp,
                "quantity_growth",
                "employment_thousands_growth",
                "BEA quantity",
                "BLS employment",
                sector_names,
                t,
                ep_dir / "bea_bls_quantity_vs_employment_endpoint_grouped.png",
                subtitle=s,
            )
        )

    if not bls_endpoint.empty:
        bls_ep_sec = bls_endpoint[bls_endpoint["industry_name"].isin(sector_names)]
        t, s = format_chart_title("Wage Growth", "endpoint", period_start, period_end)
        outputs.append(
            _sector_bar_panel(
                bls_ep_sec,
                "avg_hourly_earnings_growth",
                t,
                ep_dir / "bls_sector_wage_growth_endpoint_bars.png",
                sector_names,
                subtitle=s,
                xlabel="Avg hourly earnings growth",
            )
        )
        t, s = format_chart_title("Employment Growth", "endpoint", period_start, period_end)
        outputs.append(
            _sector_bar_panel(
                bls_ep_sec,
                "employment_thousands_growth",
                t,
                ep_dir / "bls_sector_employment_growth_endpoint_bars.png",
                sector_names,
                subtitle=s,
                xlabel="Employment growth",
            )
        )

    # --- yearly Q1 ---
    if not yearly.empty:
        yr_sectors = yearly[yearly["industry_name"].isin(sector_names)]
        t, s = format_chart_title("Price Growth", "yearly", period_start, period_end)
        outputs.append(
            _line_timeseries(
                yr_sectors,
                "price_growth",
                sector_names,
                t,
                "Price growth (%ΔP)",
                yr_dir / "bea_sector_price_growth_yearly.png",
                cfg,
                x_col="period_end",
                subtitle=s,
            )
        )
        t, s = format_chart_title("Quantity Growth", "yearly", period_start, period_end)
        outputs.append(
            _line_timeseries(
                yr_sectors,
                "quantity_growth",
                sector_names,
                t,
                "Quantity growth (%ΔQ)",
                yr_dir / "bea_sector_quantity_growth_yearly.png",
                cfg,
                x_col="period_end",
                subtitle=s,
            )
        )
        t, s = format_chart_title("Q/P Ratio", "yearly", period_start, period_end)
        outputs.append(
            _line_timeseries(
                yr_sectors,
                "qp_ratio",
                sector_names,
                t,
                "Q/P ratio (%ΔQ / %ΔP)",
                yr_dir / "bea_sector_qp_ratio_yearly.png",
                cfg,
                x_col="period_end",
                subtitle=s,
                y_clip=QP_LINE_CLIP,
                annotate_outliers=True,
            )
        )
        latest_year = int(yearly["year"].max())
        outputs.append(
            _bar_top_bottom(
                yearly[yearly["year"] == latest_year],
                "Top & Bottom Q/P Ratio",
                yr_dir / f"bea_top_bottom_qp_ratio_yearly_{latest_year}.png",
                cfg,
                subtitle=f"Q1 {latest_year}→Q1 {latest_year + 1}",
                sector_names=sector_names,
            )
        )

    if not bls_yearly.empty:
        bls_yr_sec = bls_yearly[bls_yearly["industry_name"].isin(sector_names)]
        t, s = format_chart_title("Wage Growth", "yearly", period_start, period_end)
        outputs.append(
            _line_timeseries(
                bls_yr_sec,
                "avg_hourly_earnings_growth",
                sector_names,
                t,
                "Avg hourly earnings growth",
                yr_dir / "bls_sector_wage_growth_yearly.png",
                cfg,
                x_col="period_end",
                subtitle=s,
            )
        )
        t, s = format_chart_title("Employment Growth", "yearly", period_start, period_end)
        outputs.append(
            _line_timeseries(
                bls_yr_sec,
                "employment_thousands_growth",
                sector_names,
                t,
                "Employment growth",
                yr_dir / "bls_sector_employment_growth_yearly.png",
                cfg,
                x_col="period_end",
                subtitle=s,
            )
        )

    # --- quarterly ---
    q_sectors = panel[panel["industry_name"].isin(sector_names)]
    t, s = format_chart_title("Price Growth", "quarterly", period_start, period_end)
    outputs.append(
        _line_timeseries(
            q_sectors,
            "price_growth",
            sector_names,
            t,
            "QoQ price growth (%ΔP)",
            q_dir / "bea_sector_price_growth_timeseries.png",
            cfg,
            subtitle=s,
        )
    )
    t, s = format_chart_title("Quantity Growth", "quarterly", period_start, period_end)
    outputs.append(
        _line_timeseries(
            q_sectors,
            "quantity_growth",
            sector_names,
            t,
            "QoQ quantity growth (%ΔQ)",
            q_dir / "bea_sector_quantity_growth_timeseries.png",
            cfg,
            subtitle=s,
        )
    )
    t, s = format_chart_title("Quantity & Price Decomposition", "quarterly", period_start, period_end)
    outputs.append(
        _bea_decomposition_quarterly(
            q_sectors,
            sector_names,
            t,
            q_dir / "bea_sector_quantity_price_decomposition_quarterly.png",
            cfg,
            subtitle=s,
        )
    )
    t, s = format_chart_title("Q/P Ratio", "quarterly", period_start, period_end, scope="by Sector (faceted)")
    outputs.append(
        _qp_faceted_timeseries(
            q_sectors,
            sector_names,
            t,
            q_dir / "bea_sector_qp_ratio_timeseries_faceted.png",
            cfg,
            subtitle=s,
        )
    )
    # Top-5 volatile sectors only (legacy single-axis view)
    qp_vol = (
        q_sectors.groupby("industry_name")["qp_ratio"]
        .apply(lambda s: s.abs().max())
        .nlargest(5)
        .index.tolist()
    )
    t, s = format_chart_title("Q/P Ratio", "quarterly", period_start, period_end, scope="Top 5 volatile")
    outputs.append(
        _line_timeseries(
            q_sectors,
            "qp_ratio",
            qp_vol,
            t,
            "QoQ Q/P ratio (%ΔQ / %ΔP)",
            q_dir / "bea_sector_qp_ratio_timeseries.png",
            cfg,
            subtitle=s,
            y_clip=QP_LINE_CLIP,
            annotate_outliers=True,
        )
    )

    agg_names = ["Gross domestic product", "Private industries", "Manufacturing"]
    agg = panel[panel["industry_name"].isin(agg_names)]
    if not agg.empty:
        for col, fname, ylab, metric in [
            ("quantity_growth", "bea_aggregate_quantity_growth_timeseries.png", "QoQ quantity growth (%ΔQ)", "Quantity Growth"),
            ("price_growth", "bea_aggregate_price_growth_timeseries.png", "QoQ price growth (%ΔP)", "Price Growth"),
        ]:
            t, s = format_chart_title(metric, "quarterly", period_start, period_end, scope="Aggregates")
            outputs.append(
                _line_timeseries(
                    agg,
                    col,
                    [n for n in agg_names if n in agg["industry_name"].values],
                    t,
                    ylab,
                    q_dir / fname,
                    cfg,
                    subtitle=s,
                )
            )

    qp_src = panel[panel["industry_name"].isin(sector_names)].copy()
    qp_avg = qp_src.groupby("line_id", as_index=False).agg(
        industry_name=("industry_name", "first"),
        quantity_growth=("quantity_growth", "mean"),
        price_growth=("price_growth", "mean"),
        qp_ratio=("qp_ratio", "mean"),
    )
    if "qp_sign_case" in qp_src.columns:
        cases = qp_src.groupby("line_id", as_index=False).agg(qp_sign_case=("qp_sign_case", "first"))
        qp_avg = qp_avg.merge(cases, on="line_id", how="left")
    outputs.append(
        _bar_top_bottom(
            qp_avg,
            "Top & Bottom Avg Q/P Ratio",
            q_dir / "bea_top_bottom_qp_ratio_quarterly.png",
            cfg,
            subtitle=q_sub,
            sector_names=sector_names,
        )
    )

    bls_sec = bls_panel[bls_panel["industry_name"].isin(sector_names)]
    if not bls_sec.empty:
        t, s = format_chart_title("Wage Growth", "quarterly", period_start, period_end)
        outputs.append(
            _line_timeseries(
                bls_sec,
                "avg_hourly_earnings_growth",
                sector_names,
                t,
                "Avg hourly earnings growth",
                q_dir / "bls_sector_wage_growth_timeseries.png",
                cfg,
                subtitle=s,
            )
        )
        t, s = format_chart_title("Employment Growth", "quarterly", period_start, period_end)
        outputs.append(
            _line_timeseries(
                bls_sec,
                "employment_thousands_growth",
                sector_names,
                t,
                "Employment growth",
                q_dir / "bls_sector_employment_growth_timeseries.png",
                cfg,
                subtitle=s,
            )
        )

    # --- BEA vs BLS comparison ---
    if not endpoint.empty and not bls_endpoint.empty:
        ep_cmp = _merge_bea_bls_endpoint(endpoint, bls_endpoint)
        t, s = format_chart_title("Price vs Wage Growth", "endpoint", period_start, period_end)
        outputs.append(
            _comparison_scatter(
                ep_cmp,
                "price_growth",
                "avg_hourly_earnings_growth",
                t,
                cmp_dir / "bea_price_vs_bls_wage_endpoint.png",
                "BEA price growth (%ΔP)",
                "BLS avg hourly earnings growth",
                sector_names,
                subtitle=s,
                show_diagonal=True,
            )
        )
        t, s = format_chart_title("Quantity vs Employment Growth", "endpoint", period_start, period_end)
        outputs.append(
            _comparison_scatter(
                ep_cmp,
                "quantity_growth",
                "employment_thousands_growth",
                t,
                cmp_dir / "bea_quantity_vs_bls_employment_endpoint.png",
                "BEA quantity growth (%ΔQ)",
                "BLS employment growth",
                sector_names,
                subtitle=s,
            )
        )

    q_cmp = _merge_bea_bls_quarterly(panel, bls_panel)
    if not q_cmp.empty:
        t, s = format_chart_title("Price vs Wage Growth", "quarterly", period_start, period_end)
        outputs.append(
            _dual_metric_timeseries(
                q_cmp,
                "price_growth",
                "avg_hourly_earnings_growth",
                "BEA price",
                "BLS wage",
                sector_names,
                t,
                cmp_dir / "bea_price_vs_bls_wage_quarterly.png",
                cfg,
                subtitle=s,
            )
        )
        t, s = format_chart_title("Quantity vs Employment Growth", "quarterly", period_start, period_end)
        outputs.append(
            _dual_metric_timeseries(
                q_cmp,
                "quantity_growth",
                "employment_thousands_growth",
                "BEA quantity",
                "BLS employment",
                sector_names,
                t,
                cmp_dir / "bea_quantity_vs_bls_employment_quarterly.png",
                cfg,
                subtitle=s,
            )
        )

    if not yearly.empty and not bls_yearly.empty:
        yr_cmp = _merge_bea_bls_yearly(yearly, bls_yearly)
        if not yr_cmp.empty:
            t, s = format_chart_title("Price vs Wage Growth", "yearly", period_start, period_end)
            outputs.append(
                _dual_metric_timeseries(
                    yr_cmp,
                    "price_growth",
                    "avg_hourly_earnings_growth",
                    "BEA price",
                    "BLS wage",
                    sector_names,
                    t,
                    cmp_dir / "bea_price_vs_bls_wage_yearly.png",
                    cfg,
                    x_col="period_end",
                    subtitle=s,
                )
            )
            t, s = format_chart_title("Quantity vs Employment Growth", "yearly", period_start, period_end)
            outputs.append(
                _dual_metric_timeseries(
                    yr_cmp,
                    "quantity_growth",
                    "employment_thousands_growth",
                    "BEA quantity",
                    "BLS employment",
                    sector_names,
                    t,
                    cmp_dir / "bea_quantity_vs_bls_employment_yearly.png",
                    cfg,
                    x_col="period_end",
                    subtitle=s,
                )
            )

    return [p for p in outputs if p.exists()]
