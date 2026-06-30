"""Industry growth plots: price vs quantity (BEA) and wages vs employment (BLS)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.analysis.outlier_labels import annotate_points, flag_scatter_outliers, flag_series_peaks
from src.analysis.plot_colors import get_sector_color
from src.analysis.plot_style import (
    apply_axes_style,
    format_chart_title,
    place_legend_below,
    save_figure,
    set_chart_titles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "industries.yaml"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _filter_private(df: pd.DataFrame, industries: pd.DataFrame) -> pd.DataFrame:
    meta = industries[industries["is_private"] == 1][["line_id", "industry_name", "indent_level"]]
    if "industry_name" in df.columns and "indent_level" in df.columns:
        return df[df["line_id"].isin(meta["line_id"])]
    return df.merge(meta, on="line_id", how="inner")


def _exclude_names(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    lowered = {n.lower() for n in names}
    return df[~df["industry_name"].str.lower().isin(lowered)]


def _period_filter(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df["period"] >= start) & (df["period"] <= end)]


def _scatter_growth(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    *,
    subtitle: str | None = None,
    label_outliers: bool = True,
    outlier_top_n: int = 3,
    show_diagonal: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df = data.drop_duplicates("line_id") if "line_id" in data.columns else data
    for idx, (name, group) in enumerate(plot_df.groupby("industry_name")):
        valid = group.dropna(subset=[x_col, y_col])
        valid = valid[np.isfinite(valid[x_col]) & np.isfinite(valid[y_col])]
        if valid.empty:
            continue
        row = valid.iloc[0]
        ax.scatter(
            row[x_col],
            row[y_col],
            label=name,
            s=55,
            alpha=0.85,
            color=get_sector_color(name, idx),
        )

    if show_diagonal:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, linestyle=":", color="#bbbbbb", linewidth=1, label="wage = price")

    if label_outliers and not plot_df.empty:
        outliers = flag_scatter_outliers(plot_df, x_col, y_col, top_n=outlier_top_n)
        annotate_points(ax, outliers, x_col, y_col)

    ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    apply_axes_style(ax, xlabel=x_label, ylabel=y_label)
    set_chart_titles(fig, ax, title, subtitle)
    place_legend_below(ax, ncol=3)
    save_figure(fig, output_path, bottom=0.22)


def plot_bea_panels(
    bea_growth: pd.DataFrame,
    bea_industries: pd.DataFrame,
    figures_dir: Path | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
) -> list[Path]:
    cfg = _load_config()
    start = period_start or cfg["bea_period"]["start"]
    end = period_end or cfg["bea_period"]["end"]
    figures_dir = figures_dir or FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel = _filter_private(bea_growth, bea_industries)
    panel = _period_filter(panel, start, end)
    panel = _exclude_names(panel, cfg.get("exclude", []))
    panel = _exclude_names(panel, cfg.get("exclude_subsectors", []))

    sector_names = {n.lower() for n in cfg["sector_plot"]}
    sectors = panel[panel["industry_name"].str.lower().isin(sector_names)]

    subsectors = panel[
        (panel["indent_level"] >= cfg["subsector_plot_parent_min_indent"])
        & (panel["indent_level"] <= cfg["subsector_plot_max_indent"])
    ]
    subsectors = subsectors[~subsectors["industry_name"].str.lower().isin(sector_names)]

    outputs = []
    t, s = format_chart_title("Price vs Quantity Growth", "quarterly", start, end)
    p1 = figures_dir / "bea_sector_price_vs_quantity_growth.png"
    _scatter_growth(
        sectors,
        "price_growth",
        "quantity_growth",
        t,
        "Price growth (%ΔP)",
        "Quantity growth (%ΔQ)",
        p1,
        subtitle=s,
    )
    outputs.append(p1)

    t, s = format_chart_title("Price vs Quantity Growth", "quarterly", start, end, scope="by Subsector")
    p2 = figures_dir / "bea_subsector_price_vs_quantity_growth.png"
    _scatter_growth(
        subsectors,
        "price_growth",
        "quantity_growth",
        t,
        "Price growth (%ΔP)",
        "Quantity growth (%ΔQ)",
        p2,
        subtitle=s,
    )
    outputs.append(p2)

    return outputs


def plot_bls_panels(
    bls_quarterly: pd.DataFrame,
    bls_industries: pd.DataFrame,
    figures_dir: Path | None = None,
    period_start: str | None = None,
    period_end: str | None = None,
) -> list[Path]:
    cfg = _load_config()
    start = period_start or cfg["bea_period"]["start"]
    end = period_end or cfg["bea_period"]["end"]
    figures_dir = figures_dir or FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel = bls_quarterly.copy()
    if "industry_name" not in panel.columns or "is_private" not in panel.columns:
        panel = panel.merge(
            bls_industries[["line_id", "industry_name", "indent_level", "is_private"]],
            on="line_id",
        )
    if "is_private" in panel.columns:
        panel = panel[panel["is_private"].eq(1) | panel["is_private"].eq(True)]
    panel = _period_filter(panel, start, end)
    panel = panel.dropna(subset=["employment_thousands_growth", "avg_hourly_earnings_growth"])
    panel = _exclude_names(panel, cfg.get("exclude", []))
    panel = _exclude_names(panel, cfg.get("exclude_subsectors", []))

    sector_names = {n.lower() for n in cfg["sector_plot"]}
    sectors = panel[panel["industry_name"].str.lower().isin(sector_names)]
    subsectors = panel[
        (panel["indent_level"] >= cfg["subsector_plot_parent_min_indent"])
        & (panel["indent_level"] <= cfg["subsector_plot_max_indent"])
    ]
    subsectors = subsectors[~subsectors["industry_name"].str.lower().isin(sector_names)]

    outputs = []
    t, s = format_chart_title("Wage vs Employment Growth", "quarterly", start, end)
    p1 = figures_dir / "bls_sector_wage_vs_employment_growth.png"
    _scatter_growth(
        sectors,
        "avg_hourly_earnings_growth",
        "employment_thousands_growth",
        t,
        "Average hourly earnings growth",
        "Employment growth",
        p1,
        subtitle=s,
    )
    outputs.append(p1)

    t, s = format_chart_title("Wage vs Employment Growth", "quarterly", start, end, scope="by Subsector")
    p2 = figures_dir / "bls_subsector_wage_vs_employment_growth.png"
    _scatter_growth(
        subsectors,
        "avg_hourly_earnings_growth",
        "employment_thousands_growth",
        t,
        "Average hourly earnings growth",
        "Employment growth",
        p2,
        subtitle=s,
    )
    outputs.append(p2)

    return outputs
