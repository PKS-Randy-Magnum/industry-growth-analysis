"""Statistical outlier detection for chart annotations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def flag_univariate_outliers(series: pd.Series, method: str = "iqr", k: float = 1.5) -> pd.Series:
    """Boolean mask for outliers in a numeric series."""
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(False, index=series.index)
    if method == "zscore":
        std = valid.std()
        if std == 0 or pd.isna(std):
            return pd.Series(False, index=series.index)
        z = (s - valid.mean()).abs() / std
        return z > k
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    low, high = q1 - k * iqr, q3 + k * iqr
    return (s < low) | (s > high)


def flag_scatter_outliers(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    top_n: int = 3,
    id_col: str = "line_id",
) -> pd.DataFrame:
    """Return up to top_n rows farthest from the median (x, y)."""
    if df.empty:
        return df.iloc[0:0].copy()
    work = df.dropna(subset=[x_col, y_col]).copy()
    work = work[np.isfinite(work[x_col]) & np.isfinite(work[y_col])]
    if id_col in work.columns:
        work = work.drop_duplicates(id_col)
    if work.empty:
        return work
    mx, my = work[x_col].median(), work[y_col].median()
    sx = work[x_col].std() or 1.0
    sy = work[y_col].std() or 1.0
    work["_dist"] = np.sqrt(((work[x_col] - mx) / sx) ** 2 + ((work[y_col] - my) / sy) ** 2)
    return work.nlargest(min(top_n, len(work)), "_dist").drop(columns="_dist")


def flag_series_peaks(
    df: pd.DataFrame,
    group_col: str,
    y_col: str,
    *,
    top_n_per_series: int = 2,
    x_col: str | None = None,
) -> pd.DataFrame:
    """Return extreme points per series by absolute y."""
    if df.empty or y_col not in df.columns:
        return df.iloc[0:0].copy()
    work = df.dropna(subset=[y_col, group_col]).copy()
    work = work[np.isfinite(work[y_col])]
    if work.empty:
        return work
    work["_abs_y"] = work[y_col].abs()
    peaks = []
    for _name, group in work.groupby(group_col):
        peaks.append(group.nlargest(min(top_n_per_series, len(group)), "_abs_y"))
    out = pd.concat(peaks, ignore_index=True).drop(columns="_abs_y")
    if x_col and x_col in out.columns:
        return out
    return out


def annotate_points(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    label_col: str = "industry_name",
    x_vals: dict | None = None,
    fontsize: float = 7,
) -> None:
    """Add text labels for outlier rows on a matplotlib axes."""
    from src.analysis.plot_style import short_sector_name

    for row in df.itertuples():
        x = getattr(row, x_col) if not x_vals else x_vals.get(getattr(row, x_col, None), getattr(row, x_col))
        y = getattr(row, y_col)
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        name = getattr(row, label_col, "")
        ax.annotate(
            short_sector_name(str(name)),
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=fontsize,
            color="#333333",
            alpha=0.95,
        )
