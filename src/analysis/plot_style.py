"""Shared matplotlib styling: titles, layout, legends, save."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def format_period_label(period: str) -> str:
    if "-Q" in period:
        year, quarter = period.split("-", 1)
        return f"{year} {quarter}"
    return period


def format_date_range(start: str | None, end: str | None) -> str:
    if start and end:
        return f"{format_period_label(start)} – {format_period_label(end)}"
    if end:
        return format_period_label(end)
    return ""


def format_chart_title(
    metric: str,
    horizon: str,
    start: str | None = None,
    end: str | None = None,
    *,
    scope: str = "by Sector",
) -> tuple[str, str]:
    """Return (main_title, subtitle)."""
    main = f"{metric} {scope}".strip()
    date_part = format_date_range(start, end)
    if horizon.lower() == "endpoint":
        subtitle = f"{date_part} total change" if date_part else "Total change"
    elif horizon.lower() in {"quarterly", "qoq"}:
        subtitle = f"Quarterly · {date_part}" if date_part else "Quarterly"
    elif horizon.lower() in {"yearly", "q1→q1", "q1-q1"}:
        subtitle = f"Q1→Q1 · {date_part}" if date_part else "Q1→Q1"
    else:
        subtitle = date_part or horizon
    return main, subtitle


def short_sector_name(name: str, max_len: int = 22) -> str:
    if len(name) <= max_len:
        return name
    return name.split(",")[0][:max_len]


def apply_axes_style(ax: plt.Axes, *, ylabel: str | None = None, xlabel: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="-", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    ax.tick_params(axis="both", labelsize=8)


def set_chart_titles(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    subtitle: str | None = None,
    *,
    footnote: str | None = None,
) -> None:
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.06, ha="left", y=0.98)
    if subtitle:
        ax.set_title(subtitle, fontsize=9, color="#555555", loc="left", pad=12)
    if footnote:
        fig.text(0.06, 0.01, footnote, fontsize=8, color="#888888", style="italic", ha="left")


def place_legend_below(ax: plt.Axes, *, ncol: int = 3, fontsize: float = 7) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    # Deduplicate while preserving order (dual-metric charts label only first line).
    seen: set[str] = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels, strict=True):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    ax.legend(
        unique_handles,
        unique_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(ncol, max(1, len(unique_labels))),
        fontsize=fontsize,
        frameon=False,
    )


def save_figure(fig: plt.Figure, output_path: Path, *, bottom: float = 0.18) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=bottom)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
