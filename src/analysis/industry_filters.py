"""Shared industry filtering for plots, ML, and exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "industries.yaml"

TRUST_FUNDS_LINE_ID = 59
PROFILES = ("full", "excl_trust_funds")
PROFILE_LABELS = {
    "full": "All private industries (including trust funds)",
    "excl_trust_funds": "Private industries excluding trust funds",
}


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def trust_funds_line_ids(cfg: dict | None = None) -> list[int]:
    cfg = cfg or load_config()
    tf = cfg.get("trust_funds", {})
    line_id = tf.get("line_id", TRUST_FUNDS_LINE_ID)
    return [int(line_id)]


def profile_exclude_line_ids(profile: str, cfg: dict | None = None) -> list[int]:
    if profile == "excl_trust_funds":
        return trust_funds_line_ids(cfg)
    return []


def _exclude_names(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    if df.empty or "industry_name" not in df.columns:
        return df
    lowered = {n.lower() for n in names}
    return df[~df["industry_name"].str.lower().isin(lowered)]


def _merge_industry_meta(df: pd.DataFrame, industries: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["line_id", "industry_name", "indent_level", "is_private", "plot_level"]
    available = [c for c in meta_cols if c in industries.columns]
    meta = industries[available].drop_duplicates("line_id")
    if "line_id" not in df.columns:
        return df
    drop = [c for c in ("industry_name", "indent_level", "is_private", "plot_level") if c in df.columns]
    base = df.drop(columns=drop, errors="ignore").merge(meta, on="line_id", how="left")
    return base


def apply_profile(
    df: pd.DataFrame,
    industries: pd.DataFrame,
    profile: str,
    *,
    period_start: str | None = None,
    period_end: str | None = None,
    plot_eligible_only: bool = False,
) -> pd.DataFrame:
    """Filter panel to a profile (full vs excl_trust_funds) and yaml plot rules."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")

    cfg = load_config()
    panel = _merge_industry_meta(df.copy(), industries)

    if "is_private" in panel.columns:
        panel = panel[panel["is_private"].eq(1) | panel["is_private"].eq(True)]

    if period_start and period_end and "period" in panel.columns:
        panel = panel[(panel["period"] >= period_start) & (panel["period"] <= period_end)]

    panel = _exclude_names(panel, cfg.get("exclude", []))
    panel = _exclude_names(panel, cfg.get("exclude_subsectors", []))

    if "plot_level" in panel.columns:
        panel = panel[~panel["plot_level"].isin(["exclude", "aggregate"])]

    exclude_ids = profile_exclude_line_ids(profile, cfg)
    if exclude_ids:
        panel = panel[~panel["line_id"].isin(exclude_ids)]

    if plot_eligible_only:
        sector_names = {n.lower() for n in cfg["sector_plot"]}
        sectors = panel[panel["industry_name"].str.lower().isin(sector_names)]
        subsectors = panel[
            (panel["indent_level"] >= cfg["subsector_plot_parent_min_indent"])
            & (panel["indent_level"] <= cfg["subsector_plot_max_indent"])
            & ~panel["industry_name"].str.lower().isin(sector_names)
        ]
        panel = pd.concat([sectors, subsectors], ignore_index=True).drop_duplicates(
            subset=["line_id", "period"] if "period" in panel.columns else ["line_id"]
        )

    return panel.reset_index(drop=True)


def sector_subsector_split(panel: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or load_config()
    sector_names = {n.lower() for n in cfg["sector_plot"]}
    sectors = panel[panel["industry_name"].str.lower().isin(sector_names)]
    subsectors = panel[
        (panel["indent_level"] >= cfg["subsector_plot_parent_min_indent"])
        & (panel["indent_level"] <= cfg["subsector_plot_max_indent"])
        & ~panel["industry_name"].str.lower().isin(sector_names)
    ]
    return sectors, subsectors


def figures_profile_dir(profile: str, subfolder: str = "") -> Path:
    base = PROJECT_ROOT / "outputs" / "figures" / profile
    return base / subfolder if subfolder else base
