"""Write snapshot, excel, and horizon CSV exports for each profile."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.analysis.export_utils import pivot_growth_wide, write_excel_csv
from src.analysis.growth_horizons import (
    compute_bls_endpoint_growth,
    compute_bls_yearly_q1_growth,
    compute_endpoint_growth,
    compute_yearly_q1_growth,
    quarterly_growth_panel,
)
from src.analysis.industry_filters import PROFILES, profile_exclude_line_ids
from src.db.load_sqlite import run_query

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "snapshots"
EXCEL_DIR = PROJECT_ROOT / "data" / "excel"


def _apply_line_excludes(df: pd.DataFrame, exclude_ids: list[int]) -> pd.DataFrame:
    if not exclude_ids or df.empty:
        return df
    return df[~df["line_id"].isin(exclude_ids)].copy()


def export_all(
    db_path: Path,
    industries: pd.DataFrame,
    period_start: str,
    period_end: str,
    bea_source: str = "api",
    bls_source: str = "api",
) -> Path:
    bea_obs = run_query(db_path, "SELECT line_id, period, metric, value FROM bea_observations")
    bea_growth = run_query(
        db_path,
        "SELECT line_id, industry_name, indent_level, is_private, period, "
        "quantity_growth, price_growth, qp_ratio FROM v_bea_growth",
    )
    bls_q = run_query(db_path, "SELECT * FROM bls_quarterly_growth")

    endpoint_all = compute_endpoint_growth(bea_obs, industries, period_start, period_end)
    yearly_all = compute_yearly_q1_growth(bea_obs, industries)
    bls_endpoint_all = compute_bls_endpoint_growth(bls_q, industries, period_start, period_end)
    bls_yearly_all = compute_bls_yearly_q1_growth(bls_q, industries)
    quarterly_all = quarterly_growth_panel(bea_growth)

    manifest: dict = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "period_start": period_start,
        "period_end": period_end,
        "bea_source": bea_source,
        "bls_source": bls_source,
        "growth_formula": "QoQ from BEA tables 8+11 indexes: (I_t - I_{t-1}) / I_{t-1}",
        "profiles": [],
    }

    for profile in PROFILES:
        exclude_ids = profile_exclude_line_ids(profile)
        profile_info: dict = {"name": profile, "excludes_line_ids": exclude_ids, "files": []}

        snap_dir = SNAPSHOTS_DIR / profile
        excel_dir = EXCEL_DIR / profile
        snap_dir.mkdir(parents=True, exist_ok=True)
        excel_dir.mkdir(parents=True, exist_ok=True)

        exports = {
            "bea_observations.csv": _apply_line_excludes(bea_obs, exclude_ids),
            "bea_growth_quarterly.csv": _apply_line_excludes(quarterly_all, exclude_ids),
            "bea_growth_panel.csv": _apply_line_excludes(quarterly_all, exclude_ids),
            "bea_growth_endpoint.csv": _apply_line_excludes(endpoint_all, exclude_ids),
            "bea_growth_yearly_q1.csv": _apply_line_excludes(yearly_all, exclude_ids),
            "bls_quarterly_growth.csv": _apply_line_excludes(bls_q, exclude_ids),
            "bls_growth_endpoint.csv": _apply_line_excludes(bls_endpoint_all, exclude_ids),
            "bls_growth_yearly_q1.csv": _apply_line_excludes(bls_yearly_all, exclude_ids),
        }

        for filename, frame in exports.items():
            out_snap = snap_dir / filename
            write_excel_csv(frame, out_snap)
            profile_info["files"].append(str(out_snap.relative_to(PROJECT_ROOT)))

            if filename in {
                "bea_growth_panel.csv",
                "bea_growth_quarterly.csv",
                "bea_growth_endpoint.csv",
                "bea_growth_yearly_q1.csv",
                "bls_quarterly_growth.csv",
                "bls_growth_endpoint.csv",
                "bls_growth_yearly_q1.csv",
            }:
                out_xl = excel_dir / filename
                write_excel_csv(frame, out_xl)
                profile_info["files"].append(str(out_xl.relative_to(PROJECT_ROOT)))

        q_panel = exports["bea_growth_quarterly.csv"]
        if not q_panel.empty:
            for col, name in [
                ("quantity_growth", "bea_pivot_quantity_growth.csv"),
                ("price_growth", "bea_pivot_price_growth.csv"),
                ("qp_ratio", "bea_pivot_qp_ratio.csv"),
            ]:
                wide = pivot_growth_wide(q_panel, col)
                path = excel_dir / name
                write_excel_csv(wide, path)
                profile_info["files"].append(str(path.relative_to(PROJECT_ROOT)))

        bls_panel = exports["bls_quarterly_growth.csv"]
        if not bls_panel.empty and "period" in bls_panel.columns:
            if "industry_name" not in bls_panel.columns:
                meta = industries[["line_id", "industry_name"]].drop_duplicates("line_id")
                bls_panel = bls_panel.merge(meta, on="line_id", how="left")
            for col, name in [
                ("avg_hourly_earnings_growth", "bls_pivot_wage_growth.csv"),
                ("employment_thousands_growth", "bls_pivot_employment_growth.csv"),
            ]:
                wide = pivot_growth_wide(bls_panel, col)
                path = excel_dir / name
                write_excel_csv(wide, path)
                profile_info["files"].append(str(path.relative_to(PROJECT_ROOT)))

        profile_info["row_counts"] = {k: len(v) for k, v in exports.items()}
        manifest["profiles"].append(profile_info)

    manifest_path = SNAPSHOTS_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
