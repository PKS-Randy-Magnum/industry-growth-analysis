#!/usr/bin/env python3
"""Validate snapshot BEA growth: self-consistency + optional live API check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.env import load_project_env

load_project_env()

from src.etl.fetch_bea import fetch_bea_index_observations
from src.etl.parse_bea import compute_bea_growth_from_indexes
from src.etl.parse_crosswalk import load_registry

SNAPSHOT_OBS = PROJECT_ROOT / "data" / "snapshots" / "full" / "bea_observations.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "bea_growth_validation.json"

GROWTH_METRICS = ["quantity_growth", "price_growth", "quantity_price_ratio"]
TOLERANCE = 1e-4
START = "2021-Q1"
END = "2022-Q4"


def _growth_frame(observations, label: str) -> "pd.DataFrame":
    import pandas as pd

    obs = observations[observations["metric"].isin(GROWTH_METRICS)].copy()
    obs = obs[(obs["period"] >= START) & (obs["period"] <= END)]
    wide = obs.pivot_table(
        index=["line_id", "period"],
        columns="metric",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide["source"] = label
    return wide


def _compare_metrics(left, right, left_label: str, right_label: str) -> tuple[dict, bool]:
    merged = left.merge(right, on=["line_id", "period"], how="inner", suffixes=("_a", "_b"))
    metrics_report = []
    all_pass = True
    for metric in GROWTH_METRICS:
        a_col, b_col = f"{metric}_a", f"{metric}_b"
        pairs = merged[[a_col, b_col]].dropna()
        if pairs.empty:
            metrics_report.append({"metric": metric, "pairs": 0, "pass": False})
            all_pass = False
            continue
        diff = (pairs[a_col] - pairs[b_col]).abs()
        entry = {
            "metric": metric,
            "pairs": int(len(pairs)),
            "within_tolerance": int((diff <= TOLERANCE).sum()),
            "pct_within_tolerance": float((diff <= TOLERANCE).mean()),
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
            "pass": bool((diff <= TOLERANCE).all()),
            "compare": f"{left_label} vs {right_label}",
        }
        metrics_report.append(entry)
        if not entry["pass"]:
            all_pass = False
    return {"metrics": metrics_report, "pass": all_pass}


def main() -> int:
    import pandas as pd

    print(f"Validating BEA growth ({START} to {END})...")

    if not SNAPSHOT_OBS.exists():
        print(f"Snapshot not found: {SNAPSHOT_OBS}. Run: python run.py")
        return 1

    snap_obs = pd.read_csv(SNAPSHOT_OBS)
    stored_growth = _growth_frame(snap_obs, "stored")

    index_obs = snap_obs[snap_obs["metric"].isin(["quantity_index", "price_index"])].copy()
    recomputed = compute_bea_growth_from_indexes(index_obs[["line_id", "period", "metric", "value"]])
    recomputed_growth = _growth_frame(recomputed, "recomputed")

    self_check = _compare_metrics(recomputed_growth, stored_growth, "recomputed", "stored")

    registry = load_registry()
    name_to_line = dict(zip(registry["industry_name"], registry["line_id"]))
    api_index = fetch_bea_index_observations(start_year=2021, end_year=2022)
    api_index["line_id"] = api_index["industry_name"].map(name_to_line)
    api_index = api_index.dropna(subset=["line_id"]).copy()
    api_index["line_id"] = api_index["line_id"].astype(int)
    api_obs = compute_bea_growth_from_indexes(api_index[["line_id", "period", "metric", "value"]])
    api_growth = _growth_frame(api_obs, "api")
    api_check = _compare_metrics(recomputed_growth, api_growth, "snapshot_indexes", "live_api_2021_2022")

    report = {
        "window": f"{START} to {END}",
        "snapshot": str(SNAPSHOT_OBS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "tolerance": TOLERANCE,
        "self_consistency": self_check,
        "live_api_check": api_check,
        "pass": self_check["pass"],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {OUTPUT_PATH}")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
