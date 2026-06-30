#!/usr/bin/env python3
"""Validate crosswalk: BLS API series exist and aggregated data matches legacy CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.env import load_project_env

load_project_env()

from src.etl.aggregate_bls import build_bls_from_api
from src.etl.fetch_bls import DEFAULT_CACHE, fetch_bls_series
from src.etl.parse_bls import bls_quarterly_growth, parse_bls_csv
from src.etl.parse_crosswalk import CROSSWALK_PATH, expand_series_ids, load_crosswalk

RAW_BLS = PROJECT_ROOT / "data" / "raw" / "BLS Data.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "crosswalk_validation.json"


def validate_series_exist(series_ids: list[str], start_year: int, end_year: int) -> dict:
    missing: list[str] = []
    found: list[str] = []
    batch = 50
    for i in range(0, len(series_ids), batch):
        chunk = series_ids[i : i + batch]
        df = fetch_bls_series(chunk, start_year=start_year, end_year=end_year)
        found_chunk = set(df["series_id"].unique()) if not df.empty else set()
        for sid in chunk:
            if sid in found_chunk:
                found.append(sid)
            else:
                missing.append(sid)
    return {
        "total_series": len(series_ids),
        "found": len(found),
        "missing": missing,
        "missing_count": len(missing),
    }


def compare_to_csv(
    api_quarterly: pd.DataFrame,
    csv_quarterly: pd.DataFrame,
    tolerance: float = 0.05,
) -> dict:
    """Compare employment and wage growth for overlapping 2021-2022 quarters."""
    periods = [p for p in api_quarterly["period"].unique() if p.startswith("2021") or p.startswith("2022")]
    api = api_quarterly[api_quarterly["period"].isin(periods)].copy()
    csv = csv_quarterly[csv_quarterly["period"].isin(periods)].copy()

    merged = api.merge(
        csv,
        on=["line_id", "period"],
        suffixes=("_api", "_csv"),
        how="inner",
    )

    metrics = [
        "employment_thousands_growth",
        "avg_hourly_earnings_growth",
    ]
    comparisons: list[dict] = []
    for metric in metrics:
        api_col = f"{metric}_api"
        csv_col = f"{metric}_csv"
        if api_col not in merged.columns or csv_col not in merged.columns:
            continue
        valid = merged.dropna(subset=[api_col, csv_col])
        valid = valid[(valid[api_col].abs() < 10) & (valid[csv_col].abs() < 10)]
        if valid.empty:
            continue
        diff = (valid[api_col] - valid[csv_col]).abs()
        comparisons.append(
            {
                "metric": metric,
                "pairs": len(valid),
                "within_tolerance": int((diff <= tolerance).sum()),
                "pct_within_tolerance": float((diff <= tolerance).mean()),
                "mean_abs_diff": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
            }
        )

    return {
        "overlapping_line_periods": len(merged),
        "metrics": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BEA/BLS crosswalk against legacy CSV.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args()

    crosswalk = load_crosswalk()
    series_df = expand_series_ids(crosswalk)
    series_ids = series_df["ces_series_id"].unique().tolist()

    print(f"Validating {len(series_ids)} CES series ({crosswalk['bea_line_id'].nunique()} BEA industries)...")

    employment_ids = [s for s in series_ids if s.endswith("01")]
    existence = validate_series_exist(employment_ids, args.start_year, args.end_year)
    print(f"  Employment series found: {existence['found']}/{existence['total_series']}")
    if existence["missing_count"]:
        print(f"  Missing employment ({existence['missing_count']}): {existence['missing'][:10]}...")

    # Fetch all series; BLS omits earnings for some industries (employment-only publication).
    api_raw = fetch_bls_series(series_ids, start_year=args.start_year, end_year=args.end_year)
    if not api_raw.empty and DEFAULT_CACHE:
        DEFAULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        api_raw.to_csv(DEFAULT_CACHE, index=False)
        print(f"  Cached BLS API data -> {DEFAULT_CACHE}")
    _, _, api_quarterly = build_bls_from_api(api_raw, crosswalk)

    _, csv_obs = parse_bls_csv(RAW_BLS)
    csv_quarterly = bls_quarterly_growth(csv_obs)

    comparison = compare_to_csv(api_quarterly, csv_quarterly, tolerance=args.tolerance)

    report = {
        "existence": existence,
        "comparison": comparison,
        "pass": existence["missing_count"] == 0
        and all(m.get("pct_within_tolerance", 0) >= 0.85 for m in comparison.get("metrics", [])),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")
    print(json.dumps(report, indent=2))

    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
