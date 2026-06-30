#!/usr/bin/env python3
"""End-to-end pipeline: ETL -> SQLite -> plots -> ML evaluation."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.env import load_project_env

load_project_env()

from src.analysis.snapshot_export import export_all
from src.analysis.export_utils import add_qp_sign_case, write_excel_csv
from src.analysis.growth_horizons import (
    compute_bls_endpoint_growth,
    compute_bls_yearly_q1_growth,
    compute_endpoint_growth,
    compute_yearly_q1_growth,
)
from src.analysis.industry_filters import PROFILES, apply_profile, figures_profile_dir, load_config
from src.analysis.ml_pipeline import run_ml_pipeline
from src.analysis.plots import plot_bea_panels, plot_bls_panels
from src.analysis.timeseries_plots import plot_all_horizons
from src.db.load_sqlite import load_database, run_query
from src.etl.aggregate_bls import build_bls_from_api
from src.etl.fetch_bea import load_bea_data
from src.etl.fetch_bls import DEFAULT_CACHE, fetch_bls_series
from src.etl.parse_bea import parse_bea_csv
from src.etl.parse_bls import bls_quarterly_growth, parse_bls_csv
from src.etl.parse_crosswalk import expand_series_ids, load_crosswalk, load_registry

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

REGISTRY_COLUMNS = ["line_id", "industry_name", "indent_level", "is_private", "plot_level"]


def current_quarter() -> str:
    today = date.today()
    quarter = (today.month - 1) // 3 + 1
    return f"{today.year}-Q{quarter}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Industry growth & inflation analysis pipeline")
    parser.add_argument("--bea-source", choices=["csv", "api"], default="api")
    parser.add_argument("--bls-source", choices=["csv", "api"], default="api")
    parser.add_argument("--start", default="2019-Q1", help="Start period (YYYY-Qn)")
    parser.add_argument("--end", default=None, help="End period (YYYY-Qn); default is current quarter")
    parser.add_argument("--start-year", type=int, default=None, help="API fetch start year")
    parser.add_argument("--end-year", type=int, default=None, help="API fetch end year")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force API re-fetch, use api sources, 2019-Q1 through current quarter",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use legacy CSV sources only (no API calls)",
    )
    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help="Delete and recreate SQLite database from scratch",
    )
    parser.add_argument(
        "--export-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write data/snapshots and data/excel after run (default: on)",
    )
    return parser.parse_args()


def _apply_mode_flags(args: argparse.Namespace) -> argparse.Namespace:
    if args.refresh:
        args.bea_source = "api"
        args.bls_source = "api"
        args.start = "2019-Q1"
        args.end = current_quarter()
    if args.offline:
        args.bea_source = "csv"
        args.bls_source = "csv"
    if args.end is None:
        args.end = current_quarter()
    return args


def _year_bounds(args: argparse.Namespace) -> tuple[int, int]:
    start_year = args.start_year or int(args.start.split("-")[0])
    end_year = args.end_year or int(args.end.split("-")[0])
    return start_year, end_year


def _require_api_key(env_var: str, label: str) -> None:
    if not os.environ.get(env_var):
        raise SystemExit(
            f"Missing {env_var} in .env (required for {label} API mode). "
            "Use --offline for legacy CSV mode."
        )


def _normalize_industries(
    bea_industries: pd.DataFrame,
    bls_industries: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reg = registry[REGISTRY_COLUMNS].drop_duplicates("line_id")

    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        base = df[["line_id"]].drop_duplicates().merge(reg, on="line_id", how="left")
        for col in ("industry_name", "indent_level", "is_private"):
            if col in df.columns:
                fallback = df[["line_id", col]].drop_duplicates("line_id").set_index("line_id")[col]
                base[col] = base[col].fillna(base["line_id"].map(fallback))
        if "plot_level" not in base.columns:
            base["plot_level"] = None
        return base[REGISTRY_COLUMNS]

    return enrich(bea_industries), enrich(bls_industries)


def _load_bls(args: argparse.Namespace):
    if args.bls_source == "csv":
        bls_path = RAW_DIR / "BLS Data.csv"
        bls_industries, bls_observations = parse_bls_csv(bls_path)
        bls_quarterly = bls_quarterly_growth(bls_observations)
        return bls_industries, bls_observations, bls_quarterly

    _require_api_key("BLS_API_KEY", "BLS")
    start_year, end_year = _year_bounds(args)
    crosswalk = load_crosswalk()
    series_ids = expand_series_ids(crosswalk)["ces_series_id"].unique().tolist()
    print(f"      Fetching {len(series_ids)} CES series from BLS API ({start_year}-{end_year})...")
    try:
        api_df = fetch_bls_series(
            series_ids,
            start_year=start_year,
            end_year=end_year,
            use_cache=not args.refresh,
            force_refresh=args.refresh,
        )
    except RuntimeError as exc:
        print(f"      BLS API unavailable ({exc}); trying cache...")
        if DEFAULT_CACHE.exists():
            api_df = pd.read_csv(DEFAULT_CACHE)
            api_df = api_df[api_df["series_id"].isin(series_ids)]
            if api_df.empty:
                raise SystemExit("BLS API failed and cache has no matching series.") from exc
        else:
            raise SystemExit(f"BLS API failed and no cache available: {exc}") from exc

    if not api_df.empty:
        years = api_df["year"].astype(int)
        print(f"      BLS data years: {years.min()}-{years.max()}")
    return build_bls_from_api(api_df, crosswalk)


def _load_bea(args: argparse.Namespace):
    start_year, end_year = _year_bounds(args)
    if args.bea_source == "csv":
        return parse_bea_csv(RAW_DIR / "BEA Value Added.csv")
    _require_api_key("BEA_API_KEY", "BEA")
    print(f"      Fetching BEA GDPbyIndustry ({start_year}-{end_year})...")
    try:
        industries, observations = load_bea_data("api", start_year=start_year, end_year=end_year)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "not active" in msg or "userid" in msg:
            raise SystemExit(
                "BEA API key is inactive. Activate it at https://apps.bea.gov/API/signup/ "
                "or run with --bea-source csv / --offline."
            ) from exc
        raise
    if not observations.empty:
        periods = observations["period"].sort_values()
        print(f"      BEA data periods: {periods.iloc[0]} to {periods.iloc[-1]}")
    return industries, observations


def _effective_end_period(bea_observations: pd.DataFrame, requested_end: str) -> str:
    if bea_observations.empty or "period" not in bea_observations.columns:
        return requested_end
    latest = bea_observations["period"].max()
    return latest if latest < requested_end else requested_end


def main() -> None:
    args = _apply_mode_flags(_parse_args())
    print("== Industry Growth & Inflation Analysis ==")
    print(f"      BEA source: {args.bea_source} | BLS source: {args.bls_source}")
    print(f"      Period: {args.start} to {args.end}")

    print("[1/6] Loading BEA data...")
    bea_industries, bea_observations = _load_bea(args)

    print("[2/6] Loading BLS data...")
    bls_industries, bls_observations, bls_quarterly = _load_bls(args)

    registry = load_registry()
    bea_industries, bls_industries = _normalize_industries(bea_industries, bls_industries, registry)

    crosswalk = load_crosswalk() if (PROJECT_ROOT / "config" / "bea_bls_crosswalk.csv").exists() else None

    print("[3/6] Loading SQLite database...")
    db_path = load_database(
        bea_industries,
        bea_observations,
        bls_industries,
        bls_observations,
        bls_quarterly,
        crosswalk=crosswalk,
        rebuild_db=args.rebuild_db,
    )
    print(f"      Database: {db_path}")

    bea_growth_full = run_query(
        db_path,
        "SELECT line_id, industry_name, indent_level, is_private, period, "
        "quantity_growth, price_growth, qp_ratio FROM v_bea_growth",
    )
    bea_obs_db = run_query(db_path, "SELECT line_id, period, metric, value FROM bea_observations")

    effective_end = _effective_end_period(bea_obs_db, args.end)
    if effective_end != args.end:
        print(f"      Endpoint/plots use latest available quarter: {effective_end} (requested {args.end})")

    endpoint_all = compute_endpoint_growth(bea_obs_db, bea_industries, args.start, effective_end)
    yearly_all = compute_yearly_q1_growth(bea_obs_db, bea_industries)
    bls_endpoint_all = compute_bls_endpoint_growth(bls_quarterly, bls_industries, args.start, effective_end)
    bls_yearly_all = compute_bls_yearly_q1_growth(bls_quarterly, bls_industries)

    all_label_rows: list[pd.DataFrame] = []

    print("[4/6] Generating plots (both profiles)...")
    for profile in PROFILES:
        bea_growth = apply_profile(
            bea_growth_full,
            bea_industries,
            profile,
            period_start=args.start,
            period_end=args.end,
        )
        bea_growth = add_qp_sign_case(bea_growth)
        bls_filtered = apply_profile(
            bls_quarterly,
            bls_industries,
            profile,
            period_start=args.start,
            period_end=args.end,
        )
        endpoint = apply_profile(endpoint_all, bea_industries, profile)
        yearly = apply_profile(yearly_all, bea_industries, profile)
        bls_endpoint = apply_profile(bls_endpoint_all, bls_industries, profile)
        bls_yearly = apply_profile(bls_yearly_all, bls_industries, profile)

        scatter_dir = figures_profile_dir(profile, "scatter")
        bea_plots = plot_bea_panels(
            bea_growth,
            bea_industries,
            figures_dir=scatter_dir,
            period_start=args.start,
            period_end=args.end,
        )
        bls_plots = plot_bls_panels(
            bls_filtered,
            bls_industries,
            figures_dir=scatter_dir,
            period_start=args.start,
            period_end=args.end,
        )
        horizon_plots = plot_all_horizons(
            bea_growth,
            bea_industries,
            endpoint,
            yearly,
            bea_growth,
            bls_filtered,
            bls_industries,
            bls_endpoint,
            bls_yearly,
            figures_profile_dir(profile),
            args.start,
            args.end,
        )
        for path in bea_plots + bls_plots + horizon_plots:
            print(f"      [{profile}] Saved {path}")

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        write_excel_csv(bea_growth, PROCESSED_DIR / f"bea_growth_panel_{profile}.csv")
        write_excel_csv(bls_filtered, PROCESSED_DIR / f"bls_quarterly_growth_{profile}.csv")

        print(f"[5/6] ML pipeline ({profile})...")
        results, labels, cluster_legend = run_ml_pipeline(
            bea_growth_full,
            bea_industries,
            output_dir=OUTPUT_DIR,
            profile=profile,
            period_start=args.start,
            period_end=args.end,
        )
        all_label_rows.append(labels)
        print(f"      Industries modeled: {results['n_industries']}")
        print(f"      Silhouette score: {results['silhouette_score']:.3f}")
        print(f"      Regime distribution: {results['regime_counts']}")

    if all_label_rows:
        labels_combined = pd.concat(all_label_rows, ignore_index=True)
        with sqlite3.connect(db_path) as conn:
            labels_combined.to_sql("industry_labels", conn, if_exists="replace", index=False)

    if args.export_snapshots:
        print("[6/6] Exporting snapshots...")
        manifest = export_all(
            db_path,
            bea_industries,
            args.start,
            effective_end,
            bea_source=args.bea_source,
            bls_source=args.bls_source,
        )
        print(f"      Manifest: {manifest}")
    else:
        print("[6/6] Skipping snapshot export (--no-export-snapshots)")

    default_profile = load_config().get("default_plot_profile", "excl_trust_funds")
    top_inflation = run_query(
        db_path,
        f"""
        SELECT industry_name, AVG(price_growth) AS avg_price_growth
        FROM v_bea_growth
        WHERE is_private = 1 AND period >= '{args.start}' AND period <= '{args.end}'
          AND line_id NOT IN (1, 2, 59)
        GROUP BY industry_name
        ORDER BY avg_price_growth DESC
        LIMIT 5
        """,
    )
    print(f"\nTop 5 private-sector price growth ({args.start}–{args.end} avg, excl. aggregates/trust funds):")
    print(top_inflation.to_string(index=False))
    print(f"\nFigures: outputs/figures/{default_profile}/")
    print("Done.")


if __name__ == "__main__":
    main()
