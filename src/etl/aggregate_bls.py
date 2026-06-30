"""Aggregate multi-series BLS observations to BEA industry grain."""

from __future__ import annotations

import pandas as pd

from src.etl.parse_bls import bls_quarterly_growth
from src.etl.parse_crosswalk import expand_series_ids, load_crosswalk, load_registry


def bls_api_observations_to_tidy(
    api_df: pd.DataFrame,
    crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map raw BLS API rows to line_id / period_month / metric tidy format."""
    crosswalk = crosswalk if crosswalk is not None else load_crosswalk()

    series_map = expand_series_ids(crosswalk)[
        ["ces_series_id", "bea_line_id", "metric"]
    ].drop_duplicates()

    merged = api_df.merge(series_map, left_on="series_id", right_on="ces_series_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["line_id", "period_month", "metric", "value"])

    merged = merged.rename(columns={"bea_line_id": "line_id"})
    return merged[["line_id", "period_month", "metric", "value"]].sort_values(
        ["line_id", "period_month", "metric"]
    )


def aggregate_bls_observations_weighted(
    component_obs: pd.DataFrame,
    crosswalk: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Roll component CES series up to BEA line_id with employment-weighted earnings."""
    if component_obs.empty:
        return component_obs

    crosswalk = crosswalk if crosswalk is not None else load_crosswalk()
    multi_lines = set(
        crosswalk.groupby("bea_line_id").filter(lambda g: len(g) > 1)["bea_line_id"].astype(int)
    )

    records: list[dict] = []
    for (line_id, period_month), group in component_obs.groupby(["line_id", "period_month"]):
        line_id = int(line_id)
        row_metrics: dict[str, float] = {}
        emp_vals = group.loc[group["metric"] == "employment_thousands", "value"].dropna()
        if not emp_vals.empty:
            row_metrics["employment_thousands"] = float(emp_vals.sum())

        for metric in ("avg_hourly_earnings", "avg_weekly_earnings"):
            earn = group.loc[group["metric"] == metric, "value"].dropna()
            if earn.empty:
                continue
            if line_id in multi_lines and len(earn) > 1 and len(emp_vals) == len(earn):
                weights = emp_vals.values
                if weights.sum() > 0:
                    row_metrics[metric] = float((earn.values * weights).sum() / weights.sum())
                else:
                    row_metrics[metric] = float(earn.mean())
            elif len(earn) > 1:
                row_metrics[metric] = float(earn.mean())
            else:
                row_metrics[metric] = float(earn.iloc[0])

        for metric, value in row_metrics.items():
            records.append(
                {
                    "line_id": line_id,
                    "period_month": period_month,
                    "metric": metric,
                    "value": value,
                }
            )

    return pd.DataFrame(records).sort_values(["line_id", "period_month", "metric"])


def build_bls_from_api(
    api_df: pd.DataFrame,
    crosswalk: pd.DataFrame | None = None,
    registry: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (industries, observations, quarterly_growth) at BEA grain."""
    crosswalk = crosswalk if crosswalk is not None else load_crosswalk()
    registry = registry if registry is not None else load_registry()

    component = bls_api_observations_to_tidy(api_df, crosswalk)
    observations = aggregate_bls_observations_weighted(component, crosswalk)
    quarterly = bls_quarterly_growth(observations)

    mapped_ids = set(crosswalk["bea_line_id"].astype(int))
    industries = registry[registry["line_id"].isin(mapped_ids)].copy()

    return industries, observations, quarterly
