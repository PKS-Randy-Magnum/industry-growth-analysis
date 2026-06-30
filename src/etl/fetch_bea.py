"""Fetch BEA GDP-by-Industry chain-type indexes via the BEA Data API."""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from src.etl.env import load_project_env
from src.etl.parse_bea import compute_bea_growth_from_indexes, parse_bea_csv

load_project_env()
from src.etl.parse_crosswalk import REGISTRY_PATH, load_registry

BEA_API_URL = "https://apps.bea.gov/api/data/"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BEA_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "BEA Value Added.csv"

# GDPbyIndustry: chain-type value-added indexes (iTable TVA103 / TVA104)
TABLE_QUANTITY_INDEX = 8
TABLE_PRICE_INDEX = 11

METRIC_BY_TABLE = {
    TABLE_QUANTITY_INDEX: "quantity_index",
    TABLE_PRICE_INDEX: "price_index",
}


def _bea_get(params: dict, api_key: str) -> list[dict]:
    params = {**params, "UserID": api_key, "method": "GetData", "ResultFormat": "JSON"}
    url = f"{BEA_API_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read().decode("utf-8")
    import json

    obj = json.loads(payload)
    if "BEAAPI" not in obj:
        raise RuntimeError(f"Unexpected BEA response: {obj}")

    results = obj["BEAAPI"].get("Results")
    if isinstance(results, list):
        if not results:
            raise RuntimeError("BEA API returned empty Results")
        block = results[0]
        if isinstance(block, dict) and "Error" in block:
            raise RuntimeError(block["Error"].get("APIErrorDescription", block["Error"]))
        if isinstance(block, dict):
            return block.get("Data", [])
        raise RuntimeError(f"Unexpected BEA Results entry: {type(block)}")

    if isinstance(results, dict):
        if "Error" in results:
            raise RuntimeError(results["Error"].get("APIErrorDescription", results["Error"]))
        return results.get("Data", [])

    raise RuntimeError(f"Unexpected BEA Results type: {type(results)}")


def _normalize_bea_name(name: str) -> str:
    return " ".join(name.split())


_ROMAN_QUARTER = {"I": "1", "II": "2", "III": "3", "IV": "4"}


def _quarter_number(raw: str) -> str:
    token = str(raw).strip().upper().lstrip("Q")
    return _ROMAN_QUARTER.get(token, token)


def fetch_bea_table(
    table_id: int,
    years: list[int] | str,
    api_key: str | None = None,
) -> pd.DataFrame:
    api_key = api_key or os.environ.get("BEA_API_KEY")
    if not api_key:
        raise ValueError("BEA_API_KEY is required for API fetch")

    year_param = "ALL" if years == "ALL" else ",".join(str(y) for y in years)
    rows = _bea_get(
        {
            "datasetname": "GDPbyIndustry",
            "TableID": table_id,
            "Frequency": "Q",
            "Year": year_param,
            "Industry": "ALL",
        },
        api_key,
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["metric"] = METRIC_BY_TABLE[table_id]
    df["period"] = (
        df["Year"].astype(str)
        + "-Q"
        + df["Quarter"].astype(str).map(_quarter_number)
    )
    df["value"] = pd.to_numeric(df["DataValue"].str.replace(",", "", regex=False), errors="coerce")
    df["industry_name"] = df["IndustrYDescription"].map(_normalize_bea_name)
    return df[["industry_name", "period", "metric", "value"]]


def fetch_bea_index_observations(
    start_year: int = 2021,
    end_year: int = 2025,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch chain-type quantity and price indexes only (Tables 8 and 11)."""
    years = list(range(start_year, end_year + 1))
    frames = [
        fetch_bea_table(TABLE_QUANTITY_INDEX, years, api_key),
        fetch_bea_table(TABLE_PRICE_INDEX, years, api_key),
    ]
    return pd.concat(frames, ignore_index=True)


def fetch_bea_observations(
    start_year: int = 2021,
    end_year: int = 2025,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias: indexes only (growth computed separately)."""
    return fetch_bea_index_observations(start_year, end_year, api_key)


def build_bea_from_api(
    start_year: int = 2021,
    end_year: int = 2025,
    api_key: str | None = None,
    registry_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (industries, observations) aligned to project line_id registry."""
    registry = load_registry(registry_path or REGISTRY_PATH)
    api_obs = fetch_bea_index_observations(start_year, end_year, api_key)

    name_to_line = dict(zip(registry["industry_name"], registry["line_id"]))
    api_obs["line_id"] = api_obs["industry_name"].map(name_to_line)
    api_obs = api_obs.dropna(subset=["line_id"]).copy()
    api_obs["line_id"] = api_obs["line_id"].astype(int)

    index_obs = api_obs[["line_id", "period", "metric", "value"]].dropna(subset=["value"])
    observations = compute_bea_growth_from_indexes(index_obs)

    industries = registry.copy()
    return industries, observations


def load_bea_data(
    source: str = "csv",
    csv_path: Path | None = None,
    start_year: int = 2021,
    end_year: int = 2025,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source == "csv":
        return parse_bea_csv(csv_path or BEA_CSV_PATH)
    if source == "api":
        return build_bea_from_api(start_year, end_year, api_key)
    raise ValueError(f"Unknown BEA source: {source}")
