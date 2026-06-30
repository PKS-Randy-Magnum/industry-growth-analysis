"""Fetch CES series from the BLS public API."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.etl.env import load_project_env

load_project_env()

BLS_API_V2 = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_API_V1 = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
BATCH_SIZE = 25
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "processed" / "bls_api_monthly.csv"


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cache_covers_range(
    cached: pd.DataFrame,
    series_ids: list[str],
    start_year: int | None,
    end_year: int | None,
) -> bool:
    subset = cached[cached["series_id"].isin(series_ids)]
    if subset.empty:
        return False
    years = subset["year"].astype(int)
    if start_year is not None and years.min() > start_year:
        return False
    if end_year is not None and years.max() < end_year:
        return False
    return True


def _fetch_from_api(
    series_ids: list[str],
    start_year: int | None,
    end_year: int | None,
    api_key: str | None,
    pause_seconds: float,
) -> pd.DataFrame:
    urls_to_try: list[str] = []
    if api_key:
        urls_to_try.append(BLS_API_V2)
    urls_to_try.append(BLS_API_V1)

    records: list[dict] = []
    active_url = urls_to_try[0]

    for batch_start in range(0, len(series_ids), BATCH_SIZE):
        batch = series_ids[batch_start : batch_start + BATCH_SIZE]
        payload: dict = {"seriesid": batch}
        if start_year is not None:
            payload["startyear"] = str(start_year)
        if end_year is not None:
            payload["endyear"] = str(end_year)
        if api_key and active_url == BLS_API_V2:
            payload["registrationkey"] = api_key

        response = None
        last_error: Exception | None = None
        for url in urls_to_try:
            try:
                response = _post_json(url, payload)
                active_url = url
                if response.get("status") == "REQUEST_SUCCEEDED":
                    break
                messages = response.get("message", response.get("Messages", []))
                msg_text = " ".join(str(m) for m in messages).lower()
                if "threshold" in msg_text or "exceed" in msg_text:
                    payload.pop("registrationkey", None)
                    continue
                raise RuntimeError(f"BLS API error: {messages}")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                payload.pop("registrationkey", None)
                continue
        else:
            raise RuntimeError(f"BLS API failed for batch: {last_error}")

        for series in response.get("Results", {}).get("series", []):
            series_id = series["seriesID"]
            for point in series.get("data", []):
                year = int(point["year"])
                period = point["period"]
                if not period.startswith("M"):
                    continue
                month = int(period[1:])
                records.append(
                    {
                        "series_id": series_id,
                        "year": year,
                        "period": period,
                        "period_month": f"{year}-{month:02d}",
                        "value": float(point["value"]),
                    }
                )

        if batch_start + BATCH_SIZE < len(series_ids):
            time.sleep(pause_seconds)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values(["series_id", "period_month"]).reset_index(drop=True)


def _write_cache(cache_path: Path, df: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        prior = pd.read_csv(cache_path)
        df = pd.concat([prior, df], ignore_index=True).drop_duplicates(
            subset=["series_id", "period_month"], keep="last"
        )
    df.to_csv(cache_path, index=False)


def fetch_bls_series(
    series_ids: Iterable[str],
    start_year: int | None = None,
    end_year: int | None = None,
    api_key: str | None = None,
    pause_seconds: float = 1.0,
    cache_path: Path | None = DEFAULT_CACHE,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return monthly CES observations: series_id, year, period, value."""
    series_ids = list(dict.fromkeys(series_ids))
    if not series_ids:
        return pd.DataFrame(columns=["series_id", "year", "period", "period_month", "value"])

    cached: pd.DataFrame | None = None
    if use_cache and not force_refresh and cache_path and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if not cached.empty and _cache_covers_range(cached, series_ids, start_year, end_year):
            return cached[cached["series_id"].isin(series_ids)].copy()

    api_key = api_key if api_key is not None else os.environ.get("BLS_API_KEY")
    if not api_key:
        if cached is not None and not cached.empty:
            subset = cached[cached["series_id"].isin(series_ids)].copy()
            if not subset.empty:
                print("      Warning: BLS_API_KEY missing; using cache only.")
                return subset
        raise RuntimeError("BLS_API_KEY is required for BLS API fetch (set in .env).")

    df = _fetch_from_api(series_ids, start_year, end_year, api_key, pause_seconds)
    if df.empty and cached is not None and not cached.empty:
        return cached[cached["series_id"].isin(series_ids)].copy()

    if cache_path is not None and not df.empty:
        _write_cache(cache_path, df)
        if cache_path.exists():
            full = pd.read_csv(cache_path)
            return full[full["series_id"].isin(series_ids)].copy()
    return df
