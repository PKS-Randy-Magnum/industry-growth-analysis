"""Check that .env API keys load and respond (no secrets printed)."""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _check_env_format() -> None:
    text = (PROJECT_ROOT / ".env").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        issues: list[str] = []
        if value.startswith(('"', "'")) or value.endswith(('"', "'")):
            issues.append("remove quotes around value")
        if value != value.strip():
            issues.append("trim whitespace")
        label = ", ".join(issues) if issues else "format OK"
        print(f"  {key.strip()}: {label}")


def _parse_bea_results(obj: dict) -> tuple[str | None, list]:
    """Normalize BEA JSON; Results may be a dict or a one-element list."""
    bea = obj.get("BEAAPI")
    if not bea:
        return f"Unexpected BEA response: {obj}", []

    results = bea.get("Results")
    if isinstance(results, list):
        if not results:
            return "BEA API returned empty Results", []
        block = results[0]
        if isinstance(block, dict) and "Error" in block:
            err = block["Error"]
            return err.get("APIErrorDescription", str(err)), []
        if isinstance(block, dict):
            return None, block.get("Data", [])
        return f"Unexpected BEA Results entry: {type(block)}", []

    if isinstance(results, dict):
        if "Error" in results:
            err = results["Error"]
            return err.get("APIErrorDescription", str(err)), []
        return None, results.get("Data", [])

    return f"Unexpected BEA Results type: {type(results)}", []


def _test_bea(key: str) -> str:
    params = {
        "UserID": key,
        "method": "GetData",
        "datasetname": "GDPbyIndustry",
        "TableID": 8,
        "Frequency": "Q",
        "Year": "2022",
        "Industry": "ALL",
        "ResultFormat": "JSON",
    }
    url = f"https://apps.bea.gov/api/data/?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        obj = json.loads(resp.read().decode())
    err, data = _parse_bea_results(obj)
    if err:
        return err
    return f"OK ({len(data)} rows)"


def _test_bls(key: str) -> str:
    payload = json.dumps(
        {
            "seriesid": ["CES313362000001"],
            "startyear": "2022",
            "endyear": "2022",
            "registrationkey": key,
        }
    ).encode()
    req = urllib.request.Request(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        data=payload,
        headers={"Content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        obj = json.loads(resp.read().decode())
    if obj.get("status") != "REQUEST_SUCCEEDED":
        return str(obj.get("message", obj.get("Messages")))
    return "OK"


def main() -> int:
    bea = os.environ.get("BEA_API_KEY", "")
    bls = os.environ.get("BLS_API_KEY", "")
    print("== API key check ==")
    print(f"BEA_API_KEY: {'set' if bea else 'MISSING'}")
    print(f"BLS_API_KEY: {'set' if bls else 'MISSING'}")
    print("\n.env formatting:")
    _check_env_format()

    if bea:
        print(f"\nBEA API: {_test_bea(bea)}")
    if bls:
        print(f"BLS API: {_test_bls(bls)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
