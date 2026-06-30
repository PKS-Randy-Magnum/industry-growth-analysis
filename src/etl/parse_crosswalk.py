"""Parse CES CODES.docx and align BEA industry names to BLS CES industry codes."""

from __future__ import annotations

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BEA_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "BEA Value Added.csv"
CROSSWALK_PATH = PROJECT_ROOT / "config" / "bea_bls_crosswalk.csv"
REGISTRY_PATH = PROJECT_ROOT / "config" / "bea_industry_registry.csv"


def resolve_docx_path() -> Path | None:
    return Path(os.environ["CES_DOCX_PATH"]) if os.environ.get("CES_DOCX_PATH") else None


CODE_RE = re.compile(r"\b(\d{8})\*?")
CES_CODE_RE = re.compile(r"CES(\d{7,8})XX?", re.IGNORECASE)

# Docx labels that do not fuzzy-match BEA export names cleanly.
DOCX_TO_BEA_ALIASES: dict[str, str] = {
    "agriculture forestry fishing and hunting": "Agriculture, forestry, fishing, and hunting",
    "fabricated metal products": "Fabricated metal products",
    "computer and electronic products": "Computer and electronic products",
    "other services except government": "Other services, except government",
    "social assistance": "Social assistance",
    "professional scientific and technical services": "Professional, scientific, and technical services",
    "broadcasting and telecommunications": "Broadcasting and telecommunications",
    "arts entertainment and recreation": "Arts, entertainment, and recreation",
    "miscellaneous manufacturing": "Miscellaneous manufacturing",
    "educational services": "Educational services",
    "health care and social assistance": "Health care and social assistance",
    "accommodation and food services": "Accommodation and food services",
    "food services and drinking places": "Food services and drinking places",
    "finance and insurance": "Finance and insurance",
    "real estate and rental and leasing": "Real estate and rental and leasing",
    "management of companies and enterprises": "Management of companies and enterprises",
    "administrative and waste management services": "Administrative and waste management services",
    "administrative and support services": "Administrative and support services",
    "waste management and remediation services": "Waste management and remediation services",
    "transportation and warehousing": "Transportation and warehousing",
    "other transportation and support activities": "Other transportation and support activities",
    "data processing, internet publishing, and other information services": (
        "Data processing, internet publishing, and other information services"
    ),
    "federal reserve banks, credit intermediation, and related activities": (
        "Federal Reserve banks, credit intermediation, and related activities"
    ),
    "securities, commodity contracts, and investments": (
        "Securities, commodity contracts, and investments"
    ),
    "insurance carriers and related activities": "Insurance carriers and related activities",
    "funds, trusts, and other financial vehicles": "Funds, trusts, and other financial vehicles",
    "rental and leasing services and lessors of intangible assets": (
        "Rental and leasing services and lessors of intangible assets"
    ),
    "miscellaneous professional, scientific, and technical services": (
        "Miscellaneous professional, scientific, and technical services"
    ),
    "performing arts, spectator sports, museums, and related activities": (
        "Performing arts, spectator sports, museums, and related activities"
    ),
    "amusements, gambling, and recreation industries": (
        "Amusements, gambling, and recreation industries"
    ),
    "food and beverage and tobacco products": "Food and beverage and tobacco products",
    "textile mills and textile product mills": "Textile mills and textile product mills",
    "apparel and leather and allied products": "Apparel and leather and allied products",
    "motor vehicles, bodies and trailers, and parts": (
        "Motor vehicles, bodies and trailers, and parts"
    ),
    "other transportation equipment": "Other transportation equipment",
    "electrical equipment, appliances, and components": (
        "Electrical equipment, appliances, and components"
    ),
}


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _extract_docx_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    paragraphs: list[str] = []
    for para in root.iter(f"{{{W_NS}}}p"):
        text = "".join(t.text or "" for t in para.iter(f"{{{W_NS}}}t")).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def _parse_docx_mappings(paragraphs: list[str]) -> dict[str, list[str]]:
    """Return {bea_label: [ces_industry_codes]} from docx mapping section."""
    mappings: dict[str, list[str]] = {}
    stop_markers = {"main industries", "subsectors", "subsectors with multiple ids"}

    for line in paragraphs:
        lowered = line.lower().strip()
        if lowered in stop_markers:
            break
        if lowered.startswith("ces codes and sectors"):
            continue
        if lowered.startswith("question:") or "can't split" in lowered or "cant split" in lowered:
            continue

        codes = CODE_RE.findall(line)
        if not codes:
            continue

        # Split on dash-like separators before the first code.
        name_part = CODE_RE.split(line)[0]
        name_part = re.sub(r"[\s\-–—]+$", "", name_part.strip())
        if not name_part:
            continue

        key = _normalize_name(name_part)
        existing = mappings.setdefault(key, [])
        for code in codes:
            if code not in existing:
                existing.append(code)

    # CES-prefixed lines in subsector / multi-id sections (employment base codes).
    in_multi = False
    for line in paragraphs:
        lowered = line.lower().strip()
        if lowered == "subsectors with multiple ids":
            in_multi = True
            continue
        if lowered in {"main industries", "subsectors"}:
            continue

        m = CES_CODE_RE.match(line.strip())
        if not m:
            continue
        code = m.group(1)
        if len(code) == 7:
            code = code + "0"  # e.g. 3133601 -> pad if needed; docx uses 31336001
        if len(code) != 8:
            continue
        # Multi-id section codes are already covered by first-section mappings.
        if in_multi:
            continue

    return mappings


def _build_name_lookup(bea_industries: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in bea_industries["industry_name"]:
        lookup[_normalize_name(name)] = name
    for docx_key, bea_name in DOCX_TO_BEA_ALIASES.items():
        lookup[docx_key] = bea_name
    return lookup


def _infer_plot_level(indent_level: int, industry_name: str) -> str:
    lowered = industry_name.lower()
    if "government" in lowered or "gross domestic product" in lowered:
        return "exclude"
    if indent_level <= 4 and industry_name in {
        "Private industries",
        "Manufacturing",
    }:
        return "aggregate"
    if 4 <= indent_level <= 6:
        return "sector"
    if 8 <= indent_level <= 12:
        return "subsector"
    return "other"


def build_bea_industry_registry(bea_csv: Path | None = None) -> pd.DataFrame:
    from src.etl.parse_bea import parse_bea_csv

    bea_csv = bea_csv or BEA_CSV_PATH
    industries, _ = parse_bea_csv(bea_csv)
    industries = industries.copy()
    industries["plot_level"] = industries.apply(
        lambda r: _infer_plot_level(int(r["indent_level"]), r["industry_name"]),
        axis=1,
    )
    return industries[
        ["line_id", "industry_name", "indent_level", "is_private", "plot_level"]
    ].sort_values("line_id")


def build_crosswalk(
    docx_path: Path | None = None,
    bea_csv: Path | None = None,
) -> pd.DataFrame:
    docx_path = docx_path or resolve_docx_path()
    if docx_path is None:
        raise ValueError(
            "CES docx path required. Set CES_DOCX_PATH or pass docx_path= to build_crosswalk()."
        )
    bea_csv = bea_csv or BEA_CSV_PATH

    registry = build_bea_industry_registry(bea_csv)
    name_lookup = _build_name_lookup(registry)
    docx_mappings = _parse_docx_mappings(_extract_docx_paragraphs(docx_path))

    rows: list[dict] = []
    for docx_key, codes in docx_mappings.items():
        bea_name = name_lookup.get(docx_key)
        if not bea_name:
            # Try alias pass-through
            for alias_key, alias_name in DOCX_TO_BEA_ALIASES.items():
                if docx_key == alias_key:
                    bea_name = alias_name
                    break
        if not bea_name:
            continue

        match = registry[registry["industry_name"] == bea_name]
        if match.empty:
            continue
        bea_row = match.iloc[0]
        agg = "multi" if len(codes) > 1 else "direct"
        notes = ""
        if "*" in " ".join(codes):
            notes = "docx_flagged_split"

        for code in codes:
            rows.append(
                {
                    "bea_line_id": int(bea_row["line_id"]),
                    "bea_industry_name": bea_name,
                    "ces_industry_code": code,
                    "plot_level": bea_row["plot_level"],
                    "aggregation": agg,
                    "notes": notes,
                }
            )

    crosswalk = pd.DataFrame(rows).drop_duplicates(
        subset=["bea_line_id", "ces_industry_code"]
    )
    return crosswalk.sort_values(["bea_line_id", "ces_industry_code"]).reset_index(drop=True)


def ces_series_id(industry_code: str | int, datatype: str) -> str:
    """Build full CES series ID: CES + 8-digit industry code + 2-digit datatype."""
    code = str(industry_code).strip().replace("*", "").zfill(8)
    return f"CES{code}{datatype}"


def expand_series_ids(crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Expand crosswalk rows to employment/hourly/weekly CES series IDs."""
    datatypes = {
        "01": "employment_thousands",
        "03": "avg_hourly_earnings",
        "11": "avg_weekly_earnings",
    }
    rows: list[dict] = []
    for _, row in crosswalk.iterrows():
        for dt, metric in datatypes.items():
            rows.append(
                {
                    **row.to_dict(),
                    "ces_series_id": ces_series_id(row["ces_industry_code"], dt),
                    "metric": metric,
                }
            )
    return pd.DataFrame(rows)


def load_crosswalk(path: Path | None = None) -> pd.DataFrame:
    path = path or CROSSWALK_PATH
    return pd.read_csv(path)


def load_registry(path: Path | None = None) -> pd.DataFrame:
    path = path or REGISTRY_PATH
    return pd.read_csv(path)


def write_crosswalk_files(
    crosswalk_path: Path | None = None,
    registry_path: Path | None = None,
) -> tuple[Path, Path]:
    crosswalk_path = crosswalk_path or CROSSWALK_PATH
    registry_path = registry_path or REGISTRY_PATH

    registry = build_bea_industry_registry()
    crosswalk = build_crosswalk()

    crosswalk_path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(registry_path, index=False)
    crosswalk.to_csv(crosswalk_path, index=False)
    return crosswalk_path, registry_path


if __name__ == "__main__":
    from src.etl.env import load_project_env

    load_project_env()
    cw, reg = write_crosswalk_files()
    crosswalk = pd.read_csv(cw)
    print(f"Wrote {len(crosswalk)} crosswalk rows -> {cw}")
    print(f"Wrote industry registry -> {reg}")
    print(f"Unique BEA industries mapped: {crosswalk['bea_line_id'].nunique()}")
