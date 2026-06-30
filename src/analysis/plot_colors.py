"""Fixed sector colors for consistent charts across static PNGs and Streamlit."""

from __future__ import annotations

# Paul Tol–style distinct hues; no two greens adjacent.
SECTOR_COLORS: dict[str, str] = {
    "Agriculture, forestry, fishing, and hunting": "#332288",
    "Mining": "#117733",
    "Utilities": "#44AA99",
    "Construction": "#88CCEE",
    "Durable goods": "#CC6677",
    "Nondurable goods": "#AA4499",
    "Wholesale trade": "#DDCC77",
    "Retail trade": "#999933",
    "Transportation and warehousing": "#661100",
    "Information": "#E69F00",
    "Finance, insurance, real estate, rental, and leasing": "#56B4E9",
    "Professional and business services": "#009E73",
    "Educational services, health care, and social assistance": "#F0E442",
    "Arts, entertainment, recreation, accommodation, and food services": "#D55E00",
    "Other services, except government": "#0072B2",
}

_FALLBACK_PALETTE = [
    "#332288",
    "#117733",
    "#44AA99",
    "#CC6677",
    "#DDCC77",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#0072B2",
    "#AA4499",
    "#661100",
    "#999933",
    "#88CCEE",
    "#F0E442",
]


def get_sector_color(name: str, fallback_idx: int = 0) -> str:
    if name in SECTOR_COLORS:
        return SECTOR_COLORS[name]
    lowered = name.lower()
    for key, color in SECTOR_COLORS.items():
        if key.lower() == lowered:
            return color
    return _FALLBACK_PALETTE[fallback_idx % len(_FALLBACK_PALETTE)]
