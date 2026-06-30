# Industry Growth & Inflation Decomposition

Portfolio data science project on industry-level macro trends. This repository decomposes **inflation** (BEA chain-type price indexes) versus **real output growth** (quantity indexes), and compares those patterns to **BLS labor market dynamics** (employment and wage growth).

## Research Question

When industries experience inflation, is it driven by price increases holding back quantity growth, or by genuine output expansion? How do wage and employment patterns from BLS align with BEA value-added decomposition?

## Data Sources

| Source | Content | Frequency |
|--------|---------|-----------|
| [BEA GDP by Industry](https://www.bea.gov/data/gdp/gdp-industry) | Chain-type quantity & price indexes, growth rates | Quarterly |
| [BLS CES](https://www.bls.gov/ces/) | Employment, avg hourly/weekly earnings | Monthly |

## How the data is built

BEA and BLS do not publish a join key at this industry granularity. This project **pairs** BEA value-added industries with matching BLS **CES industry codes**, then fetches and aggregates labor series to BEA grain.

| Layer | Source | What this repo does |
|-------|--------|---------------------|
| **BEA** | GDP-by-industry API (Tables 8 & 11) | Pulls chain-type **quantity** and **price** indexes; computes QoQ and Q1→Q1 growth in Python (`src/etl/fetch_bea.py`, `parse_bea.py`) |
| **BLS** | CES API | Fetches employment, avg hourly earnings, and avg weekly earnings per mapped industry (`src/etl/fetch_bls.py`) |
| **Crosswalk** | `config/bea_bls_crosswalk.csv` | **Manually constructed** mapping: each BEA `line_id` → one or more **8-digit CES industry codes**; `expand_series_ids()` builds full series IDs (`CES` + code + `01` / `03` / `11`) |

Where a BEA sector spans multiple CES codes, `src/etl/aggregate_bls.py` rolls series up to BEA grain. The crosswalk and industry registry are versioned under `config/`. To rebuild the crosswalk from a local CES code document, set `CES_DOCX_PATH` (see [docs/PIPELINE.md](docs/PIPELINE.md)).

## What's included vs what you need to run yourself

### Included in the repo (dashboard works without API keys)

- **`data/snapshots/`** — bundled **2019-Q1 through 2026-Q1** panels (`manifest.json` records `bea_source` / `bls_source`: api)
- **`config/bea_bls_crosswalk.csv`** and **`bea_industry_registry.csv`** — the BEA↔CES mapping used by ETL and the dashboard
- **`data/excel/`** — pivot-friendly exports for `full` and `excl_trust_funds` profiles
- **`outputs/`** — ML feature tables and evaluation JSON from the last pipeline run
- **Streamlit app** — reads snapshots only; no live API calls at view time

### Requires your API keys and `python run.py --refresh`

- Quarters **after** the bundled snapshot end date
- Full rebuild from BEA/BLS APIs (`BEA_API_KEY`, `BLS_API_KEY` in `.env`)

`data/raw/` CSV exports are **legacy reference only**; the default pipeline uses API + snapshots.

### Validate yourself

| Script / artifact | Purpose |
|-------------------|---------|
| [`scripts/validate_crosswalk.py`](scripts/validate_crosswalk.py) | Confirms CES series exist via API; checks aggregated BLS against legacy reference CSV where periods overlap |
| [`scripts/validate_bea_growth.py`](scripts/validate_bea_growth.py) | Checks BEA growth in snapshots is self-consistent; optional live API spot-check (writes local report, gitignored) |
| [`outputs/crosswalk_validation.json`](outputs/crosswalk_validation.json) | Summary from the last crosswalk validation run (committed) |
| [`scripts/check_api_keys.py`](scripts/check_api_keys.py) | Quick `.env` key check before a refresh |

## Quick Start

```bash
cd industry-growth-inflation-analysis
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add `BEA_API_KEY` and `BLS_API_KEY` if you plan to refresh data. Never commit `.env`.

```bash
# Refresh from APIs (needs keys) — updates snapshots through current quarter
python run.py --refresh

# Dashboard — works from bundled snapshots; no keys required
pip install streamlit plotly statsmodels
streamlit run streamlit_app.py
```

For CLI flags, snapshot file layouts, and crosswalk rebuild steps, see [docs/PIPELINE.md](docs/PIPELINE.md).

## Dashboard

Interactive Streamlit app with sector pickers, multiple chart types, and **annual Q1→Q1 SARIMA forecasts** through 2027-Q1 (baseline, 80% prediction band, optional price shock overlays).

- **Profiles** — `excl_trust_funds` (default) vs `full` (includes trust funds line)
- **Chart types** — Q/P decomposition, wage−price spread, BEA vs BLS comparison, scatter plots, SARIMA forecast
- **Deploy** — [docs/DEPLOY.md](docs/DEPLOY.md) (Docker + Nginx)

**Forecasting caveats**

- With a **2019-Q1** start you have roughly **six annual Q1→Q1 points** per sector — enough for a demo fit, not a long-horizon structural model
- COVID-era years still move annual steps; shaded bands are **statistical** (80% interval), not economic best/worst cases
- Shock overlays (+N **percentage points per year**) adjust the **BEA price growth** baseline only — they are not causal tariff or CPI models

## Limitations & planned extensions

**Current limits**

- History starts **2019-Q1** — short for stable long-run SARIMA and pre-COVID baselines
- CES / NAICS industry definitions change over time; the crosswalk reflects the current mapping
- Trust funds and aggregates are handled via the `full` vs `excl_trust_funds` profile (line 59 excluded in the default profile)

**Planned** (see [docs/ROADMAP.md](docs/ROADMAP.md) for detail)

- Extend history toward **1999** (and audit whether earlier 1990s data are feasible **per sector** — CES series start dates and NAICS breaks differ by industry)
- Document gaps where BLS or BEA coverage does not align; set sector-specific acceptable start years
- Postgres backing store for refresh jobs; optional log-price-index forecast mode

## Project Layout

```
config/          # industries.yaml, BEA↔BLS crosswalk CSVs
data/snapshots/  # API-derived tidy CSVs (dashboard reads these)
data/excel/      # Excel-friendly pivot panels per profile
data/processed/  # SQLite (local runs; gitignored DB)
outputs/         # ML features, validation JSON, static figures
src/etl/         # BEA/BLS fetch, crosswalk, aggregation
src/dashboard/   # Streamlit charts and data loaders
run.py           # One-command ETL → SQLite → plots → ML
streamlit_app.py # Interactive dashboard
```

## License

MIT. [BEA](https://www.bea.gov/) and [BLS](https://www.bls.gov/) data — use per their terms of use.
