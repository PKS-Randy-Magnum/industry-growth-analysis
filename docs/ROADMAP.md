# Roadmap (pinned items)

## Done

- Annual Q1→Q1 SARIMA forecast (replaces quarterly forecast for the dashboard chart)
- Docker + generic deploy docs (GitHub, Docker Compose, Nginx)

## Pinned — do later

### Extend history toward 1999

**Goal:** More annual (and quarterly) observations for stabler SARIMA and longer-run charts.

**Steps (high effort, research-heavy):**

1. `python run.py --refresh --start 2007-Q1` (or 1999-Q1 if APIs allow) — low code change.
2. Run `scripts/validate_crosswalk.py` and audit BLS CES series start dates per sector.
3. Document gaps: sectors with missing early BLS, NAICS/industry definition changes.
4. Decide acceptable start year per sector (may not be uniform).
5. Re-export snapshots; update default dashboard start in README.

**Risks:** CES code consistency, BEA industry line changes, trust funds line, pre-2007 BEA table coverage.

### PostgreSQL on production server

**Goal:** Replace SQLite + CSV snapshots with Postgres for refresh jobs and optional multi-app access.

**Scope when started:**

- Schema mirroring `bea_observations`, `bls_quarterly_growth`, views.
- ETL writes to Postgres; dashboard reads via SQLAlchemy or pre-aggregated views.
- Connection via `.env` on server (`DATABASE_URL`), not committed.

**Not started** — current deploy uses bundled CSV snapshots intentionally for simplicity.

### Log price index forecast

Optional second forecast mode on BEA `price_index` levels for level-persistence stories. See prior design notes.
