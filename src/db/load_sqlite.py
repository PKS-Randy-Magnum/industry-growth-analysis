"""Load parsed data into SQLite and materialize analytical tables."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
VIEWS_PATH = PROJECT_ROOT / "sql" / "views.sql"
CROSSWALK_PATH = PROJECT_ROOT / "config" / "bea_bls_crosswalk.csv"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "industry_analysis.db"

INDUSTRY_COLUMNS = [
    "source",
    "line_id",
    "industry_name",
    "indent_level",
    "is_private",
    "plot_level",
]


def _exec_script(conn: sqlite3.Connection, path: Path) -> None:
    conn.executescript(path.read_text(encoding="utf-8"))


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after first DB creation."""
    rows = conn.execute("PRAGMA table_info(industries)").fetchall()
    if rows and "plot_level" not in {row[1] for row in rows}:
        conn.execute("ALTER TABLE industries ADD COLUMN plot_level TEXT")


def _replace_reference_table(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    conn.execute(f"DELETE FROM {table}")
    if not df.empty:
        df.to_sql(table, conn, if_exists="append", index=False)


def _upsert_rows(conn: sqlite3.Connection, table: str, df: pd.DataFrame, columns: list[str]) -> None:
    if df.empty:
        return
    payload = df[columns].copy()
    placeholders = ", ".join("?" * len(columns))
    col_list = ", ".join(columns)
    sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
    conn.executemany(sql, payload.itertuples(index=False, name=None))


def _prepare_industries(
    bea_industries: pd.DataFrame,
    bls_industries: pd.DataFrame,
) -> pd.DataFrame:
    bea_i = bea_industries.copy()
    bea_i["source"] = "BEA"
    bls_i = bls_industries.copy()
    bls_i["source"] = "BLS"
    industries = pd.concat([bea_i, bls_i], ignore_index=True)
    for col in INDUSTRY_COLUMNS:
        if col not in industries.columns:
            industries[col] = None
    return industries[INDUSTRY_COLUMNS]


def load_database(
    bea_industries: pd.DataFrame,
    bea_observations: pd.DataFrame,
    bls_industries: pd.DataFrame,
    bls_observations: pd.DataFrame,
    bls_quarterly: pd.DataFrame,
    crosswalk: pd.DataFrame | None = None,
    db_path: Path | None = None,
    rebuild_db: bool = False,
) -> Path:
    db_path = db_path or DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild_db and db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        _exec_script(conn, SCHEMA_PATH)
        _migrate_schema(conn)

        industries = _prepare_industries(bea_industries, bls_industries)
        _replace_reference_table(conn, "industries", industries)

        _upsert_rows(
            conn,
            "bea_observations",
            bea_observations,
            ["line_id", "period", "metric", "value"],
        )
        _upsert_rows(
            conn,
            "bls_observations",
            bls_observations,
            ["line_id", "period_month", "metric", "value"],
        )
        quarterly_cols = [
            "line_id",
            "year",
            "quarter",
            "period",
            "employment_thousands",
            "avg_hourly_earnings",
            "avg_weekly_earnings",
            "employment_thousands_growth",
            "avg_hourly_earnings_growth",
            "avg_weekly_earnings_growth",
        ]
        _upsert_rows(conn, "bls_quarterly_growth", bls_quarterly, quarterly_cols)

        if crosswalk is None and CROSSWALK_PATH.exists():
            crosswalk = pd.read_csv(CROSSWALK_PATH)
        if crosswalk is not None and not crosswalk.empty:
            _replace_reference_table(conn, "bea_bls_crosswalk", crosswalk)

        _exec_script(conn, VIEWS_PATH)

        conn.commit()
    finally:
        conn.close()

    return db_path


def run_query(db_path: Path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)
