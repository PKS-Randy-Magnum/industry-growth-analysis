-- SQLite schema for BEA/BLS industry growth analysis

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS industries (
    source          TEXT NOT NULL CHECK (source IN ('BEA', 'BLS')),
    line_id         INTEGER NOT NULL,
    industry_name   TEXT NOT NULL,
    indent_level    INTEGER NOT NULL,
    is_private      INTEGER NOT NULL CHECK (is_private IN (0, 1)),
    plot_level      TEXT,
    PRIMARY KEY (source, line_id)
);

CREATE TABLE IF NOT EXISTS bea_observations (
    observation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id         INTEGER NOT NULL,
    period          TEXT NOT NULL,
    metric          TEXT NOT NULL,
    value           REAL NOT NULL,
    UNIQUE (line_id, period, metric)
);

CREATE TABLE IF NOT EXISTS bls_observations (
    observation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id         INTEGER NOT NULL,
    period_month    TEXT NOT NULL,
    metric          TEXT NOT NULL,
    value           REAL NOT NULL,
    UNIQUE (line_id, period_month, metric)
);

CREATE TABLE IF NOT EXISTS bls_quarterly_growth (
    line_id                         INTEGER NOT NULL,
    year                            INTEGER NOT NULL,
    quarter                         INTEGER NOT NULL,
    period                          TEXT NOT NULL,
    employment_thousands            REAL,
    avg_hourly_earnings             REAL,
    avg_weekly_earnings             REAL,
    employment_thousands_growth     REAL,
    avg_hourly_earnings_growth      REAL,
    avg_weekly_earnings_growth      REAL,
    PRIMARY KEY (line_id, period)
);

CREATE TABLE IF NOT EXISTS industry_labels (
    source          TEXT NOT NULL,
    line_id         INTEGER NOT NULL,
    growth_regime   TEXT,
    cluster_id      INTEGER,
    PRIMARY KEY (source, line_id)
);

CREATE TABLE IF NOT EXISTS bea_bls_crosswalk (
    bea_line_id         INTEGER NOT NULL,
    bea_industry_name   TEXT NOT NULL,
    ces_industry_code   TEXT NOT NULL,
    plot_level          TEXT,
    aggregation         TEXT NOT NULL,
    notes               TEXT,
    PRIMARY KEY (bea_line_id, ces_industry_code)
);
