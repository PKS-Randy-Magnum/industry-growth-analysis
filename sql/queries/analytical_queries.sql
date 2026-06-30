-- Analytical SQL queries (portfolio / assignment deliverables)
-- Run against data/processed/industry_analysis.db after pipeline execution.

-- 1) Private-sector industries with highest price growth (2022 average)
SELECT
    industry_name,
    AVG(price_growth) AS avg_price_growth,
    AVG(quantity_growth) AS avg_quantity_growth,
    AVG(qp_ratio) AS avg_qp_ratio
FROM v_bea_growth
WHERE is_private = 1
  AND period BETWEEN '2022-Q1' AND '2022-Q4'
  AND price_growth IS NOT NULL
GROUP BY industry_name
ORDER BY avg_price_growth DESC
LIMIT 15;

-- 2) Manufacturing durable vs nondurable: price vs quantity dynamics
SELECT
    industry_name,
    period,
    quantity_growth,
    price_growth,
    qp_ratio
FROM v_bea_growth
WHERE industry_name IN ('Durable goods', 'Nondurable goods')
ORDER BY industry_name, period;

-- 3) Industries where prices rose but real output fell (stagflation-like quarters)
SELECT
    industry_name,
    period,
    price_growth,
    quantity_growth
FROM v_bea_growth
WHERE is_private = 1
  AND price_growth > 0
  AND quantity_growth < 0
ORDER BY price_growth - quantity_growth DESC;

-- 4) BLS: wage growth vs employment growth by quarter (2022)
SELECT
    industry_name,
    period,
    employment_thousands_growth,
    avg_hourly_earnings_growth
FROM bls_quarterly_growth
WHERE period LIKE '2022-%'
  AND employment_thousands_growth IS NOT NULL
  AND avg_hourly_earnings_growth IS NOT NULL
ORDER BY industry_name, period;

-- 5) Cross-source: sectors where output prices and wages both accelerated
SELECT
    industry_name,
    period,
    price_growth,
    avg_hourly_earnings_growth,
    quantity_growth,
    employment_thousands_growth
FROM v_bea_bls_comparison
WHERE period BETWEEN '2022-Q1' AND '2022-Q4'
  AND price_growth > 0.02
  AND avg_hourly_earnings_growth > 0.02
ORDER BY price_growth + avg_hourly_earnings_growth DESC;
