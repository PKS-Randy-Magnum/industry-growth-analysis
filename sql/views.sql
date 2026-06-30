CREATE VIEW IF NOT EXISTS v_bea_growth AS
SELECT
    i.line_id,
    i.industry_name,
    i.indent_level,
    i.is_private,
    b.period,
    MAX(CASE WHEN b.metric = 'quantity_growth' THEN b.value END) AS quantity_growth,
    MAX(CASE WHEN b.metric = 'price_growth' THEN b.value END) AS price_growth,
    MAX(CASE WHEN b.metric = 'quantity_price_ratio' THEN b.value END) AS qp_ratio
FROM industries i
JOIN bea_observations b ON b.line_id = i.line_id
WHERE i.source = 'BEA'
GROUP BY i.line_id, i.industry_name, i.indent_level, i.is_private, b.period;

CREATE VIEW IF NOT EXISTS v_bls_monthly AS
SELECT
    i.line_id,
    i.industry_name,
    i.indent_level,
    i.is_private,
    o.period_month,
    MAX(CASE WHEN o.metric = 'employment_thousands' THEN o.value END) AS employment_thousands,
    MAX(CASE WHEN o.metric = 'avg_hourly_earnings' THEN o.value END) AS avg_hourly_earnings,
    MAX(CASE WHEN o.metric = 'avg_weekly_earnings' THEN o.value END) AS avg_weekly_earnings
FROM industries i
JOIN bls_observations o ON o.line_id = i.line_id
WHERE i.source = 'BLS'
GROUP BY i.line_id, i.industry_name, i.indent_level, i.is_private, o.period_month;

CREATE VIEW IF NOT EXISTS v_crosswalk_coverage AS
SELECT
    bi.line_id AS bea_line_id,
    bi.industry_name AS bea_industry_name,
    COUNT(DISTINCT x.ces_industry_code) AS ces_code_count,
    CASE WHEN COUNT(DISTINCT x.ces_industry_code) > 0 THEN 1 ELSE 0 END AS has_crosswalk
FROM industries bi
LEFT JOIN bea_bls_crosswalk x ON x.bea_line_id = bi.line_id
WHERE bi.source = 'BEA'
GROUP BY bi.line_id, bi.industry_name;

CREATE VIEW IF NOT EXISTS v_bea_bls_comparison AS
SELECT
    bi.line_id AS bea_line_id,
    bi.line_id AS bls_line_id,
    bi.industry_name,
    bg.period,
    bg.quantity_growth,
    bg.price_growth,
    bl.employment_thousands_growth,
    bl.avg_hourly_earnings_growth
FROM industries bi
JOIN v_bea_growth bg ON bg.line_id = bi.line_id
JOIN bls_quarterly_growth bl ON bl.line_id = bi.line_id AND bl.period = bg.period
WHERE bi.source = 'BEA'
  AND bi.is_private = 1
  AND bi.line_id IN (SELECT DISTINCT bea_line_id FROM bea_bls_crosswalk);
