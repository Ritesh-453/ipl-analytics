-- Query 6: Highest Individual Score by a Batsman in Each Season
-- Shows the best individual batting performance per season
-- Tables used: deliveries, matches

SELECT
    m.season,
    d.batter,
    d.batting_team,
    SUM(d.batsman_runs)                                      AS runs_scored,
    COUNT(*)                                                 AS balls_faced,
    ROUND(SUM(d.batsman_runs) * 100.0 / COUNT(*), 2)        AS strike_rate
FROM deliveries d
JOIN matches m ON d.match_id = m.id
GROUP BY m.season, d.match_id, d.batter
ORDER BY m.season ASC, runs_scored DESC
-- Pick highest score per season
LIMIT 100;