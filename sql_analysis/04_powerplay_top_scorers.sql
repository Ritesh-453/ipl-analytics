-- Query 4: Top 10 Batsmen by Runs Scored in Powerplay (Overs 1-6)
-- Powerplay = first 6 overs of each innings
-- Shows which batters are most aggressive at the start
-- Tables used: deliveries

SELECT
    batter,
    SUM(batsman_runs)                                        AS powerplay_runs,
    COUNT(*)                                                 AS balls_faced,
    ROUND(SUM(batsman_runs) * 100.0 / COUNT(*), 2)          AS strike_rate,
    SUM(CASE WHEN batsman_runs = 4 THEN 1 ELSE 0 END)       AS fours,
    SUM(CASE WHEN batsman_runs = 6 THEN 1 ELSE 0 END)       AS sixes
FROM deliveries
WHERE over BETWEEN 0 AND 5
GROUP BY batter
ORDER BY powerplay_runs DESC
LIMIT 10;