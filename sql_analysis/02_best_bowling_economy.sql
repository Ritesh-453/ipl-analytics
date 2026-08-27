-- Query 2: Top 10 Bowlers by Best Economy Rate (minimum 50 overs bowled)
-- Economy = runs conceded per over
-- Min 50 overs filter removes bowlers with very few deliveries
-- Tables used: deliveries

SELECT
    bowler,
    COUNT(*)                                             AS balls_bowled,
    ROUND(COUNT(*) / 6.0, 1)                            AS overs_bowled,
    SUM(total_runs)                                      AS runs_conceded,
    COUNT(CASE WHEN batsman_runs = 0 AND extra_runs = 0
               THEN 1 END)                               AS dot_balls,
    ROUND(SUM(total_runs) * 6.0 / COUNT(*), 2)          AS economy
FROM deliveries
GROUP BY bowler
HAVING overs_bowled >= 50
ORDER BY economy ASC
LIMIT 10;