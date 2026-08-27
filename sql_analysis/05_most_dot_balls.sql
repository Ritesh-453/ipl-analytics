-- Query 5: Top 10 Bowlers by Most Dot Balls Bowled
-- Dot ball = delivery where no run is scored (batsman + extras = 0)
-- Shows which bowlers are most economical and put pressure on batsmen
-- Tables used: deliveries

SELECT
    bowler,
    COUNT(*)                                                  AS total_balls,
    SUM(CASE WHEN total_runs = 0 THEN 1 ELSE 0 END)          AS dot_balls,
    ROUND(
        SUM(CASE WHEN total_runs = 0 THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 2
    )                                                         AS dot_ball_pct
FROM deliveries
GROUP BY bowler
HAVING total_balls >= 300
ORDER BY dot_balls DESC
LIMIT 10;