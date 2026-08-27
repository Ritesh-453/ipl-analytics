-- Query 1: Top 10 Batsmen by Batting Average (minimum 20 innings)
-- Average = total runs / number of times dismissed
-- Min 20 innings filter removes players with very few matches
-- Tables used: deliveries

SELECT
    batter,
    SUM(batsman_runs)                                            AS total_runs,
    COUNT(DISTINCT match_id)                                     AS innings,
    COUNT(CASE WHEN player_dismissed = batter THEN 1 END)        AS dismissals,
    ROUND(
        SUM(batsman_runs) * 1.0 /
        NULLIF(COUNT(CASE WHEN player_dismissed = batter THEN 1 END), 0)
    , 2)                                                         AS batting_average
FROM deliveries
GROUP BY batter
HAVING innings >= 20
ORDER BY batting_average DESC
LIMIT 10;