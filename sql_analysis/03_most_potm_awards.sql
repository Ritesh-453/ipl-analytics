-- Query 3: Top 10 Players with Most Player of the Match Awards
-- Shows: how dominant certain players have been across IPL history
-- Tables used: matches

SELECT
    player_of_match,
    COUNT(*)                                             AS potm_awards
FROM matches
WHERE player_of_match IS NOT NULL
  AND player_of_match != 'Unknown'
GROUP BY player_of_match
ORDER BY potm_awards DESC
LIMIT 10;