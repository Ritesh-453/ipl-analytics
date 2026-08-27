# SQL Analysis — IPL Dataset

This folder contains SQL queries written on the IPL dataset (2008–2024).
Data is loaded into a local SQLite database using `load_to_sqlite.py`.

## Setup

```bash
python sql_analysis/load_to_sqlite.py
```

This creates `ipl.db` — a SQLite database with two tables:
- `matches` — 1,095 rows, match-level data
- `deliveries` — 260,920 rows, ball-by-ball data

## Queries

| File | Question Answered |
|---|---|
| 01_best_batting_average.sql | Top 10 batsmen by batting average (min 20 innings) |
| 02_best_bowling_economy.sql | Top 10 bowlers by economy rate (min 50 overs) |
| 03_most_potm_awards.sql | Top 10 Player of the Match award winners |
| 04_powerplay_top_scorers.sql | Top 10 run scorers in powerplay overs (1-6) |
| 05_most_dot_balls.sql | Top 10 bowlers by dot balls bowled |
| 06_season_highest_score.sql | Highest individual score in each IPL season |

## Key Findings

- **AB de Villiers** leads all-time Player of the Match awards with 25
- **Bhuvneshwar Kumar** bowls the most dot balls — 40.2% of deliveries
- **BB McCullum** holds the 2008 season record with 158* off 73 balls in the very first IPL match
- **CH Gayle** holds the all-time highest IPL score — 175* in 2013
- **KL Rahul** has the best batting average among high-volume batsmen
- **Anil Kumble** has the best economy rate among bowlers with 50+ overs