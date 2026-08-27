# IPL Analytics Dashboard

An end-to-end data analysis project on Indian Premier League (IPL) data from 2008 to 2024. Covers team performance, player stats, venue analysis, SQL-based data exploration, and two machine learning models — a pre-match winner predictor and a live in-match win probability calculator.

**Live Demo:** [Add your Streamlit link here after deployment]

---

## Project Structure

```
CA_improved/
│
├── dashboard/
│   └── app.py                        # Streamlit dashboard (main file)
│
├── data/
│   ├── raw/
│   │   ├── matches.csv               # Raw match data
│   │   └── deliveries.csv            # Raw ball-by-ball data
│   └── processed/
│       ├── matches_clean.csv         # Cleaned match data
│       └── deliveries_clean.csv      # Cleaned delivery data
│
├── notebooks/
│   ├── 01_data_cleaning/             # Data cleaning and preprocessing
│   ├── 02_team_analysis/             # Team and batting EDA
│   └── 03_ml_model/                  # ML model training and evaluation
│
├── sql_analysis/
│   ├── load_to_sqlite.py             # Loads CSVs into SQLite database
│   ├── ipl.db                        # SQLite database (auto-generated)
│   ├── 01_best_batting_average.sql
│   ├── 02_best_bowling_economy.sql
│   ├── 03_most_potm_awards.sql
│   ├── 04_powerplay_top_scorers.sql
│   ├── 05_most_dot_balls.sql
│   ├── 06_season_highest_score.sql
│   └── README.md
│
├── src/
│   ├── models/
│   │   ├── future_match_predictor.joblib   # Pre-match predictor model
│   │   └── live_win_predictor.joblib       # Live win probability model
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── ml_model.py
│
├── reports/figures/                  # Saved charts and visualizations
├── requirements.txt
└── README.md
```

---

## Dataset

- **Source:** [Kaggle — IPL Complete Dataset 2008–2024](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- **matches.csv** — 1,095 matches, 20 columns (teams, venue, toss, result, etc.)
- **deliveries.csv** — 260,920 ball-by-ball records, 18 columns

---

## Key Insights

1. **Mumbai Indians** are the most successful IPL team with **144 wins** across all seasons.
2. **Virat Kohli** is the all-time leading run scorer with 8,014 runs at a strike rate of 128.51.
3. **Yuzvendra Chahal** is the highest wicket-taker across all IPL seasons.
4. **AB de Villiers** leads all-time Player of the Match awards with 25.
5. **CH Gayle** holds the all-time highest IPL score — 175* in 2013.
6. **Bhuvneshwar Kumar** bowls the most dot balls — 40.2% of all deliveries he bowls.
7. Teams that chose to **field after winning the toss** have a higher win rate (~48%) than teams that chose to bat (~40%).
8. Pre-match winner prediction with limited features performs near baseline (~53%), showing that match outcomes require richer real-time data to predict reliably.

---

## Dashboard Features

### Teams Tab
- Total wins, win rate metrics per team
- Season-wise wins bar chart for selected team
- Toss decision vs win rate comparison
- Head-to-head record between any two teams (win share pie + season-wise breakdown)

### Batting Tab
- Top 15 run scorers overall
- Individual player card: runs, innings, average, strike rate, fours, sixes
- Dismissal breakdown by type (caught, bowled, lbw, run out) as percentages
- Season-wise runs trend (area chart)
- Dismissal type donut chart

### Bowling Tab
- Top 15 wicket takers overall
- Individual player card: wickets, economy, average, strike rate, overs
- Wicket type breakdown (caught, bowled, lbw, stumped) as percentages
- Season-wise wickets trend (area chart)
- Wicket type donut chart

### Venues Tab
- Matches played, average 1st innings score, bat-first win %, chase win %
- Average score trend per season at selected venue
- Toss decision split at venue
- Top teams and top scorers at specific venue

### Match Predictor Tab
- Enter both Playing XIs (up to 11 players each), venue, toss winner, toss decision
- Predicts winner using a Random Forest model trained on career batting strike rates and bowling economies
- Shows win probability bar chart for both teams
- Dew factor checkbox adjusts probability for night matches
- Model performance section: accuracy, toss-winner baseline, confusion matrix

### Live Win % Tab
- Enter match state: target, current score, overs completed
- Select batting and bowling XIs, mark dismissed batters, select current bowler
- Gradient Boosting model predicts chase win probability using: runs scored, wickets fallen, balls left, required run rate, current run rate
- Gauge chart + team probability cards
- Player influence table: remaining batters with strike rates, bowlers with economy and overs remaining

---

## SQL Analysis

IPL data is loaded into a local **SQLite database** for structured querying. All queries use standard SQL compatible with MySQL and PostgreSQL.

### Setup

```bash
python sql_analysis/load_to_sqlite.py
```

### Queries

| File | Question Answered |
|---|---|
| 01_best_batting_average.sql | Top 10 batsmen by batting average (min 20 innings) |
| 02_best_bowling_economy.sql | Top 10 bowlers by economy rate (min 50 overs) |
| 03_most_potm_awards.sql | Top 10 Player of the Match award winners |
| 04_powerplay_top_scorers.sql | Top 10 run scorers in powerplay overs (1-6) |
| 05_most_dot_balls.sql | Top 10 bowlers by dot balls bowled |
| 06_season_highest_score.sql | Highest individual score in each IPL season |

### SQL Concepts Used
- Aggregations: `SUM`, `COUNT`, `ROUND`, `AVG`
- Filtering: `WHERE`, `HAVING`
- Conditional aggregation: `CASE WHEN`
- Joins: `JOIN` across matches and deliveries tables
- Grouping: `GROUP BY` with multiple columns
- Null handling: `NULLIF`, `IS NOT NULL`

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10 |
| Data Analysis | Pandas, NumPy |
| Database | SQLite (via Python sqlite3) |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Machine Learning | Scikit-learn (Random Forest, Gradient Boosting) |
| Model Storage | Joblib |
| Notebooks | Jupyter |

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ipl-analytics.git
cd ipl-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set up SQL database
python sql_analysis/load_to_sqlite.py

# 4. Run the dashboard
streamlit run dashboard/app.py
```

The app will open at `http://localhost:8501`

---

## ML Models

### Pre-Match Winner Predictor (Random Forest)
- **Features:** Venue, toss winner, toss decision, team batting strike rates, team bowling economies
- **Target:** Binary — Team 1 wins or Team 2 wins
- **Accuracy:** ~53% (baseline: ~55% using toss winner alone)
- **Note:** Pre-match prediction with limited features is inherently difficult. Match outcomes depend on factors unavailable before the game — pitch conditions, player form, weather. The model demonstrates this limitation honestly rather than overfitting.

### Live Win Probability (Gradient Boosting)
- **Features:** Runs scored, wickets fallen, balls done, balls left, runs needed, required run rate, current run rate, target
- **Target:** Binary — chasing team wins or not
- **Trained on:** Ball-by-ball data from all 2nd innings across 1,095 matches
- **Adjustments:** Player strike rates and bowler economies are used to adjust raw model probability based on who is currently batting and bowling

---

## About

Built by **Ritesh** | 3rd Year IT Engineering Student

This project was independently built to demonstrate end-to-end data analyst skills: data cleaning, SQL-based data exploration, exploratory analysis, interactive dashboard development, and machine learning model building and deployment.