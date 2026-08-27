"""
feature_engineering.py
----------------------
Creates derived features from cleaned IPL data
for use in EDA and the ML win predictor model.
"""

import pandas as pd


def add_toss_advantage(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a binary column: did the toss winner also win the match?
    """
    matches = matches.copy()
    matches["toss_win_match_win"] = (
        matches["toss_winner"] == matches["winner"]
    ).astype(int)
    return matches


def add_batting_phase(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Labels each delivery with a batting phase based on over number.
    Powerplay: 0-5  |  Middle: 6-14  |  Death: 15-19
    (Already added in data_cleaning, kept here for reference.)
    """
    deliveries = deliveries.copy()

    def phase(over):
        if over <= 5:
            return "Powerplay"
        elif over <= 14:
            return "Middle"
        else:
            return "Death"

    deliveries["phase"] = deliveries["over"].apply(phase)
    return deliveries


def get_batsman_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-batsman summary:
    total runs, balls faced, strike rate, 4s, 6s.
    """
    stats = deliveries.groupby("batter").agg(
        runs=("batsman_runs", "sum"),
        balls=("ball", "count"),
        fours=("batsman_runs", lambda x: (x == 4).sum()),
        sixes=("batsman_runs", lambda x: (x == 6).sum()),
    ).reset_index()

    stats["strike_rate"] = (stats["runs"] / stats["balls"] * 100).round(2)
    return stats.sort_values("runs", ascending=False)


def get_bowler_stats(deliveries: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-bowler summary:
    wickets, runs conceded, economy rate, bowling average.
    Excludes run-outs and retired hurt from wicket count.
    """
    non_bowler_dismissals = ["run out", "retired hurt", "obstructing the field"]

    wickets = (
        deliveries[
            deliveries["dismissal_kind"].notna()
            & ~deliveries["dismissal_kind"].isin(non_bowler_dismissals)
        ]
        .groupby("bowler")["dismissal_kind"]
        .count()
        .reset_index()
        .rename(columns={"dismissal_kind": "wickets"})
    )

    runs_balls = deliveries.groupby("bowler").agg(
        runs_conceded=("total_runs", "sum"),
        balls_bowled=("ball", "count"),
    ).reset_index()

    stats = runs_balls.merge(wickets, on="bowler", how="left")
    stats["wickets"] = stats["wickets"].fillna(0).astype(int)
    stats["overs"] = (stats["balls_bowled"] / 6).round(2)
    stats["economy"] = (stats["runs_conceded"] / stats["overs"]).round(2)
    stats["bowling_avg"] = (
        stats["runs_conceded"] / stats["wickets"].replace(0, float("nan"))
    ).round(2)

    return stats.sort_values("wickets", ascending=False)


if __name__ == "__main__":
    deliveries = pd.read_csv("data/processed/deliveries_clean.csv")
    matches = pd.read_csv("data/processed/matches_clean.csv")

    matches = add_toss_advantage(matches)
    bat_stats = get_batsman_stats(deliveries)
    bowl_stats = get_bowler_stats(deliveries)

    print("Toss advantage rate:", matches["toss_win_match_win"].mean().round(3))
    print("\nTop 5 batsmen:\n", bat_stats.head())
    print("\nTop 5 bowlers:\n", bowl_stats.head())
