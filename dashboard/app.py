import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="IPL Analytics", page_icon="", layout="wide")

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1a1a2e;
}
.stApp { background-color: #f5f6fa; }

/* Fix all input text visibility */
input, textarea, [data-baseweb="select"] * {
    color: #1a1a2e !important;
    background-color: #ffffff !important;
}
[data-baseweb="select"] [data-testid="stMarkdownContainer"] { color: #1a1a2e !important; }
.stSelectbox div[data-baseweb="select"] > div { background-color: #ffffff !important; color: #1a1a2e !important; }
div[data-baseweb="popover"] { background: #ffffff !important; }
div[data-baseweb="menu"] { background: #ffffff !important; }
li[role="option"] { color: #1a1a2e !important; background: #ffffff !important; }
li[role="option"]:hover { background: #f3f4f6 !important; }
.stNumberInput input { color: #1a1a2e !important; background: #ffffff !important; border: 1px solid #d1d5db !important; border-radius: 8px !important; }
.stCheckbox label { color: #1a1a2e !important; }
[data-testid="stMultiSelect"] span { color: #1a1a2e !important; }

/* Tab bar */
[data-testid="stTabs"] [role="tablist"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #e8eaf0;
    gap: 2px;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #6b7280;
    padding: 8px 16px;
    border: none;
    background: transparent;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1a1a2e;
    color: #ffffff;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stMetric"] label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #1a1a2e !important;
}

/* Buttons */
[data-testid="stButton"] > button {
    background: #1a1a2e;
    color: #ffffff;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 10px 24px;
    transition: all 0.2s ease;
}
[data-testid="stButton"] > button:hover {
    background: #2d2d4e;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(26,26,46,0.25);
}

/* Popover filter button */
[data-testid="stPopover"] > button {
    background: #1a1a2e !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 8px 18px !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e8eaf0;
    border-radius: 12px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e8eaf0;
}

hr { border: none; border-top: 1px solid #e8eaf0; margin: 1.5rem 0; }

.section-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
}
.page-header { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.25rem; }
.page-subtitle { font-size: 0.85rem; color: #6b7280; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
* { color: #1a1a2e !important; }
select, input, textarea { 
    color: #1a1a2e !important; 
    background: white !important; 
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────
PALETTE   = ["#1a1a2e","#16213e","#0f3460","#533483","#e94560",
             "#2196f3","#00bcd4","#4caf50","#ff9800","#9c27b0"]
MONO_BLUE = ["#e8f4fd","#bee3f8","#90cdf4","#63b3ed","#4299e1",
             "#3182ce","#2b6cb0","#2c5282","#2a4365","#1A365D"]

def apply_theme(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a1a2e", family="Inter"),
                   x=0, xanchor="left", pad=dict(l=0)),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter", color="#374151", size=12),
        margin=dict(l=0, r=0, t=48, b=0),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11, color="#6b7280")),
        xaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb", tickfont=dict(size=11, color="#6b7280")),
        yaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb", tickfont=dict(size=11, color="#6b7280")),
        coloraxis_showscale=False,
    )
    return fig

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_PATH      = os.path.join(BASE_DIR, "data", "processed", "matches_clean.csv")
DELIVERIES_PATH   = os.path.join(BASE_DIR, "data", "processed", "deliveries_clean.csv")
FUTURE_MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "future_match_predictor.joblib")
LIVE_MODEL_PATH   = os.path.join(BASE_DIR, "src", "models", "live_win_predictor.joblib")

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    matches    = pd.read_csv(MATCHES_PATH)
    deliveries = pd.read_csv(DELIVERIES_PATH)
    return matches, deliveries

matches, deliveries = load_data()

def clean_season(s):
    s = str(s)
    if "/" in s:
        return int(s.split("/")[0]) + 1
    return int(s)

matches["season"] = matches["season"].apply(clean_season)

# ─────────────────────────────────────────────────────────────
# PLAYER LOOKUPS
# ─────────────────────────────────────────────────────────────
@st.cache_data
def build_player_lookups(deliveries):
    bat = (deliveries.groupby("batter")
           .agg(total_runs=("batsman_runs","sum"), balls=("batsman_runs","count"))
           .reset_index())
    bat["bat_sr"] = bat["total_runs"] / bat["balls"] * 100

    non_bowler = ["run out","retired hurt","obstructing the field"]
    wk = (deliveries[deliveries["dismissal_kind"].notna() &
                     ~deliveries["dismissal_kind"].isin(non_bowler)]
          .groupby("bowler")["dismissal_kind"].count().reset_index()
          .rename(columns={"dismissal_kind":"wickets"}))
    bowl = (deliveries.groupby("bowler")
            .agg(runs_given=("total_runs","sum"), balls_b=("total_runs","count"))
            .reset_index().merge(wk, on="bowler", how="left").fillna(0))
    bowl["economy"] = bowl["runs_given"] / (bowl["balls_b"] / 6)
    return dict(zip(bat["batter"], bat["bat_sr"])), dict(zip(bowl["bowler"], bowl["economy"]))

bat_lookup, bowl_lookup = build_player_lookups(deliveries)
all_players = sorted(pd.concat([deliveries["batter"], deliveries["bowler"]]).dropna().unique())

# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_future_model(matches, deliveries):
    if os.path.exists(FUTURE_MODEL_PATH):
        return joblib.load(FUTURE_MODEL_PATH)
    mf = matches[matches["winner"].notna()].copy()
    mp = pd.concat([
        deliveries[["match_id","batting_team","batter"]].rename(columns={"batting_team":"team","batter":"player"}),
        deliveries[["match_id","bowling_team","bowler"]].rename(columns={"bowling_team":"team","bowler":"player"}),
    ]).drop_duplicates()
    def team_feats(mid, team):
        players = mp[(mp["match_id"]==mid) & (mp["team"]==team)]["player"].unique()
        return (np.mean([bat_lookup.get(p,100) for p in players]) if len(players) else 100,
                np.mean([bowl_lookup.get(p,8)  for p in players]) if len(players) else 8)
    rows = []
    for _, r in mf.iterrows():
        t1_bs, t1_ec = team_feats(r["id"], r["team1"])
        t2_bs, t2_ec = team_feats(r["id"], r["team2"])
        rows.append({"venue_raw":r["venue"], "toss_winner_is_t1":int(r["toss_winner"]==r["team1"]),
                     "toss_bat":int(r["toss_decision"]=="bat"),
                     "team1_bat_sr":t1_bs,"team2_bat_sr":t2_bs,
                     "team1_economy":t1_ec,"team2_economy":t2_ec,
                     "winner_is_t1":int(r["winner"]==r["team1"])})
    df = pd.DataFrame(rows)
    le = LabelEncoder()
    df["venue_enc"] = le.fit_transform(df["venue_raw"])
    feat = ["venue_enc","toss_winner_is_t1","toss_bat","team1_bat_sr","team2_bat_sr","team1_economy","team2_economy"]
    X, y = df[feat], df["winner_is_t1"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, ytr)
    toss_base = accuracy_score(yte, Xte["toss_winner_is_t1"])
    payload = {"model":clf,"le_venue":le,"feat_cols":feat,"bat_lookup":bat_lookup,
               "bowl_lookup":bowl_lookup,"X_test":Xte,"y_test":yte,"toss_baseline_acc":toss_base}
    os.makedirs(os.path.dirname(FUTURE_MODEL_PATH), exist_ok=True)
    joblib.dump(payload, FUTURE_MODEL_PATH)
    return payload

@st.cache_resource
def get_live_model(matches, deliveries):
    if os.path.exists(LIVE_MODEL_PATH):
        return joblib.load(LIVE_MODEL_PATH)
    mf = matches[matches["winner"].notna()][["id","winner","target_runs"]].copy()
    d2 = deliveries[deliveries["inning"]==2].merge(mf, left_on="match_id", right_on="id")
    rows = []
    for match_id, grp in d2.groupby("match_id"):
        grp    = grp.sort_values(["over","ball"]).reset_index(drop=True)
        target = grp["target_runs"].iloc[0]
        bat_t  = grp["batting_team"].iloc[0]
        winner = grp["winner"].iloc[0]
        chasing_wins = int(winner == bat_t)
        cum_r = cum_w = 0
        for _, ball in grp.iterrows():
            cum_r += ball["total_runs"]
            if ball["is_wicket"]: cum_w += 1
            bd = ball["over"]*6 + ball["ball"]
            bl = max(120-bd, 0)
            rn = target - cum_r
            ol = bl/6
            if bl <= 0 or rn <= 0 or cum_w >= 10: break
            rrr = min(rn/ol if ol>0 else 99, 36)
            crr = min(cum_r/(bd/6) if bd>0 else 0, 36)
            if ball["ball"] == 6:
                rows.append({"runs_scored":cum_r,"wickets_fallen":cum_w,"balls_done":bd,
                             "balls_left":bl,"runs_needed":rn,"rrr":rrr,"crr":crr,
                             "target":target,"chasing_wins":chasing_wins})
    df = pd.DataFrame(rows)
    feat = ["runs_scored","wickets_fallen","balls_done","balls_left","runs_needed","rrr","crr","target"]
    X, y = df[feat], df["chasing_wins"]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    clf.fit(Xtr, ytr)
    payload = {"model":clf,"feat_cols":feat}
    os.makedirs(os.path.dirname(LIVE_MODEL_PATH), exist_ok=True)
    joblib.dump(payload, LIVE_MODEL_PATH)
    return payload

with st.spinner("Loading models..."):
    future_pkg = get_future_model(matches, deliveries)
    live_pkg   = get_live_model(matches, deliveries)

# ─────────────────────────────────────────────────────────────
# REFERENCE LISTS
# ─────────────────────────────────────────────────────────────
all_seasons = sorted(matches["season"].dropna().unique())
all_teams   = sorted(pd.concat([matches["team1"], matches["team2"]]).dropna().unique())
all_batters = sorted(deliveries["batter"].dropna().unique())
all_bowlers = sorted(deliveries["bowler"].dropna().unique())
non_bowler_dismissals = ["run out","retired hurt","obstructing the field"]

# ─────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:1.5rem 0 0.5rem 0;border-bottom:1px solid #e8eaf0;margin-bottom:1.5rem;'>
    <div class='page-header'>IPL Analytics Dashboard</div>
    <div class='page-subtitle'>Indian Premier League · 2008–2024 · 1,095 Matches</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Teams", "Batting", "Bowling",
    "Venues", "Match Predictor", "Live Win %"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — TEAMS
# ═══════════════════════════════════════════════════════════
with tab1:
    # Filter button — top right
    fcol1, fcol2 = st.columns([5,1])
    with fcol1:
        st.markdown('<div class="page-header" style="font-size:1.2rem;">Team Performance</div>', unsafe_allow_html=True)
    with fcol2:
        with st.popover("Filters"):
            selected_seasons_t1 = st.multiselect("Season", all_seasons, default=all_seasons, key="t1_seasons")
            selected_team = st.selectbox("Team", ["All Teams"] + all_teams, key="t1_team")

    # Apply filters
    mf1 = matches[matches["season"].isin(selected_seasons_t1)] if selected_seasons_t1 else matches.copy()
    team_matches = mf1[(mf1["team1"]==selected_team)|(mf1["team2"]==selected_team)] if selected_team != "All Teams" else mf1

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches", len(team_matches))
    if selected_team != "All Teams":
        wins_count = (team_matches["winner"] == selected_team).sum()
        win_pct    = round(wins_count / len(team_matches) * 100, 1) if len(team_matches) else 0
        col2.metric("Wins", wins_count)
        col3.metric("Win Rate", f"{win_pct}%")
    else:
        col2.metric("Seasons", len(selected_seasons_t1) if selected_seasons_t1 else len(all_seasons))
        col3.metric("Teams", team_matches[["team1","team2"]].stack().nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    c_l, c_r = st.columns(2)
    with c_l:
        if selected_team != "All Teams":
            season_wins = (team_matches[team_matches["winner"]==selected_team]
                           .groupby("season")["winner"].count().reset_index())
            season_wins.columns = ["Season","Wins"]
            fig = px.bar(season_wins, x="Season", y="Wins", color_discrete_sequence=["#1a1a2e"])
            apply_theme(fig, f"{selected_team} — Wins per Season")
        else:
            wins = team_matches["winner"].value_counts().reset_index()
            wins.columns = ["Team","Wins"]
            fig = px.bar(wins.head(10), x="Wins", y="Team", orientation="h",
                         color_discrete_sequence=["#1a1a2e"])
            apply_theme(fig, "Top 10 Teams by Total Wins", height=400)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c_r:
        toss_df = team_matches.copy()
        toss_df["toss_won_match"] = toss_df["toss_winner"] == toss_df["winner"]
        toss_rate = toss_df.groupby("toss_decision")["toss_won_match"].mean().reset_index()
        toss_rate.columns = ["Decision","Win Rate"]
        toss_rate["Win Rate"] = (toss_rate["Win Rate"]*100).round(1)
        fig2 = px.bar(toss_rate, x="Decision", y="Win Rate", text="Win Rate", color="Decision",
                      color_discrete_sequence=["#1a1a2e","#e94560"])
        fig2.update_traces(texttemplate="%{text}%", textposition="outside", marker_line_width=0)
        apply_theme(fig2, "Toss Decision vs Win Rate (%)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Head-to-Head
    st.markdown('<div class="section-title">Head-to-Head</div>', unsafe_allow_html=True)
    h2h_col1, h2h_col2 = st.columns(2)
    with h2h_col1:
        team_a = st.selectbox("Team A", all_teams, key="h2h_a")
    with h2h_col2:
        team_b = st.selectbox("Team B", [t for t in all_teams if t != team_a], key="h2h_b")

    h2h = mf1[
        ((mf1["team1"]==team_a)&(mf1["team2"]==team_b)) |
        ((mf1["team1"]==team_b)&(mf1["team2"]==team_a))
    ].copy()

    if h2h.empty:
        st.info("No matches found between these teams in the selected seasons.")
    else:
        a_wins = (h2h["winner"]==team_a).sum()
        b_wins = (h2h["winner"]==team_b).sum()
        ties   = h2h["winner"].isna().sum()

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Matches", len(h2h))
        m2.metric(team_a, a_wins)
        m3.metric(team_b, b_wins)
        m4.metric("No Result", ties)

        hc1, hc2 = st.columns(2)
        with hc1:
            fig_h2h = px.pie(names=[team_a,team_b,"No Result"], values=[a_wins,b_wins,ties],
                             hole=0.55, color_discrete_sequence=["#1a1a2e","#e94560","#d1d5db"])
            fig_h2h.update_traces(textinfo="percent+label", textfont_size=12)
            apply_theme(fig_h2h, "Win Share")
            st.plotly_chart(fig_h2h, use_container_width=True)
        with hc2:
            h2h_season = (h2h[h2h["winner"].notna()]
                          .groupby(["season","winner"]).size().reset_index(name="Wins"))
            fig_h2h2 = px.bar(h2h_season, x="season", y="Wins", color="winner",
                              barmode="group", color_discrete_sequence=["#1a1a2e","#e94560"])
            apply_theme(fig_h2h2, "Season-wise Wins")
            fig_h2h2.update_traces(marker_line_width=0)
            st.plotly_chart(fig_h2h2, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2 — BATTING
# ═══════════════════════════════════════════════════════════
with tab2:
    fcol1, fcol2 = st.columns([5,1])
    with fcol1:
        st.markdown('<div class="page-header" style="font-size:1.2rem;">Batting Analysis</div>', unsafe_allow_html=True)
    with fcol2:
        with st.popover("Filters"):
            selected_seasons_t2 = st.multiselect("Season", all_seasons, default=all_seasons, key="t2_seasons")
            selected_batter = st.selectbox("Batter", ["All Batters"] + all_batters, key="t2_batter")

    mf2 = matches[matches["season"].isin(selected_seasons_t2)] if selected_seasons_t2 else matches.copy()
    match_ids_f2 = mf2["id"].unique()
    deliveries_f2 = deliveries[deliveries["match_id"].isin(match_ids_f2)]

    if selected_batter != "All Batters":
        batter_del   = deliveries_f2[deliveries_f2["batter"]==selected_batter]
        total_runs   = batter_del["batsman_runs"].sum()
        balls_faced  = len(batter_del)
        fours        = (batter_del["batsman_runs"]==4).sum()
        sixes        = (batter_del["batsman_runs"]==6).sum()
        strike_rate  = round(total_runs/balls_faced*100, 2) if balls_faced else 0
        innings_played = batter_del["match_id"].nunique()
        dismissals   = deliveries_f2[
            (deliveries_f2["player_dismissed"]==selected_batter) &
            (deliveries_f2["dismissal_kind"].notna())
        ]["match_id"].nunique()
        avg = round(total_runs/dismissals, 2) if dismissals else "—"

        st.markdown(f"""
        <div style='background:#1a1a2e;border-radius:14px;padding:20px 28px;margin-bottom:1.5rem;'>
            <div style='font-size:1.4rem;font-weight:700;color:#ffffff;'>{selected_batter}</div>
            <div style='font-size:0.78rem;color:#9ca3af;margin-top:4px;'>Batting Career · IPL {min(selected_seasons_t2) if selected_seasons_t2 else ""}–{max(selected_seasons_t2) if selected_seasons_t2 else ""}</div>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Runs", total_runs)
        c2.metric("Innings", innings_played)
        c3.metric("Average", avg)
        c4.metric("Strike Rate", strike_rate)
        c5.metric("Fours", fours)
        c6.metric("Sixes", sixes)

        dismissal_counts = deliveries_f2[deliveries_f2["player_dismissed"]==selected_batter]["dismissal_kind"].value_counts()
        if not dismissal_counts.empty:
            total_d = dismissal_counts.sum()
            st.markdown('<div class="section-title" style="margin-top:1rem;">Dismissal Breakdown</div>', unsafe_allow_html=True)
            d_cols = st.columns(4)
            for col, dtype in zip(d_cols, ["caught","bowled","lbw","run out"]):
                pct = round(dismissal_counts.get(dtype,0)/total_d*100,1)
                col.metric(dtype.title(), f"{pct}%")

        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        with ch1:
            season_map  = mf2[["id","season"]].rename(columns={"id":"match_id"})
            season_runs = (deliveries_f2[deliveries_f2["batter"]==selected_batter]
                           .merge(season_map, on="match_id")
                           .groupby("season")["batsman_runs"].sum().reset_index())
            season_runs.columns = ["Season","Runs"]
            fig = px.area(season_runs, x="Season", y="Runs", color_discrete_sequence=["#1a1a2e"])
            fig.update_traces(fill="tozeroy", fillcolor="rgba(26,26,46,0.1)", line_width=2)
            apply_theme(fig, "Runs per Season")
            st.plotly_chart(fig, use_container_width=True)
        with ch2:
            dismissal_data = deliveries_f2[deliveries_f2["player_dismissed"]==selected_batter]["dismissal_kind"].value_counts().reset_index()
            dismissal_data.columns = ["Dismissal Type","Count"]
            if not dismissal_data.empty:
                fig2 = px.pie(dismissal_data, names="Dismissal Type", values="Count",
                              hole=0.55, color_discrete_sequence=PALETTE)
                fig2.update_traces(textinfo="percent+label", textfont_size=11)
                apply_theme(fig2, "How They Get Out")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        batting = (deliveries_f2.groupby("batter")["batsman_runs"]
                   .sum().sort_values(ascending=False).head(15).reset_index())
        batting.columns = ["Batter","Runs"]
        fig = px.bar(batting, x="Batter", y="Runs", color="Runs", color_continuous_scale=MONO_BLUE)
        apply_theme(fig, "Top 15 Run Scorers", height=420)
        fig.update_traces(marker_line_width=0)
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — BOWLING
# ═══════════════════════════════════════════════════════════
with tab3:
    fcol1, fcol2 = st.columns([5,1])
    with fcol1:
        st.markdown('<div class="page-header" style="font-size:1.2rem;">Bowling Analysis</div>', unsafe_allow_html=True)
    with fcol2:
        with st.popover("Filters"):
            selected_seasons_t3 = st.multiselect("Season", all_seasons, default=all_seasons, key="t3_seasons")
            selected_bowler = st.selectbox("Bowler", ["All Bowlers"] + all_bowlers, key="t3_bowler")

    mf3 = matches[matches["season"].isin(selected_seasons_t3)] if selected_seasons_t3 else matches.copy()
    match_ids_f3 = mf3["id"].unique()
    deliveries_f3 = deliveries[deliveries["match_id"].isin(match_ids_f3)]

    if selected_bowler != "All Bowlers":
        bowler_del  = deliveries_f3[deliveries_f3["bowler"]==selected_bowler]
        wickets_del = bowler_del[bowler_del["dismissal_kind"].notna() &
                                 ~bowler_del["dismissal_kind"].isin(non_bowler_dismissals)]
        total_wickets = len(wickets_del)
        runs_conceded = bowler_del["total_runs"].sum()
        balls_bowled  = len(bowler_del)
        overs         = round(balls_bowled/6, 2)
        economy       = round(runs_conceded/overs, 2) if overs else 0
        bowling_avg   = round(runs_conceded/total_wickets, 2) if total_wickets else "—"
        bowling_sr    = round(balls_bowled/total_wickets, 2) if total_wickets else "—"

        st.markdown(f"""
        <div style='background:#1a1a2e;border-radius:14px;padding:20px 28px;margin-bottom:1.5rem;'>
            <div style='font-size:1.4rem;font-weight:700;color:#ffffff;'>{selected_bowler}</div>
            <div style='font-size:0.78rem;color:#9ca3af;margin-top:4px;'>Bowling Career · IPL {min(selected_seasons_t3) if selected_seasons_t3 else ""}–{max(selected_seasons_t3) if selected_seasons_t3 else ""}</div>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Wickets", total_wickets)
        c2.metric("Economy", economy)
        c3.metric("Average", bowling_avg)
        c4.metric("Strike Rate", bowling_sr)
        c5.metric("Overs", overs)

        if total_wickets > 0:
            d_counts = wickets_del["dismissal_kind"].value_counts()
            st.markdown('<div class="section-title" style="margin-top:1rem;">Wicket Type Breakdown</div>', unsafe_allow_html=True)
            d_cols = st.columns(4)
            for col, dtype in zip(d_cols, ["caught","bowled","lbw","stumped"]):
                pct = round(d_counts.get(dtype,0)/total_wickets*100,1)
                col.metric(dtype.title(), f"{pct}%")

        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        with ch1:
            season_map = mf3[["id","season"]].rename(columns={"id":"match_id"})
            season_wickets = (wickets_del.merge(season_map, on="match_id")
                              .groupby("season")["dismissal_kind"].count().reset_index())
            season_wickets.columns = ["Season","Wickets"]
            fig = px.area(season_wickets, x="Season", y="Wickets", color_discrete_sequence=["#e94560"])
            fig.update_traces(fill="tozeroy", fillcolor="rgba(233,69,96,0.1)", line_width=2)
            apply_theme(fig, "Wickets per Season")
            st.plotly_chart(fig, use_container_width=True)
        with ch2:
            dismissal_data = wickets_del["dismissal_kind"].value_counts().reset_index()
            dismissal_data.columns = ["Dismissal Type","Count"]
            if not dismissal_data.empty:
                fig2 = px.pie(dismissal_data, names="Dismissal Type", values="Count",
                              hole=0.55, color_discrete_sequence=PALETTE)
                fig2.update_traces(textinfo="percent+label", textfont_size=11)
                apply_theme(fig2, "Wicket Types")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        wickets = deliveries_f3[deliveries_f3["dismissal_kind"].notna() &
                                ~deliveries_f3["dismissal_kind"].isin(non_bowler_dismissals)]
        top_bowlers = (wickets.groupby("bowler")["dismissal_kind"]
                       .count().sort_values(ascending=False).head(15).reset_index())
        top_bowlers.columns = ["Bowler","Wickets"]
        fig = px.bar(top_bowlers, x="Bowler", y="Wickets",
                     color="Wickets", color_continuous_scale=["#fee2e2","#e94560","#7f1d1d"])
        apply_theme(fig, "Top 15 Wicket Takers", height=420)
        fig.update_traces(marker_line_width=0)
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 4 — VENUES
# ═══════════════════════════════════════════════════════════
with tab4:
    fcol1, fcol2 = st.columns([5,1])
    with fcol1:
        st.markdown('<div class="page-header" style="font-size:1.2rem;">Venue Analysis</div>', unsafe_allow_html=True)
    with fcol2:
        with st.popover("Filters"):
            selected_seasons_t4 = st.multiselect("Season", all_seasons, default=all_seasons, key="t4_seasons")
            all_venues = sorted(matches["venue"].dropna().unique())
            sel_venue  = st.selectbox("Venue", ["All Venues"] + list(all_venues), key="t4_venue")

    mf4 = matches[matches["season"].isin(selected_seasons_t4)] if selected_seasons_t4 else matches.copy()
    vm = mf4[mf4["venue"]==sel_venue] if sel_venue != "All Venues" else mf4
    venue_del = deliveries[deliveries["match_id"].isin(vm["id"].unique())]

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Matches Played", len(vm))
    avg_first = vm["target_runs"].dropna().mean()
    m2.metric("Avg 1st Innings", int(avg_first) if not np.isnan(avg_first) else "—")
    bat_first_total = len(vm[vm["toss_decision"]=="bat"])
    bat_win_pct = round(vm[vm["toss_decision"]=="bat"]["winner"].notna().sum()/bat_first_total*100,1) if bat_first_total else 0
    m3.metric("Bat First Win %", f"{bat_win_pct}%")
    field_total = len(vm[vm["toss_decision"]=="field"])
    chase_win_pct = round(vm[vm["toss_decision"]=="field"]["winner"].notna().sum()/field_total*100,1) if field_total else 0
    m4.metric("Chase Win %", f"{chase_win_pct}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        avg_by_season = vm.groupby("season")["target_runs"].mean().reset_index()
        avg_by_season.columns = ["Season","Avg Score"]
        fig_v1 = px.line(avg_by_season, x="Season", y="Avg Score",
                         color_discrete_sequence=["#1a1a2e"], markers=True)
        fig_v1.update_traces(line_width=2, marker_size=6)
        apply_theme(fig_v1, "Avg 1st Innings Score per Season")
        st.plotly_chart(fig_v1, use_container_width=True)
    with col_r:
        toss_counts = vm["toss_decision"].value_counts().reset_index()
        toss_counts.columns = ["Decision","Count"]
        fig_v2 = px.pie(toss_counts, names="Decision", values="Count",
                        hole=0.55, color_discrete_sequence=["#1a1a2e","#e94560"])
        fig_v2.update_traces(textinfo="percent+label", textfont_size=12)
        apply_theme(fig_v2, "Toss Decision Split")
        st.plotly_chart(fig_v2, use_container_width=True)

    if sel_venue != "All Venues":
        vb1, vb2 = st.columns(2)
        with vb1:
            team_wins_v = vm["winner"].value_counts().reset_index().head(8)
            team_wins_v.columns = ["Team","Wins"]
            fig_v3 = px.bar(team_wins_v, x="Team", y="Wins", color="Wins", color_continuous_scale=MONO_BLUE)
            apply_theme(fig_v3, f"Most Wins at {sel_venue}")
            fig_v3.update_traces(marker_line_width=0)
            st.plotly_chart(fig_v3, use_container_width=True)
        with vb2:
            top_bat = (venue_del.groupby("batter")["batsman_runs"]
                       .sum().sort_values(ascending=False).head(10).reset_index())
            top_bat.columns = ["Batter","Runs"]
            fig_v4 = px.bar(top_bat, x="Batter", y="Runs",
                            color="Runs", color_continuous_scale=["#dcfce7","#4ade80","#15803d"])
            apply_theme(fig_v4, f"Top Scorers at {sel_venue}")
            fig_v4.update_traces(marker_line_width=0)
            fig_v4.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_v4, use_container_width=True)
    else:
        vb1, vb2 = st.columns(2)
        with vb1:
            top_v = mf4["venue"].value_counts().head(10).reset_index()
            top_v.columns = ["Venue","Matches"]
            fig_ov = px.bar(top_v, x="Matches", y="Venue", orientation="h",
                            color="Matches", color_continuous_scale=MONO_BLUE)
            apply_theme(fig_ov, "Top 10 Venues by Matches", height=400)
            fig_ov.update_traces(marker_line_width=0)
            st.plotly_chart(fig_ov, use_container_width=True)
        with vb2:
            avg_venue = (mf4.groupby("venue")["target_runs"]
                         .mean().sort_values(ascending=False).head(10).reset_index())
            avg_venue.columns = ["Venue","Avg Score"]
            fig_av = px.bar(avg_venue, x="Avg Score", y="Venue", orientation="h",
                            color="Avg Score", color_continuous_scale=["#ede9fe","#8b5cf6","#4c1d95"])
            apply_theme(fig_av, "Top 10 Venues by Avg Score", height=400)
            fig_av.update_traces(marker_line_width=0)
            st.plotly_chart(fig_av, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 5 — MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="page-header" style="font-size:1.2rem;">Match Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter both squads, venue and toss details to predict the winner</div>', unsafe_allow_html=True)

    le_venue   = future_pkg["le_venue"]
    clf_future = future_pkg["model"]
    X_test_f   = future_pkg["X_test"]
    y_test_f   = future_pkg["y_test"]
    toss_base  = future_pkg["toss_baseline_acc"]

    with st.expander("Model Performance & Confusion Matrix", expanded=False):
        model_acc = accuracy_score(y_test_f, clf_future.predict(X_test_f))
        pm1,pm2,pm3 = st.columns(3)
        pm1.metric("Model Accuracy",       f"{round(model_acc*100,1)}%")
        pm2.metric("Toss-Winner Baseline", f"{round(toss_base*100,1)}%")
        pm3.metric("Improvement",          f"+{round((model_acc-toss_base)*100,1)}%")
        cm = confusion_matrix(y_test_f, clf_future.predict(X_test_f))
        fig_cm = px.imshow(cm, text_auto=True,
                           labels=dict(x="Predicted",y="Actual"),
                           x=["Team 2 Wins","Team 1 Wins"],
                           y=["Team 2 Wins","Team 1 Wins"],
                           color_continuous_scale=MONO_BLUE)
        apply_theme(fig_cm, "Confusion Matrix", height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    venue_list   = sorted(matches["venue"].dropna().unique())
    known_venues = list(le_venue.classes_)

    ms1,ms2,ms3,ms4 = st.columns(4)
    with ms1: t1_venue = st.selectbox("Venue", venue_list, key="f_venue")
    with ms2: t1_toss  = st.selectbox("Toss Winner", ["Team 1","Team 2"], key="f_toss")
    with ms3: t1_dec   = st.selectbox("Toss Decision", ["bat","field"], key="f_dec")
    with ms4:
        st.markdown("<br>", unsafe_allow_html=True)
        dew = st.checkbox("Dew Expected", key="f_dew")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div style='background:#f8fafc;border:1px solid #e8eaf0;border-radius:12px;
                    padding:16px 20px;margin-bottom:1rem;'>
                    <div style='font-weight:700;color:#1a1a2e;margin-bottom:12px;'>Team 1 — Playing XI</div>
                    </div>""", unsafe_allow_html=True)
        t1_players = []
        for i in range(11):
            p = st.selectbox(f"T1 Player {i+1}", [""] + all_players,
                             key=f"t1p{i}", label_visibility="collapsed")
            if p: t1_players.append(p)
        st.caption(f"{len(t1_players)}/11 players selected")

    with col2:
        st.markdown("""<div style='background:#f8fafc;border:1px solid #e8eaf0;border-radius:12px;
                    padding:16px 20px;margin-bottom:1rem;'>
                    <div style='font-weight:700;color:#1a1a2e;margin-bottom:12px;'>Team 2 — Playing XI</div>
                    </div>""", unsafe_allow_html=True)
        t2_players = []
        for i in range(11):
            p = st.selectbox(f"T2 Player {i+1}", [""] + all_players,
                             key=f"t2p{i}", label_visibility="collapsed")
            if p: t2_players.append(p)
        st.caption(f"{len(t2_players)}/11 players selected")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Predict Winner", key="future_btn"):
        if len(t1_players) < 5 or len(t2_players) < 5:
            st.warning("Please select at least 5 players per team.")
        else:
            t1_bs = np.mean([bat_lookup.get(p,100) for p in t1_players])
            t1_ec = np.mean([bowl_lookup.get(p,8)  for p in t1_players])
            t2_bs = np.mean([bat_lookup.get(p,100) for p in t2_players])
            t2_ec = np.mean([bowl_lookup.get(p,8)  for p in t2_players])
            venue_enc  = le_venue.transform([t1_venue])[0] if t1_venue in known_venues else 0
            toss_is_t1 = int(t1_toss=="Team 1")
            toss_bat   = int(t1_dec=="bat")
            inp = pd.DataFrame([{"venue_enc":venue_enc,"toss_winner_is_t1":toss_is_t1,
                                  "toss_bat":toss_bat,"team1_bat_sr":t1_bs,"team2_bat_sr":t2_bs,
                                  "team1_economy":t1_ec,"team2_economy":t2_ec}])
            proba = clf_future.predict_proba(inp)[0]
            p_t2, p_t1 = proba[0], proba[1]
            if dew:
                dew_bonus = 0.03
                if t1_dec=="bat": p_t2=min(1.0,p_t2+dew_bonus); p_t1=1-p_t2
                else: p_t1=min(1.0,p_t1+dew_bonus); p_t2=1-p_t1
            winner_label = "Team 1" if p_t1>=p_t2 else "Team 2"
            win_pct = round(max(p_t1,p_t2)*100,1)
            st.markdown(f"""
            <div style='background:#1a1a2e;border-radius:14px;padding:24px 32px;
                        display:flex;align-items:center;justify-content:space-between;margin:1rem 0;'>
                <div>
                    <div style='font-size:0.75rem;color:#9ca3af;font-weight:600;text-transform:uppercase;'>Predicted Winner</div>
                    <div style='font-size:1.8rem;font-weight:700;color:#ffffff;margin-top:4px;'>{winner_label}</div>
                </div>
                <div style='text-align:right;'>
                    <div style='font-size:0.75rem;color:#9ca3af;font-weight:600;text-transform:uppercase;'>Win Probability</div>
                    <div style='font-size:1.8rem;font-weight:700;color:#4ade80;margin-top:4px;'>{win_pct}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            fig_pred = go.Figure(go.Bar(
                x=["Team 1","Team 2"], y=[round(p_t1*100,1),round(p_t2*100,1)],
                marker_color=["#1a1a2e","#e94560"],
                text=[f"{round(p_t1*100,1)}%",f"{round(p_t2*100,1)}%"],
                textposition="outside", marker_line_width=0,
            ))
            apply_theme(fig_pred, "Win Probability Breakdown", height=320)
            fig_pred.update_layout(yaxis_range=[0,110], yaxis_title="Probability (%)")
            st.plotly_chart(fig_pred, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 6 — LIVE WIN %
# ═══════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="page-header" style="font-size:1.2rem;">Live Win Probability</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Chase scenario — enter match state for real-time prediction</div>', unsafe_allow_html=True)

    clf_live       = live_pkg["model"]
    feat_cols_live = live_pkg["feat_cols"]
    venue_list_live = sorted(matches["venue"].dropna().unique())

    c1,c2,c3 = st.columns(3)
    with c1: batting_team = st.selectbox("Batting Team (Chasing)", all_teams, key="live_bat")
    with c2: bowling_team = st.selectbox("Bowling Team (Defending)", [t for t in all_teams if t!=batting_team], key="live_bowl")
    with c3: live_venue   = st.selectbox("Venue", venue_list_live, key="live_venue")

    st.markdown("<br>", unsafe_allow_html=True)
    c4,c5,c6 = st.columns(3)
    with c4: target        = st.number_input("Target", min_value=1, max_value=300, value=180)
    with c5: current_score = st.number_input("Current Score", min_value=0, max_value=299, value=80)
    with c6: overs_done    = st.number_input("Overs Completed", min_value=0.0, max_value=20.0, value=10.0, step=0.1)

    st.markdown("---")
    lc1, lc2 = st.columns(2)

    with lc1:
        st.markdown(f"**{batting_team} — Playing XI**")
        bat_xi = []
        bat_cols = st.columns(2)
        for i in range(11):
            p = bat_cols[i%2].selectbox(f"Batter {i+1}", [""] + all_players, key=f"lbat_{i}")
            if p: bat_xi.append(p)
        dismissed = st.multiselect("Dismissed batters", options=bat_xi, key="live_dismissed") if bat_xi else []
        wickets_fallen    = len(dismissed)
        remaining_batters = [p for p in bat_xi if p not in dismissed]
        st.markdown(f"""
        <div style='display:flex;gap:12px;margin:12px 0;'>
            <div style='background:#fef2f2;border-radius:8px;padding:8px 14px;font-size:0.82rem;font-weight:600;color:#dc2626;'>Wickets: {wickets_fallen}</div>
            <div style='background:#f0fdf4;border-radius:8px;padding:8px 14px;font-size:0.82rem;font-weight:600;color:#16a34a;'>Remaining: {len(remaining_batters)}</div>
        </div>
        """, unsafe_allow_html=True)
        cr1,cr2 = st.columns(2)
        with cr1: batter1 = st.selectbox("Batter 1 (on crease)", [""] + remaining_batters, key="live_b1")
        with cr2: batter2 = st.selectbox("Batter 2 (on crease)", [""] + remaining_batters, key="live_b2")

    with lc2:
        st.markdown(f"**{bowling_team} — Playing XI**")
        bowl_xi = []
        bowl_overs = {}
        bowl_cols = st.columns(2)
        for i in range(11):
            col = bowl_cols[i%2]
            p = col.selectbox(f"Bowler {i+1}", [""] + all_players, key=f"lbwl_{i}")
            if p:
                bowl_xi.append(p)
                ov = col.number_input(f"Overs bowled", min_value=0.0, max_value=4.0, value=0.0, step=0.1, key=f"lbwl_ov_{i}")
                bowl_overs[p] = ov
        bowler_now = st.selectbox("Current Bowler", [""] + bowl_xi if bowl_xi else [""], key="live_bwl")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Calculate Win Probability", type="primary"):
        balls_done  = int(overs_done*6)
        balls_left  = max(120-balls_done, 0)
        runs_needed = max(target-current_score, 0)
        overs_left  = balls_left/6 if balls_left>0 else 0.01
        rrr  = min(runs_needed/overs_left, 36)
        crr  = min(current_score/(balls_done/6) if balls_done>0 else 0, 36)

        remaining_srs = [bat_lookup.get(p,100) for p in remaining_batters] or [100]
        b1_sr = bat_lookup.get(batter1,100)
        b2_sr = bat_lookup.get(batter2,100)
        crease_sr = np.mean([b1_sr,b2_sr]) if (batter1 or batter2) else np.mean(remaining_srs)

        remaining_bowlers = [p for p in bowl_xi if bowl_overs.get(p,0)<4]
        bowl_economies    = [bowl_lookup.get(p,8) for p in remaining_bowlers] or [8]
        avg_bowl_ec       = np.mean(bowl_economies)
        cur_bowl_ec       = bowl_lookup.get(bowler_now, avg_bowl_ec)

        inp_live = pd.DataFrame([{"runs_scored":current_score,"wickets_fallen":wickets_fallen,
                                   "balls_done":balls_done,"balls_left":balls_left,
                                   "runs_needed":runs_needed,"rrr":rrr,"crr":crr,"target":target}])
        proba_live    = clf_live.predict_proba(inp_live)[0]
        raw_prob      = proba_live[1]
        sr_advantage  = (crease_sr - cur_bowl_ec*6) / 200
        depth_factor  = (len(remaining_batters)-2)*0.005
        adjusted_prob = np.clip(raw_prob+sr_advantage+depth_factor, 0.05, 0.95)
        defend_prob   = 1-adjusted_prob

        km1,km2,km3,km4,km5 = st.columns(5)
        km1.metric("Runs Needed",  runs_needed)
        km2.metric("Balls Left",   balls_left)
        km3.metric("Required RR",  round(rrr,2))
        km4.metric("Current RR",   round(crr,2))
        km5.metric("Wickets Down", wickets_fallen)

        st.markdown("<br>", unsafe_allow_html=True)
        gc1, gc2 = st.columns([2,1])
        with gc1:
            fig_live = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(adjusted_prob*100,1),
                title={"text":f"{batting_team} Win Probability","font":{"size":14,"color":"#1a1a2e","family":"Inter"}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#6b7280"},
                       "bar":{"color":"#1a1a2e"},"bgcolor":"white",
                       "borderwidth":1,"bordercolor":"#e8eaf0",
                       "steps":[{"range":[0,33],"color":"#fee2e2"},
                                 {"range":[33,66],"color":"#fef9c3"},
                                 {"range":[66,100],"color":"#dcfce7"}],
                       "threshold":{"line":{"color":"#1a1a2e","width":3},"value":50}},
                number={"suffix":"%","font":{"size":36,"color":"#1a1a2e","family":"Inter"}},
            ))
            fig_live.update_layout(height=300, paper_bgcolor="white",
                                   font=dict(family="Inter"), margin=dict(l=20,r=20,t=60,b=20))
            st.plotly_chart(fig_live, use_container_width=True)
        with gc2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='background:#f0fdf4;border-radius:12px;padding:20px;margin-bottom:12px;border:1px solid #bbf7d0;text-align:center;'>
                <div style='font-size:0.7rem;font-weight:700;color:#16a34a;text-transform:uppercase;'>{batting_team}</div>
                <div style='font-size:2rem;font-weight:700;color:#15803d;margin-top:6px;'>{round(adjusted_prob*100,1)}%</div>
            </div>
            <div style='background:#fef2f2;border-radius:12px;padding:20px;border:1px solid #fecaca;text-align:center;'>
                <div style='font-size:0.7rem;font-weight:700;color:#dc2626;text-transform:uppercase;'>{bowling_team}</div>
                <div style='font-size:2rem;font-weight:700;color:#b91c1c;margin-top:6px;'>{round(defend_prob*100,1)}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Player Influence</div>', unsafe_allow_html=True)
        pt1, pt2 = st.columns(2)
        with pt1:
            st.markdown("**Remaining Batters**")
            bat_inf = [{"Player":p,"Strike Rate":round(bat_lookup.get(p,100),1),
                        "Status":"At Crease" if p in [batter1,batter2] else "Yet to bat"}
                       for p in remaining_batters]
            if bat_inf:
                st.dataframe(pd.DataFrame(bat_inf), use_container_width=True, hide_index=True)
            if dismissed:
                st.markdown(f"**Dismissed:** {', '.join(dismissed)}")
        with pt2:
            st.markdown("**Bowling Quota**")
            bowl_inf = [{"Bowler":p,"Economy":round(bowl_lookup.get(p,8),1),
                         "Overs Used":bowl_overs.get(p,0),"Overs Left":round(4-bowl_overs.get(p,0),1),
                         "Status":"Bowling" if p==bowler_now else ""}
                        for p in bowl_xi]
            if bowl_inf:
                st.dataframe(pd.DataFrame(bowl_inf), use_container_width=True, hide_index=True)