"""
ml_model.py
-----------
Trains a Random Forest Classifier to predict the IPL match winner
based on pre-match information: teams, venue, toss, and season.

Run this script to retrain and save the model:
    python src/ml_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_PATH = os.path.join(BASE_DIR, "data", "processed", "matches_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "ipl_win_predictor.joblib")


# ─── Load & prepare data ─────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    matches = pd.read_csv(path)
    df = matches[matches["result"].isin(["runs", "wickets"])].copy()
    df = df[["season", "team1", "team2", "toss_winner",
             "toss_decision", "venue", "winner"]].dropna()
    return df


# ─── Build pipeline ──────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    categorical = ["team1", "team2", "toss_winner", "toss_decision", "venue"]
    numeric = ["season"]

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )),
    ])
    return pipeline


# ─── Train & evaluate ────────────────────────────────────────────────────────

def train(df: pd.DataFrame) -> Pipeline:
    X = df.drop("winner", axis=1)
    y = df["winner"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return pipeline


# ─── Save model ──────────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved to: {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df = load_data(MATCHES_PATH)
    print(f"Dataset: {df.shape[0]} matches, {df['winner'].nunique()} teams\n")

    print("Training model...")
    pipeline = train(df)

    save_model(pipeline, MODEL_PATH)
