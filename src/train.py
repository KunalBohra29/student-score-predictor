"""
train.py
Trains Linear Regression and Random Forest models on the student dataset,
evaluates them, and saves the best model to models/best_model.pkl
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math


FEATURES = [
    "study_hours",
    "attendance_pct",
    "prev_score",
    "sleep_hours",
    "assignments_done"
]
TARGET   = "final_score"
DATA_PATH  = "data/students.csv"
MODEL_PATH = "models/best_model.pkl"


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'.\n"
            "Run:  python data/generate_dataset.py"
        )
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES]
    y  = df[TARGET]
    return X, y


def evaluate(name, model, X_test, y_test, scaler=None):
    X_input = scaler.transform(X_test) if scaler else X_test
    preds   = model.predict(X_input)
    mae     = mean_absolute_error(y_test, preds)
    rmse    = math.sqrt(mean_squared_error(y_test, preds))
    r2      = r2_score(y_test, preds)
    print(f"\n{'─'*40}")
    print(f"  Model : {name}")
    print(f"  MAE   : {mae:.2f}")
    print(f"  RMSE  : {rmse:.2f}")
    print(f"  R²    : {r2:.4f}")
    return r2, preds


def main():
    print("Loading data …")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Linear Regression (needs scaling) ──────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    r2_lr, _ = evaluate("Linear Regression", lr, X_test, y_test, scaler)

    # ── Random Forest ───────────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    r2_rf, _ = evaluate("Random Forest", rf, X_test, y_test)

    # ── Save best model ─────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    if r2_rf >= r2_lr:
        best = {"model": rf, "scaler": None, "name": "Random Forest"}
    else:
        best = {"model": lr, "scaler": scaler, "name": "Linear Regression"}

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best, f)

    print(f"\n{'─'*40}")
    print(f"  Best model : {best['name']}  (R² = {max(r2_lr, r2_rf):.4f})")
    print(f"  Saved to   : {MODEL_PATH}")
    print(f"{'─'*40}\n")


if __name__ == "__main__":
    main()
