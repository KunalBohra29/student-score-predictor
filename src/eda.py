"""
eda.py
Exploratory Data Analysis — prints summary statistics and feature correlations.
"""

import os
import pandas as pd

DATA_PATH = "data/students.csv"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'.\n"
            "Run:  python data/generate_dataset.py"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n" + "═"*55)
    print("   📋  Exploratory Data Analysis — Student Dataset")
    print("═"*55)

    print(f"\n  Rows : {len(df)}   |   Columns : {len(df.columns)}")
    print("\n── Column Names ──────────────────────────────────────")
    for col in df.columns:
        print(f"  • {col}")

    print("\n── Summary Statistics ────────────────────────────────")
    print(df.describe().round(2).to_string())

    print("\n── Correlation with Final Score ──────────────────────")
    corr = df.corr()["final_score"].drop("final_score").sort_values(ascending=False)
    for feat, val in corr.items():
        bar = "█" * int(abs(val) * 20)
        print(f"  {feat:<20} {val:+.3f}  {bar}")

    print("\n── Missing Values ────────────────────────────────────")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values found ✓")
    else:
        print(missing[missing > 0])

    print("\n" + "═"*55 + "\n")


if __name__ == "__main__":
    main()
