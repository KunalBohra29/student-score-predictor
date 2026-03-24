"""
predict.py
Interactive CLI tool — takes student inputs and predicts their final score.
"""

import os
import pickle

MODEL_PATH = "models/best_model.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'.\n"
            "Run:  python src/train.py"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def get_float(prompt, lo, hi):
    while True:
        try:
            val = float(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"  ✖  Please enter a value between {lo} and {hi}.")
        except ValueError:
            print("  ✖  Invalid input — please enter a number.")


def get_int(prompt, lo, hi):
    while True:
        try:
            val = int(input(prompt))
            if lo <= val <= hi:
                return val
            print(f"  ✖  Please enter a whole number between {lo} and {hi}.")
        except ValueError:
            print("  ✖  Invalid input — please enter a whole number.")


def grade(score):
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"


def main():
    print("\n" + "═"*50)
    print("   🎓  Student Score Predictor")
    print("═"*50)

    bundle = load_model()
    model  = bundle["model"]
    scaler = bundle["scaler"]
    print(f"   Model loaded: {bundle['name']}\n")

    while True:
        print("Enter the student's details below:\n")
        study_hours      = get_float("  Study hours per day       (1–10)  : ", 1, 10)
        attendance_pct   = get_float("  Attendance percentage    (50–100) : ", 50, 100)
        prev_score       = get_float("  Previous exam score      (0–100)  : ", 0, 100)
        sleep_hours      = get_float("  Sleep hours per day       (4–10)  : ", 4, 10)
        assignments_done = get_int  ("  Assignments completed     (0–10)  : ", 0, 10)

        features = [[study_hours, attendance_pct, prev_score,
                     sleep_hours, assignments_done]]

        if scaler:
            features = scaler.transform(features)

        predicted = model.predict(features)[0]
        predicted = max(0, min(100, predicted))

        print("\n" + "─"*50)
        print(f"  📊  Predicted Final Score : {predicted:.1f} / 100")
        print(f"  🏅  Grade                 : {grade(predicted)}")
        print("─"*50)

        again = input("\nPredict for another student? (y/n): ").strip().lower()
        if again != "y":
            print("\nGoodbye! 👋\n")
            break
        print()


if __name__ == "__main__":
    main()
