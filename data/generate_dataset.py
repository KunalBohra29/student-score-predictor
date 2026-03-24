import numpy as np
import pandas as pd

np.random.seed(42)
N = 500

study_hours     = np.random.uniform(1, 10, N)
attendance_pct  = np.random.uniform(50, 100, N)
prev_score      = np.random.uniform(40, 100, N)
sleep_hours     = np.random.uniform(4, 10, N)
assignments_done = np.random.randint(0, 11, N)          # out of 10

# Target: weighted combination + noise
score = (
    4.5  * study_hours
    + 0.3  * attendance_pct
    + 0.25 * prev_score
    + 1.2  * sleep_hours
    + 1.8  * assignments_done
    + np.random.normal(0, 3, N)
)
score = np.clip(score, 0, 100)

df = pd.DataFrame({
    "study_hours":      np.round(study_hours, 2),
    "attendance_pct":   np.round(attendance_pct, 2),
    "prev_score":       np.round(prev_score, 2),
    "sleep_hours":      np.round(sleep_hours, 2),
    "assignments_done": assignments_done,
    "final_score":      np.round(score, 2),
})

df.to_csv("data/students.csv", index=False)
print(f"Dataset saved → data/students.csv  ({N} rows)")
