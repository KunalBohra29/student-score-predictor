import numpy as np
import pandas as pd
N = 500  # number of students 
# generating features
study_hours = np.random.uniform(1, 10, N)          # hours studied per day
attendance = np.random.uniform(50, 100, N)         # attendance percentage
previous_score = np.random.uniform(40, 100, N)     # last exam score
sleep_hours = np.random.uniform(4, 10, N)          # average sleep
assignments_completed = np.random.randint(0, 11, N)  # out of 10
final_score = (
    4.2 * study_hours
    + 0.35 * attendance
    + 0.3 * previous_score
    + 1.1 * sleep_hours
    + 1.5 * assignments_completed
)
noise = np.random.normal(0, 4, N)
final_score = final_score + noise
final_score -= np.where(attendance < 60, 5, 0)
final_score = np.clip(final_score, 0, 100)
df = pd.DataFrame({
    "study_hours": np.round(study_hours, 2),
    "attendance": np.round(attendance, 2),
    "previous_score": np.round(previous_score, 2),
    "sleep_hours": np.round(sleep_hours, 2),
    "assignments_completed": assignments_completed,
    "final_score": np.round(final_score, 2),
})
output_path = "data/students.csv"
df.to_csv(output_path, index=False)

print(f"Dataset saved at {output_path} with {N} entries")
