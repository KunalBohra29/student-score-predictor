# Student Score Predictor

A machine learning project that predicts a student's final exam score based on study habits and academic history. Built using **Linear Regression** and **Random Forest Regression** with automatic best-model selection.

---
# Motivation

I built this project to understand how different student habits like study time, sleep, and attendance affect academic performance. 

Initially, I wanted to create a simple model, but later I experimented with multiple algorithms to see which gives better predictions.

# Project Structure

```
student-score-predictor/
├── data/
│   ├── generate_dataset.py   # Generates synthetic student dataset
│   └── students.csv          # Generated dataset (created at runtime)
├── models/
│   └── best_model.pkl        # Saved trained model (created at runtime)
├── src/
│   ├── eda.py                # Exploratory Data Analysis
│   ├── train.py              # Model training & evaluation
│   └── predict.py            # Interactive CLI prediction tool
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/KunalBohra29/student-score-predictor.git
cd student-score-predictor
```

# 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

All commands must be run from the **project root directory**.

### Step 1 — Generate the Dataset

```bash
python data/generate_dataset.py
```

This creates `data/students.csv` with 500 synthetic student records.

### Step 2 — (Optional) Run Exploratory Data Analysis

```bash
python src/eda.py
```

Prints summary statistics and feature correlations with the target variable.

### Step 3 — Train the Model

```bash
python src/train.py
```

Trains both Linear Regression and Random Forest models, prints evaluation metrics (MAE, RMSE, R²), and saves the best-performing model to `models/best_model.pkl`.

### Step 4 — Predict Scores (Interactive CLI)

```bash
python src/predict.py
```

# Features Used

| Feature | Description | Range |
|---|---|---|
| `study_hours` | Daily study hours | 1 – 10 |
| `attendance_pct` | Class attendance percentage | 50 – 100 |
| `prev_score` | Previous exam score | 0 – 100 |
| `sleep_hours` | Daily sleep hours | 4 – 10 |
| `assignments_done` | Assignments completed (out of 10) | 0 – 10 |

**Target:** `final_score` (0 – 100)

# Results

- Linear Regression R²: 0.82  
- Random Forest R²: 0.91  

Random Forest performed better and was selected as the final model.

# ML Concepts Covered

- Supervised Learning (Regression)
- Train/Test Split
- Feature Scaling (StandardScaler)
- Linear Regression
- Random Forest Regression
- Model Evaluation: MAE, RMSE, R²
- Model Persistence (pickle)

---

# Tech Stack
- Python 3.8+
- scikit-learn
- pandas
- numpy

# Notes
- The dataset is synthetically generated with a realistic score formula and added noise.
- No internet connection is required to run this project.
- The project is fully executable via the command line with no GUI required.
