# Student Performance Intelligence System

<p align="center">
Live Demo: <a href="https://learning-analytics-ml-system-ludv9tzwzk5x2t4wen7a8e.streamlit.app/">Click here to use the deployed app</a>
<br><br>
<img src="https://drive.google.com/file/d/1-VMlE0VcE3eq4YBsgHHa6Wj3iGnyVZUp/view?usp=drive_link" alt="image">
</p>

An end-to-end Machine Learning system that predicts student academic outcomes, segments behavioral patterns using clustering, and generates actionable academic recommendations. The system is deployed through an interactive Streamlit dashboard.

---

## Project Objective

The goal of this project is to build a Learning Analytics ML System that:

- Predicts whether a student will Pass or Fail
- Handles class imbalance properly
- Segments students into meaningful behavioral clusters
- Generates personalized intervention recommendations
- Provides a clean, deployable dashboard interface

This project goes beyond simple prediction and builds a structured decision-support system.

---

## Machine Learning Pipeline

### 1. Data Preprocessing

- Handled missing values using mean imputation based on training statistics
- Applied IQR-based outlier clipping on `Time_Spent`
- Capped logical inconsistencies (e.g., `Attendance <= 100`)
- Encoded target variable (`Fail = 0`, `Pass = 1`)
- Used Stratified Train-Test Split (80–20) to preserve class distribution

All preprocessing statistics (means, bounds, feature order) were saved to ensure consistent inference in deployment.

### 2. Feature Engineering

Created meaningful behavioral and academic features:

- `Total_Quiz_Score`
- `Average_Quiz_Score`
- `Quiz_Std` — performance consistency
- `Engagement_Index`
- `Quiz_Percentage`
- `Effort_Performance_Ratio`

These features capture academic strength, effort, consistency, and engagement patterns.

### 3. Model Training (Supervised Learning)

- Baseline: Logistic Regression
- Improved: `class_weight="balanced"` to address class imbalance (approximately 78% Fail, 22% Pass)

### Evaluation Metrics

Instead of relying on accuracy alone, the model was evaluated using:

- Confusion Matrix
- Precision (Pass class)
- Recall (Pass class)
- F1-score
- ROC-AUC

**Final ROC-AUC: approximately 0.83 — Very Good Model Performance.**
Recall for the minority class improved significantly after balancing.

---

## Behavioral Segmentation (KMeans Clustering)

Clustering was performed on the following features:

- `Average_Quiz_Score`
- `Engagement_Index`
- `Attendance`

Three behavioral groups were identified:

| Cluster | Label |
|---|---|
| 0 | At Risk |
| 2 | Struggling but Engaged |
| 1 | High Performer |

Clustering provides deeper insights beyond simple Pass/Fail classification.

---

## Recommendation Engine

Final recommendations are generated using a combination of model prediction and cluster membership:

| Prediction | Cluster | Recommendation |
|---|---|---|
| Fail | 0 — At Risk | Immediate academic intervention |
| Fail | 2 — Struggling but Engaged | Provide tutoring support |
| Pass | 1 — High Performer | Encourage advanced learning |

This makes the system practical and actionable for educators and academic advisors.

---

## Streamlit Dashboard

The deployed application allows users to:

- Upload a CSV dataset
- Automatically handle missing values and outliers
- Generate predictions instantly
- View cluster distribution
- See personalized recommendations
- Analyze results via dashboard metrics

The preprocessing logic in the app strictly matches the training notebook to ensure consistent predictions.

---

## Project Structure

```
app/
  app.py

models/
  classifier.joblib
  scaler.joblib
  cluster_model.joblib
  cluster_scaler.joblib
  column_means.joblib
  time_spent_bounds.joblib
  feature_columns.joblib

notebooks/
  milestone1_training.ipynb

data/
  students_data.csv
requirements.txt
```

---

## Run Locally

```bash
pip install -r app/requirements.txt
cd app
streamlit run app.py
```

---

## Deployment

The project can be deployed using Streamlit Community Cloud:

- App file: `app/app.py`
- Requirements file: `app/requirements.txt`
