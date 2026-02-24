# Learning Analytics ML System

A machine learning-powered student performance analytics platform that predicts academic outcomes, segments learners into behavioral clusters, and generates actionable intervention recommendations. Built with Logistic Regression for classification, K-Means for clustering, and served through an interactive Streamlit dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Classification Model](#classification-model)
  - [Clustering Model](#clustering-model)
  - [Recommendation Engine](#recommendation-engine)
- [Serialized Models](#serialized-models)
- [Installation](#installation)
- [Usage](#usage)
- [Dashboard Features](#dashboard-features)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

This system addresses the challenge of early identification of at-risk students in academic settings. By analyzing quiz performance, time investment, assignment completion, and attendance patterns, it provides educators with:

- **Pass/Fail predictions** for individual students using a trained classification model.
- **Behavioral clustering** that groups students into meaningful segments (High Performer, At Risk, Struggling but Engaged).
- **Targeted recommendations** that combine prediction and cluster assignment to suggest specific interventions.

The platform is designed to operate on uploaded CSV data, making it adaptable to different cohorts and semesters without retraining.

---

## Architecture

```
                    +---------------------+
                    |   CSV Upload (UI)   |
                    +----------+----------+
                               |
                    +----------v----------+
                    |   Data Validation   |
                    |   & Preprocessing   |
                    +----------+----------+
                               |
              +----------------+----------------+
              |                                 |
   +----------v----------+          +-----------v-----------+
   | Logistic Regression |          |   K-Means Clustering  |
   |   (Pass/Fail)       |          |   (3 Segments)        |
   +----------+----------+          +-----------+-----------+
              |                                 |
              +----------------+----------------+
                               |
                    +----------v----------+
                    | Recommendation      |
                    | Engine (Rule-Based) |
                    +----------+----------+
                               |
                    +----------v----------+
                    | Streamlit Dashboard |
                    | (Metrics, Charts,   |
                    |  Download Results)  |
                    +---------------------+
```

---

## Project Structure

```
learning-analytics-ml-system/
|
|-- app/
|   +-- app.py                    # Streamlit web application (dashboard + inference)
|
|-- data/
|   +-- students_data.csv         # Training dataset (1,001 student records)
|
|-- models/
|   |-- classifier.joblib         # Trained Logistic Regression model
|   |-- scaler.joblib             # StandardScaler for classification features
|   |-- cluster_model.joblib      # Trained K-Means clustering model (k=3)
|   |-- cluster_scaler.joblib     # StandardScaler for clustering features
|   |-- feature_columns.joblib    # Ordered list of feature column names
|   |-- column_means.joblib       # Column means for missing value imputation
|   +-- time_spent_bounds.joblib  # IQR-based bounds for outlier capping
|
|-- notebooks/
|   +-- milestone1_training.ipynb # Full training pipeline (EDA, training, evaluation)
|
|-- requirements.txt              # Python dependency list
|-- .gitignore                    # Git ignore rules
+-- README.md                    # Project documentation
```

---

## Dataset

The training data (`data/students_data.csv`) contains **1,001 records** with the following schema:

| Column        | Type    | Range / Description                          |
|---------------|---------|----------------------------------------------|
| Student_ID    | String  | Unique identifier (S1000 -- S1999)           |
| Quiz1         | Integer | Score on Quiz 1 (0 -- 100)                   |
| Quiz2         | Float   | Score on Quiz 2 (0 -- 100), contains nulls   |
| Quiz3         | Integer | Score on Quiz 3 (0 -- 100)                   |
| Time_Spent    | Float   | Hours spent on coursework, contains outliers  |
| Assignments   | Integer | Number of assignments completed (0 -- 10)    |
| Attendance    | Float   | Attendance percentage (0 -- 100), contains nulls |
| Final_Result  | String  | Target variable: "Pass" or "Fail"            |

**Data characteristics:**
- The dataset exhibits class imbalance, with a significantly higher proportion of "Fail" outcomes.
- Missing values are present across `Quiz2`, `Time_Spent`, `Attendance`, and other numeric columns.
- Outliers exist in the `Time_Spent` column (values reaching 99.0).

---

## Machine Learning Pipeline

### Data Preprocessing

1. **Missing Value Imputation**: Numeric columns are filled with their respective column means, computed during training and serialized in `column_means.joblib`.
2. **Outlier Handling**: The `Time_Spent` column is capped using IQR-based bounds (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR). Bounds are stored in `time_spent_bounds.joblib`.
3. **Attendance Clamping**: Attendance values are clipped to the valid range of 0 to 100.
4. **Feature Scaling**: StandardScaler normalization is applied before model inference. Separate scalers are used for the classifier and the clustering model.

### Feature Engineering

Six derived features are computed from the raw input columns:

| Engineered Feature          | Formula                                                   |
|-----------------------------|-----------------------------------------------------------|
| Total_Quiz_Score            | Quiz1 + Quiz2 + Quiz3                                     |
| Average_Quiz_Score          | Total_Quiz_Score / 3                                      |
| Quiz_Std                    | Standard deviation across Quiz1, Quiz2, Quiz3             |
| Engagement_Index            | Time_Spent * 0.5 + Assignments * 0.3 + Attendance * 0.2  |
| Quiz_Percentage             | (Total_Quiz_Score / 300) * 100                            |
| Effort_Performance_Ratio    | Total_Quiz_Score / (Time_Spent + 1)                       |

### Classification Model

- **Algorithm**: Logistic Regression with `class_weight="balanced"` to address the class imbalance in the dataset.
- **Train/Test Split**: 80/20 stratified split with `random_state=42`.
- **Output**: Binary prediction (0 = Fail, 1 = Pass).
- **Evaluation**: The model is evaluated using confusion matrix, classification report (precision, recall, F1-score), and ROC-AUC score. The balanced variant was selected as the final model after comparing against the default (unbalanced) Logistic Regression.

### Clustering Model

- **Algorithm**: K-Means with `n_clusters=3` and `random_state=42`.
- **Input Features**: Average_Quiz_Score, Engagement_Index, Attendance.
- **Cluster Assignments**:

| Cluster ID | Label                    | Description                                             |
|------------|--------------------------|---------------------------------------------------------|
| 0          | At Risk                  | Low quiz scores, low engagement, low attendance         |
| 1          | High Performer           | Strong quiz performance, high engagement and attendance |
| 2          | Struggling but Engaged   | Moderate scores with higher effort and attendance       |

### Recommendation Engine

The recommendation system applies a rule-based decision matrix that crosses the classification prediction with cluster assignment:

| Prediction | Cluster                  | Recommendation                            |
|------------|--------------------------|-------------------------------------------|
| Fail       | At Risk                  | Immediate academic intervention required  |
| Fail       | Struggling but Engaged   | Provide tutoring support                  |
| Fail       | High Performer           | Investigate performance inconsistency     |
| Pass       | At Risk                  | Monitor engagement closely                |
| Pass       | Struggling but Engaged   | Strengthen weak concepts                  |
| Pass       | High Performer           | Encourage advanced learning               |

---

## Serialized Models

All trained artifacts are persisted using `joblib` and stored in the `models/` directory:

| File                    | Contents                                                      |
|-------------------------|---------------------------------------------------------------|
| classifier.joblib       | Logistic Regression model (balanced class weights)            |
| scaler.joblib           | StandardScaler fitted on all training features                |
| cluster_model.joblib    | K-Means model (3 clusters)                                   |
| cluster_scaler.joblib   | StandardScaler fitted on clustering input features            |
| feature_columns.joblib  | List of feature column names expected by the classifier       |
| column_means.joblib     | Mean values per numeric column for missing value imputation   |
| time_spent_bounds.joblib| Dictionary with "lower" and "upper" IQR bounds for capping   |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aniket-bit7/learning-analytics-ml-system.git
   cd learning-analytics-ml-system
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Dashboard

```bash
streamlit run app/app.py
```

The application will launch in your default browser at `http://localhost:8501`.

### Input Requirements

Upload a CSV file containing the following **required columns**:

| Column      | Type    | Description                 |
|-------------|---------|-----------------------------|
| Quiz1       | Numeric | Score on Quiz 1 (0 -- 100)  |
| Quiz2       | Numeric | Score on Quiz 2 (0 -- 100)  |
| Quiz3       | Numeric | Score on Quiz 3 (0 -- 100)  |
| Time_Spent  | Numeric | Hours spent on coursework   |
| Assignments | Numeric | Assignments completed       |
| Attendance  | Numeric | Attendance percentage       |

Missing values in these columns are handled automatically through mean imputation.

### Output

The dashboard produces a results table with the following appended columns for each student:

- `Prediction_Label` -- Pass or Fail
- `Cluster_Label` -- At Risk, High Performer, or Struggling but Engaged
- `Recommendation` -- Specific intervention guidance
- All engineered features (Total_Quiz_Score, Engagement_Index, etc.)

Results can be downloaded as a CSV file directly from the dashboard.

---

## Dashboard Features

- **Data Preview**: Displays the first rows of the uploaded dataset for verification.
- **Summary Metrics**: Shows total student count, predicted pass count, and predicted fail count.
- **Prediction Distribution Chart**: Bar chart visualizing the pass/fail split.
- **Cluster Distribution Chart**: Bar chart visualizing the student segmentation.
- **Complete Results Table**: Full dataset with all predictions, clusters, and recommendations.
- **CSV Export**: One-click download of the complete results table.

---

## Technologies Used

| Technology     | Purpose                                       |
|----------------|-----------------------------------------------|
| Python         | Core programming language                     |
| Streamlit      | Interactive web dashboard framework           |
| pandas         | Data manipulation and analysis                |
| NumPy          | Numerical computing                           |
| scikit-learn   | Machine learning (Logistic Regression, KMeans, StandardScaler) |
| joblib         | Model serialization and deserialization        |
| Matplotlib     | Visualization during training (notebook)      |
| Seaborn        | Statistical visualization during training     |

---

## License

This project is developed as part of an AI and ML academic project on Student Performance Analytics.
