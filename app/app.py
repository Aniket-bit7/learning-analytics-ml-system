import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------
# Page Config (MUST BE FIRST STREAMLIT COMMAND)
# ------------------------------------------------
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ------------------------------------------------
# Custom Styling
# ------------------------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        h1, h2, h3 {
            color: #4CAF50;
        }
        .stMetric {
            background-color: #1E1E1E;
            padding: 15px;
            border-radius: 12px;
        }
        .stDataFrame {
            border-radius: 10px;
        }
        .css-1d391kg {
            background-color: #111827;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models")

# ------------------------------------------------
# Load Models
# ------------------------------------------------
try:
    classifier = joblib.load(os.path.join(MODEL_PATH, "classifier.joblib"))
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.joblib"))
    kmeans = joblib.load(os.path.join(MODEL_PATH, "cluster_model.joblib"))
    scaler_cluster = joblib.load(os.path.join(MODEL_PATH, "cluster_scaler.joblib"))

    feature_columns = joblib.load(os.path.join(MODEL_PATH, "feature_columns.joblib"))
    column_means = joblib.load(os.path.join(MODEL_PATH, "column_means.joblib"))
    time_spent_bounds = joblib.load(os.path.join(MODEL_PATH, "time_spent_bounds.joblib"))

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ------------------------------------------------
# Header
# ------------------------------------------------
st.markdown("""
# 🎓 Student Performance Intelligence Dashboard  
### AI-Powered Academic Performance Prediction System
Upload student data to generate predictions, clusters, and smart recommendations.
""")

st.markdown("---")

# ------------------------------------------------
# File Upload
# ------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    required_columns = [
        "Quiz1", "Quiz2", "Quiz3",
        "Time_Spent", "Assignments", "Attendance"
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.stop()

    # ------------------------------------------------
    # Data Preview
    # ------------------------------------------------
    st.markdown("## 📄 Uploaded Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ------------------------------------------------
    # Preprocessing
    # ------------------------------------------------
    df.fillna(column_means, inplace=True)

    df["Time_Spent"] = df["Time_Spent"].clip(
        time_spent_bounds["lower"],
        time_spent_bounds["upper"]
    )
    df["Attendance"] = df["Attendance"].clip(0, 100)

    # ------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------
    df["Total_Quiz_Score"] = df["Quiz1"] + df["Quiz2"] + df["Quiz3"]
    df["Average_Quiz_Score"] = df["Total_Quiz_Score"] / 3
    df["Quiz_Std"] = df[["Quiz1", "Quiz2", "Quiz3"]].std(axis=1)

    df["Engagement_Index"] = (
        df["Time_Spent"] * 0.5 +
        df["Assignments"] * 0.3 +
        df["Attendance"] * 0.2
    )

    df["Quiz_Percentage"] = (df["Total_Quiz_Score"] / 300) * 100
    df["Effort_Performance_Ratio"] = (
        df["Total_Quiz_Score"] / (df["Time_Spent"] + 1)
    )

    # ------------------------------------------------
    # Classification
    # ------------------------------------------------
    X = df[feature_columns]
    X_scaled = scaler.transform(X)

    df["Prediction"] = classifier.predict(X_scaled)
    df["Prediction_Label"] = df["Prediction"].map({0: "Fail", 1: "Pass"})

    # ------------------------------------------------
    # Clustering
    # ------------------------------------------------
    cluster_input = df[[
        "Average_Quiz_Score",
        "Engagement_Index",
        "Attendance"
    ]]

    cluster_scaled = scaler_cluster.transform(cluster_input)
    df["Cluster"] = kmeans.predict(cluster_scaled)

    cluster_map = {
        0: "At Risk",
        1: "High Performer",
        2: "Struggling but Engaged"
    }

    df["Cluster_Label"] = df["Cluster"].map(cluster_map)

    # ------------------------------------------------
    # Recommendation Logic
    # ------------------------------------------------
    def generate_recommendation(row):
        if row["Prediction_Label"] == "Fail":
            if row["Cluster_Label"] == "At Risk":
                return "Immediate academic intervention required"
            elif row["Cluster_Label"] == "Struggling but Engaged":
                return "Provide tutoring support"
            else:
                return "Investigate performance inconsistency"
        else:
            if row["Cluster_Label"] == "At Risk":
                return "Monitor engagement closely"
            elif row["Cluster_Label"] == "Struggling but Engaged":
                return "Strengthen weak concepts"
            else:
                return "Encourage advanced learning"

    df["Recommendation"] = df.apply(generate_recommendation, axis=1)

    # ------------------------------------------------
    # Dashboard Metrics
    # ------------------------------------------------
    st.markdown("## 📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("👥 Total Students", len(df))

    with col2:
        st.metric("✅ Predicted Pass", (df["Prediction_Label"] == "Pass").sum())

    with col3:
        st.metric("❌ Predicted Fail", (df["Prediction_Label"] == "Fail").sum())

    st.markdown("---")

    # ------------------------------------------------
    # Charts
    # ------------------------------------------------
    st.markdown("## 📈 Insights")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Prediction Distribution")
        st.bar_chart(df["Prediction_Label"].value_counts())

    with colB:
        st.subheader("Cluster Distribution")
        st.bar_chart(df["Cluster_Label"].value_counts())

    st.markdown("---")

    # ------------------------------------------------
    # Final Results
    # ------------------------------------------------
    st.markdown("## 📑 Final Results")
    st.dataframe(df, use_container_width=True)

    # Download Button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="student_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("📌 Please upload a CSV file to begin.")

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.markdown("Developed for AI & ML Project | Student Performance Analytics 🚀")