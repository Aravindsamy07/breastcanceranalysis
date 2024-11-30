import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("breast_cancer_ann_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load feature names from sklearn
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
feature_names = data.feature_names

# Streamlit App
st.title("Breast Cancer Prediction App")
st.write("""
This app predicts whether a breast cancer tumor is **Malignant** or **Benign** based on user-provided inputs.
""")

# Sidebar for user input
st.sidebar.header("Input Features")
user_inputs = {}
for feature in feature_names:
    user_inputs[feature] = st.sidebar.number_input(f"{feature}", value=float(data.frame[feature].mean()))

# Convert user inputs to DataFrame
input_df = pd.DataFrame([user_inputs])

# Preprocess the input
input_scaled = scaler.transform(input_df)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Display results
    st.write(f"### Prediction: {'Malignant' if prediction[0] == 0 else 'Benign'}")
    st.write(f"### Confidence: {prediction_proba[0][prediction[0]]:.2f}")
