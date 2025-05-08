import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    with open("xg_boost.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("📞 Call Center Prediction App")

st.markdown("Enter the details below to get the prediction from the trained XGBoost model.")

# --- Inputs ---
customer_age = st.number_input("Customer Age", min_value=10, max_value=100, value=30)

sector = st.selectbox("Sector", ["Finance", "Healthcare", "Retail", "Telecom"])
sector_mapping = {"Finance": 0, "Healthcare": 1, "Retail": 2, "Telecom": 3}
sector_encoded = sector_mapping[sector]

operator_id = st.number_input("Operator ID", min_value=1, max_value=999, value=5)
location_id = st.number_input("Location ID", min_value=1, max_value=999, value=101)

call_duration = st.slider("Call Duration (mins)", min_value=0, max_value=60, value=5)
satisfaction_score = st.slider("Satisfaction Score (0.0 to 1.0)", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

# --- Construct input for model ---
input_data = np.array([[customer_age, sector_encoded, operator_id, location_id, call_duration, satisfaction_score]])

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"✅ Model Prediction: {prediction[0]}")
