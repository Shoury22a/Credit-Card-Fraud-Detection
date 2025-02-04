import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model (ensure you have exported it from your Jupyter Notebook)
model = joblib.load("fraud_detection_model.pkl")

st.title("Credit Card Fraud Detection App")
st.write("This app predicts whether a transaction is fraudulent or not.")

# Define the feature names based on the dataset
feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

def user_input_features():
    input_data = {}
    for col in feature_names:
        input_data[col] = st.number_input(f"{col}", value=0.0, step=0.01)
    return pd.DataFrame([input_data])

input_df = user_input_features()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    result = "Fraudulent Transaction! 🚨" if prediction == 1 else "Legit Transaction ✅"
    st.write(result)

st.write("Ensure that the model is trained and saved using the uploaded Jupyter Notebook.")