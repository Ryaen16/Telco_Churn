import pandas as pd
import numpy as np
import streamlit as st
import pickle
import joblib


model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")


#with open('xgb_model.pkl', 'rb') as file:
#    model = pickle.load(file)

#with open('scaler.pkl', 'rb') as file:
#    scaler = pickle.load(file)

st.title("📞 Telco Customer Churn Prediction")
st.info("Predict whether a customer will churn based on their details")


gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=MonthlyCharges, value=MonthlyCharges)


input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}])

with st.expander("📋 Customer Input"):
    st.dataframe(input_data)

input_encoded = pd.get_dummies(
    input_data)
input_encoded = input_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0)

with st.expander("📋 Customer Input Encoded"):
    st.dataframe(input_encoded)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

with st.expander("📋 Customer Input Encoded After Scaling"):
    st.dataframe(input_encoded)

#prediction = model.predict(input_encoded)[0]
#probability = model.predict_proba(input_encoded)[0][1]

if st.button("Predict Churn"):



    prediction = model.predict(input_encoded)
    prediction_proba = model.predict_proba(input_encoded)



    st.subheader("🔮 Prediction Result")

    labels = ["No Churn", "Churn"]

    st.success(f"Prediction: {labels[prediction[0]]}")

    st.write("No churn Probability : {}".format(prediction_proba[0][0]))

    st.write("Churn Probability : {}".format(prediction_proba[0][1]))