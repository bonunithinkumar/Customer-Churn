import streamlit as st
import pickle
import pandas as pd

# ── Load model and encoders ──────────────────────────────────────────
with open('rfc_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

model         = model_data['model']
feature_names = model_data['feature_names']

# ── Page config ──────────────────────────────────────────────────────
st.title("📡 Customer Churn Predictor")
st.write("Fill in the customer details below to predict churn.")

# ── Input fields ─────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    gender          = st.selectbox("Gender",          ["Male", "Female"])
    SeniorCitizen   = st.selectbox("Senior Citizen",  [0, 1])
    Partner         = st.selectbox("Partner",         ["Yes", "No"])
    Dependents      = st.selectbox("Dependents",      ["Yes", "No"])
    tenure          = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    PhoneService    = st.selectbox("Phone Service",   ["Yes", "No"])
    MultipleLines   = st.selectbox("Multiple Lines",  ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service",["DSL", "Fiber optic", "No"])
    OnlineSecurity  = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup    = st.selectbox("Online Backup",   ["Yes", "No", "No internet service"])

with col2:
    DeviceProtection = st.selectbox("Device Protection",["Yes", "No", "No internet service"])
    TechSupport      = st.selectbox("Tech Support",     ["Yes", "No", "No internet service"])
    StreamingTV      = st.selectbox("Streaming TV",     ["Yes", "No", "No internet service"])
    StreamingMovies  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract         = st.selectbox("Contract",         ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing",["Yes", "No"])
    PaymentMethod    = st.selectbox("Payment Method",   [
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges   = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    TotalCharges     = st.number_input("Total Charges ($)",   min_value=0.0, value=500.0)

# ── Predict button ───────────────────────────────────────────────────
if st.button("🔍 Predict Churn"):

    # Build a dataframe from inputs
    input_dict = {
        'gender': gender, 'SeniorCitizen': SeniorCitizen,
        'Partner': Partner, 'Dependents': Dependents,
        'tenure': tenure, 'PhoneService': PhoneService,
        'MultipleLines': MultipleLines, 'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport,
        'StreamingTV': StreamingTV, 'StreamingMovies': StreamingMovies,
        'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod, 'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    input_df = pd.DataFrame([input_dict])

    # Apply label encoding using the saved encoders
    for col, enc in encoders.items():
        if col in input_df.columns:
            input_df[col] = enc.transform(input_df[col])

    # Reorder columns to match training data
    input_df = input_df[feature_names]

    # Make prediction
    prediction    = model.predict(input_df)[0]
    probability   = model.predict_proba(input_df)[0][1]

    # Show result
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN  (Probability: {probability:.1%})")
    else:
        st.success(f"✅ This customer is NOT likely to churn  (Probability: {probability:.1%})")