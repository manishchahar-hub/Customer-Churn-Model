import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model and Threshold
# -----------------------------
model = joblib.load("churn_model_xgb_v1.pkl")
threshold = joblib.load("churn_threshold.pkl")


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict the probability of churn.")


# -----------------------------
# User Inputs
# -----------------------------

CreditScore = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

Geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

Tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

Balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

NumOfProducts = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

HasCrCard = st.selectbox(
    "Has Credit Card?",
    [0, 1]
)

IsActiveMember = st.selectbox(
    "Is Active Member?",
    [0, 1]
)

EstimatedSalary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Churn"):

    input_data = pd.DataFrame({
        "CreditScore": [CreditScore],
        "Geography": [Geography],
        "Gender": [Gender],
        "Age": [Age],
        "Tenure": [Tenure],
        "Balance": [Balance],
        "NumOfProducts": [NumOfProducts],
        "HasCrCard": [HasCrCard],
        "IsActiveMember": [IsActiveMember],
        "EstimatedSalary": [EstimatedSalary]
    })

    # Get churn probability
    churn_probability = model.predict_proba(input_data)[0][1]

    # Apply optimized threshold
    prediction = int(churn_probability >= threshold)

    st.subheader("Prediction Result")

    st.write(
        f"**Churn Probability:** {churn_probability:.2%}"
    )

    st.write(
        f"**Decision Threshold:** {threshold:.4f}"
    )

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")