import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import math
# Load the trained GradientBoostingRegressor model
model_gb = joblib.load('gb_regressor.pkl')

# Function to preprocess input data


def preprocess_input(age, income, credit_score, ec, num_existing_loans, loan_amount, loan_tenure, ltv_ratio):
    input_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Credit Score': [credit_score],
        'Number of Existing Loans': [num_existing_loans],
        'Loan Amount': [loan_amount],
        'Loan Tenure': [loan_tenure],
        'Existing Customer': [ec],
        'LTV Ratio': [ltv_ratio]
    })
    return input_data


# Streamlit app
st.title('Credit Assessment App')

col1, col2 = st.columns(2)

# Input fields for user data in the first column
with col1:
    age = st.number_input('Age', min_value=18, max_value=100, step=1, value=31)
    income = st.number_input('Income', min_value=0, step=1000, value=36000)
    credit_score = st.number_input(
        'Credit Score', min_value=0, step=1, value=604)
    choice = st.radio("Existing Customer", ("Yes", "No"), index=1)

# Input fields for user data in the second column
with col2:
    num_existing_loans = st.number_input(
        'Number of Existing Loans', min_value=0, step=1, value=5)
    loan_amount = st.number_input(
        'Loan Amount', min_value=0, step=1000, value=109373)
    loan_tenure = st.number_input('Loan Tenure', min_value=1, step=1, value=221)
    ltv_ratio = st.number_input(
        'LTV Ratio', min_value=0.0, step=0.01, value=90.943430)
# Convert the selected option to a numerical value
if choice == "Yes":
    ec = 1
else:
    ec = 0

prediction = 0
# Predict button
if st.button('Predict'):
    # Preprocess input data
    input_data = preprocess_input(
        age, income, credit_score, ec, num_existing_loans, loan_amount, loan_tenure, ltv_ratio)

    # Make prediction using the loaded model
    prediction = model_gb.predict(input_data)
    st.caption(f'Predicted Profile Score: {round(prediction[0],2)}')
    threshold = 70
    if prediction >= threshold:
        st.success("Bank should give credit.")
    else:
        st.error("Bank should not give credit.")


