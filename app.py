import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('churn_model.pkl', 'rb'))

st.title("Customer Churn Prediction")

st.write("Enter customer details:")

# Inputs
tenure = st.slider("Tenure Months", 0, 72)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")
cltv = st.number_input("CLTV")

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
tech_support = st.selectbox("Tech Support", ["No", "Yes"])
online_security = st.selectbox("Online Security", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

# Convert inputs to numeric (IMPORTANT)
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
binary_map = {"No": 0, "Yes": 1}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}

# Prediction
if st.button("Predict"):
    data = np.array([[
        tenure,
        monthly_charges,
        contract_map[contract],
        total_charges,
        cltv,
        binary_map[tech_support],
        binary_map[online_security],
        payment_map[payment_method],
        binary_map[dependents]
    ]])

    prediction = model.predict(data)
    prob = model.predict_proba(data)

    if prediction[0] == 1:
        st.error(f"Customer likely to churn ({prob[0][1]*100:.2f}%)")
    else:
        st.success(f"Customer likely to stay ({prob[0][0]*100:.2f}%)")
    st.write("### Key Factors Affecting Churn:")
    st.write("- Short tenure increases churn risk")
    st.write("- Month-to-month contracts are riskier")
    st.write("- Lack of support/security increases churn")