import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np

# Load the background image
background_image = "websitebg.jpg"

# Set the background image as the app's background
st.markdown(
    f"""
    <style>
        body {{
            background-image: url("{background_image}");
            background-size: cover;
        }}
    </style>
    """,
    unsafe_allow_html=True
)
# Load the XGBoost model
model = xgb.Booster()
# Replace 'your_model_path.model' with the path to your saved XGBoost model
model.load_model('XgboostModel_JSCB.model')

# Streamlit UI


logo_image = st.image('JSCBlogo.png', use_column_width=True)
st.title('Loan Approval Predictor')
# User input form
st.header('Enter Customer Information:')
cust_name = st.text_input("Enter Customer First Name:" , key = "cust_name")
pan_id = st.text_input("Enter Customer's Pan Card Details:",key = "panid")
ac_no = st.text_input("Enter Customer Account Number:",0,key="acno")
loan_ac_no = st.text_input("Enter Customer Loan Account Number:",0,key="loanacno")
ph_no = st.text_input("Enter Customer Phone Number:")
total_disbursed_amount = st.number_input('Total Disbursed Amount')
total_emi_amount = st.number_input('Total EMI Amount')
net_salary = st.number_input('Net Salary')
total_outstanding_balance = st.number_input('Total Outstanding Balance')
current_cibil_score = st.number_input('Current CIBIL Score')
overdue_principal = st.number_input('Overdue Principal')
overdue_interest = st.number_input('Overdue Interest')
npa_classification = st.selectbox('NPA Classification [Account Type]', ["Standard", "Non-Performing Asset(NPA)"])

# Calculate TakeHome based on NetSalary
take_home = 0.30 * net_salary

# Calculate CIBIL Eligibility
cibil_eligibility = 1 if current_cibil_score >= 700 else 0
income_eligibility = 1 if (net_salary - total_emi_amount)>=take_home else 0

if(npa_classification == "Standard"):
    npa_val = 0
else:
    npa_val = 1

# Prepare data for prediction
new_data = pd.DataFrame({
    'TotalDisbursedAmount': [total_disbursed_amount],
    'NetSalary': [net_salary],
    'TakeHome': [take_home],
    'TotalEMIAmount': [total_emi_amount],
    'IncomeEligibility':[income_eligibility],
    'TotalOutstandingBalance': [total_outstanding_balance],
    'CurrentCIBILScore': [current_cibil_score],
    'CIBILEligibility': [cibil_eligibility],
    'OverduePrincipal': [overdue_principal],
    'OverdueInterest': [overdue_interest],
    'NPAClassification': [npa_val]
})

# Prediction button
if st.button('Predict Loan Approval'):
    # Make predictions
    new_data_matrix = np.array(new_data).reshape(1, -1)
    prediction = model.predict(xgb.DMatrix(new_data_matrix))[0]

    # Display the prediction
    st.header('Loan Approval Prediction:')
    if prediction == 1:
        st.success('Loan Approved!')
    else:
        st.error('Loan Denied.')

# Optional: Display additional information or insights based on the prediction if needed
