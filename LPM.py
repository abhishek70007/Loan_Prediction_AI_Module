import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the background image
background_image = "websitebg.jpg"

# Set the background image as the app's background
st.markdown(
    f"""
    <style>
        body {{
            background-image: url("{background_image}");
            background-size: cover;
            color: #333;
        }}
        .stButton>button {{
            background-color: #8A2BE2;  /* Violet color */
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .stButton>button:hover {{
            background-color: #800080;  /* Darker violet on hover */
        }}
        .stSuccess {{
            color: #4CAF50;
            font-size: 18px;
        }}
        .stError {{
            color: #FF6347;
            font-size: 18px;
        }}
        .stSubheader {{
            color: #8A2BE2;  /* Violet color */
            font-size: 20px;
            margin-top: 20px;
        }}
        .stPieChart {{
            max-width: 500px;
            margin-top: 20px;
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
st.header('  ')
st.header('  ')
st.header('  ')
# User input form
st.markdown(
    """
    <p style='font-weight: bold; color: #f5caa2;font-family: Arial, sans-serif; font-size: 32px;'>Enter Customer's Information:</p>
    """,
    unsafe_allow_html=True
)
cust_name = st.text_input("Enter Customer's Full Name:", key="cust_name")
pan_id = st.text_input("Enter Customer's Pan Card Details:", key="panid")
ac_no = st.text_input("Enter Customer's Account Number:", 0, key="acno")
loan_ac_no = st.text_input(
    "Enter Customer's Loan Account Number:", 0, key="loanacno")
ph_no = st.text_input("Enter Customer's Phone Number:")
total_disbursed_amount = st.number_input('Total Disbursed Amount')
total_emi_amount = st.number_input('Total EMI Amount')
net_salary = st.number_input('Net Salary')
total_outstanding_balance = st.number_input('Total Outstanding Balance')
current_cibil_score = st.number_input('Current CIBIL Score')
overdue_principal = st.number_input('Overdue Principal')
overdue_interest = st.number_input('Overdue Interest')
npa_classification = st.selectbox('NPA Classification [Account Type]', [
                                  "Standard", "Non-Performing Asset(NPA)"])
is_writeoff = st.selectbox(
    'Is the Customer had any WriteOff Accounts with other/our Bank/s?', ["Yes", "No"])
is_ots = st.selectbox(
    "Is the Customer had One Time Settlement(OTS) with other/our Bank/s?", ["Yes", "No"])

# writeoff
if is_writeoff == "Yes":
    writeoff_val = True
else:
    writeoff_val = False

# ots
if is_ots == "Yes":
    ots_val = True
else:
    ots_val = False

# Calculate TakeHome based on NetSalary
take_home = 0.30 * net_salary

# Calculate CIBIL Eligibility
cibil_eligibility = 1 if current_cibil_score >= 700 else 0
income_eligibility = 1 if (net_salary - total_emi_amount) >= take_home else 0

if npa_classification == "Standard":
    npa_val = 0
else:
    npa_val = 1

# Prepare data for prediction
new_data = pd.DataFrame({
    'TotalDisbursedAmount': [total_disbursed_amount],
    'NetSalary': [net_salary],
    'TakeHome': [take_home],
    'TotalEMIAmount': [total_emi_amount],
    'IncomeEligibility': [income_eligibility],
    'TotalOutstandingBalance': [total_outstanding_balance],
    'CurrentCIBILScore': [current_cibil_score],
    'CIBILEligibility': [cibil_eligibility],
    'OverduePrincipal': [overdue_principal],
    'OverdueInterest': [overdue_interest],
    'NPAClassification': [npa_val]
})

new_data_matrix = np.array(new_data).reshape(1, -1)
prediction = model.predict(xgb.DMatrix(new_data_matrix))[0]

# Prediction button
if st.button('Predict Loan Approval', key='predict_button', help="Click to predict loan approval"):
    # Make predictions
    new_data_matrix = np.array(new_data).reshape(1, -1)
    prediction = model.predict(xgb.DMatrix(new_data_matrix))[0]

    # Display the prediction
    st.header('Loan Approval Prediction:')
    if prediction == 1:
        st.success('Loan Application Should be Approved!')
    else:
        st.error('Loan Application Should be Denied.')

    # Optional: Display additional information or insights based on the prediction
    remark = ""
    point_ind = 1
    bad_loan = 0
    dynString = ""
    if prediction == 0:
        if income_eligibility == 0:
            dynamic_string = f"Net Income is Low as per Eligibility.\n"
            remark += dynamic_string
            point_ind = point_ind + 1
            bad_loan += 0.88
        if cibil_eligibility == 0:
            dynamic_string = f"CIBIL Score is Low as per Eligibility (<= 700).\n"
            remark += dynamic_string
            point_ind = point_ind + 1
            bad_loan += 0.64
        if overdue_interest > 0 or overdue_principal > 0 or npa_val == 1:
            if overdue_interest > 0 and overdue_principal == 0 and npa_val == 0:
                dynString = f"Overdue Interest of ₹{overdue_interest}"
                bad_loan += 0.64
            elif overdue_interest > 0 and overdue_principal == 0 and npa_val == 1:
                dynString = f"Overdue Interest of ₹{overdue_interest} and the Account is classified as NPA"
                bad_loan += 0.64 + 0.79
            elif overdue_principal > 0 and overdue_interest == 0 and npa_val == 0:
                dynString = f"Overdue Principal of ₹{overdue_principal}"
                bad_loan += 0.64
            elif overdue_principal > 0 and overdue_interest == 0 and npa_val == 1:
                dynString = f"Overdue Principal of ₹{overdue_principal} and the Account is classified as NPA"
                bad_loan += 0.64 + 0.79
            elif overdue_principal > 0 and overdue_interest > 0 and npa_val == 0:
                dynString = f"Overdue Principal of ₹{overdue_principal} and Overdue Interest of ₹{overdue_interest}"
                bad_loan += 0.64 + 0.64
            elif overdue_principal > 0 and overdue_interest > 0 and npa_val == 1:
                dynString = f"Overdue Principal of ₹{overdue_principal} and Overdue Interest of ₹{overdue_interest} and the Account is classified as NPA"
                bad_loan += 0.64 + 0.64 + 0.79
            elif overdue_principal == 0 and overdue_interest == 0 and npa_val == 1:
                dynString = f"no Overdue Interest and Overdue Principal but the Account is classified as NPA.Account needs to be manually checked and processed"
            dynamic_string = f"Customer has {dynString}.\n"
            remark += dynamic_string
            point_ind = point_ind + 1
        if ots_val == True:
            dynamic_string = f"Previously Customer had One Time Settlement(OTS) with other/our Bank/s.\n"
            remark += dynamic_string
            point_ind = point_ind + 1
            bad_loan += 0.5
        if writeoff_val == True:
            dynamic_string = f"Previously Customer had a Writeoff with other/our Bank/s.\n"
            remark += dynamic_string
            point_ind = point_ind + 1
            bad_loan += 0.5
        else:
            remark = "Error! in generating Remarks for the Customer."
        bad_loan_per = (bad_loan / 4.47) * 100

    # Display the percentage graph for bad_loan_per
    # st.subheader("Risk Assessment:")
    # st.subheader("Risk Assessment:")
    st.markdown(
        """
        <p style='font-weight: bold; color: #de2858; font-size: 29px;'>Risk Assessment:</p>
        """,
        unsafe_allow_html=True
    )
    st.success(f"Probability of Bad Loan: {bad_loan_per:.2f}%")

    # Display pie chart
    fig, ax = plt.subplots()
    labels = ['Risk', 'Safe']
    sizes = [bad_loan_per, 100 - bad_loan_per]
    colors = ['#FF6347', '#4CAF50']
    explode = (0.1, 0)
    textprops = {'color': '#1c1a18', 'fontsize': 12, 'weight': 'bold'}
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
           startangle=90, colors=colors, explode=explode, textprops=textprops)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    st.pyplot(fig)

    # Display remarks
    # st.subheader("Remarks:")
    st.markdown(
        """
        <p style='font-weight: bold; color: #43fa96; font-size: 29px;'>Remark:</p>
        """,
        unsafe_allow_html=True
    )

# Split remarks into a list and format each remark with line breaks
# Split remarks into a list and format each remark with line breaks
    formatted_remarks = []
    point_ind = 1
    for remark_line in remark.split('\n'):
        if remark_line:
            formatted_remarks.append(f"{point_ind}. {remark_line}")
            point_ind += 1

    # Combine formatted remarks into a single string with line breaks
    formatted_remarks_str = '<br>'.join(formatted_remarks)

    # Apply styles to the formatted remarks string
    styled_remarks = f"<p style='color:#d0f7e2; font-size:16px; font-weight:bold; font-family: Arial, sans-serif;'>{formatted_remarks_str}</p>"
    st.markdown(styled_remarks, unsafe_allow_html=True)
