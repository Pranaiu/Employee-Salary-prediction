import streamlit as st
import pandas as pd
import joblib

# Load the trained model
with open('/content/best_model.pkl', 'rb') as f:
    model = joblib.load(f)

st.title("Employee Salary Prediction")
st.markdown("Enter employee details to predict their salary")

# Collect input data from user
# The model was trained on multiple features, so you need to collect all of them.
# Based on the training code, the input features are:
# 'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
# 'occupation', 'relationship', 'race', 'gender', 'capital-gain',
# 'capital-loss', 'hours-per-week', 'native-country'

# You'll need to get the numerical values for the categorical features
# (workclass, marital-status, occupation, relationship, race, gender, native-country)
# as they were label encoded during training.

# Example of how to collect input for all features (you'll need to adjust this)
# This is just a placeholder, you should use Streamlit widgets to get user input for each feature
age = st.number_input("Age", min_value=17, max_value=75, value=30)
workclass = st.selectbox("Workclass (Encoded)", list(range(7))) # Adjust range based on unique encoded values
fnlwgt = st.number_input("fnlwgt", value=200000) # Example value, adjust as needed
educational_num = st.number_input("Educational Number", min_value=5, max_value=16, value=10)
marital_status = st.selectbox("Marital Status (Encoded)", list(range(7))) # Adjust range
occupation = st.selectbox("Occupation (Encoded)", list(range(15))) # Adjust range
relationship = st.selectbox("Relationship (Encoded)", list(range(6))) # Adjust range
race = st.selectbox("Race (Encoded)", list(range(5))) # Adjust range
gender = st.selectbox("Gender (Encoded)", [0, 1]) # 0 or 1
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
native_country = st.selectbox("Native Country (Encoded)", list(range(41))) # Adjust range


# Prepare input data as a DataFrame with the correct column order and names
input_data = pd.DataFrame([[
    age, workclass, fnlwgt, educational_num, marital_status, occupation,
    relationship, race, gender, capital_gain, capital_loss, hours_per_week,
    native_country
]], columns=[
    'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
    'occupation', 'relationship', 'race', 'gender', 'capital-gain',
    'capital-loss', 'hours-per-week', 'native-country'
])


# Predict button
if st.button("Predict Salary Class"):
    # The model pipeline includes a StandardScaler, so the input data should be in a DataFrame
    prediction = model.predict(input_data)
    # Assuming the target variable 'income' was encoded to 0 and 1
    label = ">50K" if prediction[0] == 1 else "<=50K"
    st.success(f"Predicted Salary Class: {label}")
