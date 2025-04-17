import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load model
model = joblib.load("logistic_regression_model.pkl")

# App Config
st.set_page_config(page_title="Student Mental Health Predictor", page_icon="🧠", layout="wide")

# Custom Banner with Logo and Title
col1, col2 = st.columns([1, 8])
with col1:
    logo = Image.open("logo.png")  # Optional: add your logo in the same folder and name it 'logo.png'
    st.image(logo, width=70)
with col2:
    st.markdown("## **Student Mental Health Predictor**")
    st.markdown("Predict if a student is likely to have a mental illness based on their responses.")

st.markdown("---")

# Sidebar Inputs
st.sidebar.header("📝 Enter Student Details")

age = st.sidebar.slider("Age", 18, 35, 21)
cgpa = st.sidebar.number_input("What is your CGPA?", min_value=0.0, max_value=4.0, step=0.01)

gender = st.sidebar.radio("Choose your gender", ["Male", "Female"])
marital_status = st.sidebar.radio("Are you married?", ["Yes", "No"])
depression = st.sidebar.radio("Do you have Depression?", ["Yes", "No"])
anxiety = st.sidebar.radio("Do you have Anxiety?", ["Yes", "No"])
panic_attack = st.sidebar.radio("Do you have Panic Attack?", ["Yes", "No"])
treatment = st.sidebar.radio("Did you seek any specialist for treatment?", ["Yes", "No"])

course = st.sidebar.selectbox("What is your course?", [
    'ALA', 'Accounting', 'BCS', 'BENL', 'BIT', 'Banking Studies', 'Biomedical science',
    'Biotechnology', 'Business Administration', 'CTS', 'Communication', 'DIPLOMA TESL',
    'Diploma Nursing', 'ENM', 'Econs', 'Engineering', 'Fiqh', 'Fiqh fatwa', 'Human Resources',
    'Human Sciences', 'IT', 'Irkhs', 'Islamic Education', 'KENMS', 'KIRKHS', 'KOE', 'Kop', 'Law',
    'Laws', 'MHSC', 'Malcom', 'Marine science', 'Mathemathics', 'Nursing', 'Pendidikan Islam',
    'Psychology', 'Radiography', 'TAASL', 'Usuluddin'
])

study_year = st.sidebar.selectbox("Your current year of Study", [
    'Year 1', 'Year 2', 'Year 3', 'Year 4'
])

# One-hot Encoding Manual
features = {
    'Age': age,
    'What is your CGPA?': cgpa,
    'Do you have Depression?': 1 if depression == "Yes" else 0,
    'Do you have Anxiety?': 1 if anxiety == "Yes" else 0,
    'Do you have Panic attack?': 1 if panic_attack == "Yes" else 0,
    'Did you seek any specialist for a treatment?': 1 if treatment == "Yes" else 0,
    'Choose your gender_Female': 1 if gender == "Female" else 0,
    'Choose your gender_Male': 1 if gender == "Male" else 0,
    'Marital status_Yes': 1 if marital_status == "Yes" else 0,
    'Marital status_No': 1 if marital_status == "No" else 0
}

# Add all one-hot encoded course and year features
all_courses = [
    'ALA', 'Accounting', 'BCS', 'BENL', 'BIT', 'Banking Studies', 'Biomedical science',
    'Biotechnology', 'Business Administration', 'CTS', 'Communication', 'DIPLOMA TESL',
    'Diploma Nursing', 'ENM', 'Econs', 'Engineering', 'Fiqh', 'Fiqh fatwa', 'Human Resources',
    'Human Sciences', 'IT', 'Irkhs', 'Islamic Education', 'KENMS', 'KIRKHS', 'KOE', 'Kop', 'Law',
    'Laws', 'MHSC', 'Malcom', 'Marine science', 'Mathemathics', 'Nursing', 'Pendidikan Islam',
    'Psychology', 'Radiography', 'TAASL', 'Usuluddin'
]
for c in all_courses:
    features[f'What is your course?_{c}'] = 1 if course == c else 0

all_years = ['Year 1', 'Year 2', 'Year 3', 'Year 4']
for y in all_years:
    features[f'Your current year of Study_{y}'] = 1 if study_year == y else 0

# Convert to DataFrame
input_df = pd.DataFrame([features])

# Ensure that the columns of input_df match the model's expected feature names
expected_columns = model.feature_names_in_
missing_cols = set(expected_columns) - set(input_df.columns)
extra_cols = set(input_df.columns) - set(expected_columns)

# Add missing columns with value 0 (if they are not in input_df)
for col in missing_cols:
    input_df[col] = 0

# Remove extra columns (if any)
input_df = input_df[expected_columns]

# Prediction
st.markdown("## 🎯 Prediction Result")
if st.button("Predict Mental Health Status"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("🚨 **At Risk of Mental Illness**. Please consult a specialist.")
    else:
        st.success("✅ **No Mental Illness Detected**. Keep up your mental well-being!")

    st.markdown("---")
    st.markdown("📊 **Model Details**")
    st.write("This prediction is based on a logistic regression model trained with survey data from students.")














# Footer
st.markdown(""" 
---
🎓 *Developed by Ayan* | 💻 *Powered by Streamlit* | 🔐 *Model: Logistic Regression*
""")
