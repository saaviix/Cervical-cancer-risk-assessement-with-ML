import streamlit as st
import joblib
import numpy as np
import os

# Streamlit App Title with a feminine theme
st.title("💖 Health Prediction Application")

# Choose machine learning model with a sleek radio button
choice = st.radio(
    "🔬 Choose a Model:",
    ("Random Forest Classifier", "GBoost Classifier", "SVM", "CatBoost Classifier"),
    key="model_choice",
    index=0,
    help="Select the machine learning model to predict your health outcome.",
)

# Load selected model
model_mapping = {
    "Random Forest Classifier": "genModels/random_forest.pkl",
    "GBoost Classifier": "genModels/xgboost.pkl",
    "SVM": "genModels/svm.pkl",
    "CatBoost Classifier": "genModels/catboost.pkl",
}
model = joblib.load(model_mapping.get(choice))

# Custom CSS for feminine theme appearance with better contrast
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f7e4e9; /* Light pink background */
            color: #000000; /* Black text color */
        }
        .stRadio > div {
            display: flex;
            justify-content: space-around;
            background-color: #fbe2f1; /* Soft pink background for radio buttons */
            border-radius: 10px;
            padding: 10px;
        }
        .stTextInput, .stNumberInput, .stRadio {
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 2px solid #e2a1d3; /* Soft pink border */
            background-color: #ffffff;
        }
        .stButton {
            background-color: #f56fa1; /* Light pink button */
            color: white;
            font-size: 18px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton:hover {
            background-color: #f1a7c3; /* Slightly darker pink on hover */
        }
        .stSubheader {
            font-size: 18px;
            color: #e14d78; /* Soft pinkish color for subheaders */
            font-weight: bold;
            text-align: left;
        }
        .stTitle {
            color: #e14d78; /* Same soft pink as subheaders */
            font-size: 36px;
            text-align: center;
            margin-bottom: 30px;
        }
        .stText, .stMarkdown {
            font-size: 16px;
            color: #000000; /* Black text color */
            text-align: left;
        }
        .stRadio label {
            font-size: 18px;
            color: #000000; /* Black text color */
            font-weight: bold;
        }
        .stNumberInput input {
            font-size: 16px;
        }
        .stMarkdown {
            text-align: center;
            font-size: 20px;
            color: #8a5e7d;
        }
        .stButton, .stRadio, .stTextInput, .stNumberInput {
            background-color: #ffffff;
            border: 2px solid #e2a1d3;
        }
    </style>
""", unsafe_allow_html=True)

# User Inputs Section
st.header("📝 Enter Your Health Data")
st.write("Please enter your personal health and lifestyle information to receive a prediction.")

# Input Fields as per your updated list

# Basic health information
age = st.number_input("Age", min_value=0, step=1, key="age")
num_sexual_partners = st.number_input("Number of Sexual Partners", min_value=0, step=1, key="num_sexual_partners")
first_sexual_intercourse = st.number_input("First Sexual Intercourse (Age)", min_value=0, step=1, key="first_sexual_intercourse")
num_pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1, key="num_pregnancies")

# Lifestyle and Contraceptive History
st.subheader("💊 Lifestyle and Contraceptive History")
smokes = st.radio("Do you smoke?", ("Yes", "No"), key="smokes")
smokes_years = st.number_input("Years of Smoking", min_value=0, step=1, key="smokes_years") if smokes == "Yes" else 0
smokes_packs_per_year = st.number_input("Packs per Year", min_value=0.0, step=0.1, key="smokes_packs_per_year") if smokes == "Yes" else 0

hormonal_contraceptives = st.radio("Do you use hormonal contraceptives?", ("Yes", "No"), key="hormonal_contraceptives")
hormonal_contraceptives_years = st.number_input("Years of Hormonal Contraceptive Use", min_value=0, step=1, key="hormonal_contraceptives_years") if hormonal_contraceptives == "Yes" else 0

iud = st.radio("Do you use an IUD?", ("Yes", "No"), key="iud")
iud_years = st.number_input("Years of IUD Use", min_value=0, step=1, key="iud_years") if iud == "Yes" else 0

# STDs section with detailed options
st.subheader("🦠 STDs History")
stds = st.radio("Have you ever had an STD?", ("Yes", "No"), key="stds")
stds_number = st.number_input("Number of STDs", min_value=0, step=1, key="stds_number") if stds == "Yes" else 0

# Diagnosis fields (for Cancer, CIN, HPV, etc.)
st.subheader("🔬 Diagnosis Information")
dx_cancer = st.radio("Diagnosed with Cancer?", ("Yes", "No"), key="dx_cancer")
dx_cin = st.radio("Diagnosed with CIN?", ("Yes", "No"), key="dx_cin")
dx_hpv = st.radio("Diagnosed with HPV?", ("Yes", "No"), key="dx_hpv")
dx_hinselmann = st.radio("Diagnosed with Hinselmann?", ("Yes", "No"), key="dx_hinselmann")
dx_schiller = st.radio("Diagnosed with Schiller?", ("Yes", "No"), key="dx_schiller")
dx_citology = st.radio("Diagnosed with Abnormal Citology?", ("Yes", "No"), key="dx_citology")

# Prepare the input list
input_list = [age, num_sexual_partners, first_sexual_intercourse, num_pregnancies, smokes, smokes_years,
              smokes_packs_per_year, hormonal_contraceptives, hormonal_contraceptives_years, iud, iud_years, stds,
              stds_number, dx_cancer, dx_cin, dx_hpv, dx_hinselmann, dx_schiller, dx_citology]

# Convert "Yes" to 1 and "No" to 0
for i in range(len(input_list)):
    if isinstance(input_list[i], str):
        input_list[i] = 1 if input_list[i] == "Yes" else 0

# Submit Button
if st.button("🛑 Submit", key="submit_button"):
    # Show entered information
    st.subheader("📝 Entered Information:")
    for feature, value in zip(
        ["Age", "Number of Sexual Partners", "First Sexual Intercourse (Age)", "Number of Pregnancies",
         "Smokes", "Years of Smoking", "Packs per Year", "Hormonal Contraceptives",
         "Years of Hormonal Contraceptive Use", "IUD", "Years of IUD Use", "STDs",
         "Number of STDs", "Diagnosed with Cancer", "Diagnosed with CIN", "Diagnosed with HPV",
         "Diagnosed with Hinselmann", "Diagnosed with Schiller", "Diagnosed with Citology Abnormality"],
        [age, num_sexual_partners, first_sexual_intercourse, num_pregnancies, smokes, smokes_years,
         smokes_packs_per_year, hormonal_contraceptives, hormonal_contraceptives_years,
         iud, iud_years, stds, stds_number, dx_cancer, dx_cin, dx_hpv, dx_hinselmann, dx_schiller, dx_citology]
    ):
        st.write(f"**{feature}:** {value}")

    # Make prediction
    prediction = model.predict([input_list])
    if prediction == 0:
        prediction = "🔴 You don't have cervical cancer (precision: 74%)."
    elif prediction == 1:
        prediction = "🟢 You have cervical cancer (precision: 64%)."

    # Display prediction
    st.write("-" * 40)
    st.subheader("⚡ Prediction:")
    st.markdown(f"<p style='font-size: 24px; color: #000000; font-weight: bold; text-align: center;'>{prediction}</p>", unsafe_allow_html=True)