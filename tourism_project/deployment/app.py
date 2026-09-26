"""Streamlit front-end for the Wellness Tourism Package prediction model."""
import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

from dotenv import load_dotenv
load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME", "<your-hf-username>")
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-package-prediction-model"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="best_model.joblib")
    return joblib.load(model_path)

st.set_page_config(page_title="Wellness Package Predictor", page_icon="\U0001F3DD\uFE0F")
st.title("\U0001F3DD\uFE0F Wellness Tourism Package \u2014 Purchase Predictor")
st.write("Enter the customer's details below to predict how likely they are to purchase the new Wellness Tourism Package.")

model = load_model()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=15)
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
with col2:
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
    passport = st.selectbox("Holds Passport?", ["Yes", "No"])
    pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
    num_children = st.number_input("Number of Children Visiting (<5 yrs)", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)

input_df = pd.DataFrame([{
    "Age": age, "TypeofContact": type_of_contact, "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch, "Occupation": occupation, "Gender": gender,
    "NumberOfPersonVisiting": num_persons, "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched, "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status, "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0, "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0, "NumberOfChildrenVisiting": num_children,
    "Designation": designation, "MonthlyIncome": monthly_income,
}])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.success(f"Likely to purchase the Wellness Package (probability: {probability:.1%})")
    else:
        st.warning(f"Unlikely to purchase the Wellness Package (probability: {probability:.1%})")
