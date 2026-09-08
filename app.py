import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from utils import engineer_features

# Page config
st.set_page_config(page_title="Diabetes Predictor Pro", layout="wide", page_icon="🩺")
st.title("🩺 Advanced Diabetes Early Detection System")
st.markdown("**Powered by XGBoost + SMOTE + SHAP explainability**")

# Load artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        medians = joblib.load('medians.pkl')
        return model, scaler, feature_names, medians
    except FileNotFoundError as e:
        st.error(f"Missing artifact: {e}. Please run train_model.py first.")
        st.stop()

model, scaler, feature_names, medians = load_artifacts()

# Preprocessing
def preprocess_user_input(data_dict):
    df = pd.DataFrame([data_dict])
    for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        if df[col].iloc[0] == 0:
            df[col] = medians[col]
    df_eng = engineer_features(df)
    for col in feature_names:
        if col not in df_eng.columns:
            df_eng[col] = 0
    df_eng = df_eng[feature_names]
    scaled = scaler.transform(df_eng)
    return pd.DataFrame(scaled, columns=feature_names)

# UI
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("👤 Patient Demographics")
    age = st.slider("Age (years)", 0, 120, 30)
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)

    st.subheader("📊 Clinical Measurements")
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=250, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=180, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level (µU/mL)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)

with col2:
    st.subheader("📈 Prediction & Risk Analysis")
    if st.button("🔍 Analyze Risk", type="primary"):
        user_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        try:
            scaled_input = preprocess_user_input(user_data)
            proba = model.predict_proba(scaled_input)[0][1]
            pred = int(proba >= 0.5)

            st.metric("Diabetes Risk Probability", f"{proba*100:.1f}%")
            if pred == 1:
                st.error("⚠️ **High Risk** – Please consult a healthcare professional.")
            else:
                st.success("✅ **Low Risk** – Maintain a healthy lifestyle.")

            st.subheader("🔎 What factors influenced this prediction?")
            col_shap1, col_shap2 = st.columns(2)
            with col_shap1:
                try:
                    st.image('shap_bar.png', caption="Global Feature Importance (Bar)")
                except:
                    st.warning("SHAP bar plot not found.")
            with col_shap2:
                try:
                    st.image('shap_summary.png', caption="Feature Impact Distribution")
                except:
                    st.warning("SHAP summary plot not found.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")