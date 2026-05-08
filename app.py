import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =========================
# LOAD MODEL
# =========================
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
features = joblib.load("features.pkl")

# =========================
# PAGE UI
# =========================
st.set_page_config(page_title="AI Health Diagnosis", layout="centered")
st.title("🧠 AI Patient Diagnosis System")

# =========================
# INPUTS (MATCH TRAINING RANGE)
# =========================

age = st.slider("Age", 1, 100, 30)

gender = st.radio("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

sysbp = st.slider("Systolic BP", 80, 200, 120)
diabp = st.slider("Diastolic BP", 50, 130, 80)

temp = st.slider("Body Temperature (°C)", 35.0, 42.0, 36.5)
heartrate = st.slider("Heart Rate", 40, 150, 75)

bmi = st.slider("BMI", 10.0, 50.0, 22.0)

# =========================
# SYMPTOMS (CONTROLLED INPUT)
# =========================

st.subheader("Select Symptoms")

symptoms = st.multiselect(
    "Choose symptoms",
    ["Fever", "Cough", "Fatigue", "Headache", 
     "Chest Pain", "Shortness of Breath",
     "Dizziness", "Nausea"]
)

# IMPORTANT: Normalize symptom count
symptoms_count = min(len(symptoms), 5)   # cap value (prevents extreme input)

# =========================
# PREDICTION
# =========================

if st.button("🔍 Predict Disease"):

    input_dict = {
        'Age': age,
        'Gender': gender,
        'SysBP': sysbp,
        'DiaBP': diabp,
        'Temp': temp,
        'HeartRate': heartrate,
        'BMI': bmi,
        'Symptoms': symptoms_count
    }

    input_df = pd.DataFrame([input_dict])

    # 🔥 EXACT FEATURE ALIGNMENT
    input_df = input_df.reindex(columns=features, fill_value=0)

    # 🔥 SCALING (SAME AS TRAINING)
    input_scaled = scaler.transform(input_df)

    # 🔥 PREDICTION
    pred = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0]

    disease = le.inverse_transform(pred)[0]

    # =========================
    # TOP 3 OUTPUT
    # =========================
    top3_idx = np.argsort(proba)[-3:][::-1]
    top3_diseases = le.inverse_transform(top3_idx)

    st.subheader("🧠 Diagnosis Result")

    st.success(f"Primary: {disease} ({proba[top3_idx[0]]:.2f})")

    st.write("### Other Possibilities:")
    for i, idx in enumerate(top3_idx):
        st.write(f"{i+1}. {top3_diseases[i]} → {proba[idx]:.2f}")

    # =========================
    # CONFIDENCE CHECK
    # =========================
    max_prob = proba[top3_idx[0]]
    diff = proba[top3_idx[0]] - proba[top3_idx[1]]

    if max_prob < 0.6:
        st.warning("⚠️ Low confidence prediction")

    if diff < 0.2:
        st.warning("⚠️ Similar diseases detected (borderline case)")

    # =========================
    # DEBUG (VERY IMPORTANT)
    # =========================
    st.write("### 🔍 Debug Info")
    st.write(input_df)