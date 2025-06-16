import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model_v2.pkl')

# Judul
st.title("🧠 Stroke Prediction App")
st.write("A web-based application that predicts an individual's risk of stroke using machine learning and health-related data.")
st.markdown("### Fill in this form")

# 2 kolom
col1, col2 = st.columns(2)

# Input dari pengguna
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0)
    hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
    heart_disease = st.selectbox("Do you have heart disease?", ["No", "Yes"])
    ever_married = st.selectbox("Have you ever married?", ["No", "Yes"])

with col2:
    work_type = st.selectbox("What kind of work you do?", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("What area you live?", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

# Mapping kategori ke angka
gender_map = {"Male": 1, "Female": 0}
work_type_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
hypertension_map = {"No": 0, "Yes": 1}
heart_disease_map = {"No": 0, "Yes": 1}
ever_married_map = {"No": 0, "Yes": 1}
Residence_type_map = {"Urban": 1, "Rural": 0}
smoking_map = {"never smoked": 0, "formerly smoked": 2, "smokes": 1, "Unknown": 3}

# Fungsi gradien warna
def get_risk_color(percentage):
    r = int(min(255, (percentage / 100) * 255))
    g = int(min(255, (1 - percentage / 100) * 255))
    b = 50
    return f'rgb({r},{g},{b})'

# Proses prediksi
if st.button("Predict Stroke Risk"):
    input_data = np.array([[gender_map[gender], age,
                            hypertension_map[hypertension],
                            heart_disease_map[heart_disease],
                            ever_married_map[ever_married],
                            work_type_map[work_type],
                            Residence_type_map[Residence_type],
                            avg_glucose_level, bmi,
                            smoking_map[smoking_status]]]).astype(float)

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    stroke_prob = probabilities[0][1] * 100
    no_stroke_prob = probabilities[0][0] * 100

    st.subheader("Prediction Result:")
    st.write(f"🧠 Risiko terkena stroke: **{stroke_prob:.2f}%**")
    st.write(f"👍 Kemungkinan tidak terkena stroke: **{no_stroke_prob:.2f}%**")

    # Warna gradien untuk kotak
    color = get_risk_color(stroke_prob)

    st.markdown(
        f"""
        <div style="padding: 1rem; background-color: {color}; border-left: 5px solid black; border-radius: 5px; margin-bottom: 20px;">
            <strong style="color: #ffffff;">
            {"⚠️ Berdasarkan data, Anda memiliki risiko terkena stroke." if prediction[0] == 1 else "✅ Berdasarkan data, risiko Anda terkena stroke rendah."}
            </strong>
            <div style="margin-top: 0.5rem; color: #f0f0f0;">
                Prediksi risiko: <strong>{stroke_prob:.2f}%</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Warna latar belakang halaman (fixed: merah/hijau)
    bg_color = "#f8d7da" if prediction[0] == 1 else "#d4edda"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Disclaimer
    st.markdown("Please keep in mind that this prediction might not be accurate and should not replace professional medical advice 😭🙏")