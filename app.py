import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Prediksi Risiko Stroke",
    page_icon="🏥",
    layout="centered"
)

# ============================================
# LOAD MODEL & INFO MODEL
# ============================================
@st.cache_resource
def load_model():
    with open('model_terbaik.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('info_model.json', 'r') as f:
        info = json.load(f)
    return model, info

model, info = load_model()

# ============================================
# HEADER
# ============================================
st.title("🏥 Prediksi Risiko Stroke")
st.write("Masukkan data pasien di bawah ini untuk memprediksi risiko stroke menggunakan Machine Learning.")

# Tampilkan info model & akurasi
with st.expander("ℹ️ Informasi Model"):
    st.write(f"**Model yang digunakan:** {info['nama_model']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{info['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{info['precision']*100:.2f}%")
    col3.metric("Recall", f"{info['recall']*100:.2f}%")
    col4.metric("F1-Score", f"{info['f1_score']*100:.2f}%")

st.divider()

# ============================================
# FORM INPUT USER
# ============================================
st.subheader("📝 Masukkan Data Pasien")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia (tahun)", min_value=0, max_value=120, value=45)
    avg_glucose_level = st.number_input("Rata-rata Kadar Glukosa (mg/dL)", min_value=50.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"])

with col2:
    hypertension = st.selectbox("Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.selectbox("Penyakit Jantung", ["Tidak", "Ya"])
    ever_married = st.selectbox("Status Pernikahan", ["Yes", "No"])
    smoking_status = st.selectbox("Status Merokok", ["never smoked", "formerly smoked", "smokes", "Unknown"])

col3, col4 = st.columns(2)
with col3:
    work_type = st.selectbox("Jenis Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
with col4:
    residence_type = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])

st.divider()

# ============================================
# PREPROCESSING INPUT (HARUS SAMA SEPERTI SAAT TRAINING)
# ============================================
def preprocess_input():
    # Mapping encoding sesuai LabelEncoder saat training
    # NOTE: urutan alfabetis adalah default LabelEncoder dari sklearn
    gender_map = {"Female": 0, "Male": 1, "Other": 2}
    married_map = {"No": 0, "Yes": 1}
    work_map = {"Govt_job": 0, "Never_worked": 1, "Private": 2, "Self-employed": 3, "children": 4}
    residence_map = {"Rural": 0, "Urban": 1}
    smoking_map = {"Unknown": 0, "formerly smoked": 1, "never smoked": 2, "smokes": 3}

    # Feature engineering: age_group
    if age < 18:
        age_group = "Anak"
    elif age < 40:
        age_group = "Dewasa Muda"
    elif age < 60:
        age_group = "Dewasa"
    else:
        age_group = "Lansia"
    age_group_map = {"Anak": 0, "Dewasa": 1, "Dewasa Muda": 2, "Lansia": 3}

    # Feature engineering: bmi_category
    if bmi < 18.5:
        bmi_cat = "Underweight"
    elif bmi < 25:
        bmi_cat = "Normal"
    elif bmi < 30:
        bmi_cat = "Overweight"
    else:
        bmi_cat = "Obesitas"
    bmi_cat_map = {"Normal": 0, "Obesitas": 1, "Overweight": 2, "Underweight": 3}

    data = {
        "gender": gender_map[gender],
        "age": age,
        "hypertension": 1 if hypertension == "Ya" else 0,
        "heart_disease": 1 if heart_disease == "Ya" else 0,
        "ever_married": married_map[ever_married],
        "work_type": work_map[work_type],
        "Residence_type": residence_map[residence_type],
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_map[smoking_status],
        "age_group": age_group_map[age_group],
        "bmi_category": bmi_cat_map[bmi_cat],
    }

    # Urutkan kolom sesuai urutan fitur saat training
    df_input = pd.DataFrame([data])
    df_input = df_input[info['fitur']]
    return df_input

# ============================================
# TOMBOL PREDIKSI
# ============================================
if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
    X_input = preprocess_input()

    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0]

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.error(f"⚠️ **Risiko Stroke Tinggi**")
        st.write(f"Probabilitas stroke: **{probability[1]*100:.2f}%**")
        st.warning("Disarankan untuk berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut.")
    else:
        st.success(f"✅ **Risiko Stroke Rendah**")
        st.write(f"Probabilitas tidak stroke: **{probability[0]*100:.2f}%**")
        st.info("Tetap jaga gaya hidup sehat untuk meminimalkan risiko.")

    # Detail probabilitas
    st.write("**Detail Probabilitas:**")
    prob_df = pd.DataFrame({
        "Kategori": ["Tidak Stroke", "Stroke"],
        "Probabilitas": [f"{probability[0]*100:.2f}%", f"{probability[1]*100:.2f}%"]
    })
    st.table(prob_df)

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("Final Project AI & Big Data 2026 — Model: " + info['nama_model'])
