import streamlit as st
import joblib
import pandas as pd
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# =========================
# Load pipeline
# =========================
model_package = joblib.load("/Users/sulthanfatahillah/Downloads/Purwadhika/Coding/Final Project/pipe_xgbc_ros_new.joblib")
model = model_package["model"]   # pipeline
threshold = model_package["threshold"]

# Ambil sub-pipeline
preprocessor = model.named_steps['preprocessing']
classifier = model.named_steps['modeling']

# Ambil semua nama fitur hasil preprocessing
feature_names = preprocessor.get_feature_names_out()



# =========================
# Streamlit Page Config + CSS
# =========================
st.set_page_config(page_title="Hotel Booking Cancellation Predictor", page_icon="🏨", layout="wide")

st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #2E86C1;
        font-size: 38px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f9f9f9;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏨 Hotel Booking Cancellation Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Isi form berikut untuk memprediksi kemungkinan cancel booking</div>", unsafe_allow_html=True)

# =========================
# Layout: Input (kiri) - Output (kanan)
# =========================
col1, col2 = st.columns([1.2, 1.8])

with col1:
    st.subheader("🔧 Input Data Reservasi")

    input_data = {}

    # =========================
    # Input hanya X_new (18 fitur)
    # =========================
    input_data['required_car_parking_spaces'] = st.number_input("Car Parking Spaces", min_value=0, value=0)
    input_data['country'] = st.text_input("Country Code", "PRT")
    input_data['market_segment'] = st.selectbox("Market Segment", 
        ["Direct","Corporate","Online TA","Offline TA/TO","Groups","Complementary"])
    input_data['previous_cancellations'] = st.number_input("Previous Cancellations", min_value=0, value=0)
    input_data['total_of_special_requests'] = st.number_input("Total Special Requests", min_value=0, value=0)
    input_data['customer_type'] = st.selectbox("Customer Type", 
        ["Transient","Contract","Group","Transient-Party"])
    
    # Agent bisa numerikal atau 'personal'
    agent_input = st.text_input("Agent (isi angka ID atau 'personal')", "personal")
    try:
        agent_input = int(agent_input)  # jika angka, cast ke int
    except:
        agent_input = "personal"       # jika string, biarkan "personal"
    input_data['agent'] = agent_input

    input_data['lead_time'] = st.number_input("Lead Time (hari)", min_value=0, value=30)
    input_data['booking_changes'] = st.number_input("Booking Changes", min_value=0, value=0)
    input_data['previous_bookings_not_canceled'] = st.number_input("Previous Bookings Not Canceled", min_value=0, value=0)
    input_data['arrival_date_year'] = st.number_input("Arrival Year", min_value=2015, max_value=2030, value=2019)
    input_data['distribution_channel'] = st.selectbox("Distribution Channel", ["Direct","Corporate","TA/TO","GDS"])
    input_data['meal'] = st.selectbox("Meal Plan", ["BB", "HB", "SC", "FB"])
    input_data['days_in_waiting_list'] = st.number_input("Days in Waiting List", min_value=0, value=0)
    input_data['hotel'] = st.selectbox("Hotel Type", ["Resort Hotel", "City Hotel"])
    input_data['is_repeated_guest'] = st.selectbox("Repeated Guest?", [0,1])
    input_data['adults'] = st.number_input("Adults", min_value=1, value=2)
    input_data['adr'] = st.number_input("Average Daily Rate (ADR)", min_value=0.0, value=100.0)

    X_input = pd.DataFrame([input_data])

with col2:
    st.subheader("📊 Hasil Prediksi")

    if st.button("🚀 Prediksi Cancel Booking"):
        # Prediksi
        y_proba = model.predict_proba(X_input)[0]
        cancel_proba, no_cancel_proba = y_proba[1], y_proba[0]

        pred = int(cancel_proba >= threshold)
        color = "#E74C3C" if pred == 1 else "#27AE60"
        text = "❌ Berpotensi CANCEL" if pred == 1 else "✅ Kemungkinan TIDAK CANCEL"

        # Pilih angka probabilitas sesuai prediksi
        prob_display = cancel_proba if pred == 1 else no_cancel_proba
        prob_label = "Probabilitas Cancel" if pred == 1 else "Probabilitas Tidak Cancel"

        st.markdown(
            f"<div class='card' style='border-left: 8px solid {color}; padding:20px;'>"
            f"<h3 style='color:{color};'>{text}</h3>"
            f"<div style='text-align:center; font-size:32px; font-weight:bold; color:{color};'>"
            f"{prob_label} = {prob_display:.1%}"
            f"</div>"
            f"<p><b>Probabilitas Cancel:</b> {cancel_proba:.2%}</p>"
            f"<p><b>Probabilitas Tidak Cancel:</b> {no_cancel_proba:.2%}</p>"
            f"<p><b>Threshold Model:</b> {threshold}</p>"
            "</div>",
            unsafe_allow_html=True
        )

        # =========================
        # Tambahan Action Rekomendasi Deposit
        # =========================
        if cancel_proba >= 0.5:
            st.warning("💡 Rekomendasi Action: **Deposit Type = No Refund**")
        else:
            st.success("💡 Rekomendasi Action: **Deposit Type = Refundable**")

        # =========================
        # LIME Explanation
        # =========================
        st.subheader("🔍 Interpretasi Prediksi")
        with st.expander("Klik untuk melihat LIME explanation"):
            X_transformed = preprocessor.transform(X_input)

            dummy_data = np.random.normal(
                loc=X_transformed.mean(axis=0),
                scale=X_transformed.std(axis=0) + 1e-6,
                size=(200, X_transformed.shape[1])
            )

            explainer = LimeTabularExplainer(
                training_data=dummy_data,
                feature_names=feature_names,
                class_names=['No Cancel','Cancel'],
                mode='classification'
            )

            exp = explainer.explain_instance(
                data_row=X_transformed[0],
                predict_fn=lambda x: classifier.predict_proba(x),
                num_features=10
            )

            fig = exp.as_pyplot_figure()
            st.pyplot(fig)