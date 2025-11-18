import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Load model and encoders
# -------------------------------
model = joblib.load("fertilizer_xgb_model.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
fert_encoder = joblib.load("fertilizer_encoder.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fertilizer Recommendation", layout="wide")
st.title("🌾 Fertilizer Recommendation System")

# -------------------------------
# Input layout: 3 per row
# -------------------------------
st.markdown("### Enter Soil and Crop Details")

row1 = st.columns(3)
temp = row1[0].number_input("Temperature (°C)", value=0.0, format="%.2f")
humidity = row1[1].number_input("Humidity (%)", value=0.0, format="%.2f")
moisture = row1[2].number_input("Moisture (%)", value=0.0, format="%.2f")

row2 = st.columns(3)
soil = row2[0].selectbox("Soil Type", soil_encoder.classes_)
crop = row2[1].selectbox("Crop Type", crop_encoder.classes_)
nitrogen = row2[2].number_input("Nitrogen (mg/kg)", value=0.0, format="%.2f")

row3 = st.columns(3)
potassium = row3[0].number_input("Potassium (mg/kg)", value=0.0, format="%.2f")
phosphorous = row3[1].number_input("Phosphorous (mg/kg)", value=0.0, format="%.2f")

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Fertilizer"):
    # Encode inputs
    soil_encoded = soil_encoder.transform([soil])[0]
    crop_encoded = crop_encoder.transform([crop])[0]

    # Prepare input
    input_data = np.array([[temp, humidity, moisture, soil_encoded, crop_encoded,
                            nitrogen, potassium, phosphorous]])
    input_scaled = scaler.transform(input_data)

    # Predict with XGBoost
    pred_probs = model.predict_proba(input_scaled)[0]
    pred_index = np.argmax(pred_probs)
    pred_label = fert_encoder.inverse_transform([pred_index])[0]
    confidence = pred_probs[pred_index] * 100

    # -------------------------------
    # Output: Highlighted prediction
    # -------------------------------
    st.markdown(
        f"<div style='text-align:center; padding:25px; background-color:#f0f8ff; border-radius:12px;'>"
        f"<h2 style='color:#2E8B57;'>🌟 Recommended Fertilizer</h2>"
        f"<h1 style='color:#1E90FF; font-size:52px; font-weight:bold;'>{pred_label}</h1>"
        f"<p style='font-size:22px;'>Confidence: <b>{confidence:.2f}%</b></p>"
        f"</div>",
        unsafe_allow_html=True
    )

    # -------------------------------
    # Confidence charts
    # -------------------------------
    labels = fert_encoder.classes_

    chart_cols = st.columns(2)

    # Pie chart (filtered for clarity)
    with chart_cols[0]:
        st.markdown("#### Confidence Distribution (Pie Chart)")

        # Filter out classes with very low confidence
        threshold = 1.0  # Only show classes with >1% confidence
        filtered_labels = [labels[i] for i in range(len(labels)) if pred_probs[i] * 100 > threshold]
        filtered_probs = [pred_probs[i] * 100 for i in range(len(labels)) if pred_probs[i] * 100 > threshold]

        # If everything is below threshold, show top 3 anyway
        if len(filtered_labels) == 0:
            sorted_indices = np.argsort(pred_probs)[::-1][:3]
            filtered_labels = [labels[i] for i in sorted_indices]
            filtered_probs = [pred_probs[i] * 100 for i in sorted_indices]

        fig, ax = plt.subplots()
        ax.pie(filtered_probs, labels=filtered_labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)



