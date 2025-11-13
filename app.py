import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and preprocessing tools
model = load_model("fertilizer_model.keras")
scaler = joblib.load("scaler.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")

# Dropdown labels
soil_labels = list(soil_encoder.classes_)
crop_labels = list(crop_encoder.classes_)

# UI setup
st.set_page_config(page_title="Fertilizer Classifier", layout="centered")
st.title("🌾 Fertilizer Recommendation System")
st.write("Enter soil and crop features to predict the recommended fertilizer.")

# Input form
with st.form("input_form"):
    temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0)
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0)
    moisture = st.number_input("🌱 Moisture (%)", min_value=0.0)

    soil_type = st.selectbox("🧱 Soil Type", soil_labels)
    crop_type = st.selectbox("🌾 Crop Type", crop_labels)

    nitrogen = st.number_input("🧪 Nitrogen (mg/kg)", min_value=0.0)
    potassium = st.number_input("🧪 Potassium (mg/kg)", min_value=0.0)
    phosphorous = st.number_input("🧪 Phosphorous (mg/kg)", min_value=0.0)

    submit = st.form_submit_button("🔍 Predict Fertilizer")

# Prediction logic
if submit:
    try:
        # Encode categorical inputs
        soil_encoded = soil_encoder.transform([soil_type])[0]
        crop_encoded = crop_encoder.transform([crop_type])[0]

        # Prepare input
        input_array = np.array([[temperature, humidity, moisture, soil_encoded, crop_encoded,
                                 nitrogen, potassium, phosphorous]])
        input_scaled = scaler.transform(input_array)
        input_reshaped = input_scaled.reshape(-1, input_scaled.shape[1], 1)

        # Predict
        prediction = model.predict(input_reshaped)
        predicted_index = np.argmax(prediction)
        predicted_class = fertilizer_encoder.inverse_transform([predicted_index])[0]
        confidence = np.max(prediction)

        # Output
        st.success(f"✅ Recommended Fertilizer: **{predicted_class}**")
        st.info(f"Confidence: {confidence:.2f}")

        # Warn if confidence is low
        if confidence < 0.5:
            st.warning("⚠️ Low confidence prediction — please double-check your input values.")

        # Show top 2 predictions
        top_indices = prediction[0].argsort()[-2:][::-1]
        top_classes = fertilizer_encoder.inverse_transform(top_indices)
        top_scores = prediction[0][top_indices]

        st.write("🔎 Top Predictions:")
        for cls, score in zip(top_classes, top_scores):
            st.write(f"• {cls}: {score:.2f}")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
