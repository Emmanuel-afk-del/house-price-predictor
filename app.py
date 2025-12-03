# app.py
import streamlit as st
import joblib
import pandas as pd
import os

# ================================
# 🚀 Load Model
# ================================
@st.cache_resource
def load_model():
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(APP_DIR, "notebooks", "models")

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    r2_path = os.path.join(MODELS_DIR, "test_r2.txt")

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at\n`{model_path}`")
        st.stop()

    try:
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        r2_score = "N/A"
        if os.path.exists(r2_path):
            with open(r2_path, "r") as f:
                r2_score = f.read().strip()
        return model, feature_names, r2_score

    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()


model, feature_names, test_r2 = load_model()

# ================================
# 🎨 UI
# ================================
st.set_page_config(page_title="House Price Predictor", page_icon="🏡")
st.title("🏡 California House Price Predictor")
st.markdown(f"Random Forest model (R² = **{test_r2}**)")

st.sidebar.header("🏠 Enter House Features (Text Inputs)")

# Text Inputs Instead of Sliders
MedInc = st.sidebar.text_input("Median Income (× $10k)", "3.0")
HouseAge = st.sidebar.text_input("House Age (years)", "20")
AveRooms = st.sidebar.text_input("Avg Rooms", "5.5")
AveBedrms = st.sidebar.text_input("Avg Bedrooms", "1.2")
Population = st.sidebar.text_input("Population", "1425")
AveOccup = st.sidebar.text_input("Avg Occupancy", "3.0")
Latitude = st.sidebar.text_input("Latitude", "34.0")
Longitude = st.sidebar.text_input("Longitude", "-118.0")

# Predict Button
if st.sidebar.button("🔮 Predict Price", type="primary"):

    try:
        # Convert all input values to float
        input_values = {
            "MedInc": float(MedInc),
            "HouseAge": float(HouseAge),
            "AveRooms": float(AveRooms),
            "AveBedrms": float(AveBedrms),
            "Population": float(Population),
            "AveOccup": float(AveOccup),
            "Latitude": float(Latitude),
            "Longitude": float(Longitude)
        }

        input_data = pd.DataFrame([input_values], columns=feature_names)

        pred = model.predict(input_data)[0]

        st.metric("💰 Predicted House Value", f"${pred * 100_000:,.0f}")

    except ValueError:
        st.error("❌ Please enter VALID numerical values.")

st.caption("✅ Model loaded from `notebooks/models/` | Built with Streamlit")
