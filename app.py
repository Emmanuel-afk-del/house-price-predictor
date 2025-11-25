# app.py
import streamlit as st
import joblib
import pandas as pd
import os

# ================================
# 🚀 Load Model from notebooks/models/ (fixed path)
# ================================
@st.cache_resource
def load_model():
    # Get the directory where app.py is located
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # ✅ FIX: Point to notebooks/models/ where your model actually is
    MODELS_DIR = os.path.join(APP_DIR, "notebooks", "models")
    
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    features_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    r2_path = os.path.join(MODELS_DIR, "test_r2.txt")
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at:\n`{model_path}`")
        st.stop()
    
    # Load
    try:
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        r2_score = "N/A"
        if os.path.exists(r2_path):
            with open(r2_path, "r") as f:
                r2_score = f.read().strip()
        return model, feature_names, r2_score
    except Exception as e:
        st.exception(e)
        st.error(f"❌ Failed to load model: {e}")
        st.stop()


model, feature_names, test_r2 = load_model()

# ================================
# 🎨 UI (unchanged)
# ================================
st.set_page_config(page_title="House Price Predictor", page_icon="🏡")
st.title("🏡 California House Price Predictor")
st.markdown(f"Random Forest model (R² = **{test_r2}**) trained on California housing data.")

# Inputs
st.sidebar.header("🏠 Enter House Features")
MedInc = st.sidebar.slider("Median Income (× $10k)", 0.5, 15.0, 3.0, 0.1)
HouseAge = st.sidebar.slider("House Age (years)", 1, 52, 20)
AveRooms = st.sidebar.number_input("Avg Rooms", 2.0, 15.0, 5.5, 0.5)
AveBedrms = st.sidebar.number_input("Avg Bedrooms", 1.0, 5.0, 1.2, 0.1)
Population = st.sidebar.number_input("Population", 100, 10000, 1425, 100)
AveOccup = st.sidebar.number_input("Avg Occupancy", 1.0, 6.0, 3.0, 0.1)
Latitude = st.sidebar.number_input("Latitude", 32.0, 42.0, 34.0, 0.1)
Longitude = st.sidebar.number_input("Longitude", -124.5, -114.0, -118.0, 0.1)

# Predict
if st.sidebar.button("🔮 Predict Price", type="primary"):
    input_data = pd.DataFrame([{
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }], columns=feature_names)
    
    pred = model.predict(input_data)[0]
    st.metric("💰 Predicted House Value", f"${pred * 100_000:,.0f}")

st.caption("✅ Model loaded from `notebooks/models/` | Built with Streamlit")