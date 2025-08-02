import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(
    page_title="🔥 Fire Type Classifier - India (MODIS)",
    layout="centered",
    page_icon="🔥"
)

# --- Custom Background (Blue-Black Gradient) ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .css-1v0mbdj, .css-1d391kg {
        color: white !important;
    }
    label, .stSelectbox label, .stNumberInput label {
        font-size: 16px;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load model and scaler safely ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("best_fire_detection_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error("❌ Error loading model or scaler. Make sure both .pkl files are present.")
        st.stop()

model, scaler = load_model()

# --- Sidebar Info ---
with st.sidebar:
    st.title("🌍 About the App")
    st.markdown("""
    This tool predicts the **type of fire** based on MODIS satellite data:
    - Brightness & T31 Brightness  
    - FRP, Scan, Track  
    - Confidence Level  
    ---
    **Project by Nilanjan Saha**  
    AICTE Internship | 2021–2023 MODIS Data  
    """)

# --- Main Title ---
st.markdown("<h1 style='text-align: center; color: orange;'>🔥 Fire Type Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter MODIS satellite readings to classify the fire type.</p>", unsafe_allow_html=True)

# --- Input Layout ---
col1, col2 = st.columns(2)

with col1:
    brightness = st.number_input("🔥 Brightness", min_value=200.0, max_value=600.0, value=300.0)
    frp = st.number_input("🔥 Fire Radiative Power (FRP)", min_value=0.0, value=15.0)
    scan = st.number_input("📏 Scan", min_value=0.0, value=1.0)

with col2:
    bright_t31 = st.number_input("🌡️ Brightness T31", min_value=200.0, max_value=400.0, value=290.0)
    track = st.number_input("🧭 Track", min_value=0.0, value=1.0)
    confidence = st.selectbox("🔍 Confidence Level", ["low", "nominal", "high"])

# --- Preprocessing ---
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])

try:
    scaled_input = scaler.transform(input_data)
except Exception as e:
    st.error("⚠️ Scaler failed to transform input. Check your scaler.pkl file.")
    st.stop()

# --- Fire Type Mapping ---
fire_types = {
    0: ("🌲 Vegetation Fire", "#4CAF50"),
    2: ("🏭 Other Static Land Source", "#FF9800"),
    3: ("🌊 Offshore Fire", "#2196F3")
}

# --- Predict Button ---
if st.button("🚀 Predict Fire Type"):
    prediction = model.predict(scaled_input)[0]
    result_text, color = fire_types.get(prediction, ("❓ Unknown", "gray"))

    st.markdown(f"""
        <div style='text-align: center; padding: 1.2rem; 
        background-color: {color}; color: white; font-size: 1.4rem; 
        border-radius: 10px; margin-top: 20px;'>
        🔎 <strong>Predicted Fire Type:</strong> {result_text}
        </div>
    """, unsafe_allow_html=True)

    # --- SHAP Explanation ---
    st.subheader("🔍 Feature Contribution (SHAP Explanation)")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_input)

        fig, ax = plt.subplots(figsize=(8, 4))

        feature_names = ["brightness", "bright_t31", "frp", "scan", "track", "confidence"]

        shap.plots.waterfall(shap.Explanation(
            values=shap_values[prediction][0],
            base_values=explainer.expected_value[prediction],
            data=scaled_input[0],
            feature_names=feature_names
        ), max_display=6, show=False)

        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ SHAP explanation failed: {e}")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; font-size:0.85rem;'>Made with ❤️ by Nilanjan Saha | AICTE Internship Project</div>", unsafe_allow_html=True)
