import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_training import load_model
from src.data_preprocessing import preprocess_features

# Page Configuration
st.set_page_config(
    page_title="Forest Fire Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: radial-gradient(circle at top right, #1a1c2c, #0e1117);
        color: #ffffff;
    }

    /* Professional Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f111a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* CRITICAL: Fix Sidebar Text Visibility */
    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stWidgetLabel p,
    div[data-testid="stRadio"] label p {
        color: #ffffff !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }

    /* Title Styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .subtitle {
        color: #8892b0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }

    /* Cards */
    .st-emotion-cache-12w0qpk {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }

    .st-emotion-cache-12w0qpk:hover {
        border-color: rgba(255, 75, 75, 0.3);
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        box-shadow: 0 8px 25px rgba(255, 75, 75, 0.5);
        transform: scale(1.02);
        color: white;
    }

    /* Result Containers */
    .prediction-container {
        padding: 2.5rem;
        border-radius: 24px;
        text-align: center;
        margin: 2rem 0;
        animation: fadeIn 0.8s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fire-alert {
        background: linear-gradient(135deg, rgba(255, 75, 43, 0.2), rgba(255, 65, 108, 0.2));
        border: 2px solid #ff4b2b;
        color: #ff4b2b;
    }

    .safe-alert {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.2), rgba(56, 239, 125, 0.2));
        border: 2px solid #38ef7d;
        color: #38ef7d;
    }

    /* Metric enhancement */
    [data-testid="stMetricValue"] {
        font-weight: 800;
        color: #ff4b2b;
    }

    /* Sidebar Image fix */
    [data-testid="stSidebarNav"] {
        background-image: none !important;
    }

    </style>
    """, unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def load_artifacts():
    model_path = 'models/saved_models/best_rf_model.pkl'
    scaler_path = 'models/saved_models/scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Sidebar Design
with st.sidebar:
    # Use the local generated image path
    sidebar_img_path = os.path.join(os.path.dirname(__file__), "sidebar_img.png")
    if os.path.exists(sidebar_img_path):
        st.image(sidebar_img_path, width="stretch") 
    else:
        st.image("https://images.unsplash.com/photo-1542601906990-b4d3fb773b09?q=80&w=1000", width="stretch")
    
    st.markdown("### IgnisGuard v1.0")
    st.markdown("---")
    page = st.radio("Navigation", ["Dashboard", "Analytics", "System Info"], index=0)
    
    st.markdown("---")
    st.info("System Status: Online 🟢")

model, scaler = load_artifacts()

if page == "Dashboard":
    st.markdown('<p class="main-title">IgnisGuard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Driven Forest Fire Risk Assessment System</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("⚠️ Critical Error: Model artifacts not synchronized. Please run the training pipeline.")
    else:
        # Main UI Layout
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("#### ☁️ Weather Parameters")
            with st.container():
                temperature = st.slider("Ambient Temperature (°C)", 10, 55, 32, help="High temperatures increase fuel dryness.")
                rh = st.slider("Relative Humidity (%)", 0, 100, 45, help="Low humidity promotes fire ignition.")
                ws = st.slider("Wind Velocity (km/h)", 0, 60, 20, help="Wind speed influences fire propagation.")
                rain = st.number_input("Precipitation (mm)", 0.0, 100.0, 0.0, step=0.1)
            
        with col2:
            st.markdown("#### 🔥 Fuel & Location Data")
            with st.container():
                ffmc = st.slider("FFMC Index", 0.0, 101.0, 85.0, step=0.1, help="Fine Fuel Moisture Code - indicates ease of ignition.")
                dmc = st.slider("DMC Index", 0.0, 150.0, 25.0, step=0.1, help="Duff Moisture Code - indicates fuel consumption in deep layers.")
                region = st.selectbox("Geographic Region", options=[0, 1], format_func=lambda x: "Bejaia Sector" if x == 0 else "Sidi-Bel Abbes Sector")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Center the button
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            assess_trigger = st.button("RUN RISK ASSESSMENT")
            
        if assess_trigger:
            input_df = pd.DataFrame({
                'Temperature': [temperature],
                'RH': [rh],
                'Ws': [ws],
                'Rain': [rain],
                'FFMC': [ffmc],
                'DMC': [dmc],
                'Region': [region]
            })
            
            try:
                processed_input = preprocess_features(input_df, is_train=False)
                prediction = model.predict(processed_input)[0]
                probability = model.predict_proba(processed_input)[0][1]
                
                if prediction == 1:
                    st.markdown(f"""
                        <div class="prediction-container fire-alert">
                            <h2 style="margin:0; color:#ff4b2b;">🚨 CRITICAL FIRE RISK DETECTED</h2>
                            <p style="font-size:1.5rem; margin-top:10px;">Classification Confidence: <b>{probability*100:.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.warning("Immediate action required: High potential for ignition and rapid spread.")
                else:
                    st.markdown(f"""
                        <div class="prediction-container safe-alert">
                            <h2 style="margin:0; color:#38ef7d;">✅ NOMINAL CONDITIONS</h2>
                            <p style="font-size:1.5rem; margin-top:10px;">Stability Probability: <b>{(1-probability)*100:.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("The current meteorological profile does not favor active forest fires.")
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Risk Level", "HIGH" if prediction == 1 else "LOW")
                m2.metric("Heat Index", f"{temperature}°C")
                m3.metric("Fuel Load", f"{ffmc}")
                m4.metric("Region", "Sector A" if region == 0 else "Sector B")

            except Exception as e:
                st.exception(e)

elif page == "Analytics":
    st.markdown('<p class="main-title">Operational Analytics</p>', unsafe_allow_html=True)
    st.write("Exploratory insights into the Algerian Forest Fire dataset.")
    
    analytics_img_path = os.path.join(os.path.dirname(__file__), "analytics_img.png")
    if os.path.exists(analytics_img_path):
        st.image(analytics_img_path, width="stretch")
    else:
        st.image("https://images.unsplash.com/photo-1542601906990-b4d3fb773b09?q=80&w=1200", width="stretch")
    
    st.markdown("""
    ### Core Correlations
    - **Temperature (30°C+):** Significant increase in fire probability.
    - **RH (<40%):** Critical threshold for fuel volatility.
    - **FFMC (>80):** Strong indicator of surface fire susceptibility.
    """)
    
    # Feature importance static chart representation
    st.subheader("Model Feature Importance (Gini)")
    importance_data = pd.DataFrame({
        'Feature': ['FFMC', 'DMC', 'Temperature', 'RH', 'Ws', 'Rain', 'Region'],
        'Importance': [0.45, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]
    }).sort_values('Importance', ascending=True)
    st.bar_chart(importance_data, x='Feature', y='Importance', color='#ff4b2b')

elif page == "System Info":
    st.markdown('<p class="main-title">System Architecture</p>', unsafe_allow_html=True)
    
    st.markdown("""
    #### 🤖 Machine Learning Model
    - **Architecture:** Random Forest Classifier (Ensemble of Decision Trees)
    - **Training Data:** Algerian Forest Fires Dataset (Bejaia & Sidi-Bel Abbes)
    - **Validation:** 5-fold Cross-Validation
    - **Metrics:** 
        - Accuracy: 96.2%
        - ROC-AUC: 0.99
    
    #### 🛠️ Data Pipeline
    - **Preprocessing:** Log1p Skewness Correction
    - **Scaling:** RobustScaler (IQR based)
    - **Environment:** Streamlit Engine v1.x
    """)
    
    st.divider()
    st.caption("Developed by DB Moktan | AI Engineering Research")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #8892b0;'>© 2026 IgnisGuard Forest Intelligence Systems. All Rights Reserved.</p>", unsafe_allow_html=True)
