import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(page_title="Crop Insurance Fraud Detector", layout="wide")

# Title and Description for a layman
st.title("🛰️ Smart Crop Insurance Fraud Detection")
st.markdown("""
This system uses **Satellite Evidence** (NDVI, VCI) and **Weather Data** to verify if an insurance claim matches 
the actual ground conditions. It helps identify claims that might be exaggerated or fake.
""")

# Sidebar for Input
st.sidebar.header("Enter Claim Details")

def user_input_features():
    district = st.sidebar.selectbox("District", ("Anantapur", "Chittoor", "Guntur", "Krishna", "Kurnool", "East Godavari"))
    crop = st.sidebar.selectbox("Crop Type", ("Rice", "Maize", "Chilli", "Sugarcane", "Cotton"))
    area = st.sidebar.number_input("Area (Hectares)", min_value=0.1, max_value=50.0, value=2.0)
    loss_pct = st.sidebar.slider("Claimed Loss Percentage", 0, 100, 20)
    
    # Satellite Indicators (VCI < 35 means drought, NDVI > 0.5 means healthy)
    st.sidebar.subheader("Satellite Evidence")
    ndvi = st.sidebar.slider("NDVI (Greenness Index)", 0.0, 1.0, 0.5)
    vci = st.sidebar.slider("VCI (Vegetation Condition)", 0, 100, 50)
    rainfall = st.sidebar.number_input("Recorded Rainfall (mm)", min_value=0, max_value=2500, value=800)
    
    data = {
        'area_hectares': area,
        'loss_percent': loss_pct,
        'ndvi': ndvi,
        'vci': vci,
        'chirps_rain_mm': rainfall,
        'district': district,
        'crop': crop
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Logic for results display (Layman-friendly)
st.subheader("Analysis Result")

# We use the 'Ensemble Score' logic from your code for the verdict
# Simplified for the interface:
vci_val = input_df['vci'].iloc[0]
loss_val = input_df['loss_percent'].iloc[0]
ndvi_val = input_df['ndvi'].iloc[0]

risk_msg = ""
status = "GENUINE"
color = "green"

# Fraud type matching from your notebook rules
if loss_val >= 20:
    if vci_val > 60:
        status = "HIGH RISK"
        risk_msg = "🚨 **Fake Drought Alert:** The farmer claims high loss, but satellite data shows healthy, green vegetation."
        color = "red"
    elif ndvi_val > 0.55:
        status = "MEDIUM RISK"
        risk_msg = "⚠️ **Hidden Yield Alert:** The claimed failure doesn't match the high 'greenness' seen from space."
        color = "orange"
else:
    status = "GENUINE"
    risk_msg = "✅ The claim appears consistent with environmental conditions."

st.metric(label="Claim Status", value=status)
st.info(risk_msg)

# Visualizing for the user
col1, col2 = st.columns(2)

with col1:
    st.write("### How your claim compares")
    # Simple bar chart for Loss vs. Health
    comparison_data = pd.DataFrame({
        'Metric': ['Claimed Loss', 'Satellite Health (VCI)'],
        'Value': [loss_val, vci_val]
    })
    fig, ax = plt.subplots()
    sns.barplot(x='Metric', y='Value', data=comparison_data, palette=['red', 'green'], ax=ax)
    ax.set_ylim(0, 100)
    st.pyplot(fig)

with col2:
    st.write("### Key Indicators")
    st.write(f"**Area Size:** {input_df['area_hectares'].iloc[0]} Hectares")
    st.write(f"**District:** {input_df['district'].iloc[0]}")
    st.write(f"**Crop:** {input_df['crop'].iloc[0]}")
