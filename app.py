# ==========================================
# 🌾 SMART FRAUD DETECTION SYSTEM (ADVANCED)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import ee

st.set_page_config(layout="wide", page_title="Agri Fraud AI")

# ==========================================
# GEE FUNCTIONS
# ==========================================
def init_gee():
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()

def get_indices(lat, lon):
    point = ee.Geometry.Point([lon, lat])

    img = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterBounds(point)
        .filterDate('2020-01-01', '2025-01-01')
        .median()
    )

    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
    savi = img.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
        {'NIR': img.select('B8'), 'RED': img.select('B4')}
    ).rename('SAVI')

    vals = ee.Image.cat([ndvi, ndwi, savi]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=10
    ).getInfo()

    return vals

# ==========================================
# LOAD MODELS
# ==========================================
@st.cache_resource
def load_models():
    return (
        joblib.load("model.pkl"),
        joblib.load("scaler.pkl"),
        joblib.load("iso_model.pkl")
    )

rf, scaler, iso = load_models()

# ==========================================
# HEADER
# ==========================================
st.title("🛰️ AI Crop Insurance Fraud Detection System")
st.markdown("### Detect. Explain. Visualize. Prevent.")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🎛️ Controls")

file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
else:
    df = pd.read_csv("data/ap_synthetic_agri_insurance.csv")

# ==========================================
# GAME MODE HEADER
# ==========================================
st.subheader("🎮 Fraud Detective Mode")

score = 0

# ==========================================
# MODEL PREDICTION
# ==========================================
features = ['farm_size', 'rainfall', 'temperature', 'yield_loss', 'claim_amount']

if all(f in df.columns for f in features):

    X = df[features]
    X_scaled = scaler.transform(X)

    df['Fraud'] = rf.predict(X_scaled)
    df['Anomaly'] = iso.predict(X_scaled)

    # ==========================================
    # METRICS
    # ==========================================
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Claims", len(df))
    col2.metric("Frauds Detected", int(df['Fraud'].sum()))
    col3.metric("Anomalies", int((df['Anomaly'] == -1).sum()))

    # ==========================================
    # HEATMAP (INDIA STYLE)
    # ==========================================
    if 'latitude' in df.columns and 'longitude' in df.columns:

        st.subheader("🇮🇳 Fraud Risk Heatmap")

        fig = px.density_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            z="Fraud",
            radius=10,
            zoom=5,
            mapbox_style="carto-positron"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # CLAIM GRAPH
    # ==========================================
    st.subheader("📊 Claim Pattern Analysis")

    fig2 = px.scatter(
        df,
        x="yield_loss",
        y="claim_amount",
        color="Fraud",
        size="farm_size",
        hover_data=df.columns
    )

    st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# 🧠 AI EXPLANATION ENGINE
# ==========================================
st.subheader("🧠 Why is this Fraud? (AI Explanation)")

def explain(row):
    reasons = []

    if row['yield_loss'] > 50:
        reasons.append("High Yield Loss")

    if row['claim_amount'] > 100000:
        reasons.append("High Claim Amount")

    if row['rainfall'] < 20:
        reasons.append("Low Rainfall (suspicious claim)")

    return ", ".join(reasons)

if 'Fraud' in df.columns:
    df['Reason'] = df.apply(explain, axis=1)

    st.dataframe(df[['Fraud', 'Reason']].head(10))

# ==========================================
# 🎮 USER FRAUD GAME
# ==========================================
st.subheader("🎮 Can YOU Detect Fraud?")

sample = df.sample(1).iloc[0]

st.write("Farm Size:", sample['farm_size'])
st.write("Rainfall:", sample['rainfall'])
st.write("Yield Loss:", sample['yield_loss'])
st.write("Claim Amount:", sample['claim_amount'])

user_guess = st.radio("Is this Fraud?", ["Yes", "No"])

if st.button("Check Answer"):

    correct = sample['Fraud']

    if (user_guess == "Yes" and correct == 1) or (user_guess == "No" and correct == 0):
        st.success("🎉 Correct!")
    else:
        st.error("❌ Wrong!")

# ==========================================
# 🛰️ SATELLITE SECTION
# ==========================================
st.subheader("🛰️ Satellite Intelligence")

lat = st.number_input("Latitude", value=15.9)
lon = st.number_input("Longitude", value=79.7)

if st.button("Get Satellite Data"):
    init_gee()
    data = get_indices(lat, lon)

    st.metric("NDVI 🌱", round(data.get('NDVI', 0), 3))
    st.metric("NDWI 💧", round(data.get('NDWI', 0), 3))
    st.metric("SAVI 🌿", round(data.get('SAVI', 0), 3))

# ==========================================
# 📈 NDVI TREND (SIMULATED TIME SERIES)
# ==========================================
st.subheader("📈 NDVI Trend Over Time")

dates = pd.date_range(start="2020-01-01", periods=12, freq="M")
ndvi_values = np.random.uniform(0.2, 0.8, 12)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=dates, y=ndvi_values, mode='lines+markers'))

st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# DOWNLOAD
# ==========================================
st.subheader("⬇️ Export Results")

st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    "fraud_results.csv"
)

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown("🚀 Built with ML + Satellite + Streamlit | Final Year Project Ready")
