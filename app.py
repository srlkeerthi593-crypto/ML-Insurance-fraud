import streamlit as st
import pandas as pd
import ee
import geemap.foliumap as geemap

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="Agri Fraud Detection", layout="wide")

st.title("🌾 Satellite-Based Crop Insurance Fraud Detection")

# ─────────────────────────────────────────
# GEE INIT
# ─────────────────────────────────────────
try:
    ee.Initialize(project='metal-imprint-487813-d9')
except:
    ee.Authenticate()
    ee.Initialize(project='metal-imprint-487813-d9')

# ─────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload Insurance Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully")

    # ─────────────────────────────────────────
    # DISTRICT SELECT
    # ─────────────────────────────────────────
    districts = df['district_clean'].unique()
    district = st.selectbox("📍 Select District", districts)

    year = st.slider("📅 Select Year", 2019, 2025, 2022)

    # ─────────────────────────────────────────
    # DISTRICT COORDS
    # ─────────────────────────────────────────
    coords = {
        'Kurnool': (15.8281, 78.0373),
        'Anantapur': (14.6819, 77.6006),
        'Prakasam': (15.3520, 79.5740),
        'Nellore': (14.4426, 79.9865),
        'Guntur': (16.3067, 80.4365),
    }

    lat, lon = coords[district]

    # ─────────────────────────────────────────
    # NDVI MAP
    # ─────────────────────────────────────────
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(30000)

    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterBounds(region)
           .filterDate(f"{year}-06-01", f"{year}-10-31")
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))

    ndvi = col.map(lambda img: img.normalizedDifference(['B8','B4']).rename('NDVI')).median()

    Map = geemap.Map(center=[lat, lon], zoom=9)
    Map.addLayer(ndvi, {'min':0, 'max':1, 'palette':['red','yellow','green']}, 'NDVI')

    st.subheader("🛰️ NDVI Map (Crop Health)")
    Map.to_streamlit(height=500)

    # ─────────────────────────────────────────
    # FILTER DATA
    # ─────────────────────────────────────────
    df_sel = df[(df['district_clean']==district) & (df['year']==year)]

    if len(df_sel) > 0:

        row = df_sel.iloc[0]

        # Fake logic (replace with your real features if needed)
        score = 0

        if row['loss_percent'] > 50:
            score += 2
        if row['claim_amount_rs'] > 50000:
            score += 2
        if row['area_hectares'] < 2:
            score += 1

        # ─────────────────────────────────────────
        # CLASSIFICATION
        # ─────────────────────────────────────────
        if score >= 4:
            verdict = "🔴 HIGH RISK"
            reason = "High claim with suspicious patterns"
        elif score >= 2:
            verdict = "🟠 MEDIUM RISK"
            reason = "Moderate inconsistency"
        else:
            verdict = "🟢 GENUINE"
            reason = "Looks normal"

        # ─────────────────────────────────────────
        # DISPLAY
        # ─────────────────────────────────────────
        st.subheader("📊 Fraud Analysis")

        col1, col2, col3 = st.columns(3)

        col1.metric("Loss %", f"{row['loss_percent']:.1f}")
        col2.metric("Claim ₹", f"{row['claim_amount_rs']:.0f}")
        col3.metric("Area (ha)", f"{row['area_hectares']:.2f}")

        st.markdown(f"### 🧠 Verdict: {verdict}")
        st.info(f"Reason: {reason}")

    else:
        st.warning("No data available for selection")
