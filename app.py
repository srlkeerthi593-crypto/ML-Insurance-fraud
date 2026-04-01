# ============================================================
# 🌾 AGRI SHIELD — FINAL CLEAN VERSION
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="AgriShield", layout="wide")

st.title("🌾 AgriShield — Fraud Detection System")

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv("data/ap_synthetic_agri_insurance.csv")
sat = pd.read_csv("data/AP_satellite_indices.csv")

df["year"] = df["year"].astype(int)
sat["year"] = sat["year"].astype(int)

df = df.merge(sat, on=["district","year"], how="left")

# ============================================================
# FEATURES
# ============================================================
df["claim_ratio"] = df["claim_amount_rs"] / (df["insurance_premium_rs"] + 1)

le = LabelEncoder()
df["crop_encoded"] = le.fit_transform(df["crop"])

features = [
    "area_hectares","loss_percent","claim_ratio",
    "rainfall_mm","ndvi","ndwi","evi","chirps_rain_mm","crop_encoded"
]

X = df[features].fillna(df[features].median())
y = df["fraud_label"]

# ============================================================
# TRAIN MODELS
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Autoencoder
auto = MLPRegressor(hidden_layer_sizes=(16,8,16), max_iter=200)
auto.fit(X_train, X_train)

# ============================================================
# PREDICTIONS
# ============================================================
gb_preds = gb.predict(X_test)
gb_proba = gb.predict_proba(X_test)[:,1]

recon = auto.predict(X_test)
error = np.mean((X_test - recon)**2, axis=1)

threshold = np.percentile(error, 95)

# Full dataset
df["gb_score"] = gb.predict_proba(X_scaled)[:,1]

recon_full = auto.predict(X_scaled)
err_full = np.mean((X_scaled - recon_full)**2, axis=1)

df["auto_flag"] = (err_full > threshold).astype(int)

df["final_score"] = 0.7 * df["gb_score"] + 0.3 * df["auto_flag"]

def verdict(score):
    if score > 0.7: return "HIGH RISK 🚨"
    elif score > 0.4: return "MEDIUM RISK ⚠️"
    else: return "GENUINE ✅"

df["verdict"] = df["final_score"].apply(verdict)

# ============================================================
# METRICS
# ============================================================
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy_score(y_test, gb_preds)*100:.2f}%")
col2.metric("AUC", f"{roc_auc_score(y_test, gb_proba):.3f}")
col3.metric("F1 Score", f"{f1_score(y_test, gb_preds):.3f}")

# ============================================================
# SIMPLE ML EXPLANATION
# ============================================================
st.subheader("🤖 How AI Works (Simple)")

st.markdown("""
🧠 **Gradient Boosting**
- Learns patterns step by step  
- Corrects mistakes  
- Predicts fraud  

🔍 **Autoencoder**
- Learns what normal claims look like  
- If something unusual → flags it  

👉 Both combined = final fraud decision
""")

# ============================================================
# SATELLITE EXPLANATION
# ============================================================
st.subheader("🛰️ Satellite Evidence (Easy Explanation)")

st.markdown("""
Think of satellite like a **camera in the sky**

🌱 NDVI → How green crops are  
- High = healthy  
- Low = damaged  

💧 NDWI → Water  
- High = flood  
- Low = dry  

🌧️ Rainfall → actual rain  

---

### 🚨 Fraud Logic

❌ Farmer says crop failed  
✅ Satellite shows green crops  
👉 Fraud  

❌ Farmer says flood  
✅ No water seen  
👉 Fraud  

❌ Farmer says no rain  
✅ Rain was normal  
👉 Fraud  
""")

# ============================================================
# VISUALIZATION
# ============================================================
st.subheader("📈 Fraud Score Distribution")

fig = px.histogram(df, x="final_score", color="fraud_label", nbins=50)
st.plotly_chart(fig, use_container_width=True)

st.markdown("🔴 High score → Fraud | 🟢 Low score → Genuine")

# ============================================================
# MAP
# ============================================================
if "latitude" in df.columns:
    st.subheader("🗺️ Fraud Map")

    fig_map = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="final_score",
        size="claim_amount_rs",
        zoom=5,
        mapbox_style="carto-positron"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ============================================================
# TABLE
# ============================================================
st.subheader("📋 Results")
st.dataframe(df.head(50))

# ============================================================
# DOWNLOAD
# ============================================================
st.download_button(
    "Download Results",
    df.to_csv(index=False),
    "fraud_results.csv"
)
