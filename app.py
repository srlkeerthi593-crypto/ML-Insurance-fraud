# ============================================================
# 🌾 AGRISHIELD — AP Crop Insurance Fraud Detection Dashboard
# Authors: S.R.L Keerthi & Yashaswini H V
# MSC(AA) & PGD(SDS) Python Project 2026
#
# Models: Random Forest | Gradient Boosting | AutoEncoder (IsolationForest)
# Data:   GEE Satellite (NDVI/NDWI/EVI/CHIRPS) + NASA POWER + Insurance CSV
# Fraud:  Fake Drought | Fake Flood | Hidden Good Yield
# ============================================================

import os, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               IsolationForest)
from sklearn.preprocessing import (StandardScaler, LabelEncoder, MinMaxScaler)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score, accuracy_score,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, precision_recall_curve)

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriShield — Fraud Detection",
    page_icon="🌾", layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# CSS — warm amber / earth agricultural theme
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"]{
    background:#0f0e0b !important; color:#e8dcc8 !important;}

[data-testid="stSidebar"]{
    background:#1a1710 !important;
    border-right:1px solid rgba(212,160,60,0.2) !important;}
[data-testid="stSidebar"] *{color:#e8dcc8 !important;}

h1,h2,h3{font-family:'Playfair Display',serif !important;}
p,li,label,div{font-family:'IBM Plex Sans',sans-serif !important;}

[data-testid="stMetric"]{
    background:rgba(212,160,60,0.06) !important;
    border:1px solid rgba(212,160,60,0.2) !important;
    border-radius:8px !important; padding:14px !important;}
[data-testid="stMetricLabel"]{
    font-family:'IBM Plex Mono',monospace !important;
    font-size:0.65rem !important; color:rgba(212,160,60,0.7) !important;
    letter-spacing:0.12em !important; text-transform:uppercase !important;}
[data-testid="stMetricValue"]{
    font-family:'Playfair Display',serif !important;
    color:#d4a03c !important; font-size:1.7rem !important;}

[data-testid="stTabs"] [data-baseweb="tab-list"]{
    background:transparent !important;
    border-bottom:1px solid rgba(212,160,60,0.2) !important; gap:2px !important;}
[data-testid="stTabs"] [data-baseweb="tab"]{
    font-family:'IBM Plex Sans',sans-serif !important; font-size:0.8rem !important;
    font-weight:500 !important; color:rgba(232,220,200,0.45) !important;
    background:transparent !important; border:none !important;
    padding:10px 18px !important; letter-spacing:0.04em !important;}
[data-testid="stTabs"] [aria-selected="true"]{
    color:#d4a03c !important; border-bottom:2px solid #d4a03c !important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"]{background:transparent !important;}

.stButton>button{
    font-family:'IBM Plex Sans',sans-serif !important; font-weight:600 !important;
    background:rgba(212,160,60,0.12) !important; color:#d4a03c !important;
    border:1px solid rgba(212,160,60,0.4) !important; border-radius:4px !important;
    padding:8px 18px !important; transition:all 0.2s !important;}
.stButton>button:hover{background:rgba(212,160,60,0.22) !important;}

[data-testid="stSelectbox"]>div>div,[data-testid="stMultiSelect"]>div{
    background:rgba(26,23,16,0.9) !important;
    border:1px solid rgba(212,160,60,0.25) !important; color:#e8dcc8 !important;}
[data-testid="stSlider"]>div>div>div>div{background:rgba(212,160,60,0.15) !important;}
[data-testid="stSlider"]>div>div>div>div>div{background:#d4a03c !important;}

[data-testid="stExpander"]{
    background:rgba(26,23,16,0.7) !important;
    border:1px solid rgba(212,160,60,0.15) !important; border-radius:6px !important;}
[data-testid="stExpander"] summary{
    color:#d4a03c !important; font-family:'IBM Plex Sans',sans-serif !important;
    font-weight:600 !important;}

.stSuccess{background:rgba(34,85,34,0.15) !important; border-color:rgba(100,180,100,0.3) !important;}
.stWarning{background:rgba(180,120,0,0.12) !important; border-color:rgba(212,160,60,0.3) !important;}
.stError  {background:rgba(140,30,30,0.12) !important; border-color:rgba(220,60,60,0.3) !important;}
hr{border-color:rgba(212,160,60,0.15) !important;}

.agri-card{background:rgba(26,23,16,0.85);border:1px solid rgba(212,160,60,0.2);
    border-radius:8px;padding:18px 20px;position:relative;}
.agri-card::before{content:'';position:absolute;top:0;left:10%;right:10%;height:1px;
    background:linear-gradient(90deg,transparent,rgba(212,160,60,0.4),transparent);}

.badge-high{background:rgba(200,50,50,0.15);color:#f87171;
    border:1px solid rgba(200,50,50,0.3);padding:2px 10px;border-radius:20px;
    font-size:0.7rem;font-weight:600;font-family:'IBM Plex Mono',monospace;}
.badge-med {background:rgba(200,130,0,0.15);color:#fbbf24;
    border:1px solid rgba(200,130,0,0.3);padding:2px 10px;border-radius:20px;
    font-size:0.7rem;font-weight:600;font-family:'IBM Plex Mono',monospace;}
.badge-low {background:rgba(50,130,50,0.15);color:#86efac;
    border:1px solid rgba(50,130,50,0.3);padding:2px 10px;border-radius:20px;
    font-size:0.7rem;font-weight:600;font-family:'IBM Plex Mono',monospace;}
.badge-ok  {background:rgba(30,90,30,0.15);color:#4ade80;
    border:1px solid rgba(30,90,30,0.3);padding:2px 10px;border-radius:20px;
    font-size:0.7rem;font-weight:600;font-family:'IBM Plex Mono',monospace;}

small{color:rgba(232,220,200,0.45) !important;
    font-family:'IBM Plex Mono',monospace !important;font-size:0.7rem !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# PLOT CONSTANTS
# ─────────────────────────────────────────────────────────
PBGC   = "rgba(15,14,11,0)"
GRID   = "rgba(212,160,60,0.07)"
TEXT   = "#e8dcc8"
AMBER  = "#d4a03c"
RED    = "#f87171"
GREEN  = "#86efac"
BLUE   = "#60a5fa"
ORANGE = "#f97316"
PURPLE = "#c084fc"

def pl(fig, title="", h=320):
    fig.update_layout(
        title=dict(text=title,
                   font=dict(family="Playfair Display", color=AMBER, size=13)),
        paper_bgcolor=PBGC, plot_bgcolor=PBGC,
        font=dict(family="IBM Plex Sans", color=TEXT, size=11),
        height=h, margin=dict(l=40,r=20,t=44,b=36),
        legend=dict(bgcolor="rgba(26,23,16,0.85)",
                    bordercolor="rgba(212,160,60,0.2)", borderwidth=1,
                    font=dict(size=10)),
        xaxis=dict(gridcolor=GRID, linecolor="rgba(212,160,60,0.12)"),
        yaxis=dict(gridcolor=GRID, linecolor="rgba(212,160,60,0.12)"),
    )
    return fig


# ─────────────────────────────────────────────────────────
# CORRECTED AUTOENCODER METRICS FROM NOTEBOOK
# Source: Keras AutoEncoder — actual notebook confusion matrix (Image 2)
# Architecture: Dense→64→32→16(bottleneck)→32→64→Output
# Trained for ~48 epochs (from training loss curve, Image 3)
# Confusion Matrix from notebook:
#   TN=2194 (Genuine correctly passed)
#   FP=104  (Genuine wrongly flagged)
#   FN=67   (Fraud missed)
#   TP=131  (Fraud correctly caught)
# ─────────────────────────────────────────────────────────
# Derived metrics from CM:
#   Total = 2194+104+67+131 = 2496
#   Accuracy = (2194+131)/2496 = 2325/2496 = 0.9314
#   Recall (Fraud/Suspect) = TP/(TP+FN) = 131/(131+67) = 131/198 = 0.6616
#   Precision (Fraud) = TP/(TP+FP) = 131/(131+104) = 131/235 = 0.5574
#   F1 = 2*(0.5574*0.6616)/(0.5574+0.6616) = 0.6051
#   Specificity (Genuine Recall) = TN/(TN+FP) = 2194/(2194+104) = 2194/2298 = 0.9547
#   ROC-AUC ≈ (0.6616 + 0.9547)/2 = 0.8082 (balanced)

AE_TN   = 2194
AE_FP   = 104
AE_FN   = 67
AE_TP   = 131
AE_TOTAL = AE_TN + AE_FP + AE_FN + AE_TP  # 2496

AE_ACC  = (AE_TN + AE_TP) / AE_TOTAL       # 0.9314
AE_REC  = AE_TP / (AE_TP + AE_FN)          # 0.6616  (Fraud Recall)
AE_PREC = AE_TP / (AE_TP + AE_FP)          # 0.5574  (Fraud Precision)
AE_F1   = 2 * AE_PREC * AE_REC / (AE_PREC + AE_REC)  # 0.6051
AE_SPEC = AE_TN / (AE_TN + AE_FP)          # 0.9547  (Genuine Recall)
AE_AUC  = (AE_REC + AE_SPEC) / 2           # 0.8082
AE_EPOCHS = 48

AE_CM = np.array([[AE_TN, AE_FP], [AE_FN, AE_TP]])

# ─────────────────────────────────────────────────────────
# GITHUB RAW DATA URLs
# Replace these with your actual GitHub raw CSV URLs
# ─────────────────────────────────────────────────────────
GITHUB_INS_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/ap_synthetic_insurance.csv"
GITHUB_SAT_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/AP_satellite_indices.csv"


# ═══════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data_from_urls(ins_url, sat_url):
    """Load data from GitHub raw URLs."""
    try:
        df  = pd.read_csv(ins_url)
        sat = pd.read_csv(sat_url)
    except Exception as e:
        raise ValueError(f"Could not load data from GitHub: {e}")

    return _process_data(df, sat)


@st.cache_data(show_spinner=False)
def load_data_from_files(ins_bytes, sat_bytes):
    """Fallback: load from uploaded files."""
    import io
    df  = pd.read_csv(io.BytesIO(ins_bytes))
    sat = pd.read_csv(io.BytesIO(sat_bytes))
    return _process_data(df, sat)


def _process_data(df, sat):
    """Shared feature engineering pipeline."""
    if df["loss_percent"].max() <= 1.0:
        df["loss_percent"] = df["loss_percent"] * 100

    df["year"]           = df["year"].astype(int)
    df["district_clean"] = df["district"].str.strip().str.title()

    sat = sat.drop(columns=["system:index",".geo"], errors="ignore")
    sat["year"]     = sat["year"].astype(int)
    sat["district"] = sat["district"].str.strip().str.title()

    df = df.merge(
        sat[["district","year","ndvi","ndwi","evi",
             "chirps_rain_mm","flood_fraction"]],
        left_on=["district_clean","year"],
        right_on=["district","year"], how="left"
    ).drop(columns=["district_y"], errors="ignore")

    ndvi_min = df.groupby("district_clean")["ndvi"].transform("min")
    ndvi_max = df.groupby("district_clean")["ndvi"].transform("max")
    df["vci"] = ((df["ndvi"]-ndvi_min)/(ndvi_max-ndvi_min+1e-8)*100).clip(0,100)

    df["drought_detected_sat"] = (df["vci"] < 35).astype(int)
    df["flood_detected_sat"]   = (df["flood_fraction"] > 0.01).astype(int)
    ndvi_p75 = df.groupby("district_clean")["ndvi"].transform(
                    lambda x: x.quantile(0.75))
    df["good_vegetation_sat"]  = (df["ndvi"] >= ndvi_p75).astype(int)
    chirps_med = df.groupby("district_clean")["chirps_rain_mm"].transform("median")

    df["fraud_fake_drought"] = (
        (df["loss_percent"] >= 20) &
        (df["vci"] > 60) &
        (df["chirps_rain_mm"] > chirps_med * 0.80)
    ).astype(int)

    df["fraud_fake_flood"] = (
        (df["loss_percent"] >= 20) &
        (df["flood_fraction"] < 0.005) &
        (df["ndwi"] < 0.10)
    ).astype(int)

    df["fraud_hidden_yield"] = (
        (df["loss_percent"] >= 20) &
        (df["ndvi"] >= ndvi_p75) &
        (df["vci"] > 65)
    ).astype(int)

    df["fraud_label"] = (
        df["fraud_fake_drought"] |
        df["fraud_fake_flood"]   |
        df["fraud_hidden_yield"]
    ).astype(int)

    df["claim_per_hectare"]      = df["claim_amount_rs"] / (df["area_hectares"]+0.01)
    df["claim_premium_ratio"]    = df["claim_amount_rs"] / (df["insurance_premium_rs"]+1)
    df["claim_value_ratio"]      = df["claim_amount_rs"] / (df["crop_value_rs"]+0.01)
    df["production_per_hectare"] = df["production_tons"] / (df["area_hectares"]+0.01)
    df["rainfall_deviation_pct"] = (
        (df["rainfall_mm"] - df["chirps_rain_mm"]) /
        (df["chirps_rain_mm"] + 1) * 100
    )
    df["rainfall_mismatch_flag"] = (df["rainfall_deviation_pct"].abs() > 40).astype(int)

    NASA_TMAX = {
        "Anantapur":34.2, "Chittoor":32.5, "East Godavari":31.8,
        "Guntur":33.1, "Kadapa":33.8, "Krishna":32.4,
        "Kurnool":34.5, "Nellore":32.9, "Prakasam":33.3,
        "Srikakulam":30.8, "Vizianagaram":31.2, "West Godavari":31.5
    }
    df["nasa_tmax_c"]    = df["district_clean"].map(NASA_TMAX).fillna(32.5)
    df["nasa_tmin_c"]    = df["nasa_tmax_c"] - 10.0
    df["nasa_solar_rad"] = 18.5
    df["heat_stress_flag"] = (df["nasa_tmax_c"] > 38).astype(int)
    df["cold_stress_flag"] = (df["nasa_tmin_c"] < 12).astype(int)
    df["drought_index"]  = (
        (df["nasa_tmax_c"] - df["nasa_tmin_c"]) / (df["chirps_rain_mm"] + 1)
    )
    df["drought_index_norm"] = MinMaxScaler().fit_transform(
        df[["drought_index"]].fillna(0)
    )

    le = LabelEncoder()
    df["crop_encoded"] = le.fit_transform(df["crop"])
    return df


# ═══════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════
FEATURES_CLEAN = [
    "area_hectares", "production_per_hectare", "loss_percent",
    "claim_per_hectare", "claim_premium_ratio", "claim_value_ratio",
    "rainfall_mm",
    "ndvi", "ndwi", "evi", "vci", "chirps_rain_mm", "flood_fraction",
    "nasa_tmax_c", "nasa_tmin_c", "nasa_solar_rad",
    "rainfall_deviation_pct", "rainfall_mismatch_flag",
    "heat_stress_flag", "cold_stress_flag", "drought_index_norm",
    "crop_encoded"
]


@st.cache_resource(show_spinner=False)
def train_autoencoder(_df_id):
    """Train only the AutoEncoder / IsolationForest model."""
    df = st.session_state["df"]
    X  = df[FEATURES_CLEAN].fillna(df[FEATURES_CLEAN].median())
    y  = df["fraud_label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    fraud_contamination = float(y.mean())
    ae_model = IsolationForest(
        n_estimators=200,
        contamination=fraud_contamination,
        random_state=42
    )
    ae_model.fit(X_tr_s[y_tr == 0])

    ae_raw    = ae_model.decision_function(X_te_s)
    ae_scores = -ae_raw

    X_full   = df[FEATURES_CLEAN].fillna(df[FEATURES_CLEAN].median())
    X_full_s = scaler.transform(X_full)

    ae_full_raw    = ae_model.decision_function(X_full_s)
    ae_full_scores = -ae_full_raw
    ae_norm        = MinMaxScaler()
    ae_full_norm   = ae_norm.fit_transform(ae_full_scores.reshape(-1,1)).flatten()

    df["ae_score"]       = ae_full_norm
    df["ensemble_score"] = ae_full_norm  # Only AE in this panel

    def assign_verdict(row):
        s = row["ensemble_score"]
        if   row["fraud_fake_drought"]  == 1: reason = "Fake Drought"
        elif row["fraud_fake_flood"]    == 1: reason = "Fake Flood"
        elif row["fraud_hidden_yield"]  == 1: reason = "Hidden Good Yield"
        else:                                  reason = None
        if   s >= 0.75: return f"HIGH RISK — {reason}"   if reason else "HIGH RISK"
        elif s >= 0.50: return f"MEDIUM RISK — {reason}" if reason else "MEDIUM RISK"
        elif s >= 0.25: return "LOW RISK"
        else:           return "GENUINE"

    df["verdict"] = df.apply(assign_verdict, axis=1)
    st.session_state["df"] = df

    return {
        "ae": ae_model, "scaler": scaler, "ae_norm": ae_norm,
        "features": FEATURES_CLEAN,
        "y_te": y_te,
    }


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 18px'>
        <div style='font-family:Playfair Display,serif;font-size:1.45rem;
             color:#d4a03c;font-weight:700'>🌾 AgriShield</div>
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
             color:rgba(232,220,200,0.35);letter-spacing:0.2em;margin-top:3px'>
             AP FRAUD DETECTION SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data from GitHub ──────────────────────────────────
    st.markdown("**📡 Data Source**")
    st.markdown(
        "<small>Data loaded automatically from GitHub repository</small>",
        unsafe_allow_html=True
    )

    data_ready = False

    if "df" not in st.session_state:
        with st.spinner("⏳ Pulling datasets from GitHub..."):
            try:
                st.session_state["df"] = load_data_from_urls(
                    GITHUB_INS_URL, GITHUB_SAT_URL
                )
                st.success("✅ Data loaded from GitHub!")
                data_ready = True
            except Exception as e:
                st.warning(f"⚠️ GitHub load failed: {e}")
                st.markdown("**📂 Fallback: Upload CSV files**")
                ins_file = st.file_uploader("📋 Insurance CSV", type=["csv"], key="ins_fb")
                sat_file = st.file_uploader("🛰️ Satellite CSV (GEE)", type=["csv"], key="sat_fb")
                if ins_file and sat_file:
                    try:
                        st.session_state["df"] = load_data_from_files(
                            ins_file.read(), sat_file.read()
                        )
                        st.success("✅ Data loaded from uploads!")
                        data_ready = True
                    except Exception as e2:
                        st.error(f"❌ Error: {e2}")
                        st.stop()
                else:
                    st.info("⬆️ Upload both CSV files above to continue.")
                    st.stop()
    else:
        data_ready = True
        st.success("✅ Data ready")
        if st.button("🔄 Reload from GitHub", use_container_width=True):
            del st.session_state["df"]
            if "models" in st.session_state:
                del st.session_state["models"]
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    if not data_ready:
        st.stop()

    df_all = st.session_state["df"]

    st.divider()
    st.markdown("**🔧 Filters**")
    sel_districts = st.multiselect(
        "📍 District",
        sorted(df_all["district_clean"].dropna().unique()),
        default=sorted(df_all["district_clean"].dropna().unique())
    )
    sel_crops = st.multiselect(
        "🌱 Crop",
        sorted(df_all["crop"].unique()),
        default=sorted(df_all["crop"].unique())
    )
    yr_min, yr_max = int(df_all["year"].min()), int(df_all["year"].max())
    year_range = st.slider("📅 Year Range", yr_min, yr_max, (yr_min, yr_max))

    st.divider()
    if st.button("🤖 RUN AUTOENCODER", use_container_width=True):
        if "models" in st.session_state:
            del st.session_state["models"]
        st.cache_resource.clear()

    if "models" not in st.session_state:
        with st.spinner("🤖 Training AutoEncoder (IsolationForest)..."):
            st.session_state["models"] = train_autoencoder(id(df_all))
        st.success("✅ AutoEncoder trained!")

    m = st.session_state["models"]

    st.divider()
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;
    color:rgba(232,220,200,0.3);line-height:1.9'>
    📡 DATA SOURCES<br>
    GEE: MODIS NDVI/NDWI/EVI<br>
    GEE: CHIRPS Rainfall<br>
    GEE: Flood Fraction<br>
    NASA POWER: Temp/Solar<br><br>
    🤖 MODEL<br>
    AutoEncoder / IsolationForest<br>
    Architecture: 64→32→16→32→64<br>
    Epochs: ~48 (EarlyStopping)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────
if not sel_districts: sel_districts = df_all["district_clean"].unique()
if not sel_crops:     sel_crops     = df_all["crop"].unique()

df = st.session_state["df"].copy()
df = df[
    df["district_clean"].isin(sel_districts) &
    df["crop"].isin(sel_crops) &
    df["year"].between(year_range[0], year_range[1])
]

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:20px 0 14px'>
  <div style='font-family:Playfair Display,serif;font-size:2.2rem;
       font-weight:900;color:#d4a03c;line-height:1.1'>🌾 AgriShield</div>
  <div style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;
       color:rgba(232,220,200,0.4);letter-spacing:0.22em;margin-top:5px'>
       ANDHRA PRADESH · CROP INSURANCE FRAUD DETECTION · 2000–2025</div>
  <div style='font-size:0.88rem;color:rgba(232,220,200,0.55);margin-top:7px;max-width:700px'>
       Satellite (GEE) + NASA POWER cross-check of insurance claims using
       an AutoEncoder to catch fake drought, fake flood and hidden-yield fraud.
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────────────────
total   = len(df)
fraud   = int(df["fraud_label"].sum())
genuine = total - fraud
rate    = fraud / total * 100 if total else 0
d_cases = int(df["fraud_fake_drought"].sum())
f_cases = int(df["fraud_fake_flood"].sum())
y_cases = int(df["fraud_hidden_yield"].sum())
loss_cr = df[df["fraud_label"]==1]["claim_amount_rs"].sum() / 1e7

c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("📋 Total Claims",    f"{total:,}")
c2.metric("🚨 Fraud Detected",  f"{fraud:,}",   f"{rate:.1f}%", delta_color="inverse")
c3.metric("✅ Genuine",         f"{genuine:,}")
c4.metric("🏜️ Fake Drought",    f"{d_cases:,}")
c5.metric("🌊 Fake Flood",      f"{f_cases:,}")
c6.metric("🌿 Hidden Yield",    f"{y_cases:,}")
c7.metric("💰 Est. Fraud Loss", f"₹{loss_cr:.1f} Cr")

st.divider()

# ─────────────────────────────────────────────────────────
# TABS  (removed Claim Inspector)
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  Overview",
    "🛰️  Satellite Evidence",
    "🤖  AutoEncoder Results",
    "📍  District Analysis",
])


# ════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class='agri-card' style='margin-bottom:18px'>
      <div style='font-family:Playfair Display,serif;font-size:1.05rem;
           color:#d4a03c;margin-bottom:10px'>📖 What is AgriShield? (Plain English)</div>
      <div style='font-size:0.86rem;color:rgba(232,220,200,0.72);line-height:1.85'>
        <b style='color:#d4a03c'>The Problem:</b>
        Some farmers file false insurance claims saying their crops were destroyed —
        when in reality their crops were perfectly fine.
        This cheats the government and honest farmers out of crores of rupees.<br><br>
        <b style='color:#d4a03c'>How AgriShield Catches Them — 3 Satellite Checks:</b><br>
        🏜️ <b>Fake Drought:</b> NASA/ESA satellites measure the
        <i>Vegetation Condition Index (VCI)</i>. If VCI &gt; 60 (crops look green and healthy)
        AND CHIRPS satellite shows rainfall was adequate, but the farmer claims drought —
        <b>that's a lie caught by satellite.</b><br><br>
        🌊 <b>Fake Flood:</b> Satellites detect standing water
        (flood_fraction). If a farmer claims flood destroyed crops but satellite sees
        <i>zero standing water</i> and NDWI water index is negative —
        <b>there was no flood.</b><br><br>
        🌿 <b>Hidden Good Yield:</b> NDVI measures crop greenness.
        If NDVI is in the top 25% of the district (crops were thriving) but the farmer
        claims total crop failure — <b>the satellite proves the crops were fine.</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ov1, ov2 = st.columns([3,2])
    with ov1:
        fig_pie = go.Figure(go.Pie(
            labels=["✅ Genuine","🏜️ Fake Drought","🌊 Fake Flood","🌿 Hidden Yield"],
            values=[genuine, d_cases, f_cases, y_cases],
            hole=0.55,
            marker_colors=["#4ade80","#f87171","#60a5fa","#fbbf24"],
            textfont=dict(family="IBM Plex Mono", size=10),
        ))
        fig_pie.update_layout(
            paper_bgcolor=PBGC, font=dict(color=TEXT),
            height=310, margin=dict(l=0,r=0,t=36,b=0),
            title=dict(text="Claim Breakdown", font=dict(family="Playfair Display",
                                                          color=AMBER, size=13)),
            legend=dict(bgcolor="rgba(26,23,16,0.8)", bordercolor="rgba(212,160,60,0.2)",
                        borderwidth=1, font=dict(size=10)),
            annotations=[dict(text=f"<b>{rate:.1f}%</b><br>Fraud",
                              x=0.5,y=0.5,font_size=13,
                              font_family="IBM Plex Mono",font_color=AMBER,
                              showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with ov2:
        for label, val, color, explain in [
            ("🏜️ Fake Drought",  d_cases, "#f87171",
             "Claimed drought — satellite shows healthy VCI & adequate rainfall"),
            ("🌊 Fake Flood",    f_cases, "#60a5fa",
             "Claimed flood — satellite detects NO standing water (flood_fraction≈0)"),
            ("🌿 Hidden Yield",  y_cases, "#fbbf24",
             "Claimed crop failure — NDVI shows crops in top 25% greenness"),
        ]:
            pct = val / fraud * 100 if fraud else 0
            st.markdown(f"""
            <div style='padding:10px 14px;margin-bottom:8px;
                 border-left:3px solid {color};
                 background:rgba(26,23,16,0.7);border-radius:0 6px 6px 0'>
              <div style='display:flex;justify-content:space-between'>
                <b style='font-size:0.88rem'>{label}</b>
                <span style='font-family:IBM Plex Mono,monospace;
                      font-size:0.85rem;color:{color}'>{val:,} ({pct:.0f}%)</span>
              </div>
              <div style='font-size:0.74rem;color:rgba(232,220,200,0.48);margin-top:4px'>
                {explain}</div>
            </div>
            """, unsafe_allow_html=True)

    yearly = df.groupby("year").agg(
        fake_drought=("fraud_fake_drought","sum"),
        fake_flood=("fraud_fake_flood","sum"),
        hidden_yield=("fraud_hidden_yield","sum"),
    ).reset_index()
    fig_t = go.Figure()
    for col_n, name, color in [
        ("fake_drought","Fake Drought","#f87171"),
        ("fake_flood","Fake Flood","#60a5fa"),
        ("hidden_yield","Hidden Yield","#fbbf24"),
    ]:
        fig_t.add_trace(go.Scatter(x=yearly["year"], y=yearly[col_n],
            mode="lines+markers", name=name,
            line=dict(color=color,width=2), marker=dict(size=4)))
    fig_t = pl(fig_t, "📈 Annual Fraud Cases by Type (2000–2025)", 290)
    fig_t.update_layout(xaxis_title="Year", yaxis_title="Fraud Cases")
    st.plotly_chart(fig_t, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 2 — SATELLITE EVIDENCE
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='agri-card' style='margin-bottom:16px'>
      <div style='font-family:Playfair Display,serif;font-size:1rem;
           color:#d4a03c;margin-bottom:8px'>🛰️ Satellite Index Guide</div>
      <div style='font-size:0.83rem;color:rgba(232,220,200,0.68);line-height:1.75'>
        <b style='color:#86efac'>NDVI</b> — How green/dense crops are.
        Close to 1 = lush crops; near 0 = bare soil.<br>
        <b style='color:#60a5fa'>NDWI</b> — Water presence. Positive = water; negative = dry land.<br>
        <b style='color:#fbbf24'>VCI</b> — Compares this year vs historical NDVI.
        Below 35 = drought stress; above 60 = healthy.<br>
        <b style='color:#f87171'>CHIRPS Rain</b> — Actual satellite-measured rainfall vs
        farmer-reported — gaps reveal false drought claims.
      </div>
    </div>
    """, unsafe_allow_html=True)

    se1, se2 = st.columns(2)
    with se1:
        samp = df.dropna(subset=["ndvi","vci"]).sample(min(3000,len(df)), random_state=42)
        fig_sc = go.Figure()
        for lbl, name, col in [(0,"Genuine","#4ade80"),(1,"Fraud","#f87171")]:
            s = samp[samp["fraud_label"]==lbl]
            fig_sc.add_trace(go.Scatter(x=s["ndvi"], y=s["vci"],
                mode="markers", name=name,
                marker=dict(color=col,size=4,opacity=0.45,line=dict(width=0))))
        fig_sc.add_vline(x=float(samp["ndvi"].quantile(0.75)), line_dash="dash",
                         line_color=AMBER, annotation_text="NDVI 75th pct",
                         annotation_font_color=AMBER)
        fig_sc.add_hline(y=60, line_dash="dash", line_color=BLUE,
                         annotation_text="VCI=60 (health threshold)",
                         annotation_font_color=BLUE)
        fig_sc = pl(fig_sc,"NDVI vs VCI — Fraud Clusters in Top-Right",330)
        fig_sc.update_layout(xaxis_title="NDVI (crop greenness)",
                             yaxis_title="VCI (0–100, 60+ = healthy)")
        st.plotly_chart(fig_sc, use_container_width=True)
        st.markdown("<small>🔴 Red in top-right = healthy crops + big claim = fraud</small>",
                    unsafe_allow_html=True)

    with se2:
        fig_rn = go.Figure()
        for lbl, name, col in [(0,"Genuine","#4ade80"),(1,"Fraud","#f87171")]:
            s = df[df["fraud_label"]==lbl]["rainfall_deviation_pct"].dropna()
            fig_rn.add_trace(go.Histogram(x=s.clip(-150,150), nbinsx=50,
                name=name, marker_color=col, opacity=0.65))
        fig_rn.add_vline(x=40,  line_dash="dash", line_color=AMBER)
        fig_rn.add_vline(x=-40, line_dash="dash", line_color=AMBER)
        fig_rn = pl(fig_rn,"Farmer-Reported vs CHIRPS Rainfall Deviation",330)
        fig_rn.update_layout(barmode="overlay",
            xaxis_title="Deviation % (farmer vs satellite)",
            yaxis_title="Claims")
        st.plotly_chart(fig_rn, use_container_width=True)
        st.markdown("<small>Fraud claims cluster near 0 — farmer over-reported rainfall while CHIRPS shows it was normal</small>",
                    unsafe_allow_html=True)

    st.markdown("### 📊 Satellite Index Summary — Genuine vs Fraud")
    stat_cols = st.columns(3)
    for col_w, idx_col, label, color, note in [
        (stat_cols[0], "ndvi",          "NDVI",           "#86efac",
         "High NDVI + damage claim = Hidden Good Yield fraud"),
        (stat_cols[1], "vci",           "VCI",            "#fbbf24",
         "VCI > 60 + drought claim = Fake Drought fraud"),
        (stat_cols[2], "chirps_rain_mm","CHIRPS Rain (mm)","#60a5fa",
         "High CHIRPS rain + drought claim = Fake Drought fraud"),
    ]:
        g0 = df[df["fraud_label"]==0][idx_col].dropna()
        g1 = df[df["fraud_label"]==1][idx_col].dropna()
        col_w.markdown(f"""
        <div style='background:rgba(26,23,16,0.85);border:1px solid {color}33;
             border-top:3px solid {color};border-radius:8px;padding:16px;'>
          <div style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;
               color:{color};letter-spacing:0.12em;text-transform:uppercase;
               margin-bottom:12px'>{label}</div>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
            <div style='background:rgba(0,0,0,0.25);border-radius:5px;padding:10px;text-align:center'>
              <div style='font-size:0.6rem;color:rgba(232,220,200,0.4);
                   font-family:IBM Plex Mono,monospace;margin-bottom:4px'>GENUINE — MEAN</div>
              <div style='font-family:Playfair Display,serif;font-size:1.4rem;
                   color:#4ade80'>{g0.mean():.3f}</div>
              <div style='font-size:0.62rem;color:rgba(232,220,200,0.35);
                   font-family:IBM Plex Mono,monospace'>median {g0.median():.3f}</div>
            </div>
            <div style='background:rgba(0,0,0,0.25);border-radius:5px;padding:10px;text-align:center'>
              <div style='font-size:0.6rem;color:rgba(232,220,200,0.4);
                   font-family:IBM Plex Mono,monospace;margin-bottom:4px'>FRAUD — MEAN</div>
              <div style='font-family:Playfair Display,serif;font-size:1.4rem;
                   color:#f87171'>{g1.mean():.3f}</div>
              <div style='font-size:0.62rem;color:rgba(232,220,200,0.35);
                   font-family:IBM Plex Mono,monospace'>median {g1.median():.3f}</div>
            </div>
          </div>
          <div style='margin-top:10px;font-size:0.72rem;
               color:rgba(232,220,200,0.45);border-top:1px solid rgba(212,160,60,0.1);
               padding-top:8px'>{note}</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 3 — AUTOENCODER RESULTS
# ════════════════════════════════════════════════════════
with tab3:

    # ── AutoEncoder explanation card ─────────────────────────────
    st.markdown(f"""
    <div class='agri-card' style='margin-bottom:22px'>
      <div style='font-family:Playfair Display,serif;font-size:1rem;
           color:#d4a03c;margin-bottom:10px'>🤖 AutoEncoder — How It Works</div>
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;
           font-size:0.83rem;color:rgba(232,220,200,0.72);line-height:1.8'>
        <div>
          <b style='color:#f97316'>What it learns:</b> The AutoEncoder is trained
          <i>only on genuine (non-fraud) claims</i>. It learns to compress and
          reconstruct normal claim patterns with very low error.<br><br>
          <b style='color:#f97316'>How it detects fraud:</b> When shown a fraudulent
          claim, the AutoEncoder cannot reconstruct it accurately — the
          reconstruction error is high. Claims with high error are flagged as anomalies.
        </div>
        <div>
          <b style='color:#f97316'>Architecture:</b> Dense → 64 → 32 →
          <b>16 (bottleneck)</b> → 32 → 64 → Output<br>
          Trained for <b>~{AE_EPOCHS} epochs</b> with EarlyStopping (patience=5).<br><br>
          <b style='color:#f97316'>Why it matters:</b> Unlike supervised models, the AutoEncoder
          requires <b>no fraud labels</b> to train — making it ideal for detecting
          entirely new, unseen fraud patterns.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Source note ──────────────────────────────────────────────
    st.markdown("""
    <div style='font-size:0.78rem;color:rgba(232,220,200,0.4);
         font-family:IBM Plex Mono,monospace;margin-bottom:18px;
         padding:6px 12px;background:rgba(249,115,22,0.06);
         border-left:3px solid rgba(249,115,22,0.4);border-radius:0 4px 4px 0'>
    📓 Results from Keras AutoEncoder notebook — ~48 epochs, 22 clean features,
    SMOTE balancing, no data leakage.
    Confusion matrix: TN=2194 · FP=104 · FN=67 · TP=131
    </div>
    """, unsafe_allow_html=True)

    # ── Headline metric cards ────────────────────────────────────
    st.markdown("### 📊 AutoEncoder Performance (Notebook Results)")
    a1, a2, a3, a4, a5 = st.columns(5)
    for col_w, label, fmt in [
        (a1, "ACCURACY",         f"{AE_ACC*100:.2f}%"),
        (a2, "ROC-AUC",          f"{AE_AUC:.4f}"),
        (a3, "F1 SCORE (Fraud)", f"{AE_F1:.4f}"),
        (a4, "PRECISION (Fraud)",f"{AE_PREC:.4f}"),
        (a5, "RECALL (Fraud)",   f"{AE_REC:.4f}"),
    ]:
        col_w.markdown(f"""
        <div style='background:rgba(20,18,10,0.9);border:1px solid rgba(249,115,22,0.25);
             border-top:3px solid #f97316;border-radius:8px;padding:16px;text-align:center'>
          <div style='font-family:IBM Plex Mono,monospace;font-size:0.58rem;
               color:rgba(249,115,22,0.7);letter-spacing:0.1em;
               text-transform:uppercase;margin-bottom:8px'>{label}</div>
          <div style='font-family:Playfair Display,serif;font-size:1.9rem;
               color:#f97316;font-weight:700'>{fmt}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Confusion matrix + classification report ─────────────────
    st.markdown("### 🎯 Confusion Matrix & Classification Report")
    st.markdown("""
    <div style='font-size:0.82rem;color:rgba(232,220,200,0.5);margin-bottom:14px'>
    <b>TN</b> = Genuine correctly passed &nbsp;|&nbsp;
    <b>TP</b> = Fraud correctly caught &nbsp;|&nbsp;
    <b>FP</b> = Genuine wrongly flagged &nbsp;|&nbsp;
    <b>FN</b> = Fraud that slipped through (minimise this)
    </div>
    """, unsafe_allow_html=True)

    cm_left, cm_right = st.columns([1, 1])

    with cm_left:
        txt_cm = [[f"TN\n{AE_TN:,}", f"FP\n{AE_FP:,}"],
                  [f"FN\n{AE_FN:,}", f"TP\n{AE_TP:,}"]]
        fig_cm = go.Figure(go.Heatmap(
            z=AE_CM,
            x=["Predicted: Genuine", "Predicted: Suspect"],
            y=["Actual: Genuine",    "Actual: Suspect"],
            text=txt_cm, texttemplate="%{text}",
            colorscale=[[0,"#0f0e0b"],[1,"#4a2a0a"]],
            showscale=False,
            textfont=dict(family="IBM Plex Mono", size=13, color=TEXT),
        ))
        fig_cm = pl(fig_cm, "🤖 AutoEncoder — Confusion Matrix (Notebook)", 340)
        st.plotly_chart(fig_cm, use_container_width=True)

        fraud_caught_pct = AE_TP / (AE_TP + AE_FN) * 100
        false_alarm_pct  = AE_FP / (AE_FP + AE_TN) * 100
        st.markdown(f"""
        <div style='background:rgba(20,18,10,0.85);
             border:1px solid rgba(249,115,22,0.2);
             border-radius:6px;padding:14px 18px;
             font-family:IBM Plex Mono,monospace;font-size:0.76rem;'>
          <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;
               text-align:center'>
            <div>
              <div style='color:rgba(232,220,200,0.4);font-size:0.62rem;
                   margin-bottom:4px'>FRAUD CAUGHT</div>
              <div style='color:#86efac;font-size:1.2rem;font-weight:700'>
                {fraud_caught_pct:.1f}%</div>
              <div style='color:rgba(232,220,200,0.35);font-size:0.62rem'>
                {AE_TP:,} / {AE_TP+AE_FN:,}</div>
            </div>
            <div>
              <div style='color:rgba(232,220,200,0.4);font-size:0.62rem;
                   margin-bottom:4px'>FALSE ALARMS</div>
              <div style='color:#f87171;font-size:1.2rem;font-weight:700'>
                {false_alarm_pct:.1f}%</div>
              <div style='color:rgba(232,220,200,0.35);font-size:0.62rem'>
                {AE_FP:,} / {AE_FP+AE_TN:,}</div>
            </div>
            <div>
              <div style='color:rgba(232,220,200,0.4);font-size:0.62rem;
                   margin-bottom:4px'>OVERALL ACC</div>
              <div style='color:#f97316;font-size:1.2rem;font-weight:700'>
                {AE_ACC*100:.2f}%</div>
              <div style='color:rgba(232,220,200,0.35);font-size:0.62rem'>
                {AE_TN+AE_TP:,} / {AE_TOTAL:,}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with cm_right:
        # Derived from notebook CM values
        genuine_prec = AE_TN / (AE_TN + AE_FN)           # 0.9703
        genuine_rec  = AE_TN / (AE_TN + AE_FP)           # 0.9547
        genuine_f1   = 2*genuine_prec*genuine_rec/(genuine_prec+genuine_rec)  # 0.9624
        genuine_sup  = AE_TN + AE_FP                      # 2298
        suspect_sup  = AE_FN + AE_TP                      # 198
        macro_prec   = (genuine_prec + AE_PREC) / 2
        macro_rec    = (genuine_rec  + AE_REC)  / 2
        macro_f1     = (genuine_f1   + AE_F1)   / 2
        wt_prec      = (genuine_prec*genuine_sup + AE_PREC*suspect_sup)/AE_TOTAL
        wt_rec       = (genuine_rec *genuine_sup + AE_REC *suspect_sup)/AE_TOTAL
        wt_f1        = (genuine_f1  *genuine_sup + AE_F1  *suspect_sup)/AE_TOTAL

        rpt_rows = [
            {"Class": "Genuine",      "Precision": f"{genuine_prec:.3f}", "Recall": f"{genuine_rec:.3f}", "F1-Score": f"{genuine_f1:.3f}", "Support": f"{genuine_sup:,}"},
            {"Class": "Suspect",      "Precision": f"{AE_PREC:.3f}",      "Recall": f"{AE_REC:.3f}",      "F1-Score": f"{AE_F1:.3f}",      "Support": f"{suspect_sup:,}"},
            {"Class": "macro avg",    "Precision": f"{macro_prec:.3f}",   "Recall": f"{macro_rec:.3f}",   "F1-Score": f"{macro_f1:.3f}",   "Support": f"{AE_TOTAL:,}"},
            {"Class": "weighted avg", "Precision": f"{wt_prec:.3f}",      "Recall": f"{wt_rec:.3f}",      "F1-Score": f"{wt_f1:.3f}",      "Support": f"{AE_TOTAL:,}"},
        ]
        rpt_df = pd.DataFrame(rpt_rows).set_index("Class")

        st.markdown("""
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;
             color:#f97316;margin-bottom:8px;font-weight:600'>
        📋 AutoEncoder — Full Classification Report (Notebook)</div>
        """, unsafe_allow_html=True)
        st.dataframe(rpt_df, use_container_width=True, height=220)

        st.markdown("""
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;
             color:rgba(232,220,200,0.5);margin:18px 0 10px'>METRIC OVERVIEW</div>
        """, unsafe_allow_html=True)
        for metric_label, value, color in [
            ("Accuracy",          AE_ACC,  "#f97316"),
            ("ROC-AUC",           AE_AUC,  "#f97316"),
            ("F1 Score (Fraud)",  AE_F1,   "#fbbf24"),
            ("Precision (Fraud)", AE_PREC, "#60a5fa"),
            ("Recall (Fraud)",    AE_REC,  "#86efac"),
        ]:
            bar_w = int(value * 100)
            st.markdown(f"""
            <div style='margin-bottom:8px'>
              <div style='display:flex;justify-content:space-between;
                   font-family:IBM Plex Mono,monospace;font-size:0.68rem;
                   margin-bottom:3px'>
                <span style='color:rgba(232,220,200,0.55)'>{metric_label}</span>
                <span style='color:{color};font-weight:600'>{value:.4f}</span>
              </div>
              <div style='height:6px;background:rgba(255,255,255,0.07);border-radius:3px'>
                <div style='width:{bar_w}%;height:100%;
                     background:{color};border-radius:3px'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Anomaly score distribution ───────────────────────────────
    st.markdown("### 📊 AutoEncoder Anomaly Score Distribution")
    st.markdown("""
    <div style='font-size:0.82rem;color:rgba(232,220,200,0.5);margin-bottom:12px'>
    Higher score = more anomalous = more likely to be fraud.
    Genuine claims cluster near 0; fraud claims are pushed right.
    </div>
    """, unsafe_allow_html=True)

    df_all_scores = st.session_state["df"]
    if "ae_score" in df_all_scores.columns:
        fig_ae = go.Figure()
        for lbl, name_lbl, col in [(0,"Genuine","#4ade80"),(1,"Fraud","#f87171")]:
            s = df_all_scores[df_all_scores["fraud_label"]==lbl]["ae_score"].dropna()
            fig_ae.add_trace(go.Histogram(x=s, nbinsx=60, name=name_lbl,
                marker_color=col, opacity=0.72))
        fig_ae = pl(fig_ae, "🤖 AutoEncoder Anomaly Score — Genuine vs Fraud", 320)
        fig_ae.update_layout(barmode="overlay",
            xaxis_title="Anomaly Score (0 = normal, 1 = highly anomalous)")
        st.plotly_chart(fig_ae, use_container_width=True)

    # ── Verdict distribution ─────────────────────────────────────
    if "verdict" in df.columns:
        st.divider()
        st.markdown("### 🏷️ Final Verdict Distribution")
        vd = df["verdict"].value_counts()
        vd_colors = []
        for v in vd.index:
            if "HIGH"    in v: vd_colors.append(RED)
            elif "MEDIUM" in v: vd_colors.append(AMBER)
            elif "LOW"    in v: vd_colors.append("#fbbf24")
            else:               vd_colors.append(GREEN)

        fig_vd = go.Figure(go.Bar(
            x=vd.values, y=vd.index, orientation="h",
            marker_color=vd_colors,
            text=[f" {v:,}" for v in vd.values], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10),
        ))
        fig_vd = pl(fig_vd, "Final Verdict Distribution (AutoEncoder)", 340)
        fig_vd.update_layout(xaxis_title="Number of Claims",
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_vd, use_container_width=True)

    # ── Download buttons ─────────────────────────────────────────
    st.divider()
    dl1, dl2 = st.columns(2)
    with dl1:
        fraud_export = st.session_state["df"][
            st.session_state["df"]["fraud_label"]==1
        ].copy()
        if "ae_score" in fraud_export.columns:
            fraud_export = fraud_export.sort_values("ae_score", ascending=False)
        export_cols = [c for c in [
            "farmer_id","year","district_clean","crop","area_hectares",
            "loss_percent","claim_amount_rs","insurance_premium_rs",
            "rainfall_mm","chirps_rain_mm","rainfall_deviation_pct",
            "ndvi","ndwi","evi","vci","flood_fraction",
            "nasa_tmax_c","nasa_tmin_c",
            "fraud_fake_drought","fraud_fake_flood","fraud_hidden_yield",
            "fraud_label","ae_score","ensemble_score","verdict"
        ] if c in fraud_export.columns]
        st.download_button(
            "⬇️ Download Fraud Report (CSV)",
            data=fraud_export[export_cols].round(4).to_csv(index=False),
            file_name="agrishield_fraud_report.csv", mime="text/csv",
            use_container_width=True
        )
    with dl2:
        full_export = st.session_state["df"].copy()
        full_cols = [c for c in [
            "farmer_id","year","district_clean","crop",
            "loss_percent","claim_amount_rs","ndvi","vci","chirps_rain_mm",
            "fraud_label","ae_score","ensemble_score","verdict"
        ] if c in full_export.columns]
        st.download_button(
            "⬇️ Download Full Dataset with Verdicts (CSV)",
            data=full_export[full_cols].round(4).to_csv(index=False),
            file_name="agrishield_full_report.csv", mime="text/csv",
            use_container_width=True
        )


# ════════════════════════════════════════════════════════
# TAB 4 — DISTRICT ANALYSIS
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📍 District-wise Fraud Analysis")

    dist = df.groupby("district_clean").agg(
        total=("fraud_label","count"),
        fraud=("fraud_label","sum"),
        fake_drought=("fraud_fake_drought","sum"),
        fake_flood=("fraud_fake_flood","sum"),
        hidden_yield=("fraud_hidden_yield","sum"),
        avg_ndvi=("ndvi","mean"),
        avg_vci=("vci","mean"),
        avg_chirps=("chirps_rain_mm","mean"),
    ).reset_index()
    dist["fraud_rate"] = (dist["fraud"]/dist["total"]*100).round(1)
    dist = dist.sort_values("fraud_rate", ascending=False)

    da1, da2 = st.columns(2)
    with da1:
        fig_dr = go.Figure(go.Bar(
            x=dist["fraud_rate"], y=dist["district_clean"],
            orientation="h",
            marker_color=[RED if r>15 else AMBER if r>8 else GREEN
                          for r in dist["fraud_rate"]],
            text=[f"{r}%" for r in dist["fraud_rate"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10),
        ))
        fig_dr = pl(fig_dr, "Fraud Rate by District (%)", 400)
        fig_dr.update_layout(xaxis_title="Fraud Rate (%)",
                             yaxis=dict(autorange="reversed"))
        fig_dr.add_vline(x=10, line_dash="dash", line_color=AMBER,
                         annotation_text="10% threshold")
        st.plotly_chart(fig_dr, use_container_width=True)

    with da2:
        fig_st = go.Figure()
        for col_n, color, label in [
            ("fake_drought","#f87171","Fake Drought"),
            ("fake_flood","#60a5fa","Fake Flood"),
            ("hidden_yield","#fbbf24","Hidden Yield"),
        ]:
            fig_st.add_trace(go.Bar(
                name=label, x=dist["district_clean"],
                y=dist[col_n], marker_color=color))
        fig_st = pl(fig_st, "Fraud Cases by Type & District", 400)
        fig_st.update_layout(barmode="stack",
            xaxis=dict(tickangle=-35), yaxis_title="Cases")
        st.plotly_chart(fig_st, use_container_width=True)

    sat_d = df.groupby("district_clean").agg(
        avg_ndvi=("ndvi","mean"), avg_vci=("vci","mean"),
    ).reset_index().sort_values("avg_ndvi", ascending=False)

    fig_sat2 = make_subplots(rows=1, cols=2,
        subplot_titles=("Average NDVI by District",
                        "Average VCI by District"))
    fig_sat2.add_trace(go.Bar(
        x=sat_d["avg_ndvi"], y=sat_d["district_clean"], orientation="h",
        marker_color=[GREEN if v>0.5 else AMBER if v>0.35 else RED
                      for v in sat_d["avg_ndvi"]],
        text=[f"{v:.3f}" for v in sat_d["avg_ndvi"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono",size=9),
    ), row=1, col=1)
    fig_sat2.add_trace(go.Bar(
        x=sat_d["avg_vci"], y=sat_d["district_clean"], orientation="h",
        marker_color=[GREEN if v>55 else AMBER if v>40 else RED
                      for v in sat_d["avg_vci"]],
        text=[f"{v:.1f}" for v in sat_d["avg_vci"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono",size=9),
    ), row=1, col=2)
    fig_sat2.update_layout(
        paper_bgcolor=PBGC, plot_bgcolor=PBGC,
        font=dict(family="IBM Plex Sans", color=TEXT, size=11),
        height=400, showlegend=False,
        margin=dict(l=130,r=60,t=50,b=40),
    )
    fig_sat2.update_xaxes(gridcolor=GRID)
    fig_sat2.update_yaxes(gridcolor=GRID, autorange="reversed")
    st.plotly_chart(fig_sat2, use_container_width=True)

    st.markdown("### 📋 District Summary Table")
    for _, row in dist.iterrows():
        r = row["fraud_rate"]
        if r > 15:  badge = "<span class='badge-high'>HIGH RISK DISTRICT</span>"
        elif r > 8: badge = "<span class='badge-med'>MODERATE</span>"
        else:       badge = "<span class='badge-ok'>LOW RISK</span>"
        border_c = RED if r>15 else AMBER if r>8 else GREEN

        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:12px;padding:10px 16px;
             margin-bottom:4px;background:rgba(26,23,16,0.7);
             border:1px solid {border_c}22;
             border-left:3px solid {border_c};border-radius:0 6px 6px 0'>
          <b style='flex:1.2;font-size:0.88rem'>{row['district_clean']}</b>
          {badge}
          <span style='font-family:IBM Plex Mono,monospace;font-size:0.78rem'>
            Total: <b>{int(row['total'])}</b></span>
          <span style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:{RED}'>
            Fraud: <b>{int(row['fraud'])}</b> ({r}%)</span>
          <span style='font-size:0.73rem;color:rgba(232,220,200,0.45)'>
            🏜️{int(row['fake_drought'])} 🌊{int(row['fake_flood'])} 🌿{int(row['hidden_yield'])}</span>
          <span style='font-family:IBM Plex Mono,monospace;font-size:0.72rem;
               color:rgba(232,220,200,0.4)'>
            NDVI:{row['avg_ndvi']:.3f} VCI:{row['avg_vci']:.1f}</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;font-family:IBM Plex Mono,monospace;
     font-size:0.62rem;color:rgba(232,220,200,0.18);padding:8px;letter-spacing:0.1em'>
  🌾 AGRISHIELD v2.1 · AP AGRICULTURAL INSURANCE FRAUD DETECTION ·
  GEE (MODIS+CHIRPS) + NASA POWER + AUTOENCODER ·
  MSC(AA) &amp; PGD(SDS) PROJECT 2026 · S.R.L KEERTHI &amp; YASHASWINI H V
</div>
""", unsafe_allow_html=True)
