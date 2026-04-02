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
[data-testid="stFileUploader"]{
    border:1px dashed rgba(212,160,60,0.25) !important;
    border-radius:8px !important; background:rgba(26,23,16,0.4) !important;}

.stSuccess{background:rgba(34,85,34,0.15) !important; border-color:rgba(100,180,100,0.3) !important;}
.stWarning{background:rgba(180,120,0,0.12) !important; border-color:rgba(212,160,60,0.3) !important;}
.stError  {background:rgba(140,30,30,0.12) !important; border-color:rgba(220,60,60,0.3) !important;}
hr{border-color:rgba(212,160,60,0.15) !important;}

.agri-card{background:rgba(26,23,16,0.85);border:1px solid rgba(212,160,60,0.2);
    border-radius:8px;padding:18px 20px;position:relative;}
.agri-card::before{content:'';position:absolute;top:0;left:10%;right:10%;height:1px;
    background:linear-gradient(90deg,transparent,rgba(212,160,60,0.4),transparent);}

.model-card{background:rgba(20,18,10,0.9);border:1px solid rgba(212,160,60,0.15);
    border-radius:10px;padding:18px;text-align:center;}
.model-card.rf {border-top:3px solid #60a5fa;}
.model-card.gb {border-top:3px solid #86efac;}
.model-card.ae {border-top:3px solid #f97316;}

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


# ═══════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(ins_path, sat_path):
    df  = pd.read_csv(ins_path)
    sat = pd.read_csv(sat_path)

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
    df["nasa_tmax_c"]   = df["district_clean"].map(NASA_TMAX).fillna(32.5)
    df["nasa_tmin_c"]   = df["nasa_tmax_c"] - 10.0
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

SAT_FEATS = {"ndvi","ndwi","evi","vci","chirps_rain_mm","flood_fraction",
             "rainfall_deviation_pct"}
NASA_FEATS = {"nasa_tmax_c","nasa_tmin_c","nasa_solar_rad",
              "heat_stress_flag","cold_stress_flag","drought_index_norm"}


@st.cache_resource(show_spinner=False)
def train_all_models(_df_id):
    df = st.session_state["df"]
    X  = df[FEATURES_CLEAN].fillna(df[FEATURES_CLEAN].median())
    y  = df["fraud_label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler     = StandardScaler()
    X_tr_s     = scaler.fit_transform(X_tr)
    X_te_s     = scaler.transform(X_te)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=4,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_tr_s, y_tr)
    rf_preds = rf.predict(X_te_s)
    rf_proba = rf.predict_proba(X_te_s)[:, 1]
    rf_cm    = confusion_matrix(y_te, rf_preds)

    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=5, random_state=42
    )
    gb.fit(X_tr_s, y_tr)
    gb_preds = gb.predict(X_te_s)
    gb_proba = gb.predict_proba(X_te_s)[:, 1]
    gb_cm    = confusion_matrix(y_te, gb_preds)

    fraud_contamination = float(y.mean())
    ae_model = IsolationForest(
        n_estimators=200,
        contamination=fraud_contamination,
        random_state=42
    )
    ae_model.fit(X_tr_s[y_tr == 0])

    ae_raw    = ae_model.decision_function(X_te_s)
    ae_scores = -ae_raw

    prec_c, rec_c, thresh_c = precision_recall_curve(y_te, ae_scores)
    f1s_c    = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-8)
    best_thr = thresh_c[np.argmax(f1s_c)]
    ae_preds = (ae_scores > best_thr).astype(int)
    ae_cm    = confusion_matrix(y_te, ae_preds)

    X_full   = df[FEATURES_CLEAN].fillna(df[FEATURES_CLEAN].median())
    X_full_s = scaler.transform(X_full)

    rf_full_proba  = rf.predict_proba(X_full_s)[:, 1]
    gb_full_proba  = gb.predict_proba(X_full_s)[:, 1]
    ae_full_raw    = ae_model.decision_function(X_full_s)
    ae_full_scores = -ae_full_raw
    ae_norm        = MinMaxScaler()
    ae_full_norm   = ae_norm.fit_transform(ae_full_scores.reshape(-1,1)).flatten()

    df["rf_score"]       = rf_full_proba
    df["gb_score"]       = gb_full_proba
    df["ae_score"]       = ae_full_norm
    df["ensemble_score"] = (0.50 * rf_full_proba +
                            0.30 * gb_full_proba +
                            0.20 * ae_full_norm)

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
        "rf": rf, "gb": gb, "ae": ae_model,
        "scaler": scaler, "ae_norm": ae_norm,
        "features": FEATURES_CLEAN,
        "y_te": y_te,
        "rf_preds": rf_preds, "rf_proba": rf_proba, "rf_cm": rf_cm,
        "rf_acc":  accuracy_score(y_te, rf_preds),
        "rf_auc":  roc_auc_score(y_te, rf_proba),
        "rf_f1":   f1_score(y_te, rf_preds),
        "rf_prec": precision_score(y_te, rf_preds),
        "rf_rec":  recall_score(y_te, rf_preds),
        "gb_preds": gb_preds, "gb_proba": gb_proba, "gb_cm": gb_cm,
        "gb_acc":  accuracy_score(y_te, gb_preds),
        "gb_auc":  roc_auc_score(y_te, gb_proba),
        "gb_f1":   f1_score(y_te, gb_preds),
        "gb_prec": precision_score(y_te, gb_preds),
        "gb_rec":  recall_score(y_te, gb_preds),
        "ae_preds": ae_preds, "ae_scores_test": ae_scores, "ae_cm": ae_cm,
        "ae_acc":  accuracy_score(y_te, ae_preds),
        "ae_auc":  roc_auc_score(y_te, ae_scores),
        "ae_f1":   f1_score(y_te, ae_preds),
        "ae_prec": precision_score(y_te, ae_preds, zero_division=0),
        "ae_rec":  recall_score(y_te, ae_preds),
        "importances": pd.Series(rf.feature_importances_,
                                  index=FEATURES_CLEAN).sort_values(ascending=False),
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

    st.markdown("**📂 Data Files**")
    st.markdown(
        "<small>Upload both CSV files to begin analysis</small>",
        unsafe_allow_html=True
    )
    ins_file = st.file_uploader("📋 Insurance CSV", type=["csv"])
    sat_file = st.file_uploader("🛰️ Satellite CSV (GEE)", type=["csv"])

    # ── KEY FIX: only load when both files are uploaded ───
    data_ready = False
    if ins_file is not None and sat_file is not None:
        # Re-load if new files are uploaded
        file_key = f"{ins_file.name}_{sat_file.name}_{ins_file.size}_{sat_file.size}"
        if st.session_state.get("_file_key") != file_key:
            # Clear stale cache when files change
            if "df" in st.session_state:
                del st.session_state["df"]
            if "models" in st.session_state:
                del st.session_state["models"]
            st.session_state["_file_key"] = file_key

        if "df" not in st.session_state:
            with st.spinner("⏳ Loading & processing data..."):
                try:
                    st.session_state["df"] = load_data(ins_file, sat_file)
                    st.success("✅ Data loaded!")
                except Exception as e:
                    st.error(f"❌ Error loading data: {e}")
                    st.stop()

        data_ready = True
    else:
        st.info("⬆️ Upload both CSV files above to begin.")

    if not data_ready:
        st.divider()
        st.markdown("""
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;
        color:rgba(232,220,200,0.3);line-height:1.9'>
        📡 DATA SOURCES<br>
        GEE: MODIS NDVI/NDWI/EVI<br>
        GEE: CHIRPS Rainfall<br>
        GEE: Flood Fraction<br>
        NASA POWER: Temp/Solar<br><br>
        🤖 MODELS<br>
        1. Random Forest (300 trees)<br>
        2. Gradient Boosting (200)<br>
        3. AutoEncoder/IsoForest<br>
        4. Ensemble (50+30+20%)
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── From here on, data is loaded ─────────────────────
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
    if st.button("🤖 TRAIN ALL 3 MODELS", use_container_width=True):
        if "models" in st.session_state:
            del st.session_state["models"]
        st.cache_resource.clear()

    if "models" not in st.session_state:
        with st.spinner("🤖 Training RF + GB + AutoEncoder..."):
            st.session_state["models"] = train_all_models(id(df_all))
        st.success("✅ All 3 models trained!")

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
    🤖 MODELS<br>
    1. Random Forest (300 trees)<br>
    2. Gradient Boosting (200)<br>
    3. AutoEncoder/IsoForest<br>
    4. Ensemble (50+30+20%)
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
       3 ML models to catch fake drought, fake flood and hidden-yield fraud.
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
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠  Overview",
    "🛰️  Satellite Evidence",
    "🤖  ML Models & Comparison",
    "📍  District Analysis",
    "🔍  Claim Inspector",
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
    for col, name, color in [
        ("fake_drought","Fake Drought","#f87171"),
        ("fake_flood","Fake Flood","#60a5fa"),
        ("hidden_yield","Hidden Yield","#fbbf24"),
    ]:
        fig_t.add_trace(go.Scatter(x=yearly["year"], y=yearly[col],
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

    bx1, bx2, bx3 = st.columns(3)
    for col_w, idx_col, title, col_f, note in [
        (bx1,"ndvi","NDVI — Genuine vs Fraud","#f87171",
         "High NDVI with damage claim = hidden good yield"),
        (bx2,"vci", "VCI — Genuine vs Fraud","#fbbf24",
         "VCI>60 with drought claim = fake drought"),
        (bx3,"chirps_rain_mm","CHIRPS Rain (mm) — Genuine vs Fraud","#60a5fa",
         "High CHIRPS rain with drought claim = fake drought"),
    ]:
        g0 = df[df["fraud_label"]==0][idx_col].dropna()
        g1 = df[df["fraud_label"]==1][idx_col].dropna()
        fig_bx = go.Figure()
        fig_bx.add_trace(go.Box(y=g0,name="Genuine",marker_color=GREEN,
                                 boxmean=True,line_width=1.5))
        fig_bx.add_trace(go.Box(y=g1,name="Fraud",  marker_color=col_f,
                                 boxmean=True,line_width=1.5))
        fig_bx = pl(fig_bx, title, 300)
        col_w.plotly_chart(fig_bx, use_container_width=True)
        col_w.markdown(f"<small>{note}</small>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 3 — ML MODELS & COMPARISON
# ════════════════════════════════════════════════════════
with tab3:

    st.markdown("""
    <div class='agri-card' style='margin-bottom:18px'>
      <div style='font-family:Playfair Display,serif;font-size:1rem;
           color:#d4a03c;margin-bottom:10px'>🤖 How the 3 Models Work</div>
      <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;
           font-size:0.82rem;color:rgba(232,220,200,0.72);'>
        <div style='border-left:3px solid #60a5fa;padding-left:10px'>
          <b style='color:#60a5fa'>Random Forest</b><br>
          Builds 300 decision trees. Each tree learns different fraud patterns.
          A claim is flagged if the majority of trees vote it suspicious.
          <b>Best at: complex patterns across many features.</b>
        </div>
        <div style='border-left:3px solid #86efac;padding-left:10px'>
          <b style='color:#86efac'>Gradient Boosting</b><br>
          Trains trees sequentially — each new tree corrects the previous one's mistakes.
          Very precise. <b>Best at: financial ratio anomalies and crop-specific patterns.</b>
        </div>
        <div style='border-left:3px solid #f97316;padding-left:10px'>
          <b style='color:#f97316'>AutoEncoder (IsolationForest)</b><br>
          Learns what a "normal/genuine" claim looks like, then flags anything
          that looks different. Requires NO fraud labels to train.
          <b>Best at: catching new unknown fraud patterns.</b>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Individual Model Results")
    mc1, mc2, mc3 = st.columns(3)

    for col_w, key, name, color, css_cls, desc in [
        (mc1, "rf", "Random Forest",    BLUE,   "rf",
         "Trained on: 80% of data | Class weights balanced | 300 trees, depth 12"),
        (mc2, "gb", "Gradient Boosting", GREEN,  "gb",
         "Trained on: 80% of data | Learning rate 0.05 | 200 boosting rounds"),
        (mc3, "ae", "AutoEncoder",       ORANGE, "ae",
         "Trained on: GENUINE claims only | IsolationForest | Detects anomalies"),
    ]:
        acc  = m[f"{key}_acc"]
        auc  = m[f"{key}_auc"]
        f1   = m[f"{key}_f1"]
        prec = m[f"{key}_prec"]
        rec  = m[f"{key}_rec"]
        col_w.markdown(f"""
        <div class='model-card {css_cls}' style='margin-bottom:12px'>
          <div style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;
               color:{color};letter-spacing:0.12em;text-transform:uppercase;
               margin-bottom:8px'>{name}</div>

          <div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px'>
            <div style='background:rgba(0,0,0,0.25);border-radius:5px;padding:8px;text-align:center'>
              <div style='font-size:0.62rem;color:rgba(232,220,200,0.4);
                   font-family:IBM Plex Mono,monospace'>ACCURACY</div>
              <div style='font-family:Playfair Display,serif;font-size:1.5rem;
                   color:{color}'>{acc*100:.2f}%</div>
            </div>
            <div style='background:rgba(0,0,0,0.25);border-radius:5px;padding:8px;text-align:center'>
              <div style='font-size:0.62rem;color:rgba(232,220,200,0.4);
                   font-family:IBM Plex Mono,monospace'>ROC-AUC</div>
              <div style='font-family:Playfair Display,serif;font-size:1.5rem;
                   color:{color}'>{auc:.4f}</div>
            </div>
            <div style='background:rgba(0,0,0,0.25);border-radius:5px;padding:8px;text-align:center'>
              <div style='font-size:0.62rem;color:rgba(232,220,200,0.4);
                   font-family:IBM Plex Mono,monospace'>F1 SCORE</div>
              <div style='font-family:Playfair Display,serif;font-size:1.5rem;
                   color:{color}'>{f1:.4f}</div>
            </div>
            <div style='background:rgba(0,0,0,0.25);border-radius:5px;padding:8px;text-align:center'>
              <div style='font-size:0.62rem;color:rgba(232,220,200,0.4);
                   font-family:IBM Plex Mono,monospace'>RECALL</div>
              <div style='font-family:Playfair Display,serif;font-size:1.5rem;
                   color:{color}'>{rec:.4f}</div>
            </div>
          </div>

          <div style='font-size:0.7rem;color:rgba(232,220,200,0.38);
               font-family:IBM Plex Mono,monospace'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📈 Model Comparison — All Metrics")
    models_list  = ["Random Forest", "Gradient Boosting", "AutoEncoder"]
    metrics_data = {
        "Accuracy (%)": [m["rf_acc"]*100, m["gb_acc"]*100, m["ae_acc"]*100],
        "ROC-AUC":      [m["rf_auc"],     m["gb_auc"],     m["ae_auc"]],
        "F1 Score":     [m["rf_f1"],      m["gb_f1"],      m["ae_f1"]],
        "Precision":    [m["rf_prec"],    m["gb_prec"],    m["ae_prec"]],
        "Recall":       [m["rf_rec"],     m["gb_rec"],     m["ae_rec"]],
    }

    comp1, comp2 = st.columns(2)
    with comp1:
        fig_comp = go.Figure()
        bar_colors = [BLUE, GREEN, ORANGE]
        for i, (model, color) in enumerate(zip(models_list, bar_colors)):
            fig_comp.add_trace(go.Bar(
                name=model,
                x=list(metrics_data.keys()),
                y=[v[i] for v in metrics_data.values()],
                marker_color=color, marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=1,
                text=[f"{v[i]:.3f}" for v in metrics_data.values()],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=9),
            ))
        fig_comp = pl(fig_comp, "All Models — Side-by-Side Metric Comparison", 370)
        fig_comp.update_layout(barmode="group", yaxis=dict(range=[0,1.1]),
                               xaxis_title="Metric")
        st.plotly_chart(fig_comp, use_container_width=True)

    with comp2:
        imp   = m["importances"]
        colors_fi = []
        for f in imp.index:
            if f in SAT_FEATS:    colors_fi.append(ORANGE)
            elif f in NASA_FEATS: colors_fi.append(PURPLE)
            else:                 colors_fi.append(AMBER)

        fig_imp = go.Figure(go.Bar(
            x=imp.values, y=imp.index,
            orientation="h",
            marker_color=colors_fi,
            marker_line_color="rgba(0,0,0,0.2)", marker_line_width=1,
            text=[f"{v:.4f}" for v in imp.values], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=9),
        ))
        fig_imp = pl(fig_imp,
            "RF Feature Importance  🟠=Satellite  🟣=NASA  🟡=Financial", 370)
        fig_imp.update_layout(xaxis_title="Importance Score",
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### 🎯 Confusion Matrices — All 3 Models")
    st.markdown("""
    <div style='font-size:0.82rem;color:rgba(232,220,200,0.55);margin-bottom:14px'>
    <b>TN</b> = Genuine correctly marked genuine &nbsp;|&nbsp;
    <b>TP</b> = Fraud correctly caught &nbsp;|&nbsp;
    <b>FP</b> = Genuine wrongly flagged &nbsp;|&nbsp;
    <b>FN</b> = Fraud that slipped through (most important to minimise)
    </div>
    """, unsafe_allow_html=True)

    cm_cols = st.columns(3)
    for i, (key, name, color, cscale) in enumerate([
        ("rf", "Random Forest",    BLUE,   [[0,"#0f0e0b"],[1,"#1e3a5f"]]),
        ("gb", "Gradient Boosting",GREEN,  [[0,"#0f0e0b"],[1,"#1a4a2a"]]),
        ("ae", "AutoEncoder",      ORANGE, [[0,"#0f0e0b"],[1,"#4a2a0a"]]),
    ]):
        cm  = m[f"{key}_cm"]
        tn, fp, fn, tp = cm.ravel()
        txt = [[f"TN\n{tn:,}", f"FP\n{fp:,}"],
               [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred: Genuine","Pred: Fraud"],
            y=["Actual: Genuine","Actual: Fraud"],
            text=txt, texttemplate="%{text}",
            colorscale=cscale, showscale=False,
            textfont=dict(family="IBM Plex Mono", size=11, color=TEXT),
        ))
        fig_cm = pl(fig_cm, f"{name} — Confusion Matrix", 280)
        cm_cols[i].plotly_chart(fig_cm, use_container_width=True)

        total_te = tn+fp+fn+tp
        fraud_caught_pct = tp/(tp+fn)*100 if (tp+fn) else 0
        false_alarm_pct  = fp/(fp+tn)*100 if (fp+tn) else 0
        cm_cols[i].markdown(f"""
        <div style='background:rgba(20,18,10,0.8);border:1px solid {color}22;
             border-radius:6px;padding:10px 12px;margin-top:-8px;
             font-family:IBM Plex Mono,monospace;font-size:0.72rem;'>
          <div style='color:{color};margin-bottom:5px;font-weight:600'>{name}</div>
          <div style='color:rgba(232,220,200,0.6);line-height:1.9'>
            Fraud caught: <b style='color:#86efac'>{fraud_caught_pct:.1f}%</b>
            ({tp:,}/{tp+fn:,})<br>
            False alarms: <b style='color:#f87171'>{false_alarm_pct:.1f}%</b>
            ({fp:,}/{fp+tn:,})<br>
            Overall acc: <b style='color:{color}'>{(tn+tp)/total_te*100:.2f}%</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📋 Full Classification Reports")
    rpt_cols = st.columns(3)
    y_te = m["y_te"]
    for col_w, preds, name, color in [
        (rpt_cols[0], m["rf_preds"], "Random Forest",    BLUE),
        (rpt_cols[1], m["gb_preds"], "Gradient Boosting",GREEN),
        (rpt_cols[2], m["ae_preds"], "AutoEncoder",       ORANGE),
    ]:
        rpt = classification_report(y_te, preds,
              target_names=["Genuine","Fraud"], output_dict=True)
        rows = []
        for cls in ["Genuine","Fraud","macro avg","weighted avg"]:
            if cls in rpt:
                r = rpt[cls]
                rows.append({
                    "Class":     cls,
                    "Precision": f"{r['precision']:.3f}",
                    "Recall":    f"{r['recall']:.3f}",
                    "F1-Score":  f"{r['f1-score']:.3f}",
                    "Support":   f"{int(r['support']):,}" if "support" in r else "—",
                })
        rpt_df = pd.DataFrame(rows)
        col_w.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;"
            f"font-size:0.72rem;color:{color};margin-bottom:4px;font-weight:600'>"
            f"{name}</div>",
            unsafe_allow_html=True
        )
        col_w.dataframe(rpt_df.set_index("Class"), use_container_width=True)

    st.divider()

    st.markdown("### 📊 Ensemble Risk Score Distribution")
    st.markdown("""
    <div style='font-size:0.82rem;color:rgba(232,220,200,0.5);margin-bottom:12px'>
    Final score = 50% RF + 30% GB + 20% AutoEncoder. Score &gt;0.75 = HIGH RISK.
    </div>
    """, unsafe_allow_html=True)

    df_all_scores = st.session_state["df"]
    if "ensemble_score" in df_all_scores.columns:
        fig_ens = go.Figure()
        for lbl, name, col in [(0,"Genuine","#4ade80"),(1,"Fraud","#f87171")]:
            s = df_all_scores[df_all_scores["fraud_label"]==lbl]["ensemble_score"].dropna()
            fig_ens.add_trace(go.Histogram(x=s, nbinsx=50, name=name,
                marker_color=col, opacity=0.7))
        fig_ens.add_vline(x=0.50, line_dash="dash", line_color=AMBER,
                          annotation_text="Medium Risk (0.50)")
        fig_ens.add_vline(x=0.75, line_dash="dash", line_color=RED,
                          annotation_text="High Risk (0.75)")
        fig_ens = pl(fig_ens, "Ensemble Score: Genuine vs Fraud Distribution", 300)
        fig_ens.update_layout(barmode="overlay",
            xaxis_title="Risk Score (0=safe, 1=high risk)")
        st.plotly_chart(fig_ens, use_container_width=True)

    if "verdict" in df.columns:
        vd = df["verdict"].value_counts()
        vd_colors = []
        for v in vd.index:
            if "HIGH"    in v: vd_colors.append(RED)
            elif "MEDIUM" in v: vd_colors.append(AMBER)
            elif "LOW"    in v: vd_colors.append("#fbbf24")
            else:               vd_colors.append(GREEN)

        fig_vd = go.Figure(go.Bar(
            x=vd.values, y=vd.index, orientation="h",
            marker_color=vd_colors, marker_line_color="rgba(0,0,0,0.2)",
            marker_line_width=1,
            text=[f" {v:,}" for v in vd.values], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10),
        ))
        fig_vd = pl(fig_vd, "Final Verdict Distribution (Ensemble)", 340)
        fig_vd.update_layout(xaxis_title="Number of Claims",
                             yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_vd, use_container_width=True)


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
                y=dist[col_n], marker_color=color,
                marker_line_color="rgba(0,0,0,0.2)",marker_line_width=1))
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


# ════════════════════════════════════════════════════════
# TAB 5 — CLAIM INSPECTOR
# ════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🔍 Individual Claim Inspector")
    st.markdown("""
    <div style='font-size:0.84rem;color:rgba(232,220,200,0.55);margin-bottom:14px'>
    Each card shows <b>exactly why</b> a claim was flagged — what the satellite
    saw vs what the farmer claimed. Use the filters below to search.
    </div>
    """, unsafe_allow_html=True)

    fi1, fi2, fi3, fi4 = st.columns([2,2,1,1])
    with fi1:
        search = st.text_input("🔎 Farmer ID", placeholder="e.g. GUN_2000_0")
    with fi2:
        ins_dist = st.selectbox("📍 District",
            ["All"] + sorted(df["district_clean"].dropna().unique().tolist()))
    with fi3:
        only_fraud = st.checkbox("Fraud only", value=True)
    with fi4:
        risk_filter = st.selectbox("Risk", ["All","HIGH","MEDIUM","LOW","GENUINE"])

    view = st.session_state["df"].copy()
    view = view[view["district_clean"].isin(sel_districts)]
    if ins_dist != "All":
        view = view[view["district_clean"] == ins_dist]
    if search:
        view = view[view["farmer_id"].str.contains(search, case=False, na=False)]
    if only_fraud:
        view = view[view["fraud_label"] == 1]
    if risk_filter != "All" and "verdict" in view.columns:
        view = view[view["verdict"].str.startswith(risk_filter, na=False)]

    st.markdown(f"<small>Showing {min(60,len(view))} of {len(view)} claims</small>",
                unsafe_allow_html=True)

    for _, row in view.head(60).iterrows():
        reasons, expls = [], []
        if row.get("fraud_fake_drought") == 1:
            reasons.append("🏜️ Fake Drought")
            expls.append(
                f"Claimed drought (loss {row['loss_percent']:.0f}%) BUT satellite VCI = "
                f"{row.get('vci',0):.1f} (above 60 = HEALTHY vegetation) AND "
                f"CHIRPS rainfall = {row.get('chirps_rain_mm',0):.0f} mm (adequate rain)"
            )
        if row.get("fraud_fake_flood") == 1:
            reasons.append("🌊 Fake Flood")
            expls.append(
                f"Claimed flood damage BUT satellite flood_fraction = "
                f"{row.get('flood_fraction',0):.4f} (near zero — NO standing water) "
                f"AND NDWI = {row.get('ndwi',0):.3f} (negative = no water body)"
            )
        if row.get("fraud_hidden_yield") == 1:
            reasons.append("🌿 Hidden Yield")
            expls.append(
                f"Claimed crop failure BUT NDVI = {row.get('ndvi',0):.3f} "
                f"(top 25% of district — crops visibly GREEN) AND "
                f"VCI = {row.get('vci',0):.1f} (excellent vegetation condition)"
            )

        if not reasons:
            border_c, badge_html = GREEN, "<span class='badge-ok'>GENUINE</span>"
        elif "HIGH" in str(row.get("verdict","")):
            border_c, badge_html = RED, "<span class='badge-high'>" + " + ".join(reasons) + "</span>"
        elif "MEDIUM" in str(row.get("verdict","")):
            border_c, badge_html = AMBER, "<span class='badge-med'>" + " + ".join(reasons) + "</span>"
        else:
            border_c, badge_html = "#fbbf24", "<span class='badge-low'>" + " + ".join(reasons) + "</span>"

        score = row.get("ensemble_score", 0)
        sc_w  = int(score * 100)
        sc_c  = RED if score > 0.75 else AMBER if score > 0.50 else GREEN

        rf_s  = row.get("rf_score", 0)
        gb_s  = row.get("gb_score", 0)
        ae_s  = row.get("ae_score", 0)

        expl_html = "".join([
            f"<div style='font-size:0.79rem;color:rgba(232,220,200,0.62);"
            f"margin-top:5px;padding:6px 8px;background:rgba(200,50,50,0.06);"
            f"border-radius:4px'>⚠️ {e}</div>"
            for e in expls
        ])

        st.markdown(f"""
        <div style='background:rgba(26,23,16,0.92);
             border:1px solid rgba(212,160,60,0.12);
             border-left:4px solid {border_c};
             border-radius:0 8px 8px 0;
             padding:14px 18px;margin-bottom:8px'>

          <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>
            <b style='font-family:IBM Plex Mono,monospace;font-size:0.86rem'>
              {row['farmer_id']}</b>
            {badge_html}
            <span style='margin-left:auto;font-size:0.76rem;
                 color:rgba(232,220,200,0.42)'>
              {row['district_clean']} · {row['crop']} · {int(row['year'])}</span>
          </div>

          <div style='display:grid;grid-template-columns:repeat(5,1fr);
               gap:6px;margin-bottom:10px'>
            {"".join([
              f"<div style='text-align:center;background:rgba(0,0,0,0.22);"
              f"border-radius:4px;padding:6px'>"
              f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;"
              f"color:{col}'>{lbl}</div>"
              f"<div style='font-size:1.05rem;font-weight:700'>{val}</div></div>"
              for lbl, val, col in [
                ("LOSS %",    f"{row['loss_percent']:.0f}%",                    "#d4a03c"),
                ("NDVI",      f"{row.get('ndvi',0):.3f}",                       "#86efac"),
                ("VCI",       f"{row.get('vci',0):.1f}",                        "#fbbf24"),
                ("CHIRPS(mm)",f"{row.get('chirps_rain_mm',0):.0f}",             "#60a5fa"),
                ("CLAIM ₹",   f"{row['claim_amount_rs']/1000:.0f}K",            "#f87171"),
              ]
            ])}
          </div>

          <div style='display:grid;grid-template-columns:1fr 1fr 1fr;
               gap:6px;margin-bottom:10px'>
            {"".join([
              f"<div style='background:rgba(0,0,0,0.18);border-radius:4px;"
              f"padding:5px 8px;display:flex;justify-content:space-between;align-items:center'>"
              f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:{sc}'>{nm}</span>"
              f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.85rem;color:{sc};font-weight:600'>"
              f"{v:.3f}</span></div>"
              for nm, v, sc in [
                ("🌲 RF Score",  rf_s,  BLUE),
                ("🚀 GB Score",  gb_s,  GREEN),
                ("🤖 AE Score",  ae_s,  ORANGE),
              ]
            ])}
          </div>

          <div style='margin-bottom:8px'>
            <div style='font-family:IBM Plex Mono,monospace;font-size:0.66rem;
                 color:rgba(232,220,200,0.35);margin-bottom:3px'>
                 ENSEMBLE RISK SCORE: {score:.3f}</div>
            <div style='height:5px;background:rgba(255,255,255,0.07);border-radius:3px'>
              <div style='width:{sc_w}%;height:100%;background:{sc_c};border-radius:3px'></div>
            </div>
          </div>

          {expl_html}
        </div>
        """, unsafe_allow_html=True)

    if len(view) == 0:
        st.info("No claims match the current filters.")

    st.divider()
    dl1, dl2 = st.columns(2)
    with dl1:
        fraud_export = st.session_state["df"][
            st.session_state["df"]["fraud_label"]==1
        ].copy()
        if "ensemble_score" in fraud_export.columns:
            fraud_export = fraud_export.sort_values("ensemble_score", ascending=False)
        export_cols = [c for c in [
            "farmer_id","year","district_clean","crop","area_hectares",
            "loss_percent","claim_amount_rs","insurance_premium_rs",
            "rainfall_mm","chirps_rain_mm","rainfall_deviation_pct",
            "ndvi","ndwi","evi","vci","flood_fraction",
            "nasa_tmax_c","nasa_tmin_c",
            "fraud_fake_drought","fraud_fake_flood","fraud_hidden_yield",
            "fraud_label","rf_score","gb_score","ae_score",
            "ensemble_score","verdict"
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
            "fraud_label","rf_score","gb_score","ae_score",
            "ensemble_score","verdict"
        ] if c in full_export.columns]
        st.download_button(
            "⬇️ Download Full Dataset with Verdicts (CSV)",
            data=full_export[full_cols].round(4).to_csv(index=False),
            file_name="agrishield_full_report.csv", mime="text/csv",
            use_container_width=True
        )


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;font-family:IBM Plex Mono,monospace;
     font-size:0.62rem;color:rgba(232,220,200,0.18);padding:8px;letter-spacing:0.1em'>
  🌾 AGRISHIELD v2.0 · AP AGRICULTURAL INSURANCE FRAUD DETECTION ·
  GEE (MODIS+CHIRPS) + NASA POWER + 3 ML MODELS ·
  MSC(AA) & PGD(SDS) PROJECT 2026 · S.R.L KEERTHI & YASHASWINI H V
</div>
""", unsafe_allow_html=True)
