# ============================================================
# 🌾 AGRI SHIELD — Andhra Pradesh Crop Insurance Fraud Detection
# Authors: S.R.L Keerthi & Yashaswini H V
# MSC (AA) & PGD (SDS) Python Project 2026
#
# Data Sources:
#   • Synthetic AP insurance dataset (12,000 rows, 2000–2025)
#   • GEE Satellite: MODIS NDVI, NDWI, EVI + CHIRPS rainfall
#   • NASA POWER: Temperature & Solar Radiation
#
# ML Models: Random Forest + Gradient Boosting + Ensemble
# Fraud Types: Fake Drought | Fake Flood | Hidden Good Yield
# ============================================================

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                              accuracy_score, confusion_matrix,
                              precision_score, recall_score, f1_score)

# ─────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriShield — Fraud Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# CSS — AMBER/EARTH TONED PROFESSIONAL DASHBOARD
# Warm agricultural palette: deep brown, amber, cream, green
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0f0e0b !important;
    color: #e8dcc8 !important;
}
[data-testid="stMain"] { background: #0f0e0b !important; }

/* Grain texture overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    opacity: 0.4;
}

/* ── Typography ── */
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }
p, li, label, div { font-family: 'IBM Plex Sans', sans-serif !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1a1710 !important;
    border-right: 1px solid rgba(212, 160, 60, 0.2) !important;
}
[data-testid="stSidebar"] * { color: #e8dcc8 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(212, 160, 60, 0.06) !important;
    border: 1px solid rgba(212, 160, 60, 0.2) !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    color: rgba(212,160,60,0.7) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    color: #d4a03c !important;
    font-size: 1.8rem !important;
}
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(212,160,60,0.2) !important;
    gap: 4px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: rgba(232,220,200,0.5) !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 20px !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #d4a03c !important;
    border-bottom: 2px solid #d4a03c !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background: transparent !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    background: rgba(212,160,60,0.12) !important;
    color: #d4a03c !important;
    border: 1px solid rgba(212,160,60,0.4) !important;
    border-radius: 4px !important;
    padding: 8px 20px !important;
    letter-spacing: 0.08em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(212,160,60,0.22) !important;
    border-color: #d4a03c !important;
}

/* ── Select / Slider ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div {
    background: rgba(26,23,16,0.9) !important;
    border: 1px solid rgba(212,160,60,0.25) !important;
    color: #e8dcc8 !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: rgba(212,160,60,0.15) !important;
}
[data-testid="stSlider"] > div > div > div > div > div {
    background: #d4a03c !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(26,23,16,0.7) !important;
    border: 1px solid rgba(212,160,60,0.15) !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    color: #d4a03c !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Alerts ── */
.stSuccess { background: rgba(34,85,34,0.15) !important; border-color: rgba(100,180,100,0.3) !important; }
.stWarning { background: rgba(180,120,0,0.12) !important; border-color: rgba(212,160,60,0.3) !important; }
.stError   { background: rgba(140,30,30,0.12) !important; border-color: rgba(220,60,60,0.3) !important; }

/* ── HR ── */
hr { border-color: rgba(212,160,60,0.15) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(212,160,60,0.25) !important;
    border-radius: 8px !important;
    background: rgba(26,23,16,0.4) !important;
}

/* ── Custom card ── */
.agri-card {
    background: rgba(26,23,16,0.85);
    border: 1px solid rgba(212,160,60,0.2);
    border-radius: 8px;
    padding: 18px 20px;
    position: relative;
}
.agri-card::before {
    content: '';
    position: absolute; top: 0; left: 10%; right: 10%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,160,60,0.4), transparent);
}

/* ── Risk badge ── */
.badge-high   { background: rgba(200,50,50,0.15); color: #f87171; border: 1px solid rgba(200,50,50,0.3); padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.badge-medium { background: rgba(200,130,0,0.15); color: #fbbf24; border: 1px solid rgba(200,130,0,0.3); padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.badge-low    { background: rgba(50,130,50,0.15);  color: #86efac; border: 1px solid rgba(50,130,50,0.3);  padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.badge-ok     { background: rgba(30,90,30,0.15);   color: #4ade80; border: 1px solid rgba(30,90,30,0.3);   padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }

/* ── Section header ── */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #d4a03c;
    border-bottom: 1px solid rgba(212,160,60,0.2);
    padding-bottom: 8px;
    margin-bottom: 16px;
}

small { color: rgba(232,220,200,0.5) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.72rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# PLOTLY THEME — dark amber
# ─────────────────────────────────────────────────────────
PLOT_BG   = "rgba(15,14,11,0)"
PAPER_BG  = "rgba(15,14,11,0)"
GRID_COL  = "rgba(212,160,60,0.08)"
TEXT_COL  = "#e8dcc8"
AMBER     = "#d4a03c"
RED       = "#f87171"
GREEN     = "#86efac"
BLUE      = "#60a5fa"
ORANGE    = "#fb923c"

def plot_layout(fig, title="", height=320):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Playfair Display", color=AMBER, size=14)),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="IBM Plex Sans", color=TEXT_COL, size=11),
        height=height,
        margin=dict(l=40, r=20, t=45, b=40),
        legend=dict(bgcolor="rgba(26,23,16,0.8)", bordercolor="rgba(212,160,60,0.2)",
                    borderwidth=1, font=dict(size=10)),
        xaxis=dict(gridcolor=GRID_COL, linecolor="rgba(212,160,60,0.15)"),
        yaxis=dict(gridcolor=GRID_COL, linecolor="rgba(212,160,60,0.15)"),
    )
    return fig


# ─────────────────────────────────────────────────────────
# DATA LOADING & PROCESSING
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process(ins_path: str, sat_path: str) -> pd.DataFrame:
    """
    Load insurance + satellite CSVs, merge them, engineer features,
    apply 3 satellite-based fraud rules, and return the full DataFrame.
    """
    df  = pd.read_csv(ins_path)
    sat = pd.read_csv(sat_path)

    # ── Scale loss_percent if stored as 0–1 ──────────────────
    if df["loss_percent"].max() <= 1.0:
        df["loss_percent"] = df["loss_percent"] * 100

    df["year"]          = df["year"].astype(int)
    df["district_clean"] = df["district"].str.strip().str.title()

    # ── Clean satellite table ─────────────────────────────────
    sat = sat.drop(columns=["system:index", ".geo"], errors="ignore")
    sat["year"]     = sat["year"].astype(int)
    sat["district"] = sat["district"].str.strip().str.title()

    # ── Merge on district + year ──────────────────────────────
    df = df.merge(
        sat[["district","year","ndvi","ndwi","evi","chirps_rain_mm","flood_fraction"]],
        left_on=["district_clean","year"],
        right_on=["district","year"],
        how="left"
    ).drop(columns=["district_y"], errors="ignore")

    # ── VCI: Vegetation Condition Index (0–100) ───────────────
    # Measures how healthy vegetation is vs district historical range
    ndvi_min = df.groupby("district_clean")["ndvi"].transform("min")
    ndvi_max = df.groupby("district_clean")["ndvi"].transform("max")
    df["vci"] = ((df["ndvi"] - ndvi_min) / (ndvi_max - ndvi_min + 1e-8) * 100).clip(0, 100)

    # ── Satellite condition flags ─────────────────────────────
    df["drought_detected_sat"] = (df["vci"] < 35).astype(int)
    df["flood_detected_sat"]   = (df["flood_fraction"] > 0.01).astype(int)
    ndvi_p75 = df.groupby("district_clean")["ndvi"].transform(lambda x: x.quantile(0.75))
    df["good_vegetation_sat"]  = (df["ndvi"] >= ndvi_p75).astype(int)

    chirps_median = df.groupby("district_clean")["chirps_rain_mm"].transform("median")

    # ════════════════════════════════════════════════════════
    # FRAUD RULE 1: FAKE DROUGHT
    # Farmer claims drought damage (loss ≥ 20%)
    # BUT satellite shows VCI > 60 (vegetation healthy)
    # AND CHIRPS rainfall was ≥ 80% of normal
    # ════════════════════════════════════════════════════════
    df["fraud_fake_drought"] = (
        (df["loss_percent"] >= 20) &
        (df["vci"] > 60) &
        (df["chirps_rain_mm"] > chirps_median * 0.80)
    ).astype(int)

    # ════════════════════════════════════════════════════════
    # FRAUD RULE 2: FAKE FLOOD
    # Farmer claims flood damage (loss ≥ 20%)
    # BUT flood_fraction < 0.5% (satellite sees NO standing water)
    # AND NDWI < 0.10 (no water body signal)
    # ════════════════════════════════════════════════════════
    df["fraud_fake_flood"] = (
        (df["loss_percent"] >= 20) &
        (df["flood_fraction"] < 0.005) &
        (df["ndwi"] < 0.10)
    ).astype(int)

    # ════════════════════════════════════════════════════════
    # FRAUD RULE 3: HIDDEN GOOD YIELD
    # Farmer claims crop failure (loss ≥ 20%)
    # BUT NDVI is in top 25% for district (crops were GREEN)
    # AND VCI > 65 (vegetation historically excellent)
    # ════════════════════════════════════════════════════════
    ndvi_p75c = df.groupby("district_clean")["ndvi"].transform(lambda x: x.quantile(0.75))
    df["fraud_hidden_yield"] = (
        (df["loss_percent"] >= 20) &
        (df["ndvi"] >= ndvi_p75c) &
        (df["vci"] > 65)
    ).astype(int)

    # Combined fraud label
    df["fraud_label"] = (
        df["fraud_fake_drought"] |
        df["fraud_fake_flood"]   |
        df["fraud_hidden_yield"]
    ).astype(int)

    # ── Financial feature engineering ────────────────────────
    df["claim_per_hectare"]    = df["claim_amount_rs"] / (df["area_hectares"] + 0.01)
    df["claim_premium_ratio"]  = df["claim_amount_rs"] / (df["insurance_premium_rs"] + 0.01)
    df["claim_value_ratio"]    = df["claim_amount_rs"] / (df["crop_value_rs"] + 0.01)
    df["production_per_hectare"] = df["production_tons"] / (df["area_hectares"] + 0.01)
    df["rainfall_deviation_pct"] = (
        (df["rainfall_mm"] - df["chirps_rain_mm"]) /
        (df["chirps_rain_mm"] + 0.01) * 100
    )
    df["rainfall_mismatch_flag"] = (df["rainfall_deviation_pct"].abs() > 40).astype(int)

    le = LabelEncoder()
    df["crop_encoded"] = le.fit_transform(df["crop"])

    return df


@st.cache_resource(show_spinner=False)
def train_models(df_hash):
    """
    Train Random Forest + Gradient Boosting on the processed data.
    Returns fitted models, scaler, predictions, and scores.
    """
    # Get data from cache via session state
    df = st.session_state["df"]

    FEATURES = [
        "area_hectares", "production_per_hectare", "loss_percent",
        "claim_per_hectare", "claim_premium_ratio", "claim_value_ratio",
        "rainfall_mm", "ndvi", "ndwi", "evi", "vci",
        "chirps_rain_mm", "flood_fraction",
        "rainfall_deviation_pct", "rainfall_mismatch_flag", "crop_encoded"
    ]

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["fraud_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train)
    X_test_sc    = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train_sc, y_train)
    rf_preds = rf.predict(X_test_sc)
    rf_proba = rf.predict_proba(X_test_sc)[:, 1]

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
    )
    gb.fit(X_train_sc, y_train)
    gb_preds = gb.predict(X_test_sc)
    gb_proba = gb.predict_proba(X_test_sc)[:, 1]

    # Ensemble score on full dataset
    X_full    = df[FEATURES].fillna(df[FEATURES].median())
    X_full_sc = scaler.transform(X_full)
    df["rf_score"]       = rf.predict_proba(X_full_sc)[:, 1]
    df["gb_score"]       = gb.predict_proba(X_full_sc)[:, 1]
    df["ensemble_score"] = 0.6 * df["rf_score"] + 0.4 * df["gb_score"]

    # Human-readable verdict
    def assign_verdict(row):
        s = row["ensemble_score"]
        if   row["fraud_fake_drought"]   == 1: reason = "Fake Drought"
        elif row["fraud_fake_flood"]     == 1: reason = "Fake Flood"
        elif row["fraud_hidden_yield"]   == 1: reason = "Hidden Good Yield"
        else:                                   reason = None
        if   s >= 0.75: return f"HIGH RISK — {reason}"   if reason else "HIGH RISK"
        elif s >= 0.50: return f"MEDIUM RISK — {reason}" if reason else "MEDIUM RISK"
        elif s >= 0.25: return "LOW RISK"
        else:           return "GENUINE"

    df["verdict"] = df.apply(assign_verdict, axis=1)
    st.session_state["df"] = df

    return {
        "rf": rf, "gb": gb, "scaler": scaler,
        "features": FEATURES,
        "y_test": y_test,
        "rf_preds": rf_preds, "rf_proba": rf_proba,
        "gb_preds": gb_preds, "gb_proba": gb_proba,
        "rf_acc": accuracy_score(y_test, rf_preds),
        "rf_auc": roc_auc_score(y_test, rf_proba),
        "rf_f1":  f1_score(y_test, rf_preds),
        "gb_acc": accuracy_score(y_test, gb_preds),
        "gb_auc": roc_auc_score(y_test, gb_proba),
        "gb_f1":  f1_score(y_test, gb_preds),
        "rf_cm":  confusion_matrix(y_test, rf_preds),
        "gb_cm":  confusion_matrix(y_test, gb_preds),
    }


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 20px'>
        <div style='font-family:Playfair Display,serif;font-size:1.5rem;
             color:#d4a03c;font-weight:700'>🌾 AgriShield</div>
        <div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;
             color:rgba(232,220,200,0.4);letter-spacing:0.2em;margin-top:4px'>
             FRAUD DETECTION SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📂 Data Files**")

    # Default paths or uploaded
    ins_file = st.file_uploader("Insurance CSV", type=["csv"],
                                 help="ap_synthetic_agri_insurance_2000_2025_12000rows.csv")
    sat_file = st.file_uploader("Satellite CSV (GEE)", type=["csv"],
                                 help="AP_satellite_indices.csv")

    # Resolve paths
    DEFAULT_INS = "/mnt/user-data/uploads/ap_synthetic_agri_insurance_2000_2025_12000rows__1_.csv"
    DEFAULT_SAT = "/mnt/user-data/uploads/AP_satellite_indices.csv"
    ins_path = ins_file if ins_file else DEFAULT_INS
    sat_path = sat_file if sat_file else DEFAULT_SAT

    st.divider()
    st.markdown("**🔧 Filters**")

    # Load data first to get filter options
    if "df" not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state["df"] = load_and_process(ins_path, sat_path)

    df_all = st.session_state["df"]

    selected_districts = st.multiselect(
        "📍 District",
        sorted(df_all["district_clean"].dropna().unique()),
        default=sorted(df_all["district_clean"].dropna().unique())
    )
    selected_crops = st.multiselect(
        "🌱 Crop",
        sorted(df_all["crop"].unique()),
        default=sorted(df_all["crop"].unique())
    )
    year_range = st.slider(
        "📅 Year Range",
        int(df_all["year"].min()), int(df_all["year"].max()),
        (int(df_all["year"].min()), int(df_all["year"].max()))
    )
    risk_filter = st.selectbox(
        "🚨 Risk Level",
        ["All Claims", "HIGH RISK Only", "MEDIUM RISK Only", "Genuine Only"]
    )

    st.divider()

    # Model training button
    if st.button("🤖 RUN ML MODELS", use_container_width=True):
        st.session_state["models_trained"] = False

    if "models_trained" not in st.session_state:
        st.session_state["models_trained"] = False

    with st.spinner("Training models...") if not st.session_state["models_trained"] else st.empty():
        if not st.session_state["models_trained"]:
            st.session_state["model_results"] = train_models(id(df_all))
            st.session_state["models_trained"] = True

    if st.session_state["models_trained"]:
        st.success("✅ Models ready")

    st.divider()
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;
    color:rgba(232,220,200,0.3);line-height:1.8'>
    📡 DATA SOURCES<br>
    GEE: MODIS NDVI/NDWI/EVI<br>
    GEE: CHIRPS Rainfall<br>
    GEE: Flood Fraction<br>
    NASA POWER: Temp/Solar<br><br>
    🤖 MODELS<br>
    Random Forest (200 trees)<br>
    Gradient Boosting<br>
    Ensemble (60/40 mix)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────
df = st.session_state["df"].copy()
if not selected_districts: selected_districts = df["district_clean"].unique()
if not selected_crops:     selected_crops = df["crop"].unique()

df = df[
    df["district_clean"].isin(selected_districts) &
    df["crop"].isin(selected_crops) &
    df["year"].between(year_range[0], year_range[1])
]

if risk_filter == "HIGH RISK Only":
    df = df[df.get("verdict", pd.Series(dtype=str)).str.startswith("HIGH", na=False)]
elif risk_filter == "MEDIUM RISK Only":
    df = df[df.get("verdict", pd.Series(dtype=str)).str.startswith("MEDIUM", na=False)]
elif risk_filter == "Genuine Only":
    df = df[df.get("verdict", pd.Series(dtype=str)) == "GENUINE"]

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:24px 0 16px'>
    <div style='font-family:Playfair Display,serif;font-size:2.4rem;
         font-weight:900;color:#d4a03c;line-height:1.1'>
         🌾 AgriShield
    </div>
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;
         color:rgba(232,220,200,0.45);letter-spacing:0.25em;margin-top:6px'>
         ANDHRA PRADESH · CROP INSURANCE FRAUD DETECTION SYSTEM
    </div>
    <div style='font-family:IBM Plex Sans,sans-serif;font-size:0.9rem;
         color:rgba(232,220,200,0.6);margin-top:8px;max-width:700px'>
         Uses real GEE satellite imagery (NDVI, NDWI, EVI, CHIRPS) to cross-check
         insurance claims — catching farmers who claim damage when crops were actually healthy.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────────────────
total_claims  = len(df)
fraud_count   = df["fraud_label"].sum()
genuine_count = total_claims - fraud_count
fraud_rate    = fraud_count / total_claims * 100 if total_claims else 0
fake_drought  = df["fraud_fake_drought"].sum()
fake_flood    = df["fraud_fake_flood"].sum()
hidden_yield  = df["fraud_hidden_yield"].sum()
est_loss_cr   = (df[df["fraud_label"]==1]["claim_amount_rs"].sum() / 1e7)

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("📋 Total Claims",   f"{total_claims:,}")
m2.metric("🚨 Fraud Detected", f"{fraud_count:,}",   f"{fraud_rate:.1f}% rate", delta_color="inverse")
m3.metric("✅ Genuine",        f"{genuine_count:,}")
m4.metric("🏜️ Fake Drought",   f"{fake_drought:,}")
m5.metric("🌊 Fake Flood",     f"{fake_flood:,}")
m6.metric("💰 Est. Fraud Loss", f"₹{est_loss_cr:.1f} Cr")

st.divider()

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠  Overview",
    "🛰️  Satellite Evidence",
    "🤖  ML Models",
    "📍  District Analysis",
    "🔍  Claim Inspector"
])


# ════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW (for laypeople)
# ════════════════════════════════════════════════════════
with tab1:
    # Explainer for layperson
    st.markdown("""
    <div class='agri-card' style='margin-bottom:20px'>
        <div style='font-family:Playfair Display,serif;font-size:1.1rem;color:#d4a03c;margin-bottom:10px'>
            📖 How does AgriShield detect fraud?
        </div>
        <div style='font-family:IBM Plex Sans,sans-serif;font-size:0.88rem;
             color:rgba(232,220,200,0.75);line-height:1.8'>
            <b style='color:#d4a03c'>The Problem:</b>
            Some farmers file insurance claims saying their crops were destroyed by
            drought or flood — even when their fields were completely fine.
            This cheats the system and hurts honest farmers.<br><br>
            <b style='color:#d4a03c'>Our Solution — 3 Satellite Checks:</b><br>
            🏜️ <b>Fake Drought Check:</b> We look at NASA/ESA satellite images.
            If the <i>Vegetation Condition Index (VCI)</i> is above 60 — meaning crops
            were visibly GREEN and healthy — but the farmer said there was a drought,
            <b>that's a red flag.</b><br><br>
            🌊 <b>Fake Flood Check:</b> Satellites can detect standing water
            (flood fraction). If a farmer claims flood damage but the satellite
            sees <i>zero standing water</i> and NDWI (water index) is low, <b>that's suspicious.</b><br><br>
            🌿 <b>Hidden Good Yield Check:</b> NDVI measures how green and dense
            the crops are. If NDVI is in the top 25% for the district (crops were
            excellent) but the farmer claims total crop failure, <b>that's a mismatch.</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        # Fraud type pie chart
        fraud_types = {
            "🏜️ Fake Drought": fake_drought,
            "🌊 Fake Flood":   fake_flood,
            "🌿 Hidden Yield": hidden_yield,
        }
        fraud_types_clean = {k: v for k, v in fraud_types.items() if v > 0}
        genuine_only = total_claims - fraud_count

        fig_pie = go.Figure(go.Pie(
            labels=["✅ Genuine"] + list(fraud_types_clean.keys()),
            values=[genuine_only] + list(fraud_types_clean.values()),
            hole=0.55,
            marker_colors=["#4ade80", "#f87171", "#60a5fa", "#fbbf24"],
            textfont=dict(family="IBM Plex Mono", size=11),
        ))
        fig_pie.update_layout(
            title="Claims: Genuine vs Fraud Type",
            paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
            font=dict(family="IBM Plex Sans", color=TEXT_COL),
            height=320, margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(bgcolor="rgba(26,23,16,0.8)",
                        bordercolor="rgba(212,160,60,0.2)", borderwidth=1),
            title_font=dict(family="Playfair Display", color=AMBER, size=14),
            annotations=[dict(text=f"<b>{fraud_rate:.1f}%</b><br>Fraud Rate",
                              x=0.5, y=0.5, font_size=14,
                              font_family="IBM Plex Mono",
                              font_color=AMBER, showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Fraud type bar with explanation
        fig_bar = go.Figure()
        bar_data = [
            ("🏜️ Fake Drought",  fake_drought,  "#f87171",
             "Claimed drought,\nsatellite shows rain was OK"),
            ("🌊 Fake Flood",    fake_flood,    "#60a5fa",
             "Claimed flood,\nno water detected by satellite"),
            ("🌿 Hidden Yield",  hidden_yield,  "#86efac",
             "Claimed crop failed,\nsatellite shows green crops"),
        ]
        for label, val, color, _ in bar_data:
            fig_bar.add_trace(go.Bar(
                x=[val], y=[label], orientation="h",
                marker_color=color, marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=1,
                text=[f" {val:,}"],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=11, color=TEXT_COL),
                showlegend=False
            ))
        fig_bar = plot_layout(fig_bar, "Fraud Cases by Type", 280)
        fig_bar.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar, use_container_width=True)

        # What each fraud type means in plain language
        for label, val, color, desc in bar_data:
            pct = val / fraud_count * 100 if fraud_count else 0
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;padding:6px 10px;
                 margin-bottom:4px;border-left:3px solid {color};
                 background:rgba(26,23,16,0.6);border-radius:0 4px 4px 0'>
                <div>
                    <div style='font-weight:600;font-size:0.82rem'>{label}</div>
                    <div style='font-size:0.72rem;color:rgba(232,220,200,0.5)'>{desc}</div>
                </div>
                <div style='margin-left:auto;font-family:IBM Plex Mono,monospace;
                     font-size:0.85rem;color:{color}'>{pct:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # Trend over years
    st.markdown("<div class='section-header'>📈 Fraud Trend Over Time</div>", unsafe_allow_html=True)
    yearly = df.groupby("year").agg(
        total=("fraud_label","count"),
        fraud=("fraud_label","sum"),
        fake_drought=("fraud_fake_drought","sum"),
        fake_flood=("fraud_fake_flood","sum"),
        hidden_yield=("fraud_hidden_yield","sum"),
    ).reset_index()
    yearly["fraud_rate"] = yearly["fraud"] / yearly["total"] * 100

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["fake_drought"],
        mode="lines+markers", name="Fake Drought",
        line=dict(color="#f87171", width=2),
        marker=dict(size=5)
    ))
    fig_trend.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["fake_flood"],
        mode="lines+markers", name="Fake Flood",
        line=dict(color="#60a5fa", width=2),
        marker=dict(size=5)
    ))
    fig_trend.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["hidden_yield"],
        mode="lines+markers", name="Hidden Yield",
        line=dict(color="#86efac", width=2),
        marker=dict(size=5)
    ))
    fig_trend = plot_layout(fig_trend, "Annual Fraud Cases by Type (2000–2025)", 300)
    fig_trend.update_layout(xaxis_title="Year", yaxis_title="Number of Fraud Cases")
    st.plotly_chart(fig_trend, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 2 — SATELLITE EVIDENCE
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='agri-card' style='margin-bottom:18px'>
        <div style='font-family:Playfair Display,serif;font-size:1rem;
             color:#d4a03c;margin-bottom:8px'>🛰️ What the Satellites Tell Us</div>
        <div style='font-size:0.85rem;color:rgba(232,220,200,0.7);line-height:1.7'>
            <b style='color:#86efac'>NDVI</b> (Normalized Difference Vegetation Index):
            How green and dense the crops are. Values near 1.0 = lush crops; near 0 = bare ground.<br>
            <b style='color:#60a5fa'>NDWI</b> (Normalized Difference Water Index):
            Water presence in the landscape. Positive = water detected; negative = dry.<br>
            <b style='color:#fbbf24'>VCI</b> (Vegetation Condition Index):
            Compares this year's NDVI to historical range. Below 35 = drought stress; above 60 = healthy.<br>
            <b style='color:#f87171'>CHIRPS Rainfall</b>: Satellite-measured actual rainfall
            vs farmer-reported — big gaps reveal false claims.
        </div>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)

    with sc1:
        # NDVI vs VCI scatter coloured by fraud type
        sample = df.dropna(subset=["ndvi","vci"]).sample(min(2000, len(df)), random_state=42)
        color_map = {0: "#4ade80", 1: "#f87171"}
        fig_scatter = go.Figure()
        for label, name, color in [(0,"Genuine","#4ade80"),(1,"Fraud","#f87171")]:
            sub = sample[sample["fraud_label"]==label]
            fig_scatter.add_trace(go.Scatter(
                x=sub["ndvi"], y=sub["vci"],
                mode="markers", name=name,
                marker=dict(color=color, size=4, opacity=0.5,
                            line=dict(width=0)),
            ))
        fig_scatter.add_vline(x=sample["ndvi"].quantile(0.75),
                              line_dash="dash", line_color=AMBER,
                              annotation_text="NDVI 75th pct",
                              annotation_font_color=AMBER)
        fig_scatter.add_hline(y=60, line_dash="dash", line_color="#60a5fa",
                              annotation_text="VCI=60 (healthy threshold)",
                              annotation_font_color="#60a5fa")
        fig_scatter = plot_layout(fig_scatter, "NDVI vs VCI — Fraud Cluster", 340)
        fig_scatter.update_layout(
            xaxis_title="NDVI (vegetation health — higher = greener crops)",
            yaxis_title="VCI (0–100, above 60 = healthy)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown("<small>Red dots in the top-right quadrant = fraud: crops were healthy but farmer claimed damage</small>",
                    unsafe_allow_html=True)

    with sc2:
        # Rainfall mismatch: farmer reported vs CHIRPS satellite
        fig_rain = go.Figure()
        for label, name, color in [(0,"Genuine","#4ade80"),(1,"Fraud","#f87171")]:
            sub = df[df["fraud_label"]==label]["rainfall_deviation_pct"].dropna()
            fig_rain.add_trace(go.Histogram(
                x=sub.clip(-150, 150),
                nbinsx=50, name=name,
                marker_color=color, opacity=0.65,
            ))
        fig_rain.add_vline(x=40,  line_dash="dash", line_color=AMBER)
        fig_rain.add_vline(x=-40, line_dash="dash", line_color=AMBER)
        fig_rain = plot_layout(fig_rain, "Rainfall: Farmer Reported vs Satellite (CHIRPS)", 340)
        fig_rain.update_layout(
            barmode="overlay",
            xaxis_title="Deviation % (positive = farmer over-reported rainfall)",
            yaxis_title="Number of Claims",
        )
        st.plotly_chart(fig_rain, use_container_width=True)
        st.markdown("<small>Fraud claims cluster near 0 deviation — farmers over-reported rainfall (claimed drought) but CHIRPS shows rain was normal</small>",
                    unsafe_allow_html=True)

    # Satellite index averages by fraud type
    st.markdown("<div class='section-header'>📊 Satellite Indices: Genuine vs Fraud</div>",
                unsafe_allow_html=True)
    ia1, ia2, ia3 = st.columns(3)

    for col, title, fraud_col, color_fraud, explanation in [
        (ia1, "NDVI by Fraud Type", "fraud_label",
         "#f87171", "High NDVI with fraud claim = crops were green = hidden good yield"),
        (ia2, "VCI by Fraud Type", "fraud_label",
         "#fbbf24", "VCI > 60 with drought claim = vegetation healthy = fake drought"),
        (ia3, "CHIRPS Rain (mm) by Fraud Type", "fraud_label",
         "#60a5fa", "High CHIRPS rain with drought claim = rain was adequate = fake drought"),
    ]:
        idx_col = "ndvi" if "NDVI" in title else "vci" if "VCI" in title else "chirps_rain_mm"
        g0 = df[df["fraud_label"]==0][idx_col].dropna()
        g1 = df[df["fraud_label"]==1][idx_col].dropna()
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=g0, name="Genuine", marker_color="#4ade80",
                                  boxmean=True, line_width=1.5))
        fig_box.add_trace(go.Box(y=g1, name="Fraud",   marker_color=color_fraud,
                                  boxmean=True, line_width=1.5))
        fig_box = plot_layout(fig_box, title, 300)
        col.plotly_chart(fig_box, use_container_width=True)
        col.markdown(f"<small>{explanation}</small>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 3 — ML MODELS
# ════════════════════════════════════════════════════════
with tab3:
    if not st.session_state.get("models_trained"):
        st.info("👈 Click **RUN ML MODELS** in the sidebar to train models")
        st.stop()

    res = st.session_state["model_results"]

    st.markdown("""
    <div class='agri-card' style='margin-bottom:18px'>
        <div style='font-family:Playfair Display,serif;font-size:1rem;
             color:#d4a03c;margin-bottom:8px'>🤖 How the AI Models Work</div>
        <div style='font-size:0.85rem;color:rgba(232,220,200,0.7);line-height:1.7'>
            <b>Random Forest</b>: Builds 200 decision trees, each learning different patterns
            in the data. A claim is flagged if most trees vote it suspicious.<br>
            <b>Gradient Boosting</b>: Trains trees sequentially — each new tree corrects the
            mistakes of the previous one, making it progressively smarter.<br>
            <b>Ensemble Score</b>: Combines both models (60% RF + 40% GB) for the final verdict.
            This is more reliable than any single model alone.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model metrics
    ml1, ml2, ml3, ml4, ml5, ml6 = st.columns(6)
    ml1.metric("🌲 RF Accuracy",  f"{res['rf_acc']*100:.2f}%")
    ml2.metric("🌲 RF ROC-AUC",   f"{res['rf_auc']:.4f}")
    ml3.metric("🌲 RF F1 Score",  f"{res['rf_f1']:.4f}")
    ml4.metric("🚀 GB Accuracy",  f"{res['gb_acc']*100:.2f}%")
    ml5.metric("🚀 GB ROC-AUC",   f"{res['gb_auc']:.4f}")
    ml6.metric("🚀 GB F1 Score",  f"{res['gb_f1']:.4f}")

    st.divider()
    mc1, mc2 = st.columns(2)

    with mc1:
        # Model comparison bar
        models     = ["Random Forest", "Gradient Boosting"]
        accuracies = [res["rf_acc"]*100, res["gb_acc"]*100]
        aucs       = [res["rf_auc"],      res["gb_auc"]]
        f1s        = [res["rf_f1"],       res["gb_f1"]]

        fig_comp = go.Figure()
        for metric, values, color in [
            ("Accuracy (%)", accuracies, AMBER),
            ("ROC-AUC",      aucs,       "#86efac"),
            ("F1 Score",     f1s,        "#60a5fa"),
        ]:
            fig_comp.add_trace(go.Bar(
                name=metric, x=models, y=values,
                marker_color=color, marker_line_color="rgba(0,0,0,0.3)",
                marker_line_width=1,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", size=10),
            ))
        fig_comp = plot_layout(fig_comp, "Model Performance Comparison", 340)
        fig_comp.update_layout(barmode="group", yaxis=dict(range=[0.85, 1.05]))
        st.plotly_chart(fig_comp, use_container_width=True)

    with mc2:
        # Feature importance
        features = res["features"]
        importances = res["rf"].feature_importances_
        imp_df = pd.Series(importances, index=features).sort_values()
        sat_features = {"ndvi","ndwi","evi","vci","chirps_rain_mm",
                        "flood_fraction","rainfall_deviation_pct"}
        colors_fi = [RED if f in sat_features else AMBER for f in imp_df.index]
        fig_imp = go.Figure(go.Bar(
            x=imp_df.values, y=imp_df.index,
            orientation="h",
            marker_color=colors_fi,
            marker_line_color="rgba(0,0,0,0.2)",
            marker_line_width=1,
            text=[f"{v:.4f}" for v in imp_df.values],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=9),
        ))
        fig_imp = plot_layout(fig_imp, "Feature Importance (🔴 = Satellite Features)", 340)
        fig_imp.update_layout(xaxis_title="Importance Score")
        st.plotly_chart(fig_imp, use_container_width=True)

    # Confusion matrices
    st.markdown("<div class='section-header'>🎯 Confusion Matrices — How accurately do models classify?</div>",
                unsafe_allow_html=True)
    cm1_col, cm2_col = st.columns(2)

    for col, cm, title in [
        (cm1_col, res["rf_cm"], "Random Forest"),
        (cm2_col, res["gb_cm"], "Gradient Boosting"),
    ]:
        z    = cm
        text = [[f"TN: {cm[0,0]}", f"FP: {cm[0,1]}"],
                [f"FN: {cm[1,0]}", f"TP: {cm[1,1]}"]]
        fig_cm = go.Figure(go.Heatmap(
            z=z, x=["Predicted Genuine", "Predicted Fraud"],
            y=["Actual Genuine", "Actual Fraud"],
            text=text, texttemplate="%{text}",
            colorscale=[[0,"#1a1710"],[0.5,"rgba(212,160,60,0.3)"],[1,"#d4a03c"]],
            showscale=False,
            textfont=dict(family="IBM Plex Mono", size=12, color=TEXT_COL),
        ))
        fig_cm = plot_layout(fig_cm, f"{title} — Confusion Matrix", 300)
        col.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("📖 How to read the confusion matrix"):
        st.markdown("""
        - **True Negative (TN)**: Genuine claims correctly identified as genuine ✅
        - **True Positive (TP)**: Fraud claims correctly caught 🚨
        - **False Positive (FP)**: Genuine claims wrongly flagged as fraud ⚠️
        - **False Negative (FN)**: Fraud claims that slipped through — missed detections 🔴

        A good model maximises TP (catching fraud) while minimising FP (not harassing honest farmers).
        """)

    # Ensemble score distribution
    st.markdown("<div class='section-header'>📊 Ensemble Risk Score Distribution</div>",
                unsafe_allow_html=True)
    df_score = st.session_state["df"]
    if "ensemble_score" in df_score.columns:
        fig_score = go.Figure()
        for label, name, color in [(0,"Genuine","#4ade80"),(1,"Fraud","#f87171")]:
            sub = df_score[df_score["fraud_label"]==label]["ensemble_score"].dropna()
            fig_score.add_trace(go.Histogram(
                x=sub, nbinsx=50, name=name,
                marker_color=color, opacity=0.7
            ))
        fig_score.add_vline(x=0.50, line_dash="dash", line_color=AMBER,
                            annotation_text="Medium Risk threshold")
        fig_score.add_vline(x=0.75, line_dash="dash", line_color=RED,
                            annotation_text="High Risk threshold")
        fig_score = plot_layout(fig_score, "Distribution of Ensemble Risk Scores", 300)
        fig_score.update_layout(barmode="overlay", xaxis_title="Risk Score (0 = safe, 1 = high risk)")
        st.plotly_chart(fig_score, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 4 — DISTRICT ANALYSIS
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>📍 District-wise Fraud Analysis</div>",
                unsafe_allow_html=True)

    # District summary table
    dist_summary = df.groupby("district_clean").agg(
        total=("fraud_label","count"),
        fraud=("fraud_label","sum"),
        fake_drought=("fraud_fake_drought","sum"),
        fake_flood=("fraud_fake_flood","sum"),
        hidden_yield=("fraud_hidden_yield","sum"),
        avg_ndvi=("ndvi","mean"),
        avg_vci=("vci","mean"),
    ).reset_index()
    dist_summary["fraud_rate"] = (dist_summary["fraud"] / dist_summary["total"] * 100).round(1)
    dist_summary = dist_summary.sort_values("fraud_rate", ascending=False)

    da1, da2 = st.columns(2)

    with da1:
        # Fraud rate by district horizontal bar
        fig_dist = go.Figure(go.Bar(
            x=dist_summary["fraud_rate"],
            y=dist_summary["district_clean"],
            orientation="h",
            marker_color=[RED if r > 15 else AMBER if r > 8 else GREEN
                          for r in dist_summary["fraud_rate"]],
            marker_line_color="rgba(0,0,0,0.3)", marker_line_width=1,
            text=[f"{r}%" for r in dist_summary["fraud_rate"]],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=10),
        ))
        fig_dist = plot_layout(fig_dist, "Fraud Rate by District (%)", 400)
        fig_dist.update_layout(
            xaxis_title="Fraud Rate (%)",
            yaxis=dict(autorange="reversed")
        )
        fig_dist.add_vline(x=10, line_dash="dash", line_color=AMBER,
                           annotation_text="10% threshold")
        st.plotly_chart(fig_dist, use_container_width=True)

    with da2:
        # Stacked fraud type bar by district
        fig_stack = go.Figure()
        for col_name, color, label in [
            ("fake_drought", "#f87171", "Fake Drought"),
            ("fake_flood",   "#60a5fa", "Fake Flood"),
            ("hidden_yield", "#86efac", "Hidden Yield"),
        ]:
            fig_stack.add_trace(go.Bar(
                name=label,
                x=dist_summary["district_clean"],
                y=dist_summary[col_name],
                marker_color=color,
                marker_line_color="rgba(0,0,0,0.2)",
                marker_line_width=1,
            ))
        fig_stack = plot_layout(fig_stack, "Fraud Cases by Type & District", 400)
        fig_stack.update_layout(
            barmode="stack",
            xaxis=dict(tickangle=-35),
            yaxis_title="Number of Cases"
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    # Satellite health by district
    st.markdown("<div class='section-header'>🛰️ Satellite Health Indices by District</div>",
                unsafe_allow_html=True)

    sat_dist = df.groupby("district_clean").agg(
        avg_ndvi=("ndvi","mean"),
        avg_vci=("vci","mean"),
        avg_chirps=("chirps_rain_mm","mean"),
        avg_ndwi=("ndwi","mean"),
    ).reset_index().sort_values("avg_ndvi", ascending=False)

    fig_sat = make_subplots(rows=1, cols=2,
                            subplot_titles=("Average NDVI by District",
                                            "Average VCI by District"))
    fig_sat.add_trace(go.Bar(
        x=sat_dist["avg_ndvi"], y=sat_dist["district_clean"],
        orientation="h",
        marker_color=[GREEN if v > 0.5 else AMBER if v > 0.35 else RED
                      for v in sat_dist["avg_ndvi"]],
        name="NDVI", text=[f"{v:.3f}" for v in sat_dist["avg_ndvi"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9),
    ), row=1, col=1)
    fig_sat.add_trace(go.Bar(
        x=sat_dist["avg_vci"], y=sat_dist["district_clean"],
        orientation="h",
        marker_color=[GREEN if v > 55 else AMBER if v > 40 else RED
                      for v in sat_dist["avg_vci"]],
        name="VCI", text=[f"{v:.1f}" for v in sat_dist["avg_vci"]],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=9),
    ), row=1, col=2)

    fig_sat.update_layout(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family="IBM Plex Sans", color=TEXT_COL, size=11),
        height=400, showlegend=False,
        margin=dict(l=130, r=60, t=50, b=40),
    )
    fig_sat.update_xaxes(gridcolor=GRID_COL)
    fig_sat.update_yaxes(gridcolor=GRID_COL, autorange="reversed")
    st.plotly_chart(fig_sat, use_container_width=True)

    # District summary table
    st.markdown("<div class='section-header'>📋 Full District Summary Table</div>",
                unsafe_allow_html=True)
    st.markdown("<small>Sorted by fraud rate (highest first). Red = high risk district.</small>",
                unsafe_allow_html=True)

    for _, row in dist_summary.iterrows():
        rate = row["fraud_rate"]
        if   rate > 15: risk_badge = "<span class='badge-high'>HIGH RISK DISTRICT</span>"
        elif rate > 8:  risk_badge = "<span class='badge-medium'>MODERATE</span>"
        else:           risk_badge = "<span class='badge-ok'>LOW RISK</span>"

        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:12px;padding:10px 16px;
             margin-bottom:4px;
             background:rgba(26,23,16,{'0.9' if rate>15 else '0.6'});
             border:1px solid rgba({'200,50,50' if rate>15 else '212,160,60' if rate>8 else '50,130,50'},0.2);
             border-left:3px solid {'#f87171' if rate>15 else AMBER if rate>8 else '#4ade80'};
             border-radius:0 6px 6px 0'>
            <span style='font-weight:700;flex:1.2'>{row['district_clean']}</span>
            {risk_badge}
            <span style='font-family:IBM Plex Mono,monospace;font-size:0.8rem'>
                Total: <b>{int(row['total'])}</b></span>
            <span style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#f87171'>
                Fraud: <b>{int(row['fraud'])}</b> ({rate}%)</span>
            <span style='font-size:0.75rem;color:rgba(232,220,200,0.5)'>
                🏜️{int(row['fake_drought'])} &nbsp;
                🌊{int(row['fake_flood'])} &nbsp;
                🌿{int(row['hidden_yield'])}</span>
            <span style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;
                 color:rgba(232,220,200,0.5)'>
                NDVI:{row['avg_ndvi']:.3f} VCI:{row['avg_vci']:.1f}</span>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# TAB 5 — CLAIM INSPECTOR
# ════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>🔍 Individual Claim Inspector</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.85rem;color:rgba(232,220,200,0.6);margin-bottom:16px'>
    Search for a specific farmer or browse high-risk claims. Each claim shows
    <b>exactly why</b> it was flagged — what the satellite saw vs what the farmer claimed.
    </div>
    """, unsafe_allow_html=True)

    # Search / filter
    ci1, ci2, ci3 = st.columns([2, 2, 1])
    with ci1:
        search_farmer = st.text_input("🔎 Search Farmer ID", placeholder="e.g. GUN_2000_0")
    with ci2:
        inspect_district = st.selectbox("📍 District",
                                         ["All"] + sorted(df["district_clean"].dropna().unique().tolist()))
    with ci3:
        show_only_fraud = st.checkbox("Show fraud only", value=True)

    # Apply search filters
    view_df = st.session_state["df"].copy()
    view_df = view_df[view_df["district_clean"].isin(selected_districts)]
    if inspect_district != "All":
        view_df = view_df[view_df["district_clean"] == inspect_district]
    if search_farmer:
        view_df = view_df[view_df["farmer_id"].str.contains(search_farmer, case=False, na=False)]
    if show_only_fraud:
        view_df = view_df[view_df["fraud_label"] == 1]

    st.markdown(f"<small>Showing {min(50, len(view_df))} of {len(view_df)} claims</small>",
                unsafe_allow_html=True)

    # Show claims as cards
    for _, row in view_df.head(50).iterrows():
        # Determine fraud reason and explanation
        reasons = []
        explanations = []
        if row.get("fraud_fake_drought") == 1:
            reasons.append("🏜️ Fake Drought")
            explanations.append(
                f"Farmer claimed drought (loss {row['loss_percent']:.0f}%) "
                f"BUT satellite VCI = {row['vci']:.1f} (above 60 = healthy vegetation) "
                f"AND CHIRPS rainfall = {row.get('chirps_rain_mm',0):.0f} mm (adequate rain)"
            )
        if row.get("fraud_fake_flood") == 1:
            reasons.append("🌊 Fake Flood")
            explanations.append(
                f"Farmer claimed flood damage BUT satellite flood fraction = "
                f"{row.get('flood_fraction',0):.4f} (near zero — no standing water) "
                f"AND NDWI = {row.get('ndwi',0):.3f} (negative = no water body)"
            )
        if row.get("fraud_hidden_yield") == 1:
            reasons.append("🌿 Hidden Good Yield")
            explanations.append(
                f"Farmer claimed crop failure BUT NDVI = {row.get('ndvi',0):.3f} "
                f"(top 25% of district — crops were visibly GREEN) "
                f"AND VCI = {row.get('vci',0):.1f} (vegetation in excellent condition)"
            )

        if not reasons and row["fraud_label"] == 0:
            risk_col  = "#4ade80"
            badge_html = "<span class='badge-ok'>GENUINE</span>"
        elif not reasons:
            risk_col  = AMBER
            badge_html = "<span class='badge-medium'>FLAGGED</span>"
        else:
            risk_col  = RED
            badge_html = "<span class='badge-high'>" + " + ".join(reasons) + "</span>"

        score = row.get("ensemble_score", 0)
        score_bar_width = int(score * 100)
        score_color = RED if score > 0.75 else AMBER if score > 0.50 else GREEN

        st.markdown(f"""
        <div style='background:rgba(26,23,16,0.9);border:1px solid rgba(212,160,60,0.15);
             border-left:4px solid {risk_col};border-radius:0 8px 8px 0;
             padding:14px 18px;margin-bottom:8px'>

            <!-- Header row -->
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>
                <b style='font-family:IBM Plex Mono,monospace;font-size:0.88rem'>
                    {row['farmer_id']}</b>
                {badge_html}
                <span style='margin-left:auto;font-size:0.78rem;color:rgba(232,220,200,0.5)'>
                    {row['district_clean']} · {row['crop']} · {int(row['year'])}</span>
            </div>

            <!-- Key numbers -->
            <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-bottom:10px'>
                <div style='text-align:center;background:rgba(0,0,0,0.2);border-radius:4px;padding:6px'>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#d4a03c'>
                        Loss Claimed</div>
                    <div style='font-size:1.1rem;font-weight:700'>{row['loss_percent']:.0f}%</div>
                </div>
                <div style='text-align:center;background:rgba(0,0,0,0.2);border-radius:4px;padding:6px'>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#86efac'>
                        NDVI</div>
                    <div style='font-size:1.1rem;font-weight:700'>{row.get('ndvi',0):.3f}</div>
                </div>
                <div style='text-align:center;background:rgba(0,0,0,0.2);border-radius:4px;padding:6px'>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#fbbf24'>
                        VCI</div>
                    <div style='font-size:1.1rem;font-weight:700'>{row.get('vci',0):.1f}</div>
                </div>
                <div style='text-align:center;background:rgba(0,0,0,0.2);border-radius:4px;padding:6px'>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#60a5fa'>
                        CHIRPS Rain</div>
                    <div style='font-size:1.1rem;font-weight:700'>{row.get('chirps_rain_mm',0):.0f}mm</div>
                </div>
                <div style='text-align:center;background:rgba(0,0,0,0.2);border-radius:4px;padding:6px'>
                    <div style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:#f87171'>
                        Claim (₹)</div>
                    <div style='font-size:1.1rem;font-weight:700'>
                        {row['claim_amount_rs']/1000:.1f}K</div>
                </div>
            </div>

            <!-- Risk score bar -->
            <div style='margin-bottom:8px'>
                <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                     color:rgba(232,220,200,0.4);margin-bottom:3px'>
                     RISK SCORE: {score:.3f}</div>
                <div style='height:5px;background:rgba(255,255,255,0.08);border-radius:3px'>
                    <div style='width:{score_bar_width}%;height:100%;
                         background:{score_color};border-radius:3px'></div>
                </div>
            </div>

            <!-- Fraud explanation -->
            {''.join([f"<div style='font-size:0.8rem;color:rgba(232,220,200,0.65);margin-top:4px;padding:6px 8px;background:rgba(255,80,80,0.06);border-radius:4px'>⚠️ {exp}</div>" for exp in explanations]) if explanations else ""}
        </div>
        """, unsafe_allow_html=True)

    if len(view_df) == 0:
        st.info("No claims match the current filters. Try adjusting the sidebar filters.")

    # Download section
    st.divider()
    st.markdown("**📥 Download Reports**")
    dl1, dl2 = st.columns(2)

    with dl1:
        fraud_df = st.session_state["df"][st.session_state["df"]["fraud_label"]==1].copy()
        if "ensemble_score" in fraud_df.columns:
            fraud_df = fraud_df.sort_values("ensemble_score", ascending=False)
        report_cols = [c for c in [
            "farmer_id","year","district_clean","crop",
            "area_hectares","loss_percent","claim_amount_rs",
            "insurance_premium_rs","rainfall_mm","chirps_rain_mm",
            "rainfall_deviation_pct","ndvi","ndwi","evi","vci",
            "flood_fraction","fraud_fake_drought","fraud_fake_flood",
            "fraud_hidden_yield","fraud_label",
            "ensemble_score","verdict"
        ] if c in fraud_df.columns]
        csv_fraud = fraud_df[report_cols].round(4).to_csv(index=False)
        st.download_button(
            "⬇️ Download Fraud Report (CSV)",
            data=csv_fraud,
            file_name="agrishield_fraud_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    with dl2:
        full_df = st.session_state["df"].copy()
        full_cols = [c for c in [
            "farmer_id","year","district_clean","crop",
            "area_hectares","loss_percent","claim_amount_rs",
            "ndvi","vci","chirps_rain_mm","fraud_label",
            "ensemble_score","verdict"
        ] if c in full_df.columns]
        csv_full = full_df[full_cols].round(4).to_csv(index=False)
        st.download_button(
            "⬇️ Download Full Dataset with Verdicts (CSV)",
            data=csv_full,
            file_name="agrishield_full_report.csv",
            mime="text/csv",
            use_container_width=True
        )


# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;font-family:IBM Plex Mono,monospace;
     font-size:0.65rem;color:rgba(232,220,200,0.2);padding:10px;letter-spacing:0.12em'>
    🌾 AGRISHIELD v1.0 · ANDHRA PRADESH AGRICULTURAL INSURANCE FRAUD DETECTION ·
    DATA: GEE MODIS + CHIRPS + NASA POWER · MSC(AA) & PGD(SDS) PROJECT 2026 ·
    S.R.L KEERTHI & YASHASWINI H V
</div>
""", unsafe_allow_html=True)
