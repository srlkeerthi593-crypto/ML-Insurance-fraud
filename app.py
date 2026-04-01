"""
╔══════════════════════════════════════════════════════════════════╗
║   AgriShield — ML-Based Crop Insurance Fraud Detection Dashboard ║
║   GEE Satellite Indices: NDVI · NDWI · EVI · SAVI · VCI         ║
╚══════════════════════════════════════════════════════════════════╝
Run:
    pip install streamlit pandas numpy scikit-learn imbalanced-learn \
                matplotlib seaborn plotly
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriShield — Fraud Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }

h1,h2,h3 { font-family: 'Space Mono', monospace !important; letter-spacing: -0.5px; }
h1 { color: #58a6ff; font-size: 1.6rem !important; }
h2 { color: #79c0ff; font-size: 1.2rem !important; }
h3 { color: #a5d6ff; font-size: 1.0rem !important; }

.metric-card {
    background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 18px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-val { font-family:'Space Mono',monospace; font-size:2rem; font-weight:700; }
.metric-lbl { font-size:0.78rem; color:#8b949e; margin-top:4px; text-transform:uppercase; letter-spacing:1px; }

.verdict-HIGH  { color:#ff7b72; font-weight:700; }
.verdict-MED   { color:#ffa657; font-weight:700; }
.verdict-LOW   { color:#e3b341; font-weight:600; }
.verdict-GEN   { color:#56d364; font-weight:600; }

.sat-badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:0.72rem; font-weight:700; letter-spacing:0.5px; margin:2px;
}
.badge-ndvi  { background:#1a4731; color:#56d364; border:1px solid #238636; }
.badge-ndwi  { background:#0d2a4a; color:#58a6ff; border:1px solid #1f6feb; }
.badge-evi   { background:#2a1f4a; color:#bc8cff; border:1px solid #6e40c9; }
.badge-savi  { background:#4a2a0d; color:#ffa657; border:1px solid #9e6a03; }
.badge-vci   { background:#3d1f1f; color:#ff7b72; border:1px solid #da3633; }

.fraud-type-box {
    border-radius:10px; padding:16px; margin:8px 0;
    border-left: 4px solid;
}
.drought-box { border-color:#e3b341; background:#272115; }
.flood-box   { border-color:#58a6ff; background:#15202b; }
.yield-box   { border-color:#56d364; background:#1a2e1a; }

div[data-testid="stDataFrame"] { border-radius:8px; }
.stSelectbox>div>div { background:#21262d; border:1px solid #30363d; }
.stSlider>div>div { color:#58a6ff; }

/* tabs */
button[data-baseweb="tab"] { font-family:'Space Mono',monospace; font-size:0.85rem; }
button[data-baseweb="tab"][aria-selected="true"] { color:#58a6ff !important; border-bottom-color:#58a6ff !important; }

.info-pill {
    background:#21262d; border:1px solid #30363d; border-radius:8px;
    padding:10px 14px; font-size:0.83rem; color:#8b949e;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_engineer(ins_path: str, sat_path: str) -> pd.DataFrame:
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler

    df  = pd.read_csv(ins_path)
    sat = pd.read_csv(sat_path)

    # ── Clean satellite ────────────────────────────────────────
    sat = sat.drop(columns=["system:index", ".geo"], errors="ignore")
    sat["year"]     = sat["year"].astype(int)
    sat["district"] = sat["district"].str.strip()

    # ── Clean insurance ───────────────────────────────────────
    if df["loss_percent"].max() <= 1.0:
        df["loss_percent"] = df["loss_percent"] * 100
    df["year"] = (
        pd.to_datetime(df["claim_date"], errors="coerce").dt.year
        if "claim_date" in df.columns else df["year"]
    )
    df["year"] = df["year"].astype(int)
    df["district_clean"] = df["district"].str.strip().str.title()

    # ── Financial features ────────────────────────────────────
    df["claim_per_hectare"]      = df["claim_amount_rs"] / df["area_hectares"]
    df["claim_premium_ratio"]    = df["claim_amount_rs"] / (df["insurance_premium_rs"] + 1)
    df["claim_value_ratio"]      = df["claim_amount_rs"] / (df["crop_value_rs"] + 1)
    df["production_per_hectare"] = df["production_tons"] / df["area_hectares"]

    le = LabelEncoder()
    df["crop_encoded"] = le.fit_transform(df["crop"].astype(str))

    # ── Merge satellite ───────────────────────────────────────
    df = df.merge(
        sat[["district","year","ndvi","ndwi","evi","chirps_rain_mm","flood_fraction"]],
        left_on=["district_clean","year"],
        right_on=["district","year"], how="left"
    ).drop(columns=["district_y"], errors="ignore")

    # ── VCI + SAVI ────────────────────────────────────────────
    ndvi_min = df.groupby("district_clean")["ndvi"].transform("min")
    ndvi_max = df.groupby("district_clean")["ndvi"].transform("max")
    df["vci"] = ((df["ndvi"] - ndvi_min) / (ndvi_max - ndvi_min + 1e-8) * 100).clip(0, 100)

    # SAVI: Soil-Adjusted Vegetation Index  (L=0.5 typical)
    L = 0.5
    df["savi"] = ((df["ndvi"] - 0) / (df["ndvi"] + L + 1e-8)) * (1 + L)

    # ── Satellite-derived flags ───────────────────────────────
    df["drought_detected_sat"] = (df["vci"] < 35).astype(int)
    df["flood_detected_sat"]   = (df["flood_fraction"] > 0.01).astype(int)
    ndvi_p60 = df.groupby("district_clean")["ndvi"].transform(lambda x: x.quantile(0.60))
    df["good_vegetation_sat"]  = (df["ndvi"] >= ndvi_p60).astype(int)

    # ── Fraud detection rules ─────────────────────────────────
    chirps_median = df.groupby("district_clean")["chirps_rain_mm"].transform("median")
    ndvi_p75      = df.groupby("district_clean")["ndvi"].transform(lambda x: x.quantile(0.75))

    df["fraud_fake_drought"] = (
        (df["loss_percent"] >= 20) &
        (df["vci"] > 60) &
        (df["chirps_rain_mm"] > chirps_median * 0.80)
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

    df["fraud_satellite_flag"] = (
        df["fraud_fake_drought"] |
        df["fraud_fake_flood"]   |
        df["fraud_hidden_yield"]
    ).astype(int)
    df["fraud_label"] = df["fraud_satellite_flag"].copy()

    # ── Rainfall mismatch ─────────────────────────────────────
    df["rainfall_deviation_pct"] = (
        (df["rainfall_mm"] - df["chirps_rain_mm"]) /
        (df["chirps_rain_mm"] + 1) * 100
    )
    df["rainfall_mismatch_flag"] = (df["rainfall_deviation_pct"].abs() > 40).astype(int)

    return df


@st.cache_data(show_spinner=False)
def train_models(df: pd.DataFrame):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import (classification_report, roc_auc_score,
                                  accuracy_score, confusion_matrix)
    from imblearn.over_sampling import SMOTE

    FEATURES = [
        "area_hectares","production_per_hectare","loss_percent",
        "claim_per_hectare","claim_premium_ratio","claim_value_ratio",
        "rainfall_mm","ndvi","ndwi","evi","savi","vci",
        "chirps_rain_mm","flood_fraction","rainfall_deviation_pct",
        "rainfall_mismatch_flag","crop_encoded"
    ]

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["fraud_label"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_tr)
    Xte_sc = scaler.transform(X_te)

    smote = SMOTE(random_state=42, k_neighbors=5)
    Xtr_bal, ytr_bal = smote.fit_resample(Xtr_sc, y_tr)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 min_samples_leaf=5, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(Xtr_bal, ytr_bal)
    rf_p = rf.predict(Xte_sc)
    rf_pr = rf.predict_proba(Xte_sc)[:,1]

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                     max_depth=5, random_state=42)
    gb.fit(Xtr_bal, ytr_bal)
    gb_p = gb.predict(Xte_sc)
    gb_pr = gb.predict_proba(Xte_sc)[:,1]

    # ── Full-dataset ensemble score ───────────────────────────
    X_full = df[FEATURES].fillna(df[FEATURES].median())
    Xf_sc  = scaler.transform(X_full)
    rf_full = rf.predict_proba(Xf_sc)[:,1]
    gb_full = gb.predict_proba(Xf_sc)[:,1]
    ens     = 0.6 * rf_full + 0.4 * gb_full

    # normalize to 0-1 (GB can be <0 in edge cases)
    ens = np.clip(ens, 0, 1)
    df["rf_score"]       = rf_full
    df["gb_score"]       = gb_full
    df["ensemble_score"] = ens

    def verdict(row):
        sc = row["ensemble_score"]
        if   row["fraud_fake_drought"]: reason = "Fake Drought"
        elif row["fraud_fake_flood"]:   reason = "Fake Flood"
        elif row["fraud_hidden_yield"]: reason = "Hidden Good Yield"
        else:                           reason = None
        if   sc >= 0.75: tag = "HIGH RISK"
        elif sc >= 0.50: tag = "MEDIUM RISK"
        elif sc >= 0.25: tag = "LOW RISK"
        else:            tag = "GENUINE"
        return f"{tag} — {reason}" if (reason and sc >= 0.25) else tag

    df["verdict"] = df.apply(verdict, axis=1)

    metrics = {
        "rf":  {"acc": accuracy_score(y_te, rf_p),
                "auc": roc_auc_score(y_te, rf_pr),
                "cm":  confusion_matrix(y_te, rf_p),
                "rep": classification_report(y_te, rf_p,
                                              target_names=["Genuine","Fraud"],
                                              output_dict=True)},
        "gb":  {"acc": accuracy_score(y_te, gb_p),
                "auc": roc_auc_score(y_te, gb_pr),
                "cm":  confusion_matrix(y_te, gb_p),
                "rep": classification_report(y_te, gb_p,
                                              target_names=["Genuine","Fraud"],
                                              output_dict=True)},
    }
    importances = pd.Series(rf.feature_importances_, index=FEATURES)\
                    .sort_values(ascending=False)

    return df, metrics, importances, FEATURES, scaler, rf, gb


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px">
      <div style="font-family:'Space Mono',monospace;font-size:1.3rem;
                  color:#58a6ff;font-weight:700;">🛰️ AgriShield</div>
      <div style="color:#8b949e;font-size:0.75rem;margin-top:4px;">
        ML-Based Crop Insurance<br>Fraud Detection · AP India
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Upload Data")
    ins_file = st.file_uploader("Insurance Dataset (CSV)", type="csv",
                                 key="ins", help="ap_synthetic_agri_insurance_*.csv")
    sat_file = st.file_uploader("Satellite Indices (CSV)", type="csv",
                                 key="sat", help="AP_satellite_indices.csv")

    use_demo = st.checkbox("▶ Use built-in demo data", value=True,
                            help="Loads the bundled CSV files if present")

    st.markdown("---")
    st.markdown("### 🔬 Satellite Indices")
    st.markdown("""
    <span class='sat-badge badge-ndvi'>NDVI</span>
    <span class='sat-badge badge-ndwi'>NDWI</span>
    <span class='sat-badge badge-evi'>EVI</span>
    <span class='sat-badge badge-savi'>SAVI</span>
    <span class='sat-badge badge-vci'>VCI</span>
    """, unsafe_allow_html=True)
    st.caption("Derived from Google Earth Engine (GEE) · MODIS MOD13A3")

    st.markdown("---")
    st.markdown("### ⚙️ Fraud Thresholds")
    vci_thr   = st.slider("VCI Healthy Threshold", 40, 80, 60, 5,
                           help="VCI above this = 'vegetation was healthy'")
    loss_thr  = st.slider("Min Loss % to Flag", 10, 40, 20, 5,
                           help="Only examine claims above this loss %")
    ndwi_thr  = st.slider("NDWI Water Threshold", 0.05, 0.30, 0.10, 0.05,
                           help="NDWI below this = no water body")
    rain_dev  = st.slider("Rainfall Mismatch %", 20, 60, 40, 5,
                           help="Flag if farmer-reported vs CHIRPS deviates by this %")

    st.markdown("---")
    st.markdown("### 📊 View Filters")
    show_only_fraud = st.checkbox("Show only flagged claims", value=False)
    min_score       = st.slider("Min Ensemble Score", 0.0, 1.0, 0.0, 0.05)

    st.markdown("---")
    st.markdown("""
    <div class='info-pill'>
      <b>Data Sources</b><br>
      🛰️ GEE MODIS · CHIRPS v2.0<br>
      🌡️ NASA POWER (temperature)<br>
      📋 Synthetic AP Insurance data
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RESOLVE DATA PATHS
# ══════════════════════════════════════════════════════════════════════════════

import os, tempfile

ins_path_default = "ap_synthetic_agri_insurance_2000_2025_12000rows.csv"
sat_path_default = "AP_satellite_indices.csv"

def save_uploaded(uf):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(uf.read()); tmp.close()
    return tmp.name

ins_path = sat_path = None

if ins_file and sat_file:
    ins_path = save_uploaded(ins_file)
    sat_path = save_uploaded(sat_file)
elif use_demo:
    # Try both common locations
    for p in [ins_path_default,
              f"/mnt/user-data/uploads/{ins_path_default}"]:
        if os.path.exists(p):
            ins_path = p; break
    for p in [sat_path_default,
              f"/mnt/user-data/uploads/{sat_path_default}"]:
        if os.path.exists(p):
            sat_path = p; break

# ── If we have data, load it ──────────────────────────────────────────────────
if ins_path and sat_path:
    with st.spinner("🔄 Loading data & engineering features…"):
        df_raw = load_and_engineer(ins_path, sat_path)

    with st.spinner("🤖 Training ML models (RF + GB)…"):
        df, metrics, importances, FEATURES, scaler, rf_model, gb_model = train_models(df_raw.copy())

    # Re-apply sidebar threshold overrides on fraud flags
    chirps_med = df.groupby("district_clean")["chirps_rain_mm"].transform("median")
    ndvi_p75   = df.groupby("district_clean")["ndvi"].transform(lambda x: x.quantile(0.75))

    df["fraud_fake_drought"] = (
        (df["loss_percent"] >= loss_thr) &
        (df["vci"] > vci_thr) &
        (df["chirps_rain_mm"] > chirps_med * 0.80)
    ).astype(int)
    df["fraud_fake_flood"] = (
        (df["loss_percent"] >= loss_thr) &
        (df["flood_fraction"] < 0.005) &
        (df["ndwi"] < ndwi_thr)
    ).astype(int)
    df["fraud_hidden_yield"] = (
        (df["loss_percent"] >= loss_thr) &
        (df["ndvi"] >= ndvi_p75) &
        (df["vci"] > 65)
    ).astype(int)
    df["fraud_satellite_flag"] = (
        df["fraud_fake_drought"] | df["fraud_fake_flood"] | df["fraud_hidden_yield"]
    ).astype(int)
    df["rainfall_mismatch_flag"] = (df["rainfall_deviation_pct"].abs() > rain_dev).astype(int)

    # Apply view filters
    view_df = df.copy()
    if show_only_fraud:
        view_df = view_df[view_df["fraud_satellite_flag"] == 1]
    view_df = view_df[view_df["ensemble_score"] >= min_score]


    # ══════════════════════════════════════════════════════════════════════════
    # 4.  HEADER
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <h1>🛰️ AgriShield — Crop Insurance Fraud Detection Dashboard</h1>
    <p style='color:#8b949e;font-size:0.9rem;margin-top:-10px;'>
      Andhra Pradesh · 2000–2025 ·
      <span class='sat-badge badge-ndvi'>NDVI</span>
      <span class='sat-badge badge-ndwi'>NDWI</span>
      <span class='sat-badge badge-evi'>EVI</span>
      <span class='sat-badge badge-savi'>SAVI</span>
      <span class='sat-badge badge-vci'>VCI</span>
      powered by Google Earth Engine (MODIS + CHIRPS)
    </p>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 5.  KPI CARDS
    # ══════════════════════════════════════════════════════════════════════════
    total         = len(df)
    flagged       = df["fraud_satellite_flag"].sum()
    fake_drought  = df["fraud_fake_drought"].sum()
    fake_flood    = df["fraud_fake_flood"].sum()
    hidden_yield  = df["fraud_hidden_yield"].sum()
    high_risk     = (df["ensemble_score"] >= 0.75).sum()
    avg_ens       = df["ensemble_score"].mean()
    rain_mis      = df["rainfall_mismatch_flag"].sum()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpis = [
        (c1, str(total),         "Total Claims",       "#58a6ff"),
        (c2, str(flagged),       "Satellite Flagged",  "#ff7b72"),
        (c3, str(fake_drought),  "🏜️ Fake Drought",    "#e3b341"),
        (c4, str(fake_flood),    "🌊 Fake Flood",       "#79c0ff"),
        (c5, str(hidden_yield),  "🌿 Hidden Yield",    "#56d364"),
        (c6, str(high_risk),     "🚨 High Risk (≥.75)","#ffa657"),
    ]
    for col, val, lbl, clr in kpis:
        col.markdown(f"""
        <div class='metric-card'>
          <div class='metric-val' style='color:{clr}'>{val}</div>
          <div class='metric-lbl'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 6.  TABS
    # ══════════════════════════════════════════════════════════════════════════
    tabs = st.tabs([
        "📊 Overview",
        "🛰️ Satellite Indices",
        "🧠 ML Models",
        "🔍 Fraud Deep-Dive",
        "📋 Claims Table",
        "🗺️ District Map"
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("## 📊 Fraud Overview")
        row1_l, row1_r = st.columns([1,1])

        with row1_l:
            # Fraud type breakdown donut
            labels = ["Fake Drought","Fake Flood","Hidden Yield","Genuine"]
            values = [fake_drought, fake_flood, hidden_yield,
                      total - flagged]
            colors = ["#e3b341","#79c0ff","#56d364","#21262d"]
            fig = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.55, marker=dict(colors=colors,
                                       line=dict(color="#0d1117", width=2)),
                textinfo="percent+label",
                textfont=dict(size=12, color="#e6edf3")
            ))
            fig.update_layout(
                title="Fraud Type Distribution",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6edf3"), showlegend=False,
                height=350, margin=dict(l=10,r=10,t=40,b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        with row1_r:
            # Verdict distribution
            verdict_counts = df["verdict"].value_counts()
            color_map = {
                "GENUINE":       "#56d364",
                "LOW RISK":      "#e3b341",
            }
            bar_colors = []
            for v in verdict_counts.index:
                if   "GENUINE" in v:   bar_colors.append("#56d364")
                elif "LOW RISK" in v:  bar_colors.append("#e3b341")
                elif "MEDIUM" in v:    bar_colors.append("#ffa657")
                else:                  bar_colors.append("#ff7b72")

            fig2 = go.Figure(go.Bar(
                x=verdict_counts.values, y=verdict_counts.index,
                orientation="h", marker_color=bar_colors,
                text=verdict_counts.values, textposition="outside",
                textfont=dict(color="#e6edf3")
            ))
            fig2.update_layout(
                title="Verdict Distribution",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6edf3"), height=350,
                margin=dict(l=10,r=10,t=40,b=10),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d")
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Year trend
        st.markdown("### 📅 Fraud Flagging Trend by Year")
        yr = df.groupby("year").agg(
            total=("fraud_label","count"),
            flagged=("fraud_satellite_flag","sum")
        ).reset_index()
        yr["pct"] = yr["flagged"] / yr["total"] * 100

        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Bar(x=yr["year"], y=yr["flagged"],
                              name="Flagged Claims", marker_color="#ff7b72",
                              opacity=0.8), secondary_y=False)
        fig3.add_trace(go.Scatter(x=yr["year"], y=yr["pct"], mode="lines+markers",
                                   name="Fraud %", line=dict(color="#e3b341",width=2.5),
                                   marker=dict(size=6)), secondary_y=True)
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"), height=320,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=10,r=10,t=10,b=10),
            xaxis=dict(gridcolor="#21262d"),
        )
        fig3.update_yaxes(gridcolor="#21262d", secondary_y=False)
        fig3.update_yaxes(gridcolor="#21262d", secondary_y=True,
                           title_text="Fraud %")
        st.plotly_chart(fig3, use_container_width=True)

        # Crop breakdown
        st.markdown("### 🌾 Fraud Rate by Crop")
        crop_agg = df.groupby("crop").agg(
            claims=("fraud_label","count"),
            flagged=("fraud_satellite_flag","sum")
        ).reset_index()
        crop_agg["fraud_rate"] = crop_agg["flagged"] / crop_agg["claims"] * 100
        crop_agg = crop_agg.sort_values("fraud_rate", ascending=False)
        fig4 = px.bar(crop_agg, x="crop", y="fraud_rate",
                       color="fraud_rate",
                       color_continuous_scale=["#56d364","#e3b341","#ff7b72"],
                       labels={"fraud_rate": "Fraud Rate %", "crop": "Crop"})
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"), height=320,
            margin=dict(l=10,r=10,t=10,b=10),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig4, use_container_width=True)


    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — SATELLITE INDICES
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("## 🛰️ Satellite Indices from Google Earth Engine")

        # Index explanation cards
        e1,e2,e3,e4,e5 = st.columns(5)
        idx_info = [
            (e1,"NDVI","badge-ndvi","Normalised Difference Vegetation Index",
             "Measures greenness / vegetation health. Range −1 to 1. >0.4 = healthy crop."),
            (e2,"NDWI","badge-ndwi","Normalised Difference Water Index",
             "Detects open water / waterlogged soil. >0.1 = water present."),
            (e3,"EVI","badge-evi","Enhanced Vegetation Index",
             "Like NDVI but reduces soil & atmospheric noise. Better in high-biomass areas."),
            (e4,"SAVI","badge-savi","Soil-Adjusted Vegetation Index",
             "NDVI corrected for soil brightness. Useful in semi-arid AP districts."),
            (e5,"VCI","badge-vci","Vegetation Condition Index",
             "How this year compares to historical range. <35 = drought stress confirmed."),
        ]
        for col, name, badge, full, desc in idx_info:
            col.markdown(f"""
            <div class='metric-card' style='text-align:left;padding:14px'>
              <span class='sat-badge {badge}'>{name}</span>
              <div style='font-size:0.75rem;color:#8b949e;margin:8px 0 4px'>{full}</div>
              <div style='font-size:0.8rem;color:#c9d1d9'>{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # NDVI vs VCI scatter coloured by fraud
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### NDVI vs VCI  ·  Fraud overlay")
            fig = px.scatter(
                df.sample(min(3000, len(df)), random_state=42),
                x="ndvi", y="vci", color="fraud_label",
                color_discrete_map={0:"#56d364", 1:"#ff7b72"},
                opacity=0.4, size_max=5,
                labels={"ndvi":"NDVI","vci":"VCI","fraud_label":"Fraud"}
            )
            fig.add_vline(x=df["ndvi"].quantile(0.75), line_dash="dot",
                          line_color="#e3b341",
                          annotation_text="NDVI 75th pct", annotation_font_color="#e3b341")
            fig.add_hline(y=60, line_dash="dot", line_color="#58a6ff",
                          annotation_text="VCI=60", annotation_font_color="#58a6ff")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#e6edf3"), height=380,
                               legend=dict(bgcolor="rgba(0,0,0,0)"),
                               xaxis=dict(gridcolor="#21262d"),
                               yaxis=dict(gridcolor="#21262d"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("### EVI vs SAVI  ·  Fraud overlay")
            fig2 = px.scatter(
                df.sample(min(3000, len(df)), random_state=7),
                x="evi", y="savi", color="fraud_label",
                color_discrete_map={0:"#56d364", 1:"#ff7b72"},
                opacity=0.4,
                labels={"evi":"EVI","savi":"SAVI","fraud_label":"Fraud"}
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#e6edf3"), height=380,
                                legend=dict(bgcolor="rgba(0,0,0,0)"),
                                xaxis=dict(gridcolor="#21262d"),
                                yaxis=dict(gridcolor="#21262d"))
            st.plotly_chart(fig2, use_container_width=True)

        # CHIRPS vs reported rainfall
        st.markdown("### 🌧️ CHIRPS Satellite Rainfall vs Farmer-Reported Rainfall")
        fig3 = go.Figure()
        for label, clr, lbl in [(0,"#56d364","Genuine"),(1,"#ff7b72","Fraud")]:
            sub = df[df["fraud_label"]==label].sample(min(1500,sum(df["fraud_label"]==label)), random_state=1)
            fig3.add_trace(go.Scatter(
                x=sub["chirps_rain_mm"], y=sub["rainfall_mm"],
                mode="markers", marker=dict(color=clr, size=4, opacity=0.4),
                name=lbl
            ))
        # 1:1 line
        mx = df[["chirps_rain_mm","rainfall_mm"]].max().max()
        fig3.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode="lines",
                                   line=dict(color="#8b949e", dash="dash"),
                                   name="1:1 reference"))
        fig3.update_layout(
            xaxis_title="CHIRPS Satellite Rainfall (mm)",
            yaxis_title="Farmer-Reported Rainfall (mm)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"), height=380,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d")
        )
        st.plotly_chart(fig3, use_container_width=True)

        # District-wise NDVI heatmap
        st.markdown("### 🗺️ District-wise Mean NDVI by Year")
        ndvi_pivot = df.groupby(["district_clean","year"])["ndvi"].mean().reset_index()
        ndvi_wide  = ndvi_pivot.pivot(index="district_clean",
                                      columns="year", values="ndvi")
        fig4 = px.imshow(ndvi_wide, color_continuous_scale="RdYlGn",
                          aspect="auto", labels=dict(color="NDVI"))
        fig4.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"), height=500,
            xaxis_title="Year", yaxis_title="District"
        )
        st.plotly_chart(fig4, use_container_width=True)


    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — ML MODELS
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("## 🧠 Machine Learning Model Performance")

        c1, c2 = st.columns(2)

        def render_model_card(col, name, key):
            m = metrics[key]
            col.markdown(f"""
            <div class='metric-card' style='text-align:left;padding:18px;margin-bottom:16px'>
              <div style='font-size:1.1rem;font-weight:700;color:#79c0ff;
                          font-family:Space Mono,monospace;'>{name}</div>
              <div style='display:flex;gap:20px;margin-top:14px'>
                <div>
                  <div class='metric-val' style='color:#56d364;font-size:1.6rem'>{m['acc']:.3f}</div>
                  <div class='metric-lbl'>Accuracy</div>
                </div>
                <div>
                  <div class='metric-val' style='color:#58a6ff;font-size:1.6rem'>{m['auc']:.3f}</div>
                  <div class='metric-lbl'>ROC-AUC</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Confusion matrix
            cm    = m["cm"]
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=["Pred Genuine","Pred Fraud"],
                y=["True Genuine","True Fraud"],
                colorscale=[[0,"#0d1117"],[0.5,"#1f6feb"],[1,"#58a6ff"]],
                showscale=False,
                text=cm.astype(str),
                texttemplate="%{text}",
                textfont=dict(size=18, color="white")
            ))
            fig_cm.update_layout(
                title=f"Confusion Matrix — {name}",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6edf3"),
                height=280, margin=dict(l=10,r=10,t=40,b=10)
            )
            col.plotly_chart(fig_cm, use_container_width=True)

        render_model_card(c1, "🌲 Random Forest",    "rf")
        render_model_card(c2, "🚀 Gradient Boosting","gb")

        # Feature Importance
        st.markdown("### 📌 Feature Importance  (Random Forest)")
        sat_set = {"ndvi","ndwi","evi","savi","vci",
                   "chirps_rain_mm","flood_fraction","rainfall_deviation_pct"}
        colors  = ["#ffa657" if f in sat_set else "#58a6ff"
                   for f in importances.index[:15]]
        fig_fi  = go.Figure(go.Bar(
            x=importances.values[:15][::-1],
            y=importances.index[:15][::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.3f}" for v in importances.values[:15][::-1]],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=11)
        ))
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"), height=480,
            margin=dict(l=10,r=10,t=10,b=10),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d")
        )
        orange_patch = "🟠 = Satellite feature   🔵 = Financial/farm feature"
        st.caption(orange_patch)
        st.plotly_chart(fig_fi, use_container_width=True)

        # Ensemble score distribution
        st.markdown("### 📈 Ensemble Score Distribution")
        fig_es = go.Figure()
        for lbl, clr, nm in [(0,"#56d364","Genuine"),(1,"#ff7b72","Fraud")]:
            fig_es.add_trace(go.Histogram(
                x=df[df["fraud_label"]==lbl]["ensemble_score"],
                nbinsx=50, name=nm,
                marker_color=clr, opacity=0.7
            ))
        for thr, clr, ann in [(0.25,"#e3b341","Low Risk"),
                               (0.50,"#ffa657","Medium Risk"),
                               (0.75,"#ff7b72","High Risk")]:
            fig_es.add_vline(x=thr, line_dash="dot", line_color=clr,
                             annotation_text=ann, annotation_font_color=clr)
        fig_es.update_layout(
            barmode="overlay",
            xaxis_title="Ensemble Score",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"), height=340,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d")
        )
        st.plotly_chart(fig_es, use_container_width=True)


    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 — FRAUD DEEP-DIVE
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("## 🔍 Fraud Deep-Dive: Satellite Evidence")

        c1, c2, c3 = st.columns(3)

        # Fake Drought
        c1.markdown("""
        <div class='fraud-type-box drought-box'>
          <b>🏜️ Fake Drought Detection</b><br><br>
          <b>Rule:</b><br>
          • Loss% ≥ threshold<br>
          • <b>VCI &gt; 60</b> (vegetation was healthy)<br>
          • CHIRPS rain &gt; 80% of median<br><br>
          <b>Logic:</b> If VCI says crops were green and rainfall
          was adequate, a drought claim is suspicious.
        </div>""", unsafe_allow_html=True)

        c2.markdown("""
        <div class='fraud-type-box flood-box'>
          <b>🌊 Fake Flood Detection</b><br><br>
          <b>Rule:</b><br>
          • Loss% ≥ threshold<br>
          • <b>flood_fraction &lt; 0.005</b><br>
          • <b>NDWI &lt; 0.10</b> (no water signal)<br><br>
          <b>Logic:</b> If satellite shows &lt;0.5% flooded pixels
          and NDWI is low, no flood occurred.
        </div>""", unsafe_allow_html=True)

        c3.markdown("""
        <div class='fraud-type-box yield-box'>
          <b>🌿 Hidden Good Yield Detection</b><br><br>
          <b>Rule:</b><br>
          • Loss% ≥ threshold<br>
          • <b>NDVI ≥ 75th percentile</b><br>
          • <b>VCI &gt; 65</b><br><br>
          <b>Logic:</b> If NDVI is in top 25% for the district
          and VCI is high, the crop was thriving — not failing.
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Scatter: loss_percent vs ensemble_score
        st.markdown("### Loss % vs Ensemble Fraud Score")
        fig = px.scatter(
            df.sample(min(4000, len(df)), random_state=42),
            x="loss_percent", y="ensemble_score",
            color="fraud_satellite_flag",
            color_discrete_map={0:"#56d364", 1:"#ff7b72"},
            opacity=0.5, hover_data=["district_clean","crop","verdict"],
            labels={"loss_percent":"Loss %","ensemble_score":"Ensemble Score",
                    "fraud_satellite_flag":"Flagged"}
        )
        fig.add_hline(y=0.75, line_dash="dot", line_color="#ff7b72",
                      annotation_text="High Risk ≥0.75")
        fig.add_hline(y=0.50, line_dash="dot", line_color="#ffa657",
                      annotation_text="Medium Risk ≥0.50")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"), height=400,
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Rainfall deviation distribution
        st.markdown("### Rainfall Deviation: Farmer-Reported vs CHIRPS Satellite")
        c1, c2 = st.columns(2)
        with c1:
            fig2 = go.Figure()
            for fl, clr, nm in [(0,"#56d364","Genuine"),(1,"#ff7b72","Fraud")]:
                fig2.add_trace(go.Histogram(
                    x=df[df["fraud_label"]==fl]["rainfall_deviation_pct"].clip(-200,200),
                    nbinsx=60, name=nm, marker_color=clr, opacity=0.7
                ))
            for v in [-rain_dev, rain_dev]:
                fig2.add_vline(x=v, line_dash="dot", line_color="#e3b341")
            fig2.update_layout(
                barmode="overlay", title="Rainfall Deviation Distribution",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6edf3"), height=350,
                xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            # NDWI distribution for fraud types
            fd = df[df["fraud_fake_flood"]==1]["ndwi"].dropna()
            gd = df[df["fraud_label"]==0]["ndwi"].dropna()
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=gd.sample(min(len(gd),500),random_state=1),
                                         nbinsx=40, name="Genuine",
                                         marker_color="#56d364", opacity=0.7))
            fig3.add_trace(go.Histogram(x=fd, nbinsx=30, name="Fake Flood Flags",
                                         marker_color="#58a6ff", opacity=0.8))
            fig3.add_vline(x=ndwi_thr, line_dash="dot", line_color="#e3b341",
                           annotation_text=f"NDWI={ndwi_thr}")
            fig3.update_layout(
                barmode="overlay", title="NDWI Distribution: Genuine vs Fake Flood",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6edf3"), height=350,
                xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Top high-risk claims
        st.markdown("### 🚨 Top 20 Highest Risk Claims")
        top20_cols = ["farmer_id","year","district_clean","crop",
                      "area_hectares","loss_percent","ndvi","vci",
                      "chirps_rain_mm","flood_fraction",
                      "fraud_fake_drought","fraud_fake_flood","fraud_hidden_yield",
                      "ensemble_score","verdict"]
        top20 = df.sort_values("ensemble_score", ascending=False)[top20_cols].head(20)
        top20["ensemble_score"] = top20["ensemble_score"].round(4)
        top20["ndvi"]           = top20["ndvi"].round(4)
        top20["vci"]            = top20["vci"].round(2)

        def colour_verdict(val):
            if "HIGH"   in str(val): return "color: #ff7b72; font-weight:700"
            if "MEDIUM" in str(val): return "color: #ffa657; font-weight:700"
            if "LOW"    in str(val): return "color: #e3b341"
            return "color: #56d364"

        st.dataframe(
            top20.style.applymap(colour_verdict, subset=["verdict"]),
            use_container_width=True, height=560
        )


    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 — CLAIMS TABLE
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("## 📋 Full Claims Explorer")

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        districts = ["All"] + sorted(df["district_clean"].dropna().unique().tolist())
        crops     = ["All"] + sorted(df["crop"].dropna().unique().tolist())
        sel_dist  = fc1.selectbox("District", districts)
        sel_crop  = fc2.selectbox("Crop", crops)
        sel_year  = fc3.selectbox("Year", ["All"] + sorted(df["year"].unique().tolist()))

        fdf = view_df.copy()
        if sel_dist != "All": fdf = fdf[fdf["district_clean"] == sel_dist]
        if sel_crop != "All": fdf = fdf[fdf["crop"] == sel_crop]
        if sel_year != "All": fdf = fdf[fdf["year"] == sel_year]

        display_cols = [
            "farmer_id","year","district_clean","crop","area_hectares",
            "loss_percent","claim_amount_rs","ndvi","ndwi","evi","savi","vci",
            "chirps_rain_mm","flood_fraction","rainfall_deviation_pct",
            "fraud_fake_drought","fraud_fake_flood","fraud_hidden_yield",
            "ensemble_score","verdict"
        ]
        disp = fdf[[c for c in display_cols if c in fdf.columns]].copy()
        for col in ["ndvi","ndwi","evi","savi","vci","chirps_rain_mm",
                    "rainfall_deviation_pct","ensemble_score"]:
            if col in disp.columns:
                disp[col] = disp[col].round(4)

        st.markdown(f"**Showing {len(disp):,} claims**")
        st.dataframe(
            disp.style.applymap(colour_verdict, subset=["verdict"]
                                if "verdict" in disp.columns else []),
            use_container_width=True, height=500
        )

        # Download
        csv = disp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download filtered claims (CSV)", csv,
                            "filtered_claims.csv", "text/csv")


    # ─────────────────────────────────────────────────────────────────────────
    # TAB 6 — DISTRICT MAP
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("## 🗺️ District-level Fraud & Satellite Map")

        DIST_COORDS = {
            "Anantapur":(14.6819,77.6006),"Chittoor":(13.2172,79.1003),
            "East Godavari":(17.385,82.0),"Guntur":(16.3067,80.4365),
            "Krishna":(16.6107,80.648),"Kurnool":(15.8281,78.0373),
            "Nellore":(14.4426,79.9865),"Prakasam":(15.352,79.574),
            "Srikakulam":(18.2949,83.8938),"Visakhapatnam":(17.6868,83.2185),
            "Vizianagaram":(18.1066,83.3956),"West Godavari":(16.9174,81.34),
            "Ysr Kadapa":(14.4674,78.8241),"Alluri Sitharama Raju":(17.9,82.45),
            "Anakapalli":(17.691,83.0048),"Bapatla":(15.9046,80.467),
            "Eluru":(16.7107,81.0952),"Konaseema":(16.8167,81.9),
            "Nandyal":(15.4786,78.4836),"Ntr":(16.5793,80.637),
            "Palnadu":(16.5,79.5),"Parvathipuram Manyam":(18.7833,83.4333),
            "Sri Sathya Sai":(13.9,77.7667),"Tirupati":(13.6288,79.4192),
        }

        dist_agg = df.groupby("district_clean").agg(
            total=("fraud_label","count"),
            flagged=("fraud_satellite_flag","sum"),
            avg_ndvi=("ndvi","mean"),
            avg_vci=("vci","mean"),
            avg_chirps=("chirps_rain_mm","mean"),
            fake_drought=("fraud_fake_drought","sum"),
            fake_flood=("fraud_fake_flood","sum"),
            hidden_yield=("fraud_hidden_yield","sum"),
        ).reset_index()
        dist_agg["fraud_rate"] = dist_agg["flagged"] / dist_agg["total"] * 100
        dist_agg["lat"] = dist_agg["district_clean"].map(
            lambda x: DIST_COORDS.get(x,(None,None))[0])
        dist_agg["lon"] = dist_agg["district_clean"].map(
            lambda x: DIST_COORDS.get(x,(None,None))[1])
        dist_agg = dist_agg.dropna(subset=["lat","lon"])

        map_metric = st.selectbox(
            "Map colour by",
            ["fraud_rate","avg_ndvi","avg_vci","avg_chirps",
             "fake_drought","fake_flood","hidden_yield"]
        )

        fig_map = px.scatter_mapbox(
            dist_agg, lat="lat", lon="lon",
            size="total", color=map_metric,
            color_continuous_scale="RdYlGn_r" if "fraud" in map_metric else "YlGn",
            hover_name="district_clean",
            hover_data={
                "fraud_rate":True, "avg_ndvi":True, "avg_vci":True,
                "avg_chirps":True, "fake_drought":True,
                "fake_flood":True, "hidden_yield":True,
                "lat":False, "lon":False
            },
            size_max=45, zoom=6,
            mapbox_style="carto-darkmatter",
            labels={map_metric: map_metric.replace("_"," ").title()}
        )
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            margin=dict(l=0,r=0,t=10,b=0), height=560
        )
        st.plotly_chart(fig_map, use_container_width=True)

        # District table
        st.markdown("### District Summary")
        dist_show = dist_agg[[
            "district_clean","total","flagged","fraud_rate",
            "avg_ndvi","avg_vci","avg_chirps",
            "fake_drought","fake_flood","hidden_yield"
        ]].sort_values("fraud_rate", ascending=False)
        for col in ["fraud_rate","avg_ndvi","avg_vci","avg_chirps"]:
            dist_show[col] = dist_show[col].round(2)
        st.dataframe(dist_show, use_container_width=True)

else:
    # ─────────────────────────────────────────────────────────────────────────
    # LANDING / UPLOAD STATE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:80px 20px'>
      <div style='font-family:Space Mono,monospace;font-size:2.5rem;
                  color:#58a6ff;margin-bottom:20px'>🛰️ AgriShield</div>
      <div style='font-size:1.1rem;color:#8b949e;max-width:600px;margin:0 auto 30px'>
        ML-Based Crop Insurance Fraud Detection<br>
        powered by Google Earth Engine Satellite Indices
      </div>
      <div style='display:flex;justify-content:center;gap:12px;flex-wrap:wrap'>
        <span class='sat-badge badge-ndvi'>NDVI</span>
        <span class='sat-badge badge-ndwi'>NDWI</span>
        <span class='sat-badge badge-evi'>EVI</span>
        <span class='sat-badge badge-savi'>SAVI</span>
        <span class='sat-badge badge-vci'>VCI</span>
      </div>
      <br><br>
      <div style='color:#c9d1d9;font-size:0.95rem'>
        Upload <b>ap_synthetic_agri_insurance_*.csv</b> and <b>AP_satellite_indices.csv</b>
        in the sidebar, or enable <b>Use built-in demo data</b>.
      </div>
    </div>
    """, unsafe_allow_html=True)
