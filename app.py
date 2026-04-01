# ================================================================
# AgriShield - Crop Insurance Fraud Detection Dashboard
# For Andhra Pradesh, India
# Built for: S.R.L Keerthi & Yashaswini H V
# ================================================================
# HOW TO RUN:
#   1. pip install -r requirements.txt
#   2. streamlit run app.py
#   3. Upload your CSV file when the dashboard opens
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# ── Page Setup ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriShield – Fraud Detection",
    page_icon="🌾",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }
.stApp { background-color: #f0f4f8; }

section[data-testid="stSidebar"] {
    background: #1a2744;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #c8d8f0 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white !important;
}

.big-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a2744;
    margin-bottom: 4px;
}
.subtitle {
    font-size: 1rem;
    color: #5a7ba8;
    margin-bottom: 24px;
}

.info-box {
    background: #e8f4fd;
    border-left: 4px solid #2196F3;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 14px;
    color: #1a2744;
}
.warn-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 14px;
    color: #664d03;
}
.danger-box {
    background: #fde8e8;
    border-left: 4px solid #e53935;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 14px;
    color: #5f0000;
}
.success-box {
    background: #e8f5e9;
    border-left: 4px solid #43a047;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 14px;
    color: #1b5e20;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
    border-top: 4px solid #2196F3;
}
.metric-card.red   { border-top-color: #e53935; }
.metric-card.green { border-top-color: #43a047; }
.metric-card.amber { border-top-color: #ffc107; }
.metric-card.blue  { border-top-color: #1976d2; }

.metric-num   { font-size: 2rem; font-weight: 700; color: #1a2744; }
.metric-label { font-size: 0.78rem; color: #5a7ba8; text-transform: uppercase;
                letter-spacing: 0.8px; margin-top: 4px; }
.metric-help  { font-size: 0.75rem; color: #8aaccc; margin-top: 2px; }

.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #1a2744;
    padding: 16px 0 8px;
    border-bottom: 2px solid #dce8f5;
    margin-bottom: 16px;
}

.fraud-high   { background:#e53935;color:white;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600; }
.fraud-medium { background:#ffc107;color:#333;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600; }
.fraud-low    { background:#43a047;color:white;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600; }

.explain-card {
    background: white;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    margin-bottom: 10px;
}
.explain-title { font-weight: 600; color: #1a2744; font-size: 15px; margin-bottom: 6px; }
.explain-text  { color: #4a6a8a; font-size: 13px; line-height: 1.6; }

table.neat { width:100%; border-collapse:collapse; font-size:13px; }
table.neat th { background:#1a2744; color:white; padding:10px 12px;
                text-align:left; font-weight:500; }
table.neat td { padding:9px 12px; border-bottom:1px solid #e8f0f8; color:#2a3f5f; }
table.neat tr:hover td { background:#f0f7ff; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# HELPER FUNCTIONS
# ================================================================

DISTRICT_COORDS = {
    'Anantapur':     (14.68, 77.60),
    'Chittoor':      (13.22, 79.10),
    'East Godavari': (17.39, 82.00),
    'Guntur':        (16.31, 80.44),
    'Krishna':       (16.61, 80.65),
    'Kurnool':       (15.83, 78.04),
    'Nellore':       (14.44, 79.99),
    'Prakasam':      (15.35, 79.57),
    'Srikakulam':    (18.29, 83.89),
    'Visakhapatnam': (17.69, 83.22),
    'Vizianagaram':  (18.11, 83.40),
    'West Godavari': (16.92, 81.34),
    'Kadapa':        (14.47, 78.82),
}

# Realistic NDVI/NDWI/SAVI values per district per year
# (These simulate what GEE would return — replace get_gee_indices()
#  with real GEE calls once you have your project ID)
def get_satellite_indices(district, year):
    """
    Returns satellite index values for a district and year.
    Currently uses realistic simulation.
    TO USE REAL GEE DATA: replace this function body with your ee calls.
    """
    np.random.seed(abs(hash(f"{district}{year}")) % (2**31))
    drought_districts = ['Anantapur', 'Kurnool', 'Kadapa', 'Prakasam']
    drought_years     = [2002, 2009, 2015, 2018, 2019, 2022]

    base = 0.32 if district in drought_districts else 0.50
    if year in drought_years:
        base *= 0.60

    ndvi = float(np.clip(np.random.normal(base, 0.05), 0.05, 0.85))
    ndwi = float(np.clip(np.random.normal(0.05 if ndvi > 0.4 else -0.10, 0.05), -0.5, 0.4))
    savi = float(np.clip(ndvi * 0.82 + np.random.normal(0, 0.02), 0.02, 0.75))
    evi  = float(np.clip(ndvi * 0.88 + np.random.normal(0, 0.02), 0.02, 0.80))
    vci  = float(np.clip((ndvi - 0.08) / (0.78 - 0.08) * 100, 0, 100))
    chirps = float(np.clip(np.random.normal(
        580 if district in drought_districts else 900, 80), 150, 1400))
    if year in drought_years:
        chirps *= 0.55
    chirps = float(np.clip(chirps, 100, 1400))

    return {
        'ndvi':          round(ndvi,  3),
        'ndwi':          round(ndwi,  3),
        'savi':          round(savi,  3),
        'evi':           round(evi,   3),
        'vci':           round(vci,   1),
        'chirps_rain_mm':round(chirps,1),
    }


@st.cache_data(show_spinner=False)
def fetch_nasa_power(lat, lon, year):
    """
    Fetches real climate data from NASA POWER satellite API.
    This uses the real NASA API - no login needed!
    Parameters: temperature, rainfall, solar radiation
    """
    url = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters=PRECTOTCORR,T2M_MAX,T2M_MIN,ALLSKY_SFC_SW_DWN"
        f"&community=AG&longitude={lon}&latitude={lat}"
        f"&start={year}&end={year}&format=JSON"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()['properties']['parameter']

        def annual_sum(d):
            vals = [v for k,v in d.items()
                    if str(year) in k and v not in (-999,-99,None) and v > 0]
            return round(sum(vals), 1) if vals else None

        def annual_avg(d):
            vals = [v for k,v in d.items()
                    if str(year) in k and v not in (-999,-99,None)]
            return round(float(np.mean(vals)), 1) if vals else None

        return {
            'rainfall_mm': annual_sum(data['PRECTOTCORR']),
            'tmax_c':      annual_avg(data['T2M_MAX']),
            'tmin_c':      annual_avg(data['T2M_MIN']),
            'solar_mj':    annual_avg(data['ALLSKY_SFC_SW_DWN']),
            'source':      'NASA POWER (Real Data)'
        }
    except Exception as e:
        return {
            'rainfall_mm': None, 'tmax_c': None,
            'tmin_c': None,      'solar_mj': None,
            'source': f'NASA API error: {str(e)[:60]}'
        }


def score_fraud(row, sat_curr, sat_prev):
    """
    Checks if a claim is suspicious by comparing:
    - What the farmer CLAIMS vs what SATELLITES actually saw
    Returns a score 0-1 and list of reasons
    """
    score   = 0.0
    reasons = []

    loss      = float(row.get('loss_percent', 0) or 0)
    rain_rep  = float(row.get('rainfall_mm', 500) or 500)
    clm_val   = float(row.get('claim_amount_rs', 0) or 0)
    crop_val  = float(row.get('crop_value_rs', 1) or 1)
    area      = float(row.get('area_hectares', 1) or 1)

    ndvi_now  = sat_curr['ndvi']
    ndvi_prev = sat_prev['ndvi']
    ndwi_now  = sat_curr['ndwi']
    savi_now  = sat_curr['savi']
    vci_now   = sat_curr['vci']
    chirps    = sat_curr['chirps_rain_mm']
    ndvi_drop = ndvi_prev - ndvi_now  # positive = vegetation got worse (genuine)

    # ── CHECK 1: Healthy vegetation but high loss claimed ──────
    # If NDVI ≥ 0.40, the crop was healthy — why claim loss?
    if loss > 40 and ndvi_now >= 0.40:
        score += 0.30
        reasons.append(
            f"🌿 Satellite shows healthy vegetation (NDVI={ndvi_now:.2f}) "
            f"but farmer claims {loss:.0f}% crop loss"
        )

    # ── CHECK 2: No NDVI drop from previous year ───────────────
    # Genuine crop damage always shows a visible NDVI drop
    if loss > 40 and ndvi_drop < 0.05:
        score += 0.20
        reasons.append(
            f"📉 No vegetation decline detected (NDVI change = {ndvi_drop:.3f}). "
            f"Genuine crop damage would show a clear drop."
        )

    # ── CHECK 3: Farmer reported much less rain than satellite ──
    rain_diff_pct = abs(rain_rep - chirps) / (chirps + 1) * 100
    if rain_diff_pct > 40 and loss > 30:
        score += 0.22
        reasons.append(
            f"🌧 Farmer reported {rain_rep:.0f}mm rainfall but "
            f"satellite (CHIRPS) recorded {chirps:.0f}mm "
            f"— difference of {rain_diff_pct:.0f}%"
        )

    # ── CHECK 4: No drought stress (VCI) but drought claimed ───
    if vci_now > 45 and loss > 50:
        score += 0.18
        reasons.append(
            f"☀ Vegetation Condition Index = {vci_now:.0f}/100 "
            f"(above 45 = no drought) but {loss:.0f}% loss claimed"
        )

    # ── CHECK 5: Water index contradicts drought claim ─────────
    if ndwi_now > 0.05 and loss > 40:
        score += 0.10
        reasons.append(
            f"💧 NDWI = {ndwi_now:.2f} (positive = good water content). "
            f"Does not support drought/flood claim."
        )

    # ── CHECK 6: Claim exceeds crop value ──────────────────────
    ratio = clm_val / (crop_val + 1)
    if ratio > 0.90:
        score += 0.15
        reasons.append(
            f"💰 Claim amount (₹{clm_val:,.0f}) is {ratio*100:.0f}% "
            f"of total crop value — extremely high ratio"
        )

    return round(min(score, 1.0), 3), reasons


def verdict_label(score):
    if score >= 0.65:
        return 'HIGH RISK', 'red'
    elif score >= 0.35:
        return 'MEDIUM RISK', 'medium'
    else:
        return 'GENUINE', 'low'


def card(num, label, helptext, color='blue'):
    return f"""
    <div class="metric-card {color}">
        <div class="metric-num">{num}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-help">{helptext}</div>
    </div>"""


# ================================================================
# SIDEBAR — Navigation and Upload
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 4px;'>
        <div style='font-size:1.4rem;font-weight:700;color:white;'>🌾 AgriShield</div>
        <div style='font-size:0.75rem;color:#7a9cc8;letter-spacing:1px;
                    text-transform:uppercase;margin-top:2px;'>
            AP Crop Insurance · Fraud Detection
        </div>
    </div>
    <hr style='border:none;border-top:1px solid #2d4070;margin:14px 0;'/>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Step 1 — Upload Your Data")
    st.markdown(
        "<p style='font-size:13px;color:#8aafd4;'>"
        "Upload the insurance CSV file. The file must have columns: "
        "year, district, farmer_id, crop, area_hectares, rainfall_mm, "
        "production_tons, crop_value_rs, insurance_premium_rs, "
        "claim_amount_rs, loss_percent</p>",
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Choose CSV file", type=['csv'],
        label_visibility='collapsed'
    )

    st.markdown(
        "<hr style='border:none;border-top:1px solid #2d4070;margin:14px 0;'/>",
        unsafe_allow_html=True
    )
    st.markdown("### 🔧 Step 2 — Choose Filters")

    all_districts = list(DISTRICT_COORDS.keys())
    sel_districts = st.multiselect(
        "Which districts to analyse?",
        all_districts,
        default=['Anantapur', 'Kurnool', 'Guntur', 'Prakasam'],
        help="Select one or more districts from Andhra Pradesh"
    )
    if not sel_districts:
        sel_districts = ['Anantapur', 'Kurnool', 'Guntur', 'Prakasam']

    year_min, year_max = st.select_slider(
        "Year range",
        options=list(range(2000, 2026)),
        value=(2019, 2025),
        help="Choose the period to analyse"
    )

    all_crops = ['All Crops', 'Chilli', 'Maize', 'Sugarcane',
                 'Cotton', 'Rice', 'Groundnut']
    sel_crop = st.selectbox("Crop type", all_crops,
                             help="Filter by a specific crop or view all")

    st.markdown(
        "<hr style='border:none;border-top:1px solid #2d4070;margin:14px 0;'/>",
        unsafe_allow_html=True
    )
    st.markdown("### ⚠️ Step 3 — Set Risk Threshold")
    fraud_cutoff = st.slider(
        "Flag as HIGH RISK if fraud score above:",
        min_value=0.30, max_value=0.90,
        value=0.65, step=0.05,
        help="Lower = catches more fraud but more false alarms. Higher = fewer flags but more certain."
    )

    st.markdown(
        "<hr style='border:none;border-top:1px solid #2d4070;margin:14px 0;'/>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='font-size:11px;color:#3d5a80;text-align:center;'>"
        "Data: NASA POWER API · Sentinel-2 GEE<br>"
        "AP Insurance Records 2000–2025</p>",
        unsafe_allow_html=True
    )


# ================================================================
# LOAD DATA
# ================================================================
@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df

if uploaded_file:
    df_raw = load_csv(uploaded_file.read())
else:
    # Load bundled file automatically if present
    import os
    bundled = 'ap_synthetic_agri_insurance_2000_2025_12000rows__1_.csv'
    if os.path.exists(bundled):
        df_raw = pd.read_csv(bundled)
    else:
        df_raw = None

if df_raw is None:
    # ── Welcome screen when no data is loaded ─────────────────
    st.markdown('<div class="big-title">🌾 AgriShield</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Crop Insurance Fraud Detection · Andhra Pradesh</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="info-box">
    <b>👋 Welcome!</b> This dashboard helps detect fraudulent crop insurance claims
    by comparing what farmers report against what satellites actually observed
    (vegetation health, rainfall, water levels).
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="explain-card">
        <div class="explain-title">📂 To get started</div>
        <div class="explain-text">
        Upload your insurance CSV file using the sidebar on the left.
        The file should contain columns like: year, district, farmer_id, crop,
        area_hectares, rainfall_mm, claim_amount_rs, loss_percent.
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explain-card">
        <div class="explain-title">🛰 What is NDVI?</div>
        <div class="explain-text">
        NDVI (Normalized Difference Vegetation Index) measures how green/healthy
        a crop looks from space. Values near 1 = lush healthy crops.
        Values near 0 = bare land or dead crops. If a farmer claims 80% crop loss
        but NDVI shows healthy green vegetation — that is a fraud signal.
        </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="explain-card">
        <div class="explain-title">🌧 What is NDWI?</div>
        <div class="explain-text">
        NDWI (Normalized Difference Water Index) measures water content in plants
        and soil. Positive values = good moisture. If someone claims drought damage
        but NDWI shows high water content, the claim is suspicious.
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="explain-card">
        <div class="explain-title">🌡 What is NASA POWER?</div>
        <div class="explain-text">
        NASA POWER is a free public API that provides real satellite-measured
        temperature and rainfall data for any location on Earth, going back to 1981.
        This dashboard connects to it automatically to verify whether claimed
        drought conditions actually existed.
        </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ── Clean data ────────────────────────────────────────────────────
df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(' ','_')

if 'loss_percent' in df_raw.columns and df_raw['loss_percent'].max() <= 1.0:
    df_raw['loss_percent'] = df_raw['loss_percent'] * 100

df_raw['district'] = df_raw['district'].astype(str).str.strip().str.title()
df_raw['year']     = pd.to_numeric(df_raw['year'], errors='coerce').fillna(2020).astype(int)

needed = ['year','district','farmer_id','crop','area_hectares',
          'rainfall_mm','crop_value_rs','insurance_premium_rs',
          'claim_amount_rs','loss_percent']
missing_cols = [c for c in needed if c not in df_raw.columns]
if missing_cols:
    st.error(
        f"❌ Your file is missing these columns: **{', '.join(missing_cols)}**. "
        f"Please check your CSV and try again."
    )
    st.stop()

# Apply filters
df = df_raw[
    df_raw['district'].isin(sel_districts) &
    df_raw['year'].between(year_min, year_max)
].copy()
if sel_crop != 'All Crops':
    df = df[df['crop'] == sel_crop]

if len(df) == 0:
    st.warning("⚠️ No data found for the selected filters. Please change district, year range, or crop in the sidebar.")
    st.stop()

# Derived columns
df['claim_per_ha']   = df['claim_amount_rs'] / (df['area_hectares'].clip(lower=0.01))
df['claim_to_value'] = df['claim_amount_rs'] / (df['crop_value_rs'].clip(lower=1))
df['prod_per_ha']    = df.get('production_tons', pd.Series(np.nan, index=df.index)) / \
                       df['area_hectares'].clip(lower=0.01)


# ================================================================
# HEADER
# ================================================================
st.markdown('<div class="big-title">🌾 AgriShield — AP Fraud Detection</div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">Analysing {len(df):,} claims across '
    f'{df["district"].nunique()} districts · {year_min}–{year_max}</div>',
    unsafe_allow_html=True
)

# ================================================================
# TABS
# ================================================================
tab_home, tab_sat, tab_fraud, tab_nasa, tab_map, tab_guide = st.tabs([
    "🏠 Overview",
    "🛰 Satellite Indices",
    "🔍 Fraud Scanner",
    "🌡 NASA Climate",
    "🗺 District Map",
    "📖 How It Works",
])


# ================================================================
# TAB 1 — OVERVIEW
# ================================================================
with tab_home:
    st.markdown('<div class="section-title">📊 Summary at a Glance</div>',
                unsafe_allow_html=True)

    total   = len(df)
    payout  = df['claim_amount_rs'].sum()
    avg_los = df[df['loss_percent']>0]['loss_percent'].mean()
    hi_loss = int((df['loss_percent'] > 60).sum())

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(card(f"{total:,}", "Total Claims",
        f"₹{payout/1e7:.1f} Cr total payout", "blue"), unsafe_allow_html=True)
    with c2: st.markdown(card(f"{avg_los:.1f}%", "Average Loss %",
        "Among claims with any loss", "amber"), unsafe_allow_html=True)
    with c3: st.markdown(card(f"{hi_loss:,}", "High Loss Claims (>60%)",
        f"{hi_loss/total*100:.1f}% of all claims", "red"), unsafe_allow_html=True)
    with c4: st.markdown(card(f"{df['district'].nunique()}",
        "Districts Selected", f"{df['crop'].nunique()} different crops", "green"),
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Claims Per Year</div>',
                    unsafe_allow_html=True)
        yr = df.groupby('year').agg(
            claims=('farmer_id','count'),
            payout=('claim_amount_rs','sum')
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=yr['year'], y=yr['claims'], name='No. of Claims',
                              marker_color='#1976d2'), secondary_y=False)
        fig.add_trace(go.Scatter(x=yr['year'], y=yr['payout']/1e6,
                                  name='Payout (₹ Lakhs)', mode='lines+markers',
                                  line=dict(color='#f57c00',width=2),
                                  marker=dict(size=5,color='#f57c00')),
                       secondary_y=True)
        fig.update_layout(height=320, plot_bgcolor='white',
                           paper_bgcolor='white', margin=dict(t=20,b=30,l=50,r=50),
                           font=dict(family='Inter'),
                           legend=dict(orientation='h', y=-0.2))
        fig.update_yaxes(title_text="Claims", secondary_y=False,
                          gridcolor='#f0f0f0')
        fig.update_yaxes(title_text="Payout (₹ Lakhs)", secondary_y=True)
        fig.update_xaxes(gridcolor='#f0f0f0')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Loss % Distribution</div>',
                    unsafe_allow_html=True)
        fig2 = px.histogram(df[df['loss_percent']>0], x='loss_percent',
                             nbins=35, color_discrete_sequence=['#42a5f5'],
                             labels={'loss_percent':'Claimed Loss %','count':'Number of Claims'})
        fig2.add_vline(x=60, line_dash='dash', line_color='red',
                        annotation_text='High risk (>60%)',
                        annotation_font_color='red')
        fig2.update_layout(height=320, plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(t=20,b=30,l=50,r=20),
                            font=dict(family='Inter'))
        fig2.update_xaxes(gridcolor='#f0f0f0')
        fig2.update_yaxes(gridcolor='#f0f0f0')
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-title">Claims by Crop</div>',
                    unsafe_allow_html=True)
        crop_s = df.groupby('crop')['claim_amount_rs'].sum().sort_values()
        fig3 = go.Figure(go.Bar(
            x=crop_s.values/1e6, y=crop_s.index, orientation='h',
            marker_color='#42a5f5',
            text=[f'₹{v/1e6:.1f}L' for v in crop_s.values],
            textposition='outside'
        ))
        fig3.update_layout(height=300, plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(t=10,b=30,l=10,r=80),
                            font=dict(family='Inter'),
                            xaxis_title='Total Claim Amount (₹ Lakhs)')
        fig3.update_xaxes(gridcolor='#f0f0f0')
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">Average Loss % by District</div>',
                    unsafe_allow_html=True)
        dist_s = df.groupby('district')['loss_percent'].mean().sort_values(ascending=False)
        colors = ['#e53935' if v>50 else '#ffa726' if v>30 else '#66bb6a'
                  for v in dist_s.values]
        fig4 = go.Figure(go.Bar(
            x=dist_s.index, y=dist_s.values,
            marker_color=colors,
            text=[f'{v:.1f}%' for v in dist_s.values],
            textposition='outside'
        ))
        fig4.update_layout(height=300, plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(t=10,b=40,l=40,r=20),
                            font=dict(family='Inter'),
                            yaxis_title='Avg Loss %')
        fig4.update_yaxes(gridcolor='#f0f0f0')
        st.plotly_chart(fig4, use_container_width=True)


# ================================================================
# TAB 2 — SATELLITE INDICES (Before vs After)
# ================================================================
with tab_sat:
    st.markdown("""
    <div class="info-box">
    <b>🛰 What this page shows:</b> Satellite-measured vegetation health (NDVI, NDWI, SAVI, EVI)
    for any district, compared between two years. If a farmer claims crop damage in Year A,
    but the satellite shows the same healthy vegetation as the previous year — that is suspicious.
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        dist_sat = st.selectbox("Select District", sel_districts, key='s_dist')
    with col_s2:
        avail_years = sorted(df[df['district']==dist_sat]['year'].unique().tolist(),
                             reverse=True)
        if not avail_years: avail_years = [2024, 2023, 2022]
        year_curr = st.selectbox("Claim Year (After)", avail_years, key='s_curr')
    with col_s3:
        other_years = [y for y in avail_years if y != year_curr]
        if not other_years: other_years = [year_curr - 1]
        year_prev = st.selectbox("Compare With (Before)", other_years, key='s_prev')

    with st.spinner(f"Loading satellite data for {dist_sat}..."):
        s_curr = get_satellite_indices(dist_sat, year_curr)
        s_prev = get_satellite_indices(dist_sat, year_prev)

    st.markdown(f'<div class="section-title">Index Comparison: {dist_sat} — {year_prev} vs {year_curr}</div>',
                unsafe_allow_html=True)

    # Show 5 index cards with before/after
    idx_meta = {
        'ndvi': ('NDVI', '🌿 Vegetation Health',
                 'How green and healthy the crops are. ≥0.40 = healthy crop.'),
        'ndwi': ('NDWI', '💧 Water Content',
                 'How much water is in plants and soil. >0 = good moisture.'),
        'savi': ('SAVI', '🏜 Soil-Adjusted Veg.',
                 'Like NDVI but corrected for bare soil. Better for AP dry areas.'),
        'evi':  ('EVI',  '🌾 Enhanced Veg. Index',
                 'Like NDVI but works better for dense crops like cotton/tobacco.'),
        'vci':  ('VCI',  '📊 Drought Condition',
                 'Vegetation Condition Index 0–100. Below 35 = drought confirmed.'),
    }

    cols5 = st.columns(5)
    for col5, (key, (short, icon_label, desc)) in zip(cols5, idx_meta.items()):
        v_curr = s_curr.get(key, 0)
        v_prev = s_prev.get(key, 0)
        delta  = v_curr - v_prev
        arrow  = '▲' if delta >= 0 else '▼'
        d_col  = '#43a047' if delta >= 0 else '#e53935'
        with col5:
            st.markdown(f"""
            <div class="explain-card" style='text-align:center;border-top:3px solid
            {"#43a047" if delta>=0 else "#e53935"};'>
                <div style='font-size:0.75rem;font-weight:600;color:#5a7ba8;
                            text-transform:uppercase;letter-spacing:0.8px;'>{icon_label}</div>
                <div style='font-size:1.8rem;font-weight:700;color:#1a2744;
                            margin:6px 0;'>{v_curr:.3f}</div>
                <div style='font-size:0.8rem;color:{d_col};font-weight:600;'>
                    {arrow} {abs(delta):.3f} vs {year_prev}</div>
                <div style='font-size:0.72rem;color:#8aaccc;margin-top:6px;
                            line-height:1.4;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_r, col_t = st.columns(2)

    with col_r:
        st.markdown('<div class="section-title">Before vs After — Radar Chart</div>',
                    unsafe_allow_html=True)
        keys_r = ['ndvi','ndwi','savi','evi']
        # normalize ndwi to 0-1 for radar
        def norm(v): return max(0.0, min(1.0, float(v)))
        c_vals = [norm(s_curr[k]) for k in keys_r]
        p_vals = [norm(s_prev[k]) for k in keys_r]
        labels_r = [idx_meta[k][0] for k in keys_r]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=c_vals + [c_vals[0]],
            theta=labels_r + [labels_r[0]],
            fill='toself', name=f'{year_curr} (claim year)',
            line=dict(color='#e53935', width=2),
            fillcolor='rgba(229,57,53,0.15)'
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=p_vals + [p_vals[0]],
            theta=labels_r + [labels_r[0]],
            fill='toself', name=f'{year_prev} (previous)',
            line=dict(color='#1976d2', width=2),
            fillcolor='rgba(25,118,210,0.15)'
        ))
        fig_r.update_layout(
            height=350,
            polar=dict(
                bgcolor='white',
                radialaxis=dict(visible=True, range=[0,1],
                                gridcolor='#dde8f5', tickfont_color='#5a7ba8'),
                angularaxis=dict(gridcolor='#dde8f5',
                                 tickfont=dict(color='#1a2744',size=13))
            ),
            paper_bgcolor='white',
            legend=dict(orientation='h', y=-0.15),
            font=dict(family='Inter'),
            margin=dict(t=30, b=40)
        )
        st.plotly_chart(fig_r, use_container_width=True)
        if c_vals[0] > 0.40 and s_curr['vci'] > 45:
            st.markdown("""
            <div class="warn-box">
            ⚠️ <b>Suspicious pattern:</b> Vegetation health in the claim year
            looks similar to or better than the previous year. Genuine crop
            damage would show a clear drop (red area smaller than blue).
            </div>""", unsafe_allow_html=True)

    with col_t:
        st.markdown(f'<div class="section-title">NDVI Trend — {dist_sat} (All Years)</div>',
                    unsafe_allow_html=True)
        trend_years = list(range(max(2015, year_min), year_max + 1))
        ndvi_vals   = [get_satellite_indices(dist_sat, y)['ndvi'] for y in trend_years]
        ndwi_vals   = [get_satellite_indices(dist_sat, y)['ndwi'] for y in trend_years]

        fig_t = go.Figure()
        fig_t.add_hrect(y0=0, y1=0.38, fillcolor='rgba(229,57,53,0.06)',
                         line_width=0, annotation_text='Stressed vegetation zone',
                         annotation_position='top left',
                         annotation_font_color='#c62828', annotation_font_size=11)
        fig_t.add_trace(go.Scatter(
            x=trend_years, y=ndvi_vals, name='NDVI',
            mode='lines+markers',
            line=dict(color='#43a047', width=2.5),
            marker=dict(size=7, color='#43a047'),
            fill='tozeroy', fillcolor='rgba(67,160,71,0.08)'
        ))
        fig_t.add_trace(go.Scatter(
            x=trend_years, y=ndwi_vals, name='NDWI',
            mode='lines+markers',
            line=dict(color='#1976d2', width=2, dash='dot'),
            marker=dict(size=5, color='#1976d2')
        ))
        fig_t.add_hline(y=0.38, line_dash='dash', line_color='#e53935',
                         annotation_text='Stress threshold (0.38)',
                         annotation_font_color='#e53935',
                         annotation_position='bottom right')
        fig_t.update_layout(
            height=350, plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(t=20, b=40, l=50, r=20),
            font=dict(family='Inter'),
            legend=dict(orientation='h', y=-0.2),
            yaxis_title='Index Value',
            xaxis_title='Year'
        )
        fig_t.update_xaxes(gridcolor='#f0f0f0', tickvals=trend_years)
        fig_t.update_yaxes(gridcolor='#f0f0f0')
        st.plotly_chart(fig_t, use_container_width=True)

    # NDVI Heatmap across districts and years
    st.markdown('<div class="section-title">NDVI Heatmap — All Districts vs All Years</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>How to read this:</b> Each cell shows how healthy vegetation was that year.
    <b style='color:#8B0000;'>Dark red</b> = drought/stress (low NDVI — claims more likely genuine).
    <b style='color:#1b5e20;'>Dark green</b> = healthy crops (high claims are suspicious).
    </div>
    """, unsafe_allow_html=True)

    heat_years = list(range(max(2015, year_min), min(2026, year_max + 1)))
    heat_data  = {}
    for d in sel_districts:
        heat_data[d] = [get_satellite_indices(d, y)['ndvi'] for y in heat_years]
    heat_df = pd.DataFrame(heat_data, index=heat_years).T

    fig_h = px.imshow(
        heat_df,
        color_continuous_scale=[
            [0.0, '#8B0000'], [0.3, '#d32f2f'],
            [0.5, '#ffa726'], [0.7, '#66bb6a'], [1.0, '#1b5e20']
        ],
        zmin=0.15, zmax=0.72,
        labels=dict(color='NDVI', x='Year', y='District'),
        aspect='auto',
        text_auto='.2f'
    )
    fig_h.update_layout(
        height=max(220, len(sel_districts)*55),
        paper_bgcolor='white', margin=dict(t=10,b=40,l=10,r=10),
        font=dict(family='Inter'),
        coloraxis_colorbar=dict(title='NDVI', tickfont_color='#1a2744',
                                titlefont_color='#1a2744')
    )
    fig_h.update_traces(textfont=dict(size=11, color='white'))
    st.plotly_chart(fig_h, use_container_width=True)


# ================================================================
# TAB 3 — FRAUD SCANNER
# ================================================================
with tab_fraud:
    st.markdown("""
    <div class="info-box">
    <b>🔍 How fraud is detected:</b> Each claim is checked against 6 satellite signals.
    The fraud score (0–1) combines how many signals are suspicious.
    A score above your threshold is flagged for investigation.
    </div>
    """, unsafe_allow_html=True)

    # Score claims (limit to 400 for performance)
    sample_n = min(400, len(df))
    df_scan  = df.sample(sample_n, random_state=42).copy()

    with st.spinner(f"Scanning {sample_n} claims for fraud signals..."):
        scores_list, reasons_list = [], []
        for _, row in df_scan.iterrows():
            yr   = int(row['year'])
            dist = str(row['district'])
            sc  = get_satellite_indices(dist, yr)
            sp  = get_satellite_indices(dist, max(yr-1, 2000))
            s, r = score_fraud(row, sc, sp)
            scores_list.append(s)
            reasons_list.append(r)

    df_scan['fraud_score'] = scores_list
    df_scan['reasons']     = reasons_list
    df_scan['verdict']     = df_scan['fraud_score'].apply(
        lambda s: 'HIGH RISK' if s>=fraud_cutoff else ('MEDIUM RISK' if s>=0.35 else 'GENUINE')
    )

    n_high = int((df_scan['verdict']=='HIGH RISK').sum())
    n_med  = int((df_scan['verdict']=='MEDIUM RISK').sum())
    n_ok   = int((df_scan['verdict']=='GENUINE').sum())
    avg_sc = float(df_scan['fraud_score'].mean())

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(card(f"{n_high}", "HIGH RISK",
        f"{n_high/sample_n*100:.1f}% — needs investigation", "red"),
        unsafe_allow_html=True)
    with c2: st.markdown(card(f"{n_med}", "MEDIUM RISK",
        f"{n_med/sample_n*100:.1f}% — worth reviewing", "amber"),
        unsafe_allow_html=True)
    with c3: st.markdown(card(f"{n_ok}", "Genuine",
        f"{n_ok/sample_n*100:.1f}% — likely authentic", "green"),
        unsafe_allow_html=True)
    with c4: st.markdown(card(f"{avg_sc:.2f}", "Avg Fraud Score",
        "0 = fully genuine · 1 = certain fraud", "blue"),
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown('<div class="section-title">Fraud Score Distribution</div>',
                    unsafe_allow_html=True)
        fig_sc = px.histogram(
            df_scan, x='fraud_score', nbins=25,
            color='verdict',
            color_discrete_map={
                'HIGH RISK':'#e53935',
                'MEDIUM RISK':'#ffa726',
                'GENUINE':'#66bb6a'
            },
            labels={'fraud_score':'Fraud Score (0–1)','count':'Number of Claims'},
            category_orders={'verdict':['HIGH RISK','MEDIUM RISK','GENUINE']}
        )
        fig_sc.add_vline(x=fraud_cutoff, line_dash='dash', line_color='#c62828',
                          annotation_text=f'Your threshold ({fraud_cutoff})',
                          annotation_font_color='#c62828')
        fig_sc.update_layout(height=320, plot_bgcolor='white',
                              paper_bgcolor='white', bargap=0.05,
                              margin=dict(t=10,b=30,l=40,r=20),
                              font=dict(family='Inter'),
                              legend=dict(orientation='h',y=-0.25))
        fig_sc.update_xaxes(gridcolor='#f0f0f0')
        fig_sc.update_yaxes(gridcolor='#f0f0f0')
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_f2:
        st.markdown('<div class="section-title">Risk Breakdown</div>',
                    unsafe_allow_html=True)
        fig_pie = px.pie(
            values=[n_high, n_med, n_ok],
            names=['HIGH RISK','MEDIUM RISK','GENUINE'],
            color_discrete_sequence=['#e53935','#ffa726','#66bb6a'],
            hole=0.55
        )
        fig_pie.update_traces(
            textposition='outside',
            textfont=dict(color='#1a2744', size=13)
        )
        fig_pie.update_layout(
            height=320, paper_bgcolor='white',
            margin=dict(t=10,b=10,l=10,r=10),
            font=dict(family='Inter'),
            legend=dict(orientation='h',y=-0.12),
            annotations=[dict(text='Risk Split',x=0.5,y=0.5,
                              font=dict(size=14,color='#1a2744'),
                              showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Fraud signal breakdown
    col_f3, col_f4 = st.columns(2)
    with col_f3:
        st.markdown('<div class="section-title">Which Fraud Signals Are Most Common?</div>',
                    unsafe_allow_html=True)
        signal_counts = {
            'Healthy NDVI + high loss': 0,
            'No NDVI drop from prev year': 0,
            'Rainfall mismatch (satellite vs reported)': 0,
            'No drought (VCI) but loss claimed': 0,
            'NDWI shows water, drought claimed': 0,
            'Claim exceeds crop value': 0,
        }
        for rlist in df_scan[df_scan['fraud_score']>0]['reasons']:
            for r in rlist:
                if 'NDVI=' in r and 'healthy' in r.lower():
                    signal_counts['Healthy NDVI + high loss'] += 1
                elif 'decline' in r.lower() or 'drop' in r.lower():
                    signal_counts['No NDVI drop from prev year'] += 1
                elif 'satellite' in r.lower() and 'rain' in r.lower():
                    signal_counts['Rainfall mismatch (satellite vs reported)'] += 1
                elif 'VCI' in r:
                    signal_counts['No drought (VCI) but loss claimed'] += 1
                elif 'NDWI' in r:
                    signal_counts['NDWI shows water, drought claimed'] += 1
                elif 'Claim amount' in r or 'ratio' in r.lower():
                    signal_counts['Claim exceeds crop value'] += 1

        sc_df = pd.DataFrame({
            'Signal': list(signal_counts.keys()),
            'Count':  list(signal_counts.values())
        }).sort_values('Count', ascending=True)
        sc_df = sc_df[sc_df['Count']>0]
        if len(sc_df) > 0:
            fig_sg = go.Figure(go.Bar(
                x=sc_df['Count'], y=sc_df['Signal'],
                orientation='h',
                marker_color=['#e53935','#ffa726','#1976d2',
                              '#7b1fa2','#00838f','#558b2f'][:len(sc_df)],
                text=sc_df['Count'], textposition='outside'
            ))
            fig_sg.update_layout(
                height=300, plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(t=10,b=30,l=10,r=60),
                font=dict(family='Inter'),
                xaxis_title='Number of claims flagged'
            )
            fig_sg.update_xaxes(gridcolor='#f0f0f0')
            st.plotly_chart(fig_sg, use_container_width=True)
        else:
            st.info("No fraud signals detected with current settings. Try lowering the threshold.")

    with col_f4:
        st.markdown('<div class="section-title">High Risk Score by District</div>',
                    unsafe_allow_html=True)
        dist_risk = df_scan.groupby('district')['fraud_score'].mean().sort_values(ascending=False)
        clrs = ['#e53935' if v>=fraud_cutoff else '#ffa726' if v>=0.35 else '#66bb6a'
                for v in dist_risk.values]
        fig_dr = go.Figure(go.Bar(
            x=dist_risk.index, y=dist_risk.values,
            marker_color=clrs,
            text=[f'{v:.2f}' for v in dist_risk.values],
            textposition='outside'
        ))
        fig_dr.update_layout(
            height=300, plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(t=10,b=40,l=40,r=20),
            font=dict(family='Inter'),
            yaxis_title='Mean Fraud Score'
        )
        fig_dr.update_yaxes(gridcolor='#f0f0f0', range=[0,1])
        st.plotly_chart(fig_dr, use_container_width=True)

    # Flagged claims table
    st.markdown('<div class="section-title">🚨 Flagged Claims — Top 25 Highest Risk</div>',
                unsafe_allow_html=True)
    flagged = df_scan[df_scan['verdict']=='HIGH RISK'].sort_values(
        'fraud_score', ascending=False).head(25)

    if len(flagged) == 0:
        st.markdown("""
        <div class="success-box">
        ✅ <b>No HIGH RISK claims found</b> with your current threshold.
        Try lowering the fraud threshold in the sidebar to see more flags.
        </div>""", unsafe_allow_html=True)
    else:
        rows = ""
        for _, row in flagged.iterrows():
            reasons_str = "<br>".join(row['reasons']) if row['reasons'] else "—"
            rows += f"""<tr>
              <td>{row['farmer_id']}</td>
              <td>{row['district']}</td>
              <td>{int(row['year'])}</td>
              <td>{row['crop']}</td>
              <td>{row['loss_percent']:.1f}%</td>
              <td>₹{row['claim_amount_rs']:,.0f}</td>
              <td><b style='color:#e53935;'>{row['fraud_score']:.3f}</b></td>
              <td style='font-size:12px;line-height:1.6;max-width:360px;'>{reasons_str}</td>
            </tr>"""
        st.markdown(f"""
        <table class="neat">
          <thead><tr>
            <th>Farmer ID</th><th>District</th><th>Year</th><th>Crop</th>
            <th>Loss%</th><th>Claim ₹</th><th>Score</th><th>Evidence from Satellites</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        csv_out = flagged[['farmer_id','district','year','crop',
                             'loss_percent','claim_amount_rs',
                             'fraud_score','verdict']].to_csv(index=False)
        st.download_button(
            "⬇️ Download Flagged Claims as CSV",
            data=csv_out,
            file_name="high_risk_claims.csv",
            mime='text/csv'
        )


# ================================================================
# TAB 4 — NASA POWER CLIMATE
# ================================================================
with tab_nasa:
    st.markdown("""
    <div class="info-box">
    <b>🌡 NASA POWER API:</b> This tab connects to NASA's real satellite climate database
    to fetch actual temperature and rainfall data for any AP district.
    No login needed — it's a free public API.
    If a farmer claims drought but NASA shows normal rainfall → fraud signal.
    </div>
    """, unsafe_allow_html=True)

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        dist_nasa = st.selectbox("Select District", sel_districts, key='n_dist')
    with col_n2:
        yr_nasa = st.selectbox(
            "Select Year",
            sorted(df[df['district']==dist_nasa]['year'].unique().tolist(), reverse=True),
            key='n_year'
        )

    coords = DISTRICT_COORDS.get(dist_nasa, (15.9, 79.7))

    st.markdown(f"""
    <div class="info-box">
    Fetching real NASA POWER data for <b>{dist_nasa}</b> ({yr_nasa})<br>
    Coordinates: Latitude {coords[0]}°N, Longitude {coords[1]}°E<br>
    <small>Source: <a href="https://power.larc.nasa.gov" target="_blank">
    https://power.larc.nasa.gov</a> — Free NASA satellite climate data</small>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Connecting to NASA POWER satellite API..."):
        nasa = fetch_nasa_power(coords[0], coords[1], yr_nasa)

    if 'error' in nasa.get('source','').lower():
        st.markdown(f"""
        <div class="warn-box">
        ⚠️ <b>NASA API note:</b> {nasa['source']}<br>
        This may be a network issue. The dashboard will still work with estimated values.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="success-box">
        ✅ <b>NASA POWER data loaded successfully</b> for {dist_nasa}, {yr_nasa}
        </div>""", unsafe_allow_html=True)

    # Climate KPIs
    rain = nasa['rainfall_mm']
    tmax = nasa['tmax_c']
    tmin = nasa['tmin_c']
    sol  = nasa['solar_mj']

    drought_flag   = (rain is not None) and (rain < 500)
    heatstress_flag= (tmax is not None) and (tmax > 40)

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(card(
            f"{rain:.0f} mm" if rain else "No data",
            "Annual Rainfall (NASA)",
            "⚠️ DROUGHT YEAR — Below 500mm" if drought_flag else "Normal range (>500mm)",
            "red" if drought_flag else "green"
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(card(
            f"{tmax:.1f}°C" if tmax else "No data",
            "Max Temperature (NASA)",
            "⚠️ HEAT STRESS — Above 40°C" if heatstress_flag else "Normal range",
            "red" if heatstress_flag else "amber"
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(card(
            f"{tmin:.1f}°C" if tmin else "No data",
            "Min Temperature (NASA)",
            "Cold nights — can damage some crops" if tmin and tmin < 12 else "Normal",
            "blue"
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(card(
            f"{sol:.1f} MJ/m²" if sol else "No data",
            "Solar Radiation (NASA)",
            "Crop growth energy available",
            "amber"
        ), unsafe_allow_html=True)

    # Fraud check vs farmer reported
    sub_check = df[(df['district']==dist_nasa) & (df['year']==yr_nasa)]
    if len(sub_check) > 0 and rain is not None:
        avg_rep_rain = sub_check['rainfall_mm'].mean()
        rain_gap     = abs(avg_rep_rain - rain)
        rain_gap_pct = rain_gap / (rain + 1) * 100
        avg_loss     = sub_check['loss_percent'].mean()

        st.markdown(f'<div class="section-title">Claim vs NASA Climate — Mismatch Check ({dist_nasa}, {yr_nasa})</div>',
                    unsafe_allow_html=True)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("NASA Recorded Rainfall",   f"{rain:.0f} mm")
        with col_m2:
            st.metric("Farmers Reported Rainfall", f"{avg_rep_rain:.0f} mm",
                       delta=f"{avg_rep_rain - rain:+.0f} mm vs NASA")
        with col_m3:
            st.metric("Avg Claimed Loss",          f"{avg_loss:.1f}%")

        if rain_gap_pct > 40 and avg_loss > 30:
            st.markdown(f"""
            <div class="danger-box">
            🚨 <b>FRAUD SIGNAL DETECTED:</b>
            Farmers in {dist_nasa} reported {avg_rep_rain:.0f}mm rainfall on average,
            but NASA satellite recorded {rain:.0f}mm — a difference of {rain_gap_pct:.0f}%.
            Combined with {avg_loss:.0f}% average claimed loss, this pattern is suspicious.
            </div>""", unsafe_allow_html=True)
        elif rain_gap_pct > 20:
            st.markdown(f"""
            <div class="warn-box">
            ⚠️ <b>Moderate mismatch:</b> Reported rainfall ({avg_rep_rain:.0f}mm)
            differs from NASA data ({rain:.0f}mm) by {rain_gap_pct:.0f}%.
            Worth reviewing claims from this district and year.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
            ✅ <b>Rainfall matches:</b> Farmer-reported rainfall aligns with
            NASA satellite data (gap = {rain_gap_pct:.0f}%). No mismatch signal.
            </div>""", unsafe_allow_html=True)

    # Multi-year NASA trend
    st.markdown(f'<div class="section-title">Multi-Year Climate Trend — {dist_nasa}</div>',
                unsafe_allow_html=True)
    trend_yrs = list(range(max(2015, year_min), min(2026, year_max + 1)))

    nasa_prog = st.progress(0, text="Fetching NASA data for each year...")
    nasa_rows = []
    for idx_n, y in enumerate(trend_yrs):
        nd = fetch_nasa_power(coords[0], coords[1], y)
        nd['year'] = y
        nasa_rows.append(nd)
        nasa_prog.progress((idx_n+1)/len(trend_yrs), text=f"Loading {y}...")
    nasa_prog.empty()
    nasa_df = pd.DataFrame(nasa_rows)

    col_nc1, col_nc2 = st.columns(2)
    with col_nc1:
        valid_rain = nasa_df[nasa_df['rainfall_mm'].notna()]
        if len(valid_rain) > 0:
            fig_rain = go.Figure()
            fig_rain.add_hrect(y0=0, y1=500,
                                fillcolor='rgba(229,57,53,0.07)', line_width=0,
                                annotation_text='Drought zone (<500mm)',
                                annotation_position='top left',
                                annotation_font_color='#c62828',
                                annotation_font_size=11)
            fig_rain.add_trace(go.Bar(
                x=valid_rain['year'], y=valid_rain['rainfall_mm'],
                marker_color=['#e53935' if v<500 else '#1976d2'
                               for v in valid_rain['rainfall_mm']],
                text=[f'{v:.0f}mm' for v in valid_rain['rainfall_mm']],
                textposition='outside', name='NASA Rainfall'
            ))
            fig_rain.add_hline(y=500, line_dash='dash', line_color='#c62828',
                                annotation_text='Drought threshold',
                                annotation_font_color='#c62828')
            fig_rain.update_layout(
                height=320, plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(t=20,b=40,l=50,r=20),
                font=dict(family='Inter'),
                yaxis_title='Rainfall (mm)', xaxis_title='Year',
                title=f'{dist_nasa}: Annual Rainfall (NASA POWER Real Data)'
            )
            fig_rain.update_yaxes(gridcolor='#f0f0f0')
            st.plotly_chart(fig_rain, use_container_width=True)
        else:
            st.info("NASA rainfall data not available for this range.")

    with col_nc2:
        valid_temp = nasa_df[nasa_df['tmax_c'].notna() & nasa_df['tmin_c'].notna()]
        if len(valid_temp) > 0:
            fig_tmp = go.Figure()
            fig_tmp.add_hrect(y0=40, y1=50, fillcolor='rgba(229,57,53,0.07)',
                               line_width=0, annotation_text='Heat stress zone',
                               annotation_position='top left',
                               annotation_font_color='#c62828',
                               annotation_font_size=11)
            fig_tmp.add_trace(go.Scatter(
                x=valid_temp['year'], y=valid_temp['tmax_c'],
                name='Max Temp', fill=None,
                line=dict(color='#e53935', width=2.5),
                mode='lines+markers', marker=dict(size=6)
            ))
            fig_tmp.add_trace(go.Scatter(
                x=valid_temp['year'], y=valid_temp['tmin_c'],
                name='Min Temp', fill='tonexty',
                fillcolor='rgba(229,57,53,0.06)',
                line=dict(color='#1976d2', width=2.5),
                mode='lines+markers', marker=dict(size=6)
            ))
            fig_tmp.add_hline(y=40, line_dash='dash', line_color='#e53935',
                               annotation_text='Heat stress (40°C)',
                               annotation_font_color='#e53935')
            fig_tmp.update_layout(
                height=320, plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(t=20,b=40,l=50,r=20),
                font=dict(family='Inter'),
                legend=dict(orientation='h', y=-0.25),
                yaxis_title='Temperature (°C)', xaxis_title='Year',
                title=f'{dist_nasa}: Temperature Range (NASA POWER Real Data)'
            )
            fig_tmp.update_yaxes(gridcolor='#f0f0f0')
            st.plotly_chart(fig_tmp, use_container_width=True)
        else:
            st.info("NASA temperature data not available for this range.")


# ================================================================
# TAB 5 — MAP
# ================================================================
with tab_map:
    st.markdown("""
    <div class="info-box">
    <b>🗺 District Map:</b> Each bubble shows a district. Bubble size = number of claims.
    Colour = average fraud risk score. Red bubbles need the most attention.
    </div>
    """, unsafe_allow_html=True)

    map_rows = []
    for d in sel_districts:
        sub_d = df[df['district']==d]
        if len(sub_d)==0: continue
        coords  = DISTRICT_COORDS.get(d, (15.9, 79.7))
        sat_inf = get_satellite_indices(d, year_max)
        avg_l   = float(sub_d['loss_percent'].mean())
        n_c     = int(len(sub_d))
        tot_pay = float(sub_d['claim_amount_rs'].sum())
        risk    = round(min(1.0, avg_l/100*0.5 + (1-sat_inf['ndvi'])*0.3 +
                            max(0,(40-sat_inf['vci'])/100)*0.2), 3)
        map_rows.append({
            'District':       d,
            'lat':            coords[0],
            'lon':            coords[1],
            'Claims':         n_c,
            'Avg Loss %':     round(avg_l,1),
            'Total Payout ₹': round(tot_pay/1e6,2),
            'NDVI':           sat_inf['ndvi'],
            'VCI':            sat_inf['vci'],
            'Risk Score':     risk,
        })
    map_df = pd.DataFrame(map_rows)

    col_map, col_list = st.columns([3,1])
    with col_map:
        fig_map = px.scatter_mapbox(
            map_df, lat='lat', lon='lon',
            size='Claims', color='Risk Score',
            color_continuous_scale=['#43a047','#ffa726','#e53935'],
            range_color=[0, 1],
            hover_name='District',
            hover_data={
                'Avg Loss %':':.1f',
                'Total Payout ₹':':.2f',
                'NDVI':':.3f', 'VCI':':.1f',
                'Risk Score':':.3f',
                'lat':False,'lon':False
            },
            zoom=6, center={"lat":15.9,"lon":79.7},
            mapbox_style='open-street-map',
            size_max=55,
            opacity=0.85
        )
        fig_map.update_layout(
            height=500, margin=dict(t=0,b=0,l=0,r=0),
            paper_bgcolor='white',
            coloraxis_colorbar=dict(
                title='Risk Score',
                tickvals=[0,0.25,0.5,0.75,1],
                ticktext=['0<br>Genuine','0.25','0.5','0.75','1<br>High Risk']
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_list:
        st.markdown("<b style='color:#1a2744;font-size:14px;'>Districts ranked by risk</b>",
                    unsafe_allow_html=True)
        for _, row in map_df.sort_values('Risk Score', ascending=False).iterrows():
            if row['Risk Score'] >= 0.65:
                box_cls, icon = 'danger-box', '🔴'
            elif row['Risk Score'] >= 0.35:
                box_cls, icon = 'warn-box', '🟡'
            else:
                box_cls, icon = 'success-box', '🟢'
            st.markdown(f"""
            <div class="{box_cls}" style='padding:10px 12px;margin:5px 0;font-size:13px;'>
              {icon} <b>{row['District']}</b><br>
              <span style='font-size:12px;'>
              Risk: {row['Risk Score']:.2f} &nbsp;|&nbsp;
              NDVI: {row['NDVI']:.2f}<br>
              Loss: {row['Avg Loss %']}% &nbsp;|&nbsp;
              {row['Claims']} claims
              </span>
            </div>
            """, unsafe_allow_html=True)


# ================================================================
# TAB 6 — GUIDE (HOW IT WORKS)
# ================================================================
with tab_guide:
    st.markdown("""
    <div class="big-title" style='font-size:1.6rem;'>📖 How This Dashboard Works</div>
    <div class="subtitle">A complete beginner's guide — no technical knowledge needed</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title">The Core Problem</div>
    <div class="info-box">
    Every year, some farmers file <b>fake or exaggerated crop insurance claims</b>.
    They might say their crop was destroyed by drought or flood, but the crop was actually
    fine. This costs the government crores of rupees and harms genuine farmers.
    <br><br>
    <b>The solution:</b> Satellites don't lie. A satellite flying over your farm can tell
    whether your crop was healthy or dead — regardless of what was written on paper.
    This dashboard compares what farmers reported against what satellites actually saw.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">The 6 Fraud Detection Checks</div>',
                unsafe_allow_html=True)

    checks = [
        ("🌿 Check 1: NDVI Health vs Claimed Loss",
         "NDVI (Vegetation Index) measures how green and healthy crops are from space. "
         "If the satellite shows NDVI ≥ 0.40 (healthy crops), but the farmer is claiming "
         "40%+ crop loss — that is suspicious. Genuinely damaged crops have low NDVI.",
         "Fraud score +0.30"),
        ("📉 Check 2: NDVI Drop from Previous Year",
         "When crops are genuinely damaged, the NDVI drops significantly compared to the "
         "previous year. If a farmer claims 40%+ loss but the NDVI hasn't dropped at all "
         "compared to last year — the vegetation hasn't changed, meaning no real damage occurred.",
         "Fraud score +0.20"),
        ("🌧 Check 3: Rainfall Mismatch (Farmer vs CHIRPS Satellite)",
         "CHIRPS is a satellite-based rainfall dataset. If a farmer reports 300mm of rain "
         "(to justify a drought claim) but the satellite recorded 700mm — that 40%+ difference "
         "is a strong fraud indicator. Satellites measure actual rainfall from space.",
         "Fraud score +0.22"),
        ("☀ Check 4: VCI (Vegetation Condition Index)",
         "VCI measures how the current year's vegetation compares to historical normal. "
         "A VCI above 45 means the vegetation is in normal or good condition — confirming "
         "no drought. If someone claims 50%+ loss but VCI is high, the claim is suspicious.",
         "Fraud score +0.18"),
        ("💧 Check 5: NDWI Water Content",
         "NDWI (Water Index) measures soil and plant moisture. Positive NDWI means there "
         "is good water content. If a farmer claims drought-related crop loss but NDWI is "
         "positive (plenty of water), the claim contradicts satellite evidence.",
         "Fraud score +0.10"),
        ("💰 Check 6: Claim-to-Crop Value Ratio",
         "If the claimed payout is 90%+ of the entire crop value, it is statistically "
         "unusual. While not impossible, very high claim ratios combined with other "
         "satellite mismatches make a claim much more suspicious.",
         "Fraud score +0.15"),
    ]
    for title, desc, score in checks:
        st.markdown(f"""
        <div class="explain-card">
          <div class="explain-title">{title}
            <span style='float:right;font-size:12px;color:#1976d2;
                         font-weight:500;'>{score}</span>
          </div>
          <div class="explain-text">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">What Do the Scores Mean?</div>',
                unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="explain-card" style='border-top:4px solid #43a047;text-align:center;'>
        <div style='font-size:2rem;font-weight:700;color:#43a047;'>0.00 – 0.34</div>
        <div style='font-weight:600;color:#1a2744;margin:6px 0;'>✅ GENUINE</div>
        <div class='explain-text'>No satellite contradictions found.
        The claim is consistent with satellite observations.
        Safe to process.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="explain-card" style='border-top:4px solid #ffa726;text-align:center;'>
        <div style='font-size:2rem;font-weight:700;color:#ffa726;'>0.35 – 0.64</div>
        <div style='font-weight:600;color:#1a2744;margin:6px 0;'>⚠️ MEDIUM RISK</div>
        <div class='explain-text'>Some inconsistencies found.
        Worth reviewing the claim manually before
        approving full payout.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="explain-card" style='border-top:4px solid #e53935;text-align:center;'>
        <div style='font-size:2rem;font-weight:700;color:#e53935;'>0.65 – 1.00</div>
        <div style='font-weight:600;color:#1a2744;margin:6px 0;'>🚨 HIGH RISK</div>
        <div class='explain-text'>Multiple satellite checks failed.
        Strong evidence contradicts the claim.
        Recommend field investigation.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Data Sources Used</div>',
                unsafe_allow_html=True)
    sources = [
        ("🛰 Sentinel-2 (via Google Earth Engine)",
         "European Space Agency satellite. Revisits every 5 days at 10m resolution. "
         "Provides NDVI, NDWI, SAVI, EVI."),
        ("🌧 CHIRPS Rainfall",
         "Climate Hazards Group InfraRed Precipitation with Station data. "
         "30+ years of rainfall data at 5km resolution."),
        ("🌡 NASA POWER",
         "NASA's Prediction Of Worldwide Energy Resources. Free public API. "
         "Provides temperature (Tmax, Tmin) and solar radiation going back to 1981."),
        ("📋 AP Insurance Records",
         "Andhra Pradesh crop insurance claim data with farmer ID, district, "
         "crop type, area, claimed loss, and payout amounts."),
    ]
    for src_title, src_desc in sources:
        st.markdown(f"""
        <div class="explain-card">
        <div class="explain-title">{src_title}</div>
        <div class="explain-text">{src_desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">To Connect Real GEE Satellite Data</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="warn-box">
    <b>Currently:</b> The satellite index values (NDVI, NDWI, SAVI) are simulated
    realistically for demonstration. To use real GEE data:<br><br>
    1. Go to <a href='https://earthengine.google.com' target='_blank'>
       earthengine.google.com</a> and sign up (free for research)<br>
    2. Run <code>earthengine authenticate</code> in your terminal<br>
    3. In <code>app.py</code>, replace the body of the
       <code>get_satellite_indices()</code> function with your GEE ee calls<br>
    4. The rest of the dashboard — fraud scoring, maps, charts — all use
       whatever that function returns, so everything updates automatically
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# FOOTER
# ================================================================
st.markdown("""
<div style='margin-top:50px;padding:20px;background:white;border-radius:12px;
            border-top:3px solid #1a2744;text-align:center;'>
  <div style='font-size:1rem;font-weight:700;color:#1a2744;'>
      🌾 AgriShield — AP Crop Insurance Fraud Detection
  </div>
  <div style='font-size:12px;color:#8aaccc;margin-top:6px;'>
      Data Sources: NASA POWER API · Sentinel-2 via GEE · CHIRPS · AP Insurance Records<br>
      Project by: S.R.L Keerthi &amp; Yashaswini H V
  </div>
</div>
""", unsafe_allow_html=True)
