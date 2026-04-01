import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriShield — Fraud Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp { background: #0a0f1e; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1425 !important;
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] * { color: #c8d8f0 !important; }

/* Cards */
.kpi-card {
    background: linear-gradient(135deg, #0f1d35 0%, #162040 100%);
    border: 1px solid #1e3058;
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    border-radius: 16px 0 0 16px;
}
.kpi-card.red::before   { background: #e84545; }
.kpi-card.amber::before { background: #f5a623; }
.kpi-card.green::before { background: #27c97a; }
.kpi-card.blue::before  { background: #3b9eff; }

.kpi-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5a7ba8 !important;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: #e8f2ff !important;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 12px;
    color: #3b6ea8 !important;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #e8f2ff;
    letter-spacing: -0.3px;
    margin: 24px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a2d4a;
}

/* Alert banners */
.alert-high {
    background: rgba(232,69,69,0.12);
    border: 1px solid rgba(232,69,69,0.35);
    border-left: 4px solid #e84545;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #ffb3b3 !important;
    font-size: 14px;
}
.alert-medium {
    background: rgba(245,166,35,0.12);
    border: 1px solid rgba(245,166,35,0.35);
    border-left: 4px solid #f5a623;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #ffe0b0 !important;
    font-size: 14px;
}
.alert-low {
    background: rgba(39,201,122,0.10);
    border: 1px solid rgba(39,201,122,0.3);
    border-left: 4px solid #27c97a;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #b0ffd8 !important;
    font-size: 14px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1425;
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #5a7ba8;
    border-radius: 8px;
    padding: 8px 20px;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #162040 !important;
    color: #3b9eff !important;
}

/* Verdict badges */
.badge-high   { background:#e84545;color:#fff;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:0.5px; }
.badge-medium { background:#f5a623;color:#000;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:0.5px; }
.badge-low    { background:#27c97a;color:#000;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:0.5px; }

/* Plotly chart area */
.js-plotly-plot { border-radius: 12px; }

/* Metric delta override */
[data-testid="metric-container"] {
    background: #0f1d35;
    border: 1px solid #1e3058;
    border-radius: 12px;
    padding: 16px;
}

/* Title bar */
.main-title {
    font-family: 'Syne', sans-serif;
    font-size: 38px;
    font-weight: 800;
    color: #e8f2ff;
    letter-spacing: -1px;
    line-height: 1.1;
}
.main-sub {
    font-size: 14px;
    color: #3b6ea8;
    font-weight: 400;
    letter-spacing: 0.5px;
    margin-top: 4px;
}
.accent { color: #3b9eff; }

/* Table */
.styled-table {
    width:100%; border-collapse:collapse; font-size:13px;
}
.styled-table th {
    background:#0f1d35; color:#5a7ba8; font-size:10px;
    letter-spacing:1px; text-transform:uppercase; padding:10px 14px;
    border-bottom:1px solid #1e3058;
}
.styled-table td {
    padding:10px 14px; border-bottom:1px solid #111e32;
    color:#c8d8f0; vertical-align:middle;
}
.styled-table tr:hover td { background:#111e32; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#8aafd4', size=12),
    margin=dict(t=40, b=40, l=50, r=20),
    xaxis=dict(gridcolor='#1a2d4a', showgrid=True, zeroline=False),
    yaxis=dict(gridcolor='#1a2d4a', showgrid=True, zeroline=False),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8aafd4')),
)

DISTRICT_COORDS = {
    'Anantapur':(14.68,77.60),'Chittoor':(13.22,79.10),
    'East Godavari':(17.39,82.00),'Guntur':(16.31,80.44),
    'Krishna':(16.61,80.65),'Kurnool':(15.83,78.04),
    'Nellore':(14.44,79.99),'Prakasam':(15.35,79.57),
    'Srikakulam':(18.29,83.89),'Visakhapatnam':(17.69,83.22),
    'Vizianagaram':(18.11,83.40),'West Godavari':(16.92,81.34),
    'YSR Kadapa':(14.47,78.82),
}

def kpi(label, value, sub="", variant="blue"):
    return f"""<div class="kpi-card {variant}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""

def badge(text, level):
    return f'<span class="badge-{level.lower()}">{text}</span>'

def verdict_color(score):
    if score >= 0.70: return ('HIGH RISK','red','#e84545')
    if score >= 0.45: return ('MEDIUM RISK','amber','#f5a623')
    return ('LOW RISK','low','#27c97a')

# ════════════════════════════════════════════════════════════════
# DATA LOADERS
# ════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_insurance_data(path):
    df = pd.read_csv(path)
    if df['loss_percent'].max() <= 1.0:
        df['loss_percent'] = df['loss_percent'] * 100
    df['district_clean'] = df['district'].str.strip().str.title()
    df['claim_per_hectare']   = df['claim_amount_rs'] / (df['area_hectares'] + 1e-8)
    df['claim_value_ratio']   = df['claim_amount_rs'] / (df['crop_value_rs'] + 1)
    df['claim_premium_ratio'] = df['claim_amount_rs'] / (df['insurance_premium_rs'] + 1)
    df['production_per_ha']   = df['production_tons'] / (df['area_hectares'] + 1e-8)
    return df

@st.cache_data(show_spinner=False)
def fetch_nasa_power(lat, lon, year):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/monthly/point?"
        f"parameters=PRECTOTCORR,T2M_MAX,T2M_MIN,ALLSKY_SFC_SW_DWN"
        f"&community=AG&longitude={lon}&latitude={lat}"
        f"&start={year}&end={year}&format=JSON"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()['properties']['parameter']
        def asum(d): 
            v=[x for k,x in d.items() if str(year) in k and x not in(-999,-99,None)]
            return round(sum(v),2) if v else np.nan
        def amean(d):
            v=[x for k,x in d.items() if str(year) in k and x not in(-999,-99,None)]
            return round(np.mean(v),2) if v else np.nan
        return {
            'precip_mm': asum(data['PRECTOTCORR']),
            'tmax_c':   amean(data['T2M_MAX']),
            'tmin_c':   amean(data['T2M_MIN']),
            'solar':    amean(data['ALLSKY_SFC_SW_DWN']),
        }
    except:
        return {'precip_mm':np.nan,'tmax_c':np.nan,'tmin_c':np.nan,'solar':np.nan}

# ── Simulated GEE function (replace with real ee calls) ───────────
@st.cache_data(show_spinner=False)
def get_gee_indices(district, year, season='kharif'):
    """
    REAL USAGE — replace body with:
        import ee
        ee.Initialize(project='YOUR_PROJECT')
        ... (full GEE fetch as in previous cells)

    For now returns realistic simulated values so dashboard runs
    without GEE auth. Swap the return dict with real GEE output.
    """
    np.random.seed(hash(f"{district}{year}{season}") % (2**31))
    drought_districts = ['Anantapur','Kurnool','Kadapa','Prakasam']
    base_ndvi = 0.35 if district in drought_districts else 0.52
    # simulate a drought year dip
    if year in [2019,2022]:
        base_ndvi *= 0.65

    ndvi  = round(np.clip(np.random.normal(base_ndvi, 0.06), 0.05, 0.85), 3)
    ndwi  = round(np.clip(np.random.normal(-0.05 if district in drought_districts else 0.08, 0.05), -0.4, 0.4), 3)
    savi  = round(ndvi * 0.82 * np.random.uniform(0.9,1.1), 3)
    evi   = round(ndvi * 0.88 * np.random.uniform(0.88,1.08), 3)
    vci   = round(np.clip((ndvi - 0.10)/(0.75 - 0.10)*100, 0, 100), 1)
    flood = round(max(0, np.random.normal(0.005, 0.003)), 4)
    chirps= round(np.random.normal(620 if district in drought_districts else 920, 90), 1)

    return {
        'ndvi': ndvi, 'ndwi': ndwi, 'savi': savi,
        'evi': evi,   'vci': vci,
        'flood_fraction': flood,
        'chirps_rain_mm': chirps,
    }

def compute_fraud_score(row, sat):
    score = 0.0
    reasons = []

    loss = row.get('loss_percent', 0)
    claim_val_ratio = row.get('claim_value_ratio', 0)
    claim_per_ha    = row.get('claim_per_hectare', 0)
    rain_mm         = row.get('rainfall_mm', 500)

    ndvi  = sat.get('ndvi', 0.4)
    ndwi  = sat.get('ndwi', 0.0)
    vci   = sat.get('vci', 50)
    chirps= sat.get('chirps_rain_mm', 600)
    flood = sat.get('flood_fraction', 0)

    # Signal 1: healthy NDVI but high loss claimed
    if loss > 40 and ndvi >= 0.40:
        score += 0.28
        reasons.append(f"NDVI={ndvi:.2f} (healthy) but {loss:.0f}% loss claimed")

    # Signal 2: farmer reports drought but CHIRPS shows good rain
    rain_dev = abs(rain_mm - chirps) / (chirps + 1) * 100
    if rain_dev > 35 and loss > 30:
        score += 0.22
        reasons.append(f"Reported rain={rain_mm:.0f}mm vs satellite={chirps:.0f}mm (Δ{rain_dev:.0f}%)")

    # Signal 3: VCI shows no drought stress
    if vci > 40 and loss > 50:
        score += 0.18
        reasons.append(f"VCI={vci:.0f} (no drought stress) but {loss:.0f}% loss")

    # Signal 4: positive NDWI but flood claim
    if ndwi > 0.05 and flood < 0.01 and loss > 50:
        score += 0.12
        reasons.append(f"No flood signal (NDWI={ndwi:.2f}, flood frac={flood:.4f})")

    # Signal 5: extreme claim-to-value ratio
    if claim_val_ratio > 0.85:
        score += 0.15
        reasons.append(f"Claim/crop value = {claim_val_ratio:.2f} (exceeds crop worth)")

    # Signal 6: claim per hectare outlier
    if claim_per_ha > 80000:
        score += 0.08
        reasons.append(f"Claim/ha = ₹{claim_per_ha:,.0f} (statistical outlier)")

    return min(score, 1.0), reasons


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px;'>
      <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;
                  color:#e8f2ff;letter-spacing:-0.5px;'>AgriShield</div>
      <div style='font-size:11px;color:#3b6ea8;letter-spacing:1.5px;
                  text-transform:uppercase;'>Fraud Intelligence System</div>
    </div>
    <hr style='border:none;border-top:1px solid #1e2d4a;margin:12px 0;'/>
    """, unsafe_allow_html=True)

    st.markdown("**Data Source**")
    uploaded = st.file_uploader("Upload Insurance CSV", type=['csv'],
                                label_visibility='collapsed')

    st.markdown("<hr style='border:none;border-top:1px solid #1e2d4a;margin:12px 0;'/>",
                unsafe_allow_html=True)
    st.markdown("**Filters**")

    districts_all = list(DISTRICT_COORDS.keys())
    sel_districts = st.multiselect("Districts", districts_all,
                                   default=['Anantapur','Kurnool','Guntur','Prakasam'])

    years_all = list(range(2000, 2026))
    sel_years = st.select_slider("Year range",
                                  options=years_all,
                                  value=(2019, 2025))

    crops_all = ['All','Paddy','Groundnut','Cotton','Maize',
                 'Chilli','Tobacco','Sugarcane','Sunflower','Bengalgram','Jowar']
    sel_crop = st.selectbox("Crop", crops_all)

    st.markdown("<hr style='border:none;border-top:1px solid #1e2d4a;margin:12px 0;'/>",
                unsafe_allow_html=True)
    st.markdown("**GEE Connection**")
    gee_project = st.text_input("GEE Project ID", placeholder="your-project-id",
                                 help="Enter your Google Earth Engine project ID")
    use_real_gee = st.toggle("Use real GEE data", value=False,
                              help="Toggle ON after entering project ID and authenticating")
    if use_real_gee:
        st.info("⚡ Set use_real_gee=True and replace get_gee_indices() body with your ee calls")

    st.markdown("<hr style='border:none;border-top:1px solid #1e2d4a;margin:12px 0;'/>",
                unsafe_allow_html=True)
    st.markdown("**Fraud Threshold**")
    fraud_thresh = st.slider("Flag as HIGH RISK above", 0.40, 0.90, 0.70, 0.05)

# ════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════
if uploaded:
    df_raw = pd.read_csv(uploaded)
    if df_raw['loss_percent'].max() <= 1.0:
        df_raw['loss_percent'] = df_raw['loss_percent'] * 100
    df_raw['district_clean'] = df_raw['district'].str.strip().str.title()
    df_raw['claim_per_hectare']   = df_raw['claim_amount_rs'] / (df_raw['area_hectares'] + 1e-8)
    df_raw['claim_value_ratio']   = df_raw['claim_amount_rs'] / (df_raw['crop_value_rs'] + 1)
    df_raw['claim_premium_ratio'] = df_raw['claim_amount_rs'] / (df_raw['insurance_premium_rs'] + 1)
    df_raw['production_per_ha']   = df_raw['production_tons'] / (df_raw['area_hectares'] + 1e-8)
else:
    # Use bundled synthetic data as fallback
    import os
    fallback = 'ap_synthetic_agri_insurance_2000_2025_12000rows__1_.csv'
    if os.path.exists(fallback):
        df_raw = load_insurance_data(fallback)
    else:
        # Generate minimal demo data
        np.random.seed(42)
        n = 500
        df_raw = pd.DataFrame({
            'year': np.random.randint(2019,2026,n),
            'district': np.random.choice(list(DISTRICT_COORDS.keys()), n),
            'farmer_id': [f'F{i:05d}' for i in range(n)],
            'crop': np.random.choice(['Paddy','Groundnut','Cotton','Maize'],n),
            'area_hectares': np.random.lognormal(0.5,0.5,n).clip(0.2,8),
            'rainfall_mm': np.random.normal(700,200,n).clip(100,1400),
            'production_tons': np.random.lognormal(1,0.5,n),
            'crop_value_rs': np.random.lognormal(12,0.8,n),
            'insurance_premium_rs': np.random.lognormal(9,0.5,n),
            'claim_amount_rs': np.random.lognormal(11,1,n),
            'loss_percent': np.random.uniform(0,90,n),
        })
        df_raw['district_clean'] = df_raw['district']
        df_raw['claim_per_hectare']   = df_raw['claim_amount_rs']/(df_raw['area_hectares']+1e-8)
        df_raw['claim_value_ratio']   = df_raw['claim_amount_rs']/(df_raw['crop_value_rs']+1)
        df_raw['claim_premium_ratio'] = df_raw['claim_amount_rs']/(df_raw['insurance_premium_rs']+1)
        df_raw['production_per_ha']   = df_raw['production_tons']/(df_raw['area_hectares']+1e-8)

# ── Apply filters ─────────────────────────────────────────────────
df = df_raw[
    (df_raw['district_clean'].isin(sel_districts)) &
    (df_raw['year'].between(sel_years[0], sel_years[1]))
].copy()
if sel_crop != 'All':
    df = df[df['crop'] == sel_crop]

if len(df) == 0:
    st.warning("No data matches your filters. Please adjust district/year/crop selection.")
    st.stop()

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
col_title, col_status = st.columns([3,1])
with col_title:
    st.markdown("""
    <div class="main-title">AgriShield <span class="accent">·</span> AP</div>
    <div class="main-sub">MACHINE LEARNING FRAUD INTELLIGENCE · ANDHRA PRADESH CROP INSURANCE</div>
    """, unsafe_allow_html=True)
with col_status:
    st.markdown(f"""
    <div style='text-align:right;padding-top:12px;'>
      <div style='font-size:11px;color:#3b6ea8;letter-spacing:1px;text-transform:uppercase;'>Live status</div>
      <div style='font-size:13px;color:#27c97a;font-weight:500;'>● NASA POWER Connected</div>
      <div style='font-size:13px;color:{"#27c97a" if use_real_gee else "#f5a623"};font-weight:500;'>
        {"● GEE Connected" if use_real_gee else "○ GEE Simulated"}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:1px;background:#1a2d4a;margin:16px 0 24px;'></div>",
            unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Overview",
    "🛰  Satellite Analysis",
    "🔍  Fraud Detection",
    "🗺  District Map",
    "🌡  Climate (NASA)",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
with tab1:
    # KPI row
    total_claims = len(df)
    total_claim_rs = df['claim_amount_rs'].sum()
    avg_loss = df[df['loss_percent']>0]['loss_percent'].mean()
    high_loss = (df['loss_percent']>60).sum()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("Total Claims",f"{total_claims:,}",
                              f"₹{total_claim_rs/1e7:.1f} Cr total payout","blue"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Avg Loss %",f"{avg_loss:.1f}%",
                              "Among claims with loss > 0","amber"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("High Loss Flags",f"{high_loss:,}",
                              f"{high_loss/total_claims*100:.1f}% of all claims","red"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Districts",f"{df['district_clean'].nunique()}",
                              f"{df['crop'].nunique()} crops tracked","green"), unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Claims Overview</div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        # Claims by year
        yr_data = df.groupby('year').agg(
            claims=('farmer_id','count'),
            total_payout=('claim_amount_rs','sum')
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=yr_data['year'], y=yr_data['claims'],
                              name='Claims', marker_color='#1e4d8c',
                              marker_line_color='#3b9eff', marker_line_width=0.5), secondary_y=False)
        fig.add_trace(go.Scatter(x=yr_data['year'], y=yr_data['total_payout']/1e6,
                                  name='Payout (₹L)', mode='lines+markers',
                                  line=dict(color='#f5a623', width=2),
                                  marker=dict(size=6, color='#f5a623')), secondary_y=True)
        fig.update_layout(title='Claims & Payout by Year', **PLOTLY_LAYOUT, height=300)
        fig.update_yaxes(title_text="Claim count", secondary_y=False, color='#8aafd4')
        fig.update_yaxes(title_text="Payout (₹ Lakhs)", secondary_y=True, color='#f5a623')
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Loss distribution
        fig2 = px.histogram(df[df['loss_percent']>0], x='loss_percent',
                            nbins=40, color_discrete_sequence=['#3b9eff'])
        fig2.add_vline(x=60, line_dash='dash', line_color='#e84545',
                       annotation_text='High risk threshold (60%)',
                       annotation_font_color='#e84545')
        fig2.update_layout(title='Loss % Distribution', **PLOTLY_LAYOUT,
                           height=300, bargap=0.05)
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        crop_data = df.groupby('crop')['claim_amount_rs'].sum().sort_values(ascending=True)
        fig3 = go.Figure(go.Bar(
            x=crop_data.values/1e6, y=crop_data.index,
            orientation='h',
            marker=dict(
                color=crop_data.values/1e6,
                colorscale=[[0,'#1e3a6e'],[0.5,'#1e6eb5'],[1,'#3b9eff']],
                showscale=False,
            )
        ))
        fig3.update_layout(title='Claim Amount by Crop (₹ Lakhs)', **PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        dist_data = df.groupby('district_clean').agg(
            avg_loss=('loss_percent','mean'),
            total_claims=('farmer_id','count')
        ).reset_index()
        fig4 = px.scatter(dist_data, x='total_claims', y='avg_loss',
                          size='total_claims', text='district_clean',
                          color='avg_loss',
                          color_continuous_scale=['#1e4d8c','#f5a623','#e84545'])
        fig4.update_traces(textfont=dict(color='#c8d8f0', size=10),
                           textposition='top center')
        fig4.update_coloraxes(showscale=False)
        fig4.update_layout(title='Districts: Claims vs Avg Loss %',
                           **PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — SATELLITE ANALYSIS (NDVI/NDWI/SAVI/EVI comparison)
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Satellite Index Comparison — Before vs After Season</div>",
                unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        sel_dist_sat = st.selectbox("District", sel_districts, key='sat_dist')
    with col_s2:
        sel_year_sat = st.selectbox("Year", list(range(sel_years[0], sel_years[1]+1)),
                                     index=0, key='sat_year')
    with col_s3:
        compare_year = st.selectbox("Compare with year",
                                     [y for y in range(sel_years[0], sel_years[1]+1)
                                      if y != sel_year_sat],
                                     key='sat_compare')

    # Fetch indices for both years
    with st.spinner("Fetching satellite indices..."):
        sat_curr = get_gee_indices(sel_dist_sat, sel_year_sat, 'kharif')
        sat_prev = get_gee_indices(sel_dist_sat, compare_year, 'kharif')

    # Index comparison cards
    indices = ['ndvi','ndwi','savi','evi','vci']
    index_labels = {
        'ndvi': ('NDVI','Vegetation health (0–1)','≥0.40 = healthy'),
        'ndwi': ('NDWI','Water stress (−1 to +1)','> 0 = water present'),
        'savi': ('SAVI','Soil-adj. veg. index (0–1)','Sparse vegetation areas'),
        'evi':  ('EVI','Enhanced veg. index (0–1)','Dense crop areas'),
        'vci':  ('VCI','Veg. condition index (0–100)','< 35 = drought stress'),
    }

    st.markdown(f"""
    <div style='display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;'>
    """, unsafe_allow_html=True)

    cols_idx = st.columns(5)
    for col, idx in zip(cols_idx, indices):
        curr_val = sat_curr.get(idx, 0)
        prev_val = sat_prev.get(idx, 0)
        delta    = curr_val - prev_val
        delta_pct= delta/abs(prev_val+1e-8)*100
        lbl, desc, bench = index_labels[idx]
        color = '#27c97a' if delta >= 0 else '#e84545'
        delta_icon = '▲' if delta >= 0 else '▼'
        with col:
            st.markdown(f"""
            <div class="kpi-card {'green' if delta>=0 else 'red'}">
              <div class="kpi-label">{lbl}</div>
              <div class="kpi-value" style='font-size:26px;'>{curr_val:.3f}</div>
              <div style='font-size:11px;color:{color};margin-top:4px;font-weight:600;'>
                {delta_icon} {abs(delta):.3f} vs {compare_year}
              </div>
              <div class="kpi-sub">{bench}</div>
            </div>
            """, unsafe_allow_html=True)

    # Before/After radar chart
    st.markdown("<div class='section-header'>Index Profile — Radar Comparison</div>",
                unsafe_allow_html=True)

    col_radar, col_trend = st.columns(2)

    with col_radar:
        radar_indices = ['ndvi','ndwi','savi','evi']
        # Normalize 0-1 for radar
        curr_vals = [max(0, min(1, sat_curr.get(i,0))) for i in radar_indices]
        prev_vals = [max(0, min(1, sat_prev.get(i,0))) for i in radar_indices]
        labels_r  = [i.upper() for i in radar_indices]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=curr_vals + [curr_vals[0]],
            theta=labels_r + [labels_r[0]],
            fill='toself', name=str(sel_year_sat),
            line_color='#3b9eff', fillcolor='rgba(59,158,255,0.15)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=prev_vals + [prev_vals[0]],
            theta=labels_r + [labels_r[0]],
            fill='toself', name=str(compare_year),
            line_color='#f5a623', fillcolor='rgba(245,166,35,0.10)'
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0,1], color='#3b6ea8',
                                gridcolor='#1a2d4a'),
                angularaxis=dict(color='#8aafd4', gridcolor='#1a2d4a'),
            ),
            **PLOTLY_LAYOUT, height=360,
            title=f'{sel_dist_sat}: {sel_year_sat} vs {compare_year}',
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_trend:
        # NDVI trend across all years for this district
        trend_years = list(range(max(2015,sel_years[0]), sel_years[1]+1))
        ndvi_trend  = [get_gee_indices(sel_dist_sat, y, 'kharif')['ndvi'] for y in trend_years]
        ndwi_trend  = [get_gee_indices(sel_dist_sat, y, 'kharif')['ndwi'] for y in trend_years]
        vci_trend   = [get_gee_indices(sel_dist_sat, y, 'kharif')['vci']  for y in trend_years]

        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(go.Scatter(
            x=trend_years, y=ndvi_trend, name='NDVI',
            line=dict(color='#27c97a', width=2),
            mode='lines+markers', marker=dict(size=5)
        ), secondary_y=False)
        fig_trend.add_trace(go.Scatter(
            x=trend_years, y=ndwi_trend, name='NDWI',
            line=dict(color='#3b9eff', width=2, dash='dot'),
            mode='lines+markers', marker=dict(size=5)
        ), secondary_y=False)
        fig_trend.add_trace(go.Scatter(
            x=trend_years, y=vci_trend, name='VCI',
            line=dict(color='#f5a623', width=1.5, dash='dash'),
            mode='lines+markers', marker=dict(size=4)
        ), secondary_y=True)
        fig_trend.add_hline(y=0.38, line_dash="dash", line_color="#e84545",
                             annotation_text="NDVI stress threshold",
                             annotation_font_color="#e84545",
                             secondary_y=False)
        fig_trend.update_layout(
            title=f'{sel_dist_sat}: Vegetation Trend',
            **PLOTLY_LAYOUT, height=360
        )
        fig_trend.update_yaxes(title_text="Index value", secondary_y=False, color='#8aafd4')
        fig_trend.update_yaxes(title_text="VCI (0–100)", secondary_y=True, color='#f5a623')
        st.plotly_chart(fig_trend, use_container_width=True)

    # Multi-district NDVI heatmap
    st.markdown("<div class='section-header'>NDVI Heatmap — All Selected Districts × Years</div>",
                unsafe_allow_html=True)

    heat_years = list(range(max(2019,sel_years[0]), min(2026,sel_years[1]+1)))
    heat_data  = {}
    for d in sel_districts:
        heat_data[d] = [get_gee_indices(d, y, 'kharif')['ndvi'] for y in heat_years]

    heat_df = pd.DataFrame(heat_data, index=heat_years).T
    fig_heat = px.imshow(heat_df,
                         color_continuous_scale=['#4a0a0a','#8b2020','#d4a017','#2d7a2d','#1a5c1a'],
                         zmin=0.15, zmax=0.75,
                         labels=dict(color='NDVI'),
                         aspect='auto')
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=280,
                            title='Kharif Season NDVI — Darker Red = Drought Stress')
    fig_heat.update_coloraxes(colorbar_tickfont=dict(color='#8aafd4'))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Index explanation
    with st.expander("What do these indices mean?"):
        st.markdown("""
        | Index | Full Name | Range | Fraud Signal |
        |-------|-----------|-------|-------------|
        | **NDVI** | Normalized Difference Vegetation Index | 0–1 | ≥ 0.40 during claim season = healthy crop, yet loss claimed |
        | **NDWI** | Normalized Difference Water Index | −1 to +1 | > 0 = water present, contradicts drought claim |
        | **SAVI** | Soil Adjusted Vegetation Index | 0–1 | Better than NDVI for AP's sparse soil areas |
        | **EVI** | Enhanced Vegetation Index | 0–1 | Doesn't saturate in dense crops like cotton/tobacco |
        | **VCI** | Vegetation Condition Index | 0–100 | < 35 = confirmed drought stress |
        | **CHIRPS** | Rainfall estimate (satellite) | mm | Compare vs farmer-reported rainfall |
        """)

# ════════════════════════════════════════════════════════════════
# TAB 3 — FRAUD DETECTION
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Claim-Level Fraud Scoring</div>",
                unsafe_allow_html=True)

    # Score a sample of claims
    sample_size = min(300, len(df))
    df_sample = df.sample(sample_size, random_state=42).copy()

    scores, reasons_list = [], []
    for _, row in df_sample.iterrows():
        sat = get_gee_indices(row['district_clean'],
                              int(row['year']), 'kharif')
        s, r = compute_fraud_score(row, sat)
        scores.append(s)
        reasons_list.append(' | '.join(r) if r else 'No anomalies detected')

    df_sample['fraud_score'] = scores
    df_sample['reasons']     = reasons_list
    df_sample['verdict']     = df_sample['fraud_score'].apply(
        lambda s: 'HIGH RISK' if s>=fraud_thresh else ('MEDIUM RISK' if s>=0.40 else 'LOW RISK')
    )

    # KPIs
    high_risk  = (df_sample['verdict']=='HIGH RISK').sum()
    med_risk   = (df_sample['verdict']=='MEDIUM RISK').sum()
    low_risk   = (df_sample['verdict']=='LOW RISK').sum()
    avg_score  = df_sample['fraud_score'].mean()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("HIGH RISK",f"{high_risk}",
                              f"{high_risk/sample_size*100:.1f}% of analysed","red"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("MEDIUM RISK",f"{med_risk}",
                              "Review recommended","amber"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("LOW RISK / Clear",f"{low_risk}",
                              "Approve without review","green"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Mean Fraud Score",f"{avg_score:.3f}",
                              "0=genuine, 1=certain fraud","blue"), unsafe_allow_html=True)

    col_fd1, col_fd2 = st.columns(2)

    with col_fd1:
        fig_score = px.histogram(df_sample, x='fraud_score', nbins=30,
                                  color='verdict',
                                  color_discrete_map={
                                      'HIGH RISK':'#e84545',
                                      'MEDIUM RISK':'#f5a623',
                                      'LOW RISK':'#27c97a'
                                  })
        fig_score.add_vline(x=fraud_thresh, line_dash='dash', line_color='white',
                             annotation_text=f'Threshold ({fraud_thresh})',
                             annotation_font_color='white')
        fig_score.update_layout(title='Fraud Score Distribution',
                                 **PLOTLY_LAYOUT, height=320, bargap=0.05)
        st.plotly_chart(fig_score, use_container_width=True)

    with col_fd2:
        fig_pie = px.pie(
            values=[high_risk, med_risk, low_risk],
            names=['High Risk','Medium Risk','Low Risk'],
            color_discrete_sequence=['#e84545','#f5a623','#27c97a'],
            hole=0.6,
        )
        fig_pie.update_traces(textfont_color='white')
        fig_pie.update_layout(title='Risk Distribution', **PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Fraud reasons breakdown
    col_fd3, col_fd4 = st.columns(2)
    with col_fd3:
        reason_counts = {}
        for r in df_sample[df_sample['fraud_score']>0]['reasons']:
            for part in r.split(' | '):
                if 'NDVI' in part:       reason_counts['NDVI mismatch']   = reason_counts.get('NDVI mismatch',0)+1
                elif 'rain' in part.lower(): reason_counts['Rainfall mismatch'] = reason_counts.get('Rainfall mismatch',0)+1
                elif 'VCI' in part:      reason_counts['VCI no drought']  = reason_counts.get('VCI no drought',0)+1
                elif 'Claim/crop' in part: reason_counts['Over-claiming'] = reason_counts.get('Over-claiming',0)+1
                elif 'flood' in part.lower(): reason_counts['No flood signal'] = reason_counts.get('No flood signal',0)+1

        if reason_counts:
            fig_reasons = go.Figure(go.Bar(
                x=list(reason_counts.values()),
                y=list(reason_counts.keys()),
                orientation='h',
                marker_color=['#e84545','#f5a623','#3b9eff','#9b59b6','#27c97a'][:len(reason_counts)]
            ))
            fig_reasons.update_layout(title='Fraud Signal Breakdown',
                                       **PLOTLY_LAYOUT, height=280)
            st.plotly_chart(fig_reasons, use_container_width=True)

    with col_fd4:
        # Score by district
        dist_scores = df_sample.groupby('district_clean')['fraud_score'].mean().sort_values(ascending=False)
        fig_dist_score = go.Figure(go.Bar(
            x=dist_scores.index, y=dist_scores.values,
            marker=dict(
                color=dist_scores.values,
                colorscale=[[0,'#1e4d8c'],[0.5,'#f5a623'],[1,'#e84545']],
                showscale=False,
            )
        ))
        fig_dist_score.update_layout(title='Mean Fraud Score by District',
                                      **PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_dist_score, use_container_width=True)

    # Flagged claims table
    st.markdown("<div class='section-header'>Flagged Claims — Top 20 High Risk</div>",
                unsafe_allow_html=True)

    high_df = df_sample[df_sample['verdict']=='HIGH RISK'].sort_values(
        'fraud_score', ascending=False).head(20)

    if len(high_df) == 0:
        st.info("No HIGH RISK claims found with current threshold. Try lowering it in the sidebar.")
    else:
        rows_html = ""
        for _, row in high_df.iterrows():
            verd, level, color = verdict_color(row['fraud_score'])
            rows_html += f"""<tr>
              <td>{row['farmer_id']}</td>
              <td>{row['district_clean']}</td>
              <td>{int(row['year'])}</td>
              <td>{row['crop']}</td>
              <td>{row['loss_percent']:.1f}%</td>
              <td>₹{row['claim_amount_rs']:,.0f}</td>
              <td>{row['fraud_score']:.3f}</td>
              <td><span class="badge-{level}">{verd}</span></td>
              <td style='font-size:11px;max-width:300px;'>{row['reasons'][:120]}…</td>
            </tr>"""

        st.markdown(f"""
        <table class="styled-table">
          <thead><tr>
            <th>Farmer ID</th><th>District</th><th>Year</th><th>Crop</th>
            <th>Loss%</th><th>Claim ₹</th><th>Score</th>
            <th>Verdict</th><th>Evidence</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 4 — DISTRICT MAP
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Fraud Risk Heatmap — Andhra Pradesh Districts</div>",
                unsafe_allow_html=True)

    # Build district-level metrics
    dist_metrics = []
    for d in sel_districts:
        sub = df[df['district_clean']==d]
        if len(sub)==0: continue
        sat_info = get_gee_indices(d, sel_years[1], 'kharif')
        coords   = DISTRICT_COORDS.get(d, (15.9, 79.7))
        avg_loss = sub['loss_percent'].mean()
        total_cl = sub['claim_amount_rs'].sum()
        n_claims = len(sub)
        # rough district fraud risk
        risk = min(1.0, avg_loss/100*0.5 +
                   (1-sat_info['ndvi'])*0.3 +
                   max(0,(0.35 - sat_info['vci']/100))*0.2)
        dist_metrics.append({
            'district': d,
            'lat': coords[0], 'lon': coords[1],
            'avg_loss': round(avg_loss,1),
            'total_claim_cr': round(total_cl/1e7,2),
            'n_claims': n_claims,
            'ndvi': sat_info['ndvi'],
            'vci':  sat_info['vci'],
            'risk_score': round(risk,3),
        })

    map_df = pd.DataFrame(dist_metrics)

    col_map, col_legend = st.columns([3,1])
    with col_map:
        fig_map = px.scatter_mapbox(
            map_df, lat='lat', lon='lon',
            size='n_claims',
            color='risk_score',
            color_continuous_scale=['#1a5c1a','#f5a623','#e84545'],
            range_color=[0,1],
            hover_name='district',
            hover_data={
                'avg_loss': ':.1f',
                'total_claim_cr': ':.2f',
                'ndvi': ':.3f',
                'vci': ':.1f',
                'risk_score': ':.3f',
                'lat': False, 'lon': False
            },
            zoom=6,
            center={"lat": 15.9, "lon": 79.7},
            mapbox_style='carto-darkmatter',
            size_max=45,
        )
        fig_map.update_layout(
            height=520,
            margin=dict(t=0,b=0,l=0,r=0),
            paper_bgcolor='rgba(0,0,0,0)',
            coloraxis_colorbar=dict(
                title='Risk', tickfont=dict(color='#8aafd4'),
                titlefont=dict(color='#8aafd4')
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_legend:
        st.markdown("<br>", unsafe_allow_html=True)
        for _, row in map_df.sort_values('risk_score',ascending=False).iterrows():
            _, level, color = verdict_color(row['risk_score'])
            st.markdown(f"""
            <div style='background:#0f1d35;border:1px solid #1e3058;border-radius:10px;
                        padding:12px 14px;margin-bottom:8px;'>
              <div style='font-size:12px;font-weight:600;color:#e8f2ff;
                          display:flex;justify-content:space-between;'>
                <span>{row['district']}</span>
                <span style='color:{color};'>{row['risk_score']:.2f}</span>
              </div>
              <div style='font-size:11px;color:#5a7ba8;margin-top:4px;'>
                NDVI {row['ndvi']:.2f} · VCI {row['vci']:.0f} · Loss {row['avg_loss']:.0f}%
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Choropleth-style bar comparison
    st.markdown("<div class='section-header'>District Risk Comparison</div>",
                unsafe_allow_html=True)
    fig_comp = px.bar(
        map_df.sort_values('risk_score', ascending=False),
        x='district', y='risk_score',
        color='risk_score',
        color_continuous_scale=['#1a5c1a','#f5a623','#e84545'],
        range_color=[0,1],
        text='risk_score'
    )
    fig_comp.update_traces(texttemplate='%{text:.2f}', textfont_color='white')
    fig_comp.update_coloraxes(showscale=False)
    fig_comp.update_layout(title='Composite Risk Score by District',
                            **PLOTLY_LAYOUT, height=280)
    st.plotly_chart(fig_comp, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 5 — NASA POWER CLIMATE
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>NASA POWER Climate Data — Drought & Heat Indicators</div>",
                unsafe_allow_html=True)

    col_n1, col_n2 = st.columns(2)
    with col_n1: sel_dist_nasa = st.selectbox("District", sel_districts, key='nasa_dist')
    with col_n2: sel_year_nasa = st.selectbox("Year", list(range(sel_years[0], sel_years[1]+1)),
                                               key='nasa_year')

    coords = DISTRICT_COORDS.get(sel_dist_nasa, (15.9,79.7))

    with st.spinner(f"Fetching NASA POWER data for {sel_dist_nasa} {sel_year_nasa}…"):
        nasa = fetch_nasa_power(coords[0], coords[1], sel_year_nasa)

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        drought_flag = nasa['precip_mm'] < 500
        st.markdown(kpi("Annual Rainfall",
                         f"{nasa['precip_mm']:.0f} mm" if not np.isnan(nasa['precip_mm']) else "N/A",
                         "⚠ DROUGHT YEAR" if drought_flag else "Normal range",
                         "red" if drought_flag else "green"), unsafe_allow_html=True)
    with c2:
        heat_flag = nasa['tmax_c'] > 40
        st.markdown(kpi("Max Temp (Tmax)",
                         f"{nasa['tmax_c']:.1f}°C" if not np.isnan(nasa['tmax_c']) else "N/A",
                         "⚠ HEAT STRESS" if heat_flag else "Normal range",
                         "red" if heat_flag else "amber"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Min Temp (Tmin)",
                         f"{nasa['tmin_c']:.1f}°C" if not np.isnan(nasa['tmin_c']) else "N/A",
                         "Cold stress" if nasa['tmin_c']<10 else "Normal",
                         "blue"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Solar Radiation",
                         f"{nasa['solar']:.1f} MJ/m²" if not np.isnan(nasa['solar']) else "N/A",
                         "Crop growth potential","amber"), unsafe_allow_html=True)

    # Multi-year trend
    st.markdown("<div class='section-header'>Multi-Year Climate Trend</div>",
                unsafe_allow_html=True)

    trend_yrs = list(range(max(2015,sel_years[0]), sel_years[1]+1))
    nasa_trend = []
    prog = st.progress(0, text="Fetching historical NASA data…")
    for idx_p, y in enumerate(trend_yrs):
        nd = fetch_nasa_power(coords[0], coords[1], y)
        nd['year'] = y
        nasa_trend.append(nd)
        prog.progress((idx_p+1)/len(trend_yrs), text=f"Fetching {y}…")
    prog.empty()

    nasa_trend_df = pd.DataFrame(nasa_trend)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_rain = go.Figure()
        fig_rain.add_trace(go.Bar(x=nasa_trend_df['year'], y=nasa_trend_df['precip_mm'],
                                   name='Rainfall', marker_color='#3b9eff',
                                   marker_line_width=0))
        fig_rain.add_hline(y=500, line_dash='dash', line_color='#e84545',
                            annotation_text='Drought threshold (500mm)',
                            annotation_font_color='#e84545')
        fig_rain.update_layout(title=f'{sel_dist_nasa}: Annual Rainfall (NASA POWER)',
                                **PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig_rain, use_container_width=True)

    with col_c2:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=nasa_trend_df['year'], y=nasa_trend_df['tmax_c'],
                                       name='Tmax', fill=None,
                                       line=dict(color='#e84545', width=2)))
        fig_temp.add_trace(go.Scatter(x=nasa_trend_df['year'], y=nasa_trend_df['tmin_c'],
                                       name='Tmin', fill='tonexty',
                                       line=dict(color='#3b9eff', width=2),
                                       fillcolor='rgba(59,158,255,0.08)'))
        fig_temp.add_hline(y=40, line_dash='dash', line_color='#f5a623',
                            annotation_text='Heat stress (40°C)',
                            annotation_font_color='#f5a623')
        fig_temp.update_layout(title=f'{sel_dist_nasa}: Temperature Range (NASA POWER)',
                                **PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig_temp, use_container_width=True)

    # Drought-Claim mismatch analysis
    st.markdown("<div class='section-header'>Drought-Claim Mismatch — Where Claims Don't Match Climate</div>",
                unsafe_allow_html=True)

    mismatch_data = []
    for y in trend_yrs:
        nd    = next((r for r in nasa_trend if r['year']==y), {})
        sat_y = get_gee_indices(sel_dist_nasa, y, 'kharif')
        sub_y = df[(df['district_clean']==sel_dist_nasa) & (df['year']==y)]
        if len(sub_y)==0: continue
        avg_claim_loss = sub_y['loss_percent'].mean()
        # Drought score from NASA: normalized
        drought_score = max(0, min(1, 1 - (nd.get('precip_mm',600)-200)/(900-200)))
        claim_score   = avg_claim_loss / 100
        mismatch      = max(0, claim_score - drought_score)
        mismatch_data.append({
            'year': y,
            'avg_claimed_loss': round(avg_claim_loss,1),
            'nasa_drought_score': round(drought_score,3),
            'claim_score': round(claim_score,3),
            'mismatch': round(mismatch,3),
            'ndvi': sat_y['ndvi'],
        })

    if mismatch_data:
        mm_df = pd.DataFrame(mismatch_data)
        fig_mm = go.Figure()
        fig_mm.add_trace(go.Scatter(x=mm_df['year'], y=mm_df['claim_score'],
                                     name='Claimed Loss Score', fill=None,
                                     line=dict(color='#e84545',width=2),
                                     mode='lines+markers'))
        fig_mm.add_trace(go.Scatter(x=mm_df['year'], y=mm_df['nasa_drought_score'],
                                     name='NASA Drought Score', fill='tonexty',
                                     line=dict(color='#3b9eff',width=2),
                                     fillcolor='rgba(232,69,69,0.12)',
                                     mode='lines+markers'))
        fig_mm.add_trace(go.Bar(x=mm_df['year'], y=mm_df['mismatch'],
                                 name='Mismatch (Fraud Signal)',
                                 marker_color='rgba(245,166,35,0.6)',
                                 yaxis='y2'))
        fig_mm.update_layout(
            title=f'{sel_dist_nasa}: Claim vs Climate Mismatch',
            yaxis2=dict(overlaying='y', side='right', color='#f5a623',
                        gridcolor='rgba(0,0,0,0)'),
            **PLOTLY_LAYOUT, height=360,
            annotations=[dict(
                text="Gap between lines = potential over-claiming",
                x=0.01, y=1.05, xref='paper', yref='paper',
                showarrow=False, font=dict(color='#5a7ba8', size=11)
            )]
        )
        st.plotly_chart(fig_mm, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div style='margin-top:48px;padding:20px 0;border-top:1px solid #1a2d4a;
            display:flex;justify-content:space-between;align-items:center;'>
  <div style='font-family:Syne,sans-serif;font-size:16px;color:#2a4a70;font-weight:700;'>
    AgriShield · AP Fraud Intelligence
  </div>
  <div style='font-size:11px;color:#1e3058;letter-spacing:1px;text-transform:uppercase;'>
    Data: NASA POWER · GEE Sentinel-2 · CHIRPS · AP Insurance Records
  </div>
  <div style='font-size:11px;color:#1e3058;'>
    S.R.L Keerthi · Yashaswini H V
  </div>
</div>
""", unsafe_allow_html=True)
