

import pandas as pd

import numpy as np

import streamlit as st

import plotly.express as px

import math

import joblib



from datetime import date, timedelta

from pathlib import Path



DATA_DIR = Path(__file__).parent / "Data"

WEATHER_FEATURES_PATH = DATA_DIR / "sowing_window_features.csv"



st.set_page_config(

    page_title="SowingSense AI",

    page_icon="🌱",

    layout="wide"

)





st.markdown("""

<style>

    /* Main dashboard */

    .stApp {

        background: linear-gradient(135deg, #071923 0%, #0b292c 55%, #103b32 100%);

        color: #e5f3ed;

    }



    .main .block-container {

        max-width: 1500px;

        padding: 2rem 2.5rem 3rem;

    }



    /* Sidebar */

    [data-testid="stSidebar"] {

        background: #07151f;

        border-right: 1px solid #25444b;

    }



    [data-testid="stSidebar"] h1,

    [data-testid="stSidebar"] h2,

    [data-testid="stSidebar"] h3,

    [data-testid="stSidebar"] label,

    [data-testid="stSidebar"] p {

        color: #dcece6;

    }



    /* Headings */

    h1, h2, h3 {

        color: #edfff6 !important;

        letter-spacing: -0.4px;

    }



    h1 {

        font-weight: 800 !important;

    }



    /* Metric cards */

    [data-testid="stMetric"] {

        background: linear-gradient(145deg, #102d38, #0b222d);

        border: 1px solid #28505a;

        border-radius: 18px;

        padding: 20px 22px;

        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);

        min-height: 125px;

    }



    [data-testid="stMetricLabel"] {

        color: #b8d1ca !important;

    }



    [data-testid="stMetricValue"] {

        color: #effff7 !important;

        font-weight: 700;

    }



    /* Buttons */

    .stButton > button,

    .stDownloadButton > button {

        border-radius: 10px;

        min-height: 44px;

        font-weight: 600;

        transition: all 0.2s ease;

    }



    .stButton > button[kind="primary"] {

        background: #238653;

        color: white;

        border: 1px solid #39a66b;

    }



    .stButton > button[kind="primary"]:hover {

        background: #2da466;

        border-color: #68d69b;

    }



    /* Inputs */

    [data-testid="stSelectbox"],

    [data-testid="stDateInput"],

    [data-testid="stRadio"],

    [data-testid="stSlider"] {

        margin-bottom: 12px;

    }



    /* Alerts and information panels */

    [data-testid="stAlert"] {

        border-radius: 12px;

        border: 1px solid #31545a;

    }



    /* Tables */

    [data-testid="stDataFrame"],

    [data-testid="stTable"] {

        border: 1px solid #294a52;

        border-radius: 12px;

        overflow: hidden;

    }



    /* Dividers and captions */

    hr {

        border-color: #29474b;

        margin-top: 1.8rem;

        margin-bottom: 1.8rem;

    }



    [data-testid="stCaptionContainer"] {

        color: #9bb8b0;

    }



    /* Keep layout responsive */

    @media (max-width: 768px) {

        .main .block-container {

            padding: 1rem;

        }



        [data-testid="stMetric"] {

            padding: 14px;

        }

    }


    /* SowingSense premium redesign */
    :root { color-scheme: dark; }
    .stApp {
        background:
          radial-gradient(ellipse at 82% 0%, rgba(34,197,94,.12), transparent 35%),
          linear-gradient(135deg, #06131f 0%, #0a202b 55%, #0b2924 100%);
        color: #e8f5ee;
    }
    .main .block-container { max-width: 1500px; padding: 1.4rem 2rem 3rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#07131f 0%,#0a211f 100%);
        border-right: 1px solid rgba(94,234,164,.16);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.2rem; }
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,[data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p { color:#d9eee5; }
    h1,h2,h3 { color:#f0fff7 !important; letter-spacing:-.45px; }
    h1 { font-weight:850 !important; }
    p, label, .stMarkdown { color:#d4e6df; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg,rgba(19,48,59,.94),rgba(10,31,39,.94));
        border:1px solid rgba(90,166,145,.28); border-radius:18px;
        padding:18px 20px; min-height:120px;
        box-shadow:0 10px 30px rgba(0,0,0,.16);
        animation: card-in .45s ease both;
    }
    [data-testid="stMetricLabel"] { color:#a8c9bd !important; }
    [data-testid="stMetricValue"] { color:#effff5 !important; font-weight:800; }
    .stButton > button,.stDownloadButton > button {
        border-radius:12px; min-height:44px; font-weight:700;
        transition:transform .18s ease, box-shadow .18s ease, background .18s ease;
    }
    .stButton > button[kind="primary"] {
        background:linear-gradient(135deg,#16834d,#22a45e);
        color:white; border:1px solid #48c783;
        box-shadow:0 6px 18px rgba(34,197,94,.16);
    }
    .stButton > button:hover,.stDownloadButton > button:hover {
        transform:translateY(-2px); box-shadow:0 8px 22px rgba(34,197,94,.12);
    }
    [data-testid="stSelectbox"],[data-testid="stDateInput"],
    [data-testid="stRadio"],[data-testid="stSlider"] { margin-bottom:10px; }
    [data-testid="stAlert"] { border-radius:14px; border:1px solid #31545a; }
    [data-testid="stDataFrame"],[data-testid="stTable"] {
        border:1px solid #294a52; border-radius:14px; overflow:hidden;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color:rgba(94,234,164,.18) !important; border-radius:18px;
    }
    .hero-panel {
        /* Landscape image + dark overlay makes the hero closer to the reference UI.
           The image is loaded by the browser, so an internet connection is needed for it. */
        background:
          linear-gradient(90deg, rgba(4,19,29,.96) 0%, rgba(5,28,31,.86) 48%, rgba(7,42,30,.38) 100%),
          url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85") center 48% / cover no-repeat;
        border:1px solid rgba(94,234,164,.30); border-radius:24px;
        padding:clamp(26px,4vw,48px) clamp(22px,3vw,36px);
        min-height:230px; margin-bottom:22px;
        box-shadow:0 18px 42px rgba(0,0,0,.24);
        display:flex; flex-direction:column; justify-content:center;
        animation: card-in .45s ease both;
        overflow:hidden;
    }
    .hero-panel .hero-title { text-shadow:0 2px 18px rgba(0,0,0,.35); }
    .hero-panel .hero-copy { font-size:1.02rem; line-height:1.7; text-shadow:0 1px 10px rgba(0,0,0,.35); }
    [data-testid="stPlotlyChart"] {
        background:rgba(11,31,39,.55); border:1px solid rgba(94,234,164,.15);
        border-radius:18px; padding:8px 8px 0;
    }
    [data-testid="stExpander"] {
        background:rgba(10,31,39,.45); border:1px solid rgba(94,234,164,.15);
        border-radius:14px;
    }
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stDateInput"] > div > div {
        background:rgba(12,35,43,.88); border-radius:12px;
    }
    [data-testid="stRadio"] label { font-weight:600; }
    .stMarkdown h2 { margin-top:1.2rem; }

    .eyebrow { color:#65e6a0; text-transform:uppercase; font-size:.76rem;
        letter-spacing:2px; font-weight:800; }
    .hero-title { color:#f2fff7; font-size:clamp(1.8rem,3vw,2.8rem);
        font-weight:850; line-height:1.12; margin:.35rem 0 .6rem; }
    .hero-copy { color:#b9d4c9; max-width:760px; }
    .section-kicker { color:#65e6a0; font-size:.76rem; text-transform:uppercase;
        letter-spacing:1.5px; font-weight:800; margin-bottom:.2rem; }
    .small-note { color:#9bb8b0; font-size:.86rem; }
    @keyframes card-in { from { opacity:0; transform:translateY(7px); } to {opacity:1; transform:translateY(0);} }
    @media (prefers-reduced-motion: reduce) {
        *,*::before,*::after { animation:none !important; transition:none !important; }
    }
    @media (max-width:768px) {
        .main .block-container { padding:.85rem .8rem 2rem; }
        .hero-panel { padding:20px 18px; border-radius:18px; }
        [data-testid="stMetric"] { padding:13px; min-height:100px; }
        h1 { font-size:1.8rem !important; }
    }

</style>

""", unsafe_allow_html=True)



MODEL_PATH = Path(__file__).parent / "sowing_model_demo.joblib"



@st.cache_resource

def load_sowing_model():

    if MODEL_PATH.exists():

        return joblib.load(MODEL_PATH)

    return None



sowing_model = load_sowing_model()

CROPS = {

    "Paddy": {"rain_min": 60, "rain_ideal": 100, "temp_min": 22, "temp_ideal": 29, "waterlogging_sensitive": False},

    "Maize": {"rain_min": 25, "rain_ideal": 60, "temp_min": 18, "temp_ideal": 27, "waterlogging_sensitive": True},

    "Groundnut": {"rain_min": 20, "rain_ideal": 50, "temp_min": 22, "temp_ideal": 30, "waterlogging_sensitive": True},

    "Millet": {"rain_min": 15, "rain_ideal": 40, "temp_min": 24, "temp_ideal": 32, "waterlogging_sensitive": True},

    "Cotton": {"rain_min": 25, "rain_ideal": 65, "temp_min": 21, "temp_ideal": 30, "waterlogging_sensitive": True},

}



DISTRICTS = {

    "Thoothukudi": (8.7642, 78.1348),

    "Tirunelveli": (8.7139, 77.7567),

    "Madurai": (9.9252, 78.1198),

    "Chennai": (13.0827, 80.2707),

    "Coimbatore": (11.0168, 76.9558),

    "Thanjavur": (10.7870, 79.1378),

    "Salem": (11.6643, 78.1460),

}



def get_demo_weather(start_date, seed=42):

    """Clearly labelled synthetic data for an offline demo."""

    rng = np.random.default_rng(seed + start_date.timetuple().tm_yday)

    rows = []

    for i in range(42):

        d = start_date + timedelta(days=i)

        seasonal = 1.0 + 0.5 * math.sin((d.timetuple().tm_yday - 220) * 2 * math.pi / 365)

        rain = max(0, rng.gamma(1.3, 5.0) * seasonal)

        tmax = 31 + 3 * math.sin((d.timetuple().tm_yday - 80) * 2 * math.pi / 365) + rng.normal(0, 1.5)

        tmin = tmax - rng.uniform(7, 11)

        rows.append({"date": d, "rain_mm": round(float(rain), 1), "tmax_c": round(float(tmax), 1), "tmin_c": round(float(tmin), 1)})

    return pd.DataFrame(rows)



def fetch_open_meteo(lat, lon, start_date, end_date):

    """Fetch recent forecast data from Open-Meteo. If unavailable, caller uses demo data."""

    import requests

    url = "https://api.open-meteo.com/v1/forecast"

    params = {

        "latitude": lat, "longitude": lon,

        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",

        "timezone": "auto", "forecast_days": 16

    }

    response = requests.get(url, params=params, timeout=8)

    response.raise_for_status()

    data = response.json().get("daily", {})

    df = pd.DataFrame({

        "date": pd.to_datetime(data.get("time", [])),

        "rain_mm": data.get("precipitation_sum", []),

        "tmax_c": data.get("temperature_2m_max", []),

        "tmin_c": data.get("temperature_2m_min", []),

    })

    df = df.dropna()

    if df.empty:

        raise ValueError("No forecast data returned")

    return df



def fetch_current_weather(lat, lon):

    """Fetch current observed/modelled weather from Open-Meteo."""

    import requests



    url = "https://api.open-meteo.com/v1/forecast"



    params = {

        "latitude": lat,

        "longitude": lon,

        "current": (

            "temperature_2m,relative_humidity_2m,"

            "apparent_temperature,is_day,precipitation,"

            "rain,wind_speed_10m,weather_code"

        ),

        "timezone": "auto",

    }



    response = requests.get(url, params=params, timeout=15)

    response.raise_for_status()



    data = response.json()

    current = data.get("current")



    if not current:

        raise ValueError("Current weather data unavailable")



    return current



def predict_sowing_suitability(rainfall_mm, temperature_c, dry_spell_days):

    if sowing_model is None:

        return None, None



    features = pd.DataFrame([{

        "rainfall_mm": rainfall_mm,

        "temperature_c": temperature_c,

        "dry_spell_days": dry_spell_days

    }])



    prediction = int(sowing_model.predict(features)[0])

    probabilities = sowing_model.predict_proba(features)[0]

    confidence = float(max(probabilities) * 100)



    label = (

        "Favourable (demo model)"

        if prediction == 1

        else "Less favourable (demo model)"

    )



    return label, confidence

def score_window(window, crop_info):

    rain = float(window["rain_mm"].sum())

    avg_temp = float(((window["tmax_c"] + window["tmin_c"]) / 2).mean())

    rainy_days = int((window["rain_mm"] >= 1.0).sum())

    dry_spell = max_consecutive_dry(window["rain_mm"].tolist())

    rain_score = min(100, rain / crop_info["rain_ideal"] * 100)

    if rain < crop_info["rain_min"]:

        rain_score *= 0.55

    temp_score = max(0, 100 - abs(avg_temp - crop_info["temp_ideal"]) * 8)

    if avg_temp < crop_info["temp_min"]:

        temp_score *= 0.6

    dry_score = max(0, 100 - dry_spell * 12)

    excess_score = max(20, 100 - max(0, rain - crop_info["rain_ideal"]) * 0.8)

    score = round(0.35 * rain_score + 0.30 * temp_score + 0.20 * dry_score + 0.15 * excess_score)

    risks = []

    if rain < crop_info["rain_min"]:

        risks.append("Rainfall may be insufficient")

    if dry_spell >= 5:

        risks.append("Possible prolonged dry spell")

    if avg_temp > crop_info["temp_ideal"] + 3:

        risks.append("Temperature may be high")

    if rain > crop_info["rain_ideal"] * 1.4:

        risks.append("Excess rainfall / waterlogging risk")

    if not risks:

        risks.append("No major risk flagged by this simple prototype")

    return {

        "score": int(max(0, min(100, score))),

        "rain_mm": round(rain, 1),

        "avg_temp": round(avg_temp, 1),

        "dry_spell": dry_spell,

        "rainy_days": rainy_days,

        "risks": risks,

    }



def max_consecutive_dry(rain_values):

    best = current = 0

    for value in rain_values:

        if float(value) < 1.0:

            current += 1

            best = max(best, current)

        else:

            current = 0

    return best



def evaluate_windows(weather, crop_info):

    weather = weather.sort_values("date").reset_index(drop=True)

    results = []

    # Rank consecutive 7-day windows

    for start in range(max(1, len(weather) - 6)):

        window = weather.iloc[start:start+7]

        if len(window) < 7:

            continue

        metrics = score_window(window, crop_info)

        results.append({

            "Start": pd.to_datetime(window.iloc[0]["date"]).date(),

            "End": pd.to_datetime(window.iloc[-1]["date"]).date(),

            "Suitability score": metrics["score"],

            "Rainfall (mm)": metrics["rain_mm"],

            "Average temperature (°C)": metrics["avg_temp"],

            "Longest dry spell (days)": metrics["dry_spell"],

            "Rainy days": metrics["rainy_days"],

            "Risks": "; ".join(metrics["risks"]),

        })

    if not results:

        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("Suitability score", ascending=False).reset_index(drop=True)




# -------------------- Shared data and navigation --------------------
st.markdown("""
<div class="hero-panel">
  <div class="eyebrow">CLIMATE INTELLIGENCE PLATFORM · TAMIL NADU</div>
  <div class="hero-title">SowingSense AI <span style="color:#65e6a0">🌱</span></div>
  <div class="hero-copy">Make more informed sowing decisions with weather insights, crop-aware scoring and transparent experimental analytics.</div>
  <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:18px">
    <span style="display:inline-block;padding:7px 11px;border-radius:999px;background:rgba(8,25,31,.68);border:1px solid rgba(101,230,160,.30);color:#dfffee;font-size:.78rem">🌦 Weather-aware</span>
    <span style="display:inline-block;padding:7px 11px;border-radius:999px;background:rgba(8,25,31,.68);border:1px solid rgba(101,230,160,.30);color:#dfffee;font-size:.78rem">🌱 Crop-specific scoring</span>
    <span style="display:inline-block;padding:7px 11px;border-radius:999px;background:rgba(8,25,31,.68);border:1px solid rgba(101,230,160,.30);color:#dfffee;font-size:.78rem">📊 Transparent insights</span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding:8px 4px 18px">
      <div style="font-size:1.5rem;font-weight:850;color:#65e6a0">🌱 SowingSense</div>
      <div style="font-size:.78rem;color:#a9c8bd">SMARTER SOWING · HEALTHIER TOMORROW</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("WORKSPACE", [
        "Dashboard", "Sowing Recommendation", "Weather & Climate",
        "Crop Information", "Soil Analysis", "Climate Trends",
        "Insights & Alerts", "Export Data", "About"
    ], label_visibility="visible")
    st.markdown("---")
    st.markdown("#### Farm configuration")
    district = st.selectbox("District / location", list(DISTRICTS.keys()))
    crop = st.selectbox("Crop", list(CROPS.keys()))
    source = st.radio("Climate data", ["Live forecast (if available)", "Demo data (offline)"])
    start_day = st.date_input("Analysis start date", value=date.today())
    soil = st.selectbox("Soil type", ["Not specified", "Clay", "Loam", "Sandy", "Black soil"])
    run = st.button("🌱 Refresh recommendation", type="primary", use_container_width=True)
    st.caption("Prototype only · Verify advice with local agricultural experts.")

lat, lon = DISTRICTS[district]
config_key = (district, source, start_day)
if run or "weather_data" not in st.session_state or st.session_state.get("weather_config") != config_key:
    if source == "Live forecast (if available)":
        try:
            weather = fetch_open_meteo(lat, lon, start_day, start_day + timedelta(days=15))
            st.session_state["data_source_label"] = "Open-Meteo forecast"
        except Exception:
            weather = get_demo_weather(start_day)
            st.session_state["data_source_label"] = "Synthetic demo data (live forecast unavailable)"
    else:
        weather = get_demo_weather(start_day)
        st.session_state["data_source_label"] = "Synthetic demo data"
    st.session_state["weather_data"] = weather
    st.session_state["weather_config"] = config_key

weather = st.session_state["weather_data"].copy()
weather["date"] = pd.to_datetime(weather["date"])
weather = weather.sort_values("date").reset_index(drop=True)
result = evaluate_windows(weather, CROPS[crop])
weather_total_rain = float(weather["rain_mm"].sum())
avg_max_temp = float(weather["tmax_c"].mean())
avg_min_temp = float(weather["tmin_c"].mean())
rainy_days = int((weather["rain_mm"] >= 1).sum())
dry_days = int((weather["rain_mm"] < 1).sum())
data_label = st.session_state.get("data_source_label", "Demo data")
best = result.iloc[0] if not result.empty else None
second = result.iloc[1] if len(result) > 1 else best

# Shared current weather fetch: used on dashboard/weather pages only.
def render_current_weather():
    st.markdown("### Live conditions")
    try:
        current = fetch_current_weather(lat, lon)
        st.caption(f"Open-Meteo current data · Observation time: {current.get('time', 'Unavailable')}")
        cols = st.columns(4)
        cols[0].metric("Temperature", f"{current.get('temperature_2m', '—')} °C")
        cols[1].metric("Humidity", f"{current.get('relative_humidity_2m', '—')}%")
        cols[2].metric("Rainfall", f"{current.get('rain', '—')} mm")
        cols[3].metric("Wind speed", f"{current.get('wind_speed_10m', '—')} km/h")
    except Exception:
        st.warning("Live weather is currently unavailable. Current-condition cards are not filled with demo values.")

def render_weather_charts():
    chart = px.bar(weather, x="date", y="rain_mm", title="Daily rainfall forecast",
                   labels={"date":"Date","rain_mm":"Rainfall (mm)"})
    chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=15,r=15,t=55,b=15))
    chart.update_xaxes(showgrid=False)
    chart.update_yaxes(gridcolor="rgba(148,190,175,.14)", rangemode="tozero")
    st.plotly_chart(chart, use_container_width=True)
    temp = px.line(weather, x="date", y=["tmax_c","tmin_c"], markers=True,
                   title="Maximum and minimum temperature",
                   labels={"date":"Date","value":"Temperature (°C)","variable":"Series"})
    temp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=15,r=15,t=55,b=15))
    temp.update_xaxes(showgrid=False)
    temp.update_yaxes(gridcolor="rgba(148,190,175,.14)")
    st.plotly_chart(temp, use_container_width=True)

def render_recommendation():
    st.markdown("### Recommended sowing window")
    if result.empty:
        st.warning("Not enough daily weather data to evaluate a 7-day window.")
        return
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Suitability score", f"{int(best['Suitability score'])}/100")
    c2.metric("Recommended window", f"{best['Start'].strftime('%d %b')} – {best['End'].strftime('%d %b')}")
    c3.metric("7-day rainfall", f"{best['Rainfall (mm)']} mm")
    c4.metric("Average temperature", f"{best['Average temperature (°C)']} °C")
    st.success(f"Primary candidate: {best['Start'].strftime('%d %B')} – {best['End'].strftime('%d %B')} · Score {int(best['Suitability score'])}/100")
    if second is not None:
        st.info(f"Alternative candidate: {second['Start'].strftime('%d %B')} – {second['End'].strftime('%d %B')} · Score {int(second['Suitability score'])}/100")
    st.caption("This is the highest-ranked candidate in the available data, not a confirmed agronomic sowing date.")
    label, confidence = predict_sowing_suitability(float(best["Rainfall (mm)"]),
        float(best["Average temperature (°C)"]), int(best["Longest dry spell (days)"]))
    st.markdown("#### Experimental ML output")
    if label is None:
        st.warning("The demonstration model file was not found.")
    else:
        st.write(f"**Demo model output:** {label}")
        st.metric("Experimental model score (not validated)", f"{confidence:.1f}%")
    st.caption("The model was trained on synthetic demo data and is not validated for real-world agricultural decisions.")
    reasons=[]
    if float(best["Rainfall (mm)"]) >= CROPS[crop]["rain_min"]:
        reasons.append("Estimated 7-day rainfall meets the prototype's minimum threshold.")
    else:
        reasons.append("Rainfall is below the prototype's minimum threshold; moisture stress may be a concern.")
    if abs(float(best["Average temperature (°C)"]) - CROPS[crop]["temp_ideal"]) <= 3:
        reasons.append("Average temperature is near the crop's configured ideal temperature.")
    else:
        reasons.append("Average temperature differs from the crop's configured ideal temperature.")
    if int(best["Longest dry spell (days)"]) >= 5:
        reasons.append("A long dry spell is detected in this candidate window.")
    else:
        reasons.append("The longest dry spell is relatively short in this candidate window.")
    st.markdown("#### Why this window was ranked")
    for reason in reasons:
        st.write("• " + reason)
    st.markdown("#### Top candidate comparison")
    chart_df=result.head(10).copy()
    chart_df["Window"]=chart_df.apply(lambda x:f"{x['Start'].strftime('%d %b')}–{x['End'].strftime('%d %b')}",axis=1)
    fig=px.bar(chart_df,x="Window",y="Suitability score",hover_data=["Rainfall (mm)","Average temperature (°C)","Longest dry spell (days)"],range_y=[0,100])
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)

if page == "Dashboard":
    st.markdown(f"<div class='section-kicker'>Overview / {district}</div><h2>Farm dashboard</h2><p class='small-note'>A quick snapshot of conditions for <b>{crop}</b> · {data_label}</p>", unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric("Forecast rainfall",f"{weather_total_rain:.1f} mm")
    b.metric("Average max temp",f"{avg_max_temp:.1f} °C")
    c.metric("Rainy days",str(rainy_days))
    d.metric("Dry days",str(dry_days))
    render_current_weather()
    st.markdown("---")
    render_recommendation()
    st.markdown("### Weather outlook")
    render_weather_charts()
    with st.expander("How this prototype works / limitations"):
        st.write("It ranks rolling 7-day candidate windows using a transparent weighted score. Forecast data may fall back to clearly labelled synthetic demo data. Crop thresholds are simplified placeholders and require local agronomic validation. District-centre coordinates are used; soil measurements and historical yield records are not included in the score.")

elif page == "Sowing Recommendation":
    st.markdown("## Sowing recommendation")
    st.caption(f"Selected crop: {crop} · District: {district} · Soil: {soil}")
    render_recommendation()
    st.markdown("### Crop thresholds used by this prototype")
    crop_info=CROPS[crop]
    st.write(f"Minimum rainfall: **{crop_info['rain_min']} mm** · Ideal rainfall: **{crop_info['rain_ideal']} mm**")
    st.write(f"Minimum temperature: **{crop_info['temp_min']} °C** · Ideal temperature: **{crop_info['temp_ideal']} °C**")
    st.caption("These are illustrative thresholds, not validated local agricultural guidance.")

elif page == "Weather & Climate":
    st.markdown("## Weather & climate")
    st.caption(f"{district} · {data_label}")
    render_current_weather()
    w1,w2,w3,w4=st.columns(4)
    w1.metric("Total rainfall",f"{weather_total_rain:.1f} mm")
    w2.metric("Average maximum",f"{avg_max_temp:.1f} °C")
    w3.metric("Rainy days",str(rainy_days))
    w4.metric("Dry days",str(dry_days))
    render_weather_charts()
    st.markdown("### Daily weather records")
    display=weather.copy()
    display["date"]=display["date"].dt.strftime("%d %b %Y")
    st.dataframe(display,use_container_width=True,hide_index=True)

elif page == "Crop Information":
    st.markdown("## Crop information")
    st.caption("Prototype crop settings. Confirm variety-specific guidance with the local agriculture department.")
    rows=[]
    for name,info in CROPS.items():
        rows.append({"Crop":name,"Minimum rainfall (mm)":info["rain_min"],"Ideal rainfall (mm)":info["rain_ideal"],"Minimum temperature (°C)":info["temp_min"],"Ideal temperature (°C)":info["temp_ideal"],"Waterlogging sensitive":info["waterlogging_sensitive"]})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    chosen=CROPS[crop]
    st.markdown(f"### {crop}")
    x1,x2,x3=st.columns(3)
    x1.metric("Minimum rainfall",f"{chosen['rain_min']} mm")
    x2.metric("Ideal rainfall",f"{chosen['rain_ideal']} mm")
    x3.metric("Ideal temperature",f"{chosen['temp_ideal']} °C")
    if chosen["waterlogging_sensitive"]:
        st.warning("The prototype marks this crop as waterlogging-sensitive. Check field drainage and local guidance.")
    else:
        st.info("The prototype does not flag this crop as waterlogging-sensitive.")

elif page == "Soil Analysis":
    st.markdown("## Soil analysis")
    st.caption("Soil type is captured as farm context; it does not currently change the suitability score.")
    st.selectbox("Selected soil type",["Not specified","Clay","Loam","Sandy","Black soil"],index=["Not specified","Clay","Loam","Sandy","Black soil"].index(soil),disabled=True)
    notes={
        "Not specified":"Add a soil type or obtain a soil test for a more informed decision.",
        "Clay":"Clay soils can retain more water; check drainage and avoid field operations when soil is excessively wet.",
        "Loam":"Loam often balances water retention and drainage, but field conditions and soil tests still matter.",
        "Sandy":"Sandy soils usually drain quickly; monitor moisture and irrigation availability.",
        "Black soil":"Black soils can retain substantial moisture; check cracking, drainage and local crop guidance."
    }
    st.info(notes[soil])
    st.markdown("### What to record before sowing")
    st.write("- Soil pH and electrical conductivity from a soil test")
    st.write("- Available nitrogen, phosphorus and potassium")
    st.write("- Drainage, field moisture and irrigation access")
    st.warning("No laboratory soil measurements are collected by this prototype. These notes are general guidance, not a soil diagnosis.")

elif page == "Climate Trends":
    st.markdown("## Historical climate trends")
    monthly_path=DATA_DIR/"monthly_climate_summary.csv"
    if WEATHER_FEATURES_PATH.exists():
        climate_df=pd.read_csv(WEATHER_FEATURES_PATH,parse_dates=["date"])
        st.caption("Source label in project data: NASA POWER daily weather data · Thoothukudi, Tamil Nadu · 2024–2025")
        a,b,c=st.columns(3)
        a.metric("Weather records",f"{len(climate_df):,}")
        if "avg_temp_c" in climate_df: b.metric("Average temperature",f"{climate_df['avg_temp_c'].mean():.1f} °C")
        if "rain_7day_mm" in climate_df: c.metric("Average 7-day rainfall",f"{climate_df['rain_7day_mm'].mean():.1f} mm")
        if "date" in climate_df:
            climate_plot=climate_df.set_index("date")
            cols=[col for col in ["avg_temp_c","max_temp_7day_c","min_temp_7day_c"] if col in climate_plot.columns]
            if cols: st.line_chart(climate_plot[cols],use_container_width=True)
            if "rain_7day_mm" in climate_plot: st.area_chart(climate_plot[["rain_7day_mm"]],use_container_width=True)
        st.download_button("Download historical climate features",climate_df.to_csv(index=False),file_name="sowing_window_features.csv",mime="text/csv")
    else:
        st.warning("Historical weather features not found. Run build_features.py first.")
    if monthly_path.exists():
        monthly_df=pd.read_csv(monthly_path)
        st.markdown("### Monthly climate analysis")
        if "average_temperature_c" in monthly_df:
            st.bar_chart(monthly_df.set_index("month")[["average_temperature_c"]],use_container_width=True)
        if "total_rainfall_mm" in monthly_df:
            st.bar_chart(monthly_df.set_index("month")[["total_rainfall_mm"]],use_container_width=True)
        st.dataframe(monthly_df,use_container_width=True,hide_index=True)
    else:
        st.info("Monthly summary not found. Run analyze_weather.py to generate it.")

elif page == "Insights & Alerts":
    st.markdown("## Insights & alerts")
    alerts=[]
    if dry_days>=5: alerts.append(("Dry weather",f"{dry_days} days have less than 1 mm rainfall. Check consecutive dry spells before sowing."))
    if avg_max_temp>=35: alerts.append(("Heat risk",f"Average maximum temperature is {avg_max_temp:.1f} °C. Check crop heat tolerance."))
    if weather_total_rain>=150: alerts.append(("Heavy rainfall",f"Total rainfall is {weather_total_rain:.1f} mm. Check drainage and local forecasts."))
    if alerts:
        for title,msg in alerts: st.warning(f"**{title}:** {msg}")
    else:
        st.success("No major weather alert triggered by the prototype thresholds.")
    if best is not None:
        st.markdown("### Risk indicators for the top window")
        st.write(f"**Rainfall:** {best['Rainfall (mm)']} mm over 7 days")
        st.write(f"**Longest dry spell:** {int(best['Longest dry spell (days)'])} day(s)")
        st.write(f"**Rainy days:** {int(best['Rainy days'])}")
        st.write(f"**Flags:** {best['Risks']}")
    st.markdown("---")
    st.markdown("### Climate twin simulator")
    st.write("Test hypothetical changes in rainfall and temperature. This does not predict actual future climate.")
    x,y=st.columns(2)
    with x: rainfall_change=st.slider("Rainfall change (%)",-50,50,0,5)
    with y: temperature_change=st.slider("Temperature change (°C)",-5.0,5.0,0.0,0.5)
    simulated=weather.copy()
    simulated["rain_mm"]=(simulated["rain_mm"]*(1+rainfall_change/100)).clip(lower=0)
    simulated["tmax_c"]=simulated["tmax_c"]+temperature_change
    simulated["tmin_c"]=simulated["tmin_c"]+temperature_change
    simulated_result=evaluate_windows(simulated,CROPS[crop])
    sim_best=simulated_result.iloc[0] if not simulated_result.empty else None
    sim_score=int(sim_best["Suitability score"]) if sim_best is not None else 0
    orig_score=int(best["Suitability score"]) if best is not None else 0
    q1,q2,q3=st.columns(3)
    q1.metric("Original score",f"{orig_score}/100")
    q2.metric("Simulated score",f"{sim_score}/100")
    q3.metric("Score change",f"{sim_score-orig_score:+d}",delta=f"{sim_score-orig_score:+d} points")
    if sim_best is not None: st.info(f"Under this hypothetical scenario, the top-ranked window is {sim_best['Start']} to {sim_best['End']}.")
    st.caption("All simulator thresholds are illustrative and not validated for every crop or district.")

elif page == "Export Data":
    st.markdown("## Export data")
    st.caption("Download the current candidate-window results and weather data.")
    if not result.empty:
        st.markdown("### Candidate sowing windows")
        st.dataframe(result.head(20),use_container_width=True,hide_index=True)
        st.download_button("⬇️ Download recommendation report (CSV)",result.to_csv(index=False).encode("utf-8"),file_name="sowingsense_recommendations.csv",mime="text/csv",type="primary")
    st.markdown("### Weather data")
    st.download_button("⬇️ Download weather data (CSV)",weather.to_csv(index=False).encode("utf-8"),file_name="sowingsense_weather.csv",mime="text/csv")
    if WEATHER_FEATURES_PATH.exists():
        historical=pd.read_csv(WEATHER_FEATURES_PATH)
        st.download_button("⬇️ Download historical climate features",historical.to_csv(index=False).encode("utf-8"),file_name="sowing_window_features.csv",mime="text/csv")
    st.info(f"Active source: {data_label}")

elif page == "About":
    st.markdown("## About SowingSense AI")
    st.write("SowingSense AI is a climate-aware crop sowing recommendation prototype designed to explore how weather conditions may support sowing-window decisions.")
    st.markdown("### Current capabilities")
    st.write("- Open-Meteo current conditions and short-range forecast when available")
    st.write("- Clearly labelled synthetic fallback data for offline demonstrations")
    st.write("- Crop-specific, rule-based ranking of rolling seven-day weather windows")
    st.write("- Experimental ML model output, climate trends, risk alerts and CSV exports")
    st.markdown("### Limitations")
    st.warning("Scores are illustrative and are not a guarantee of yield or a substitute for local agricultural advice. Crop thresholds are simplified placeholders. The demo model was trained on synthetic data and requires validation with real agricultural data. Soil type is not currently used in the score.")
    st.caption("SowingSense AI · Smarter Sowing. Healthier Tomorrow.")
