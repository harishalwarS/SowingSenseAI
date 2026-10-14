
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

st.title("🌱 SowingSense AI")
st.markdown(
    "#### Climate intelligence for smarter sowing decisions"
)
st.caption(
    "Weather insights • Experimental ML • Crop-specific decision support"
)

st.subheader("Climate-aware crop sowing recommendation prototype")
st.caption("Decision-support demo only. Scores are illustrative and are not a guarantee of yield or a substitute for local agricultural advice.")


with st.sidebar:
    st.markdown("""
        <div style="padding: 12px 4px 24px;">
            <h2 style="margin-bottom: 0; color: #22c55e;">
                🌱 SowingSense AI
            </h2>
            <p style="font-size: 12px; color: #b7c9d8;">
                Smarter Sowing. Healthier Tomorrow.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")

    page = st.radio(
        "Go to",
        [
            "Dashboard",
            "Sowing Recommendation",
            "Weather & Climate",
            "Crop Information",
            "Soil Analysis",
            "Climate Trends",
            "Insights & Alerts",
            "Export Data",
            "About"
        ],
        label_visibility="collapsed"
    )
    st.session_state["selected_page"] = page

    st.markdown("---")
    st.markdown("### Farm Details")

    district = st.selectbox(
        "District / location",
        list(DISTRICTS.keys())
    )

    crop = st.selectbox(
        "Crop",
        list(CROPS.keys())
    )

    source = st.radio(
        "Climate data source",
        [
            "Live forecast (if available)",
            "Demo data (offline)"
        ]
    )

    start_day = st.date_input(
        "Start date for analysis",
        value=date.today()
    )

    soil = st.selectbox(
        "Soil type (optional)",
        ["Not specified", "Clay", "Loam", "Sandy", "Black soil"]
    )

    st.markdown("---")

    run = st.button(
        "🌱 Generate recommendation",
        type="primary",
        use_container_width=True
    )


lat, lon = DISTRICTS[district]
if run or "weather_data" not in st.session_state:
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

weather = st.session_state["weather_data"]

# --- Current Live Weather ---
st.markdown("---")
st.subheader(f"🌦️ Current Live Weather — {district}")

try:
    current = fetch_current_weather(lat, lon)

    st.caption(
        f"Open-Meteo current data | "
        f"Observation time: {current.get('time', 'Unavailable')}"
    )

    w1, w2, w3, w4 = st.columns(4)

    w1.metric(
        "Temperature",
        f"{current['temperature_2m']} °C"
    )
    w2.metric(
        "Humidity",
        f"{current['relative_humidity_2m']}%"
    )
    w3.metric(
        "Rainfall",
        f"{current['rain']} mm"
    )
    w4.metric(
        "Wind Speed",
        f"{current['wind_speed_10m']} km/h"
    )

except Exception as e:
    st.warning(
        "Live weather is currently unavailable. "
        "The current weather cards are not displaying demo values."
    )

result = evaluate_windows(weather, CROPS[crop])
st.info(f"Location: **{district}** · Crop: **{crop}** · Data: **{st.session_state.get('data_source_label', 'Demo data')}**")

if result.empty:
    st.warning("Not enough daily weather data to evaluate a 7-day window.")
else:
    best = result.iloc[0]
    second = result.iloc[1] if len(result) > 1 else best
   
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric(
        "🌱 Best Window Score",
        f"{int(best['Suitability score'])}/100",
        help="Prototype score based on rainfall, temperature and dry spells"
    )
    c2.metric(
        "📅 Recommended Window",
        f"{best['Start'].strftime('%d %b')} – {best['End'].strftime('%d %b')}"
    )
    c3.metric(
        "🌧️ 7-Day Rainfall",
        f"{best['Rainfall (mm)']} mm"
    )
    c4.metric(
        "🌡️ Average Temperature",
        f"{best['Average temperature (°C)']} °C"
    )


    st.markdown("### Recommendation")
    
    st.success(
        f"Primary candidate: {best['Start'].strftime('%d %B')} – "
        f"{best['End'].strftime('%d %B')} | "
        f"Score: {int(best['Suitability score'])}/100"
    )

    st.info(
        f"Alternative candidate: {second['Start'].strftime('%d %B')} – "
        f"{second['End'].strftime('%d %B')} | "
        f"Score: {int(second['Suitability score'])}/100"
    )

    st.caption("The best window is the highest-ranked candidate among the available data, not a confirmed agronomic sowing date.")
st.markdown("### Experimental ML prediction")

ml_label, ml_confidence = predict_sowing_suitability(
    rainfall_mm=float(best["Rainfall (mm)"]),
    temperature_c=float(best["Average temperature (°C)"]),
    dry_spell_days=int(best["Longest dry spell (days)"])
)

if ml_label is not None:
    st.info(f"**Demo model output:** {ml_label}")
    
    st.metric("Experimental model score (not validated)", f"{ml_confidence:.1f}%")
    st.caption(
        "This score reflects the experimental model's output, "
        "not real-world prediction accuracy. The model was trained "
        "on synthetic demo data and requires validation with real "
        "agricultural data before practical use."
    )

    st.caption(
        "Experimental output only. This model uses synthetic training data "
        "and is not validated for real-world agricultural decisions."
    )
else:
    st.warning("The demonstration model file was not found.")
    st.markdown("### Why this window was ranked")
    reasons = []
    if float(best["Rainfall (mm)"]) >= CROPS[crop]["rain_min"]:
        reasons.append("The estimated 7-day rainfall meets the prototype's minimum rainfall threshold.")
    else:
        reasons.append("Rainfall is below the prototype's minimum threshold, so moisture stress may be a concern.")
    if abs(float(best["Average temperature (°C)"]) - CROPS[crop]["temp_ideal"]) <= 3:
        reasons.append("Average temperature is near the crop's configured ideal temperature.")
    else:
        reasons.append("Average temperature differs from the crop's configured ideal temperature.")
    if int(best["Longest dry spell (days)"]) >= 5:
        reasons.append("A long dry spell is detected in this candidate window.")
    else:
        reasons.append("The longest dry spell is relatively short in this candidate window.")
    for r in reasons:
        st.write("• " + r)

    st.markdown("### Candidate window comparison")
    chart_df = result.head(10).copy()
    chart_df["Window"] = chart_df.apply(lambda x: f"{x['Start'].strftime('%d %b')}–{x['End'].strftime('%d %b')}", axis=1)
    fig = px.bar(chart_df, x="Window", y="Suitability score", hover_data=["Rainfall (mm)", "Average temperature (°C)", "Longest dry spell (days)"], range_y=[0, 100], title="Top candidate sowing windows")
    st.plotly_chart(fig, use_container_width=True)
st.markdown("### 🌦️ Live Weather Insights")

weather_insights = weather.copy()
weather_insights["date"] = pd.to_datetime(weather_insights["date"])
weather_insights = weather_insights.sort_values("date")

total_rain = float(weather_insights["rain_mm"].sum())
avg_max_temp = float(weather_insights["tmax_c"].mean())
avg_min_temp = float(weather_insights["tmin_c"].mean())
rainy_days = int((weather_insights["rain_mm"] >= 1).sum())
dry_days = int((weather_insights["rain_mm"] < 1).sum())

w1, w2, w3, w4 = st.columns(4)

w1.metric("Forecast rainfall", f"{total_rain:.1f} mm")
w2.metric("Average max temperature", f"{avg_max_temp:.1f} °C")
w3.metric("Rainy days", str(rainy_days))
w4.metric("Dry days", str(dry_days))

if st.session_state.get("data_source_label") == "Open-Meteo forecast":
    st.success("Live forecast data loaded successfully.")
else:
    st.warning(
        "Demo weather data is being displayed. "
        "Live forecast was unavailable or demo mode was selected."
    )


st.markdown("#### 🌧️ Daily Rainfall Forecast")

rain_chart = px.bar(
    weather_insights,
    x="date",
    y="rain_mm",
    title="Daily rainfall (mm)",
    labels={
        "date": "Date",
        "rain_mm": "Rainfall (mm)"
    },
    hover_data={"rain_mm": ":.1f"}
)

rain_chart.update_layout(
    template="plotly_white",
    title_x=0.02,
    xaxis_title="Forecast date",
    yaxis_title="Rainfall (mm)",
    margin=dict(l=20, r=20, t=60, b=20)
)

rain_chart.update_xaxes(showgrid=False)
rain_chart.update_yaxes(
    rangemode="tozero",
    gridcolor="#e5eee3"
)

st.plotly_chart(rain_chart, use_container_width=True)



st.markdown("#### 🌡️ Temperature Trend")

temp_chart = px.line(
    weather_insights,
    x="date",
    y=["tmax_c", "tmin_c"],
    markers=True,
    title="Daily maximum and minimum temperature",
    labels={
        "date": "Date",
        "value": "Temperature (°C)",
        "variable": "Temperature"
    }
)

temp_chart.update_layout(
    template="plotly_white",
    title_x=0.02,
    xaxis_title="Forecast date",
    yaxis_title="Temperature (°C)",
    legend_title="Temperature type",
    margin=dict(l=20, r=20, t=60, b=20)
)

temp_chart.update_xaxes(showgrid=False)
temp_chart.update_yaxes(gridcolor="#e5eee3")

st.plotly_chart(temp_chart, use_container_width=True)

st.markdown("### ⚠️ Weather Risk Alerts")

if dry_days >= 5:
    st.warning(
        f"🌵 **Dry weather alert:** {dry_days} days have less than "
        "1 mm rainfall. Check consecutive dry spells before sowing."
    )

if avg_max_temp >= 35:
    st.warning(
        f"🌡️ **Heat alert:** Average maximum temperature is "
        f"{avg_max_temp:.1f}°C. Check crop heat tolerance."
    )

if total_rain >= 150:
    st.warning(
        f"🌧️ **Heavy rainfall alert:** Total rainfall is "
        f"{total_rain:.1f} mm. Check drainage and local forecasts."
    )

if dry_days < 5 and avg_max_temp < 35 and total_rain < 150:
    st.success(
        "✅ No major weather alert triggered by the prototype thresholds."
    )


if total_rain >= 150:
    st.warning(
        "Substantial accumulated rainfall detected. "
        "Check drainage and local heavy-rain forecasts."
    )

if dry_days < 5 and avg_max_temp < 35 and total_rain < 150:
    st.markdown("### Weather data")
    weather_display = weather.copy()
    weather_display["date"] = pd.to_datetime(weather_display["date"]).dt.strftime("%d %b %Y")
    st.line_chart(weather.set_index(pd.to_datetime(weather["date"]))[["rain_mm", "tmax_c", "tmin_c"]])
    st.dataframe(weather_display, use_container_width=True, hide_index=True)

    st.markdown("### Risk indicators for the top window")
    st.write(f"- **Rainfall:** {best['Rainfall (mm)']} mm over 7 days")
    st.write(f"- **Longest dry spell:** {int(best['Longest dry spell (days)'])} day(s)")
    st.write(f"- **Rainy days:** {int(best['Rainy days'])}")
    st.write(f"- **Flags:** {best['Risks']}")

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("Download candidate-window report (CSV)", csv, file_name="sowingsense_recommendations.csv", mime="text/csv")

with st.expander("How this prototype works / limitations"):
    st.write("""
    - It evaluates rolling 7-day candidate windows and ranks them with a transparent weighted score.
    - The current scoring logic is a baseline rule-based model, not a trained machine-learning model.
    - Live mode attempts to retrieve a short-range forecast from Open-Meteo. If unavailable, the app falls back to synthetic demo data and labels it.
    - Crop thresholds are simplified placeholders for demonstration. Replace them with locally validated agronomic guidance before real-world use.
    - The current prototype uses district-centre coordinates and does not incorporate soil measurements or historical yield records into the score.
    """)
    
# ---------------- Climate Twin Simulator ----------------

st.markdown("---")
st.markdown("## 🌦️ Climate Twin Simulator")
st.write(
    "Test how changing rainfall and temperature could affect "
    "a crop's sowing suitability. This is a hypothetical simulation."
)

col1, col2 = st.columns(2)

with col1:
    rainfall_change = st.slider(
        "🌧️ Rainfall change (%)",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        help="Simulate rainfall decreasing or increasing."
    )

with col2:
    temperature_change = st.slider(
        "🌡️ Temperature change (°C)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        help="Simulate a temperature increase or decrease."
    )

simulated_rain = max(0, total_rain * (1 + rainfall_change / 100))
simulated_temp = avg_max_temp + temperature_change

# Recalculate suitability for the selected 7-day window
simulated_weather = weather.copy()

simulated_weather["rain_mm"] = (
    simulated_weather["rain_mm"] * (1 + rainfall_change / 100)
).clip(lower=0)

simulated_weather["tmax_c"] = (
    simulated_weather["tmax_c"] + temperature_change
)

simulated_weather["tmin_c"] = (
    simulated_weather["tmin_c"] + temperature_change
)


# Recalculate and rank all candidate windows under simulated climate
simulated_result = evaluate_windows(
    simulated_weather,
    CROPS[crop]
)

original_top3 = result.head(3).copy()
simulated_top3 = simulated_result.head(3).copy()

original_score = int(best["Suitability score"])

if not simulated_result.empty:
    simulated_best = simulated_result.iloc[0]
    simulated_score = int(simulated_best["Suitability score"])
    score_difference = simulated_score - original_score
else:
    simulated_best = None
    simulated_score = 0
    score_difference = 0


st.markdown("### Simulation Results")

st.markdown("### 📅 Sowing Window Comparison")

if not original_top3.empty:
    st.markdown("**Original climate — Top 3 windows**")
    st.dataframe(
        original_top3[
            ["Start", "End", "Suitability score", "Risks"]
        ],
        use_container_width=True,
        hide_index=True
    )

if not simulated_top3.empty:
    st.markdown("**Simulated climate — Top 3 windows**")
    st.dataframe(
        simulated_top3[
            ["Start", "End", "Suitability score", "Risks"]
        ],
        use_container_width=True,
        hide_index=True
    )

if simulated_best is not None:
    st.info(
        f"Under this hypothetical scenario, the top-ranked "
        f"window is {simulated_best['Start']} to "
        f"{simulated_best['End']}."
    )

st.markdown("### 🎯 Suitability Impact")

s1, s2, s3 = st.columns(3)

s1.metric("Original score", f"{original_score}/100")
s2.metric("Simulated score", f"{simulated_score}/100")
s3.metric(
    "Score change",
    f"{score_difference:+d}",
    delta=f"{score_difference:+d} points",
    delta_color="normal"
)

m1, m2 = st.columns(2)

with m1:
    st.metric(
        "Simulated 7-day rainfall",
        f"{simulated_rain:.1f} mm",
        delta=f"{rainfall_change:+d}%"
    )

with m2:
    st.metric(
        "Simulated maximum temperature",
        f"{simulated_temp:.1f} °C",
        delta=f"{temperature_change:+.1f} °C"
    )

# Illustrative indicators, not an agronomically validated model
if simulated_rain < 10:
    st.warning(
        "⚠️ Very low simulated rainfall. "
        "Check local soil moisture and irrigation availability."
    )
elif simulated_rain >= 150:
    st.warning(
        "🌧️ High simulated rainfall. "
        "Check drainage and local waterlogging risk."
    )
else:
    st.info(
        "🌱 Simulated rainfall is within this prototype's "
        "illustrative range."
    )

if simulated_temp >= 35:
    st.warning(
        "🔥 High simulated temperature. "
        "Check the selected crop's heat tolerance."
    )

st.caption(
    "Important: This simulator changes weather indicators only. "
    "It does not recalculate the original sowing-window ranking. "
    "Thresholds are illustrative and are not validated for every crop or district."
)

# --- Historical Weather Data Explorer ---
st.markdown("---")
st.subheader("🌦️ Historical Climate Data Explorer")

if WEATHER_FEATURES_PATH.exists():
    climate_df = pd.read_csv(WEATHER_FEATURES_PATH, parse_dates=["date"])

    st.caption(
        "Source: NASA POWER daily weather data | "
        "Thoothukudi, Tamil Nadu | 2024–2025"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Weather Records",
            f"{len(climate_df):,}"
        )

    with col2:
        st.metric(
            "Average Temperature",
            f"{climate_df['avg_temp_c'].mean():.1f} °C"
        )

    with col3:
        st.metric(
            "Average 7-Day Rainfall",
            f"{climate_df['rain_7day_mm'].mean():.1f} mm"
        )

    chart_df = climate_df.set_index("date")

    st.markdown("#### Temperature Trends")
    st.line_chart(
        chart_df[["avg_temp_c", "max_temp_7day_c", "min_temp_7day_c"]]
    )

    st.markdown("#### Seven-Day Rainfall")
    st.area_chart(chart_df[["rain_7day_mm"]])

    st.download_button(
        "Download Historical Climate Features",
        data=climate_df.to_csv(index=False),
        file_name="sowing_window_features.csv",
        mime="text/csv"
    )

else:
    st.warning(
        "Historical weather features not found. "
        "Run build_features.py first."
    )
    
# --- Monthly Climate Analysis ---
st.markdown("---")
st.subheader("📅 Monthly Climate Analysis")

MONTHLY_SUMMARY_PATH = DATA_DIR / "monthly_climate_summary.csv"

if MONTHLY_SUMMARY_PATH.exists():
    monthly_df = pd.read_csv(MONTHLY_SUMMARY_PATH)

    st.caption(
        "Monthly rainfall and temperature summary "
        "from historical weather data."
    )

       
    st.markdown("#### Monthly Average Temperature")
    st.bar_chart(
        monthly_df.set_index("month")[["average_temperature_c"]]
    )

    st.markdown("#### Monthly Total Rainfall")
    st.bar_chart(
        monthly_df.set_index("month")[["total_rainfall_mm"]]
    )


    st.dataframe(
        monthly_df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning(
        "Monthly climate summary not found. "
        "Run analyze_weather.py first."
    )


