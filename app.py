import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).parent / "Data"
WEATHER_FEATURES_PATH = DATA_DIR / "sowing_window_features.csv"
MONTHLY_PATH = DATA_DIR / "monthly_climate_summary.csv"
MODEL_PATH = Path(__file__).parent / "sowing_model_demo.joblib"

st.set_page_config(page_title="SowingSense AI", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")

# =====================================================================
# THEME / CSS (100% ORIGINAL DESIGN PRESERVED + SINGLE-LINE TOP NAV)
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Noto+Sans+Tamil:wght@400;500;600;700;800;900&display=swap');
:root{
  --g:#65e6a0; --g2:#22c55e; --blue:#4cc9f0; --amber:#f5c04a; --red:#ff7b72;
  --line:rgba(101,230,160,.20); --muted:#9bb8b0; color-scheme:dark;
}
html{scroll-behavior:smooth;}
.stApp{
  background:
    radial-gradient(ellipse at 88% -8%, rgba(34,197,94,.20), transparent 42%),
    radial-gradient(ellipse at 0% 60%, rgba(76,201,240,.07), transparent 40%),
    linear-gradient(135deg,#050f18 0%,#09202a 55%,#0a2822 100%);
  color:#e8f5ee;
}
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4{font-family:'Inter','Noto Sans Tamil',system-ui,sans-serif;}
.main .block-container{max-width:1500px;padding:1rem 2rem 4rem;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}
h1,h2,h3,h4{color:#f0fff7 !important;letter-spacing:-.4px;}
p,label,.stMarkdown{color:#d4e6df;}
hr{border-color:var(--line);margin:1.4rem 0;}
::-webkit-scrollbar{width:10px;} ::-webkit-scrollbar-track{background:#06131c;}
::-webkit-scrollbar-thumb{background:linear-gradient(#1c8f58,#65e6a0);border-radius:10px;}
iframe[height="0"]{display:none !important;height:0 !important;width:0 !important;border:none !important;}
div:has(> iframe[height="0"]){display:none !important;margin:0 !important;padding:0 !important;}

/* ---------- Brand ---------- */
.brand{display:flex;align-items:center;gap:10px;padding:4px 2px;}
.brand .logo{font-size:1.85rem;filter:drop-shadow(0 0 10px rgba(101,230,160,.8));}
.brand .name{font-size:1.3rem;font-weight:900;color:#fff;letter-spacing:-.3px;white-space:nowrap;}
.brand .name b{color:var(--g);text-shadow:0 0 14px rgba(101,230,160,.7);}
.brand .tag{font-size:.66rem;color:#a9c8bd;letter-spacing:.3px;white-space:nowrap;}

/* ---------- Hero ---------- */
.hero{position:relative;overflow:hidden;border-radius:26px;padding:clamp(22px,3vw,38px) clamp(20px,3vw,36px);
  margin-bottom:18px;border:1px solid rgba(101,230,160,.4);
  background:
    linear-gradient(100deg,rgba(3,14,24,.96) 0%,rgba(4,28,32,.80) 48%,rgba(5,48,30,.25) 100%),
    url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=80") center 45%/cover no-repeat,
    linear-gradient(135deg,#0b2924,#06131f);
  box-shadow:0 0 40px rgba(34,197,94,.16), 0 20px 50px rgba(0,0,0,.35);
  animation:rise .55s ease both;isolation:isolate;}
.hero::after{content:"";position:absolute;inset:0;z-index:-1;pointer-events:none;
  background:linear-gradient(115deg,transparent 25%,rgba(255,255,255,.07) 48%,transparent 70%);
  background-size:230% 100%;animation:shimmer 8s ease-in-out infinite;}
.hero-top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap;}
.hero .title{font-size:clamp(2rem,4vw,3.3rem);font-weight:900;line-height:1.05;color:#fff;display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
.hero .grad{background:linear-gradient(90deg,#fff,#9dffc9,#65e6a0,#fff);background-size:250% 100%;
  -webkit-background-clip:text;background-clip:text;color:transparent;animation:flow 6s linear infinite;
  filter:drop-shadow(0 0 16px rgba(101,230,160,.45));}
.ai-badge{font-size:1.2rem;font-weight:900;color:#7dffb5;padding:4px 14px;border-radius:12px;
  background:rgba(22,110,66,.55);border:1px solid #65e6a0;animation:glow 2.4s ease-in-out infinite;}
.hero .sub{color:#e2f5ec;font-size:1.05rem;font-weight:600;margin:.55rem 0 .2rem;max-width:560px;line-height:1.5;}
.hero .tags{color:#a9d4c0;font-size:.85rem;letter-spacing:.3px;}
.hero-right{display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
.pill{display:inline-flex;align-items:center;gap:8px;padding:9px 15px;border-radius:12px;font-size:.84rem;font-weight:600;
  color:#e8fff3;background:rgba(6,22,28,.72);border:1px solid rgba(101,230,160,.30);backdrop-filter:blur(8px);}
.pill.live{color:#8dffbc;border-color:rgba(34,197,94,.6);}
.pill.demo{color:#ffe2a3;border-color:rgba(245,192,74,.6);}
.dot{width:9px;height:9px;border-radius:50%;background:#22c55e;animation:pulse 1.6s infinite;}
.demo .dot{background:#f5c04a;}
.feat-row{display:flex;flex-wrap:wrap;gap:12px;margin-top:20px;}
.feat{display:flex;align-items:center;gap:10px;padding:10px 16px;border-radius:14px;cursor:default;
  background:rgba(8,28,30,.72);border:1px solid rgba(101,230,160,.30);backdrop-filter:blur(8px);transition:all .25s ease;}
.feat:hover{transform:translateY(-5px) scale(1.03);border-color:#65e6a0;box-shadow:0 0 24px rgba(101,230,160,.42);}
.feat .ico{font-size:1.4rem;}
.feat .t{font-size:.84rem;font-weight:700;color:#f0fff7;line-height:1.2;} .feat .s{font-size:.68rem;color:#98bdb0;}
.jump{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 14px;}
.jump a{text-decoration:none !important;color:#cdeedd !important;font-size:.8rem;font-weight:700;padding:7px 14px;border-radius:999px;
  background:rgba(101,230,160,.07);border:1px solid rgba(101,230,160,.25);transition:all .2s ease;}
.jump a:hover{background:rgba(101,230,160,.2);box-shadow:0 0 18px rgba(101,230,160,.4);transform:translateY(-2px);color:#fff !important;}

/* ---------- Glow cards (st.container(border=True)) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:linear-gradient(145deg,rgba(14,40,50,.80),rgba(7,25,33,.82));
  border:1px solid var(--line) !important;border-radius:20px !important;
  transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease;animation:rise .55s ease both;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{
  border-color:rgba(101,230,160,.75) !important;transform:translateY(-3px);
  box-shadow:0 0 28px rgba(101,230,160,.24), inset 0 0 26px rgba(101,230,160,.05);}
.ctitle{display:flex;align-items:center;gap:12px;margin-bottom:10px;}
.ctitle .ci{width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;
  background:rgba(34,197,94,.15);border:1px solid rgba(101,230,160,.4);box-shadow:0 0 16px rgba(34,197,94,.25);}
.ctitle .ct{font-size:1.12rem;font-weight:800;color:#f2fff8;} .ctitle .cs{font-size:.78rem;color:var(--muted);}
[data-testid="stPlotlyChart"]{background:transparent;border:none;padding:0;}

/* ---------- Weather card ---------- */
.wx{display:flex;gap:20px;align-items:center;flex-wrap:wrap;}
.wx-main{display:flex;align-items:center;gap:16px;min-width:230px;}
.wx-emoji{font-size:3.6rem;filter:drop-shadow(0 0 18px rgba(255,214,102,.5));animation:float 4s ease-in-out infinite;}
.wx-temp{font-size:2.9rem;font-weight:900;color:#fff;line-height:1;text-shadow:0 0 22px rgba(101,230,160,.35);}
.wx-desc{font-weight:700;color:#dff5ea;margin-top:4px;} .wx-feel{color:var(--muted);font-size:.85rem;}
.wx-grid{display:grid;grid-template-columns:repeat(4,minmax(86px,1fr));gap:10px;flex:1;min-width:300px;}
.tile{text-align:center;padding:14px 8px;border-radius:14px;background:rgba(255,255,255,.035);border:1px solid rgba(101,230,160,.15);transition:all .22s ease;}
.tile:hover{border-color:#65e6a0;box-shadow:0 0 18px rgba(101,230,160,.3);transform:translateY(-3px);}
.tile .tv{font-size:1.25rem;font-weight:800;color:#fff;} .tile .tl{font-size:.72rem;color:var(--muted);margin-top:2px;} .tile .ti{font-size:1.2rem;}

/* ---------- KPI ---------- */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:.4rem 0 1rem;}
.kpi{background:linear-gradient(145deg,rgba(19,48,59,.92),rgba(10,31,39,.92));border:1px solid var(--line);border-radius:18px;padding:16px 18px;
  transition:all .22s ease;animation:rise .5s ease both;}
.kpi:hover{transform:translateY(-4px);border-color:#65e6a0;box-shadow:0 0 26px rgba(101,230,160,.28);}
.kpi .kl{color:#a8c9bd;font-size:.72rem;font-weight:800;letter-spacing:.9px;text-transform:uppercase;}
.kpi .kv{color:#effff5;font-size:clamp(1.25rem,2vw,1.7rem);font-weight:850;margin-top:6px;white-space:nowrap;}
.kpi .ks{color:var(--muted);font-size:.78rem;margin-top:3px;}
.kpi.accent{border-color:rgba(101,230,160,.6);background:linear-gradient(145deg,rgba(22,92,60,.88),rgba(9,40,36,.95));box-shadow:0 0 24px rgba(34,197,94,.22);}

/* ---------- Alerts ---------- */
.alert-card{display:flex;gap:12px;align-items:center;padding:11px 14px;border-radius:14px;margin-bottom:9px;border:1px solid;border-left-width:4px;
  background:rgba(255,255,255,.03);transition:all .2s ease;}
.alert-card:hover{transform:translateX(5px);}
.alert-card .ai{font-size:1.35rem;} .alert-card .at{font-weight:700;font-size:.92rem;} .alert-card .ad{font-size:.8rem;color:#c3dcd1;}
.alert-card.ok{border-color:rgba(34,197,94,.5);} .alert-card.ok .at{color:#8dffbc;} .alert-card.ok:hover{box-shadow:0 0 18px rgba(34,197,94,.3);}
.alert-card.warn{border-color:rgba(245,192,74,.5);} .alert-card.warn .at{color:#ffd978;} .alert-card.warn:hover{box-shadow:0 0 18px rgba(245,192,74,.3);}
.alert-card.bad{border-color:rgba(255,123,114,.55);} .alert-card.bad .at{color:#ffa9a2;} .alert-card.bad:hover{box-shadow:0 0 18px rgba(255,123,114,.3);}
.alert-card.info{border-color:rgba(76,201,240,.5);} .alert-card.info .at{color:#9fe3fa;}

/* ---------- Suitability ---------- */
.bar-row{margin:10px 0;} .bar-top{display:flex;justify-content:space-between;font-size:.84rem;font-weight:700;color:#dff5ea;}
.bar-bg{height:8px;border-radius:99px;background:rgba(255,255,255,.07);margin-top:5px;overflow:hidden;}
.bar-fg{height:100%;border-radius:99px;background:linear-gradient(90deg,#16834d,#65e6a0);box-shadow:0 0 12px rgba(101,230,160,.6);animation:grow 1.1s ease both;}
.reco{margin-top:12px;padding:14px 16px;border-radius:16px;color:#042214;font-weight:700;
  background:linear-gradient(135deg,#9dffc9,#65e6a0);box-shadow:0 0 28px rgba(101,230,160,.45);animation:glow 2.8s ease-in-out infinite;}
.reco .rt{font-size:.78rem;opacity:.8;} .reco .rd{font-size:1.15rem;font-weight:900;} .reco .rs{font-size:.74rem;opacity:.75;}

/* ---------- Chips ---------- */
.chip-row{display:flex;flex-wrap:wrap;gap:8px;margin:.3rem 0 .8rem;}
.chip{padding:6px 12px;border-radius:999px;font-size:.82rem;font-weight:600;border:1px solid;}
.chip.ok{color:#bff5d6;background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.4);}
.chip.warn{color:#ffe2a3;background:rgba(245,192,74,.12);border-color:rgba(245,192,74,.4);}

/* ---------- Inputs / buttons ---------- */
[data-baseweb="select"] > div,[data-testid="stDateInput"] > div > div{background:rgba(10,32,40,.9) !important;border-radius:12px !important;
  border:1px solid rgba(101,230,160,.22) !important;transition:all .2s ease;}
[data-baseweb="select"] > div:hover,[data-testid="stDateInput"] > div > div:hover{border-color:rgba(101,230,160,.6) !important;box-shadow:0 0 14px rgba(101,230,160,.2);}
[data-baseweb="select"] > div:focus-within,[data-testid="stDateInput"] > div > div:focus-within{border-color:#65e6a0 !important;box-shadow:0 0 0 2px rgba(101,230,160,.35),0 0 20px rgba(101,230,160,.3);}
.stButton > button,.stDownloadButton > button{border-radius:12px;min-height:46px;font-weight:800;transition:all .2s ease;
  background:rgba(12,36,44,.9);border:1px solid rgba(101,230,160,.35);color:#e8fff3;}
.stButton > button:hover,.stDownloadButton > button:hover{transform:translateY(-3px);border-color:#65e6a0;box-shadow:0 0 24px rgba(101,230,160,.45);color:#fff;}
.stButton > button:active{transform:scale(.97);}
.stButton > button[kind="primary"]{background:linear-gradient(135deg,#15803d,#22c55e);border:1px solid #7dffb5;color:#fff;animation:btnglow 2.2s ease-in-out infinite;}
.stButton > button[kind="primary"]:hover{background:linear-gradient(135deg,#1a9a4c,#34e077);}
.stButton > button:focus-visible{outline:2px solid #65e6a0 !important;outline-offset:3px;}

/* main-area radios become glowing segmented pills */
.main [data-testid="stRadio"] [role="radiogroup"]{flex-direction:row;flex-wrap:wrap;gap:6px;}
.main [data-testid="stRadio"] label{border-radius:10px;padding:6px 14px;background:rgba(255,255,255,.04);border:1px solid rgba(101,230,160,.2);transition:all .2s ease;}
.main [data-testid="stRadio"] label > div:first-child{display:none;}
.main [data-testid="stRadio"] label:hover{border-color:#65e6a0;box-shadow:0 0 14px rgba(101,230,160,.3);}
.main [data-testid="stRadio"] label:has(input:checked){background:linear-gradient(135deg,#15803d,#22c55e);border-color:#7dffb5;box-shadow:0 0 18px rgba(34,197,94,.5);}
.main [data-testid="stRadio"] label:has(input:checked) p{color:#fff !important;font-weight:800;}
.main [data-testid="stRadio"] > label{display:none;}

[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(19,48,59,.92),rgba(10,31,39,.92));border:1px solid var(--line);border-radius:18px;padding:16px 18px;transition:all .2s ease;}
[data-testid="stMetric"]:hover{border-color:#65e6a0;box-shadow:0 0 22px rgba(101,230,160,.25);}
[data-testid="stMetricValue"]{color:#effff5 !important;font-weight:800;} [data-testid="stMetricLabel"]{color:#a8c9bd !important;}
[data-testid="stAlert"]{border-radius:14px;} [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden;}
[data-testid="stExpander"]{background:rgba(10,31,39,.5);border:1px solid var(--line);border-radius:14px;}
.stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;padding:8px 18px;color:var(--muted);font-weight:700;transition:all .2s ease;}
.stTabs [data-baseweb="tab"]:hover{color:#fff;background:rgba(101,230,160,.08);}
.stTabs [aria-selected="true"]{color:#65e6a0 !important;background:rgba(101,230,160,.10);text-shadow:0 0 12px rgba(101,230,160,.6);}
.stTabs [data-baseweb="tab-highlight"]{background:#65e6a0;box-shadow:0 0 12px #65e6a0;}
.kicker{color:var(--g);font-size:.74rem;text-transform:uppercase;letter-spacing:1.6px;font-weight:800;}
.small-note{color:var(--muted);font-size:.86rem;}

/* ---------- Fixed, aligned top navigation ---------- */
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none !important;}
.st-key-topnav{position:sticky;top:.6rem;z-index:999;background:rgba(6,19,28,.94);backdrop-filter:blur(16px);
  border:1px solid rgba(101,230,160,.30);border-radius:18px;padding:10px 14px;margin:0 0 18px;
  box-shadow:0 0 28px rgba(34,197,94,.13),0 10px 28px rgba(0,0,0,.30);}
.st-key-topnav [data-testid="stHorizontalBlock"]{align-items:center !important;gap:10px !important;}
.st-key-topnav [data-testid="column"]{min-width:0 !important;display:flex;justify-content:center;}
.st-key-topnav [data-testid="column"]:first-child{justify-content:flex-start;}
.st-key-topnav .brand{padding:0;min-width:0;}
.st-key-topnav .brand .name{font-size:clamp(.95rem,1.1vw,1.25rem);}
.st-key-topnav .brand .tag{font-size:.62rem;}
.st-key-topnav [data-testid="stRadio"]{width:100%;min-width:0;}
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"]{display:flex !important;align-items:center !important;justify-content:center !important;gap:5px !important;flex-wrap:nowrap !important;width:100%;}
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] > label{display:flex !important;align-items:center !important;justify-content:center !important;
  flex:0 1 auto !important;min-width:0 !important;margin:0 !important;padding:8px 9px !important;border-radius:10px !important;
  white-space:nowrap !important;background:transparent;border:1px solid transparent;}
/* Hide Streamlit radio circles without hiding the option text */
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] > label > div:first-child,
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] > label input,
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] > label svg{display:none !important;width:0 !important;margin:0 !important;}
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] > label p{font-size:clamp(.70rem,.78vw,.82rem) !important;line-height:1.2 !important;white-space:nowrap !important;font-weight:750 !important;margin:0 !important;}
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] > label:hover{border-color:rgba(101,230,160,.45);background:rgba(101,230,160,.08);}
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked){background:linear-gradient(135deg,#15803d,#22c55e) !important;border-color:#7dffb5 !important;box-shadow:0 0 14px rgba(34,197,94,.32);}
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p{color:#fff !important;}
.st-key-topnav [data-testid="stRadio"] > label,.st-key-topnav [data-testid="stSelectbox"] > label{display:none !important;}
.st-key-topnav [data-testid="stSelectbox"]{width:100%;min-width:0;}
.st-key-topnav [data-baseweb="select"] > div{min-height:40px !important;background:rgba(255,255,255,.045) !important;
  border-radius:11px !important;border:1px solid rgba(101,230,160,.28) !important;font-size:.78rem !important;font-weight:700 !important;}
.st-key-topnav [data-testid="stSelectbox"] [data-testid="stMarkdownContainer"] p{font-size:.78rem !important;}
.foot{text-align:center;color:#7fa196;font-size:.78rem;margin-top:2.5rem;padding-top:1rem;border-top:1px solid rgba(101,230,160,.14);}
@media (max-width:1450px){
  .st-key-topnav{position:relative;top:0;padding:10px 12px;}
  .st-key-topnav [data-testid="stHorizontalBlock"]{flex-wrap:wrap !important;}
  .st-key-topnav [data-testid="column"]{flex:1 1 auto !important;}
  .st-key-topnav [data-testid="column"]:first-child{flex:1 1 220px !important;}
  .st-key-topnav [data-testid="column"]:nth-child(2){flex:3 1 500px !important;}
  .st-key-topnav [data-testid="column"]:nth-child(3){flex:0 1 170px !important;}
  .st-key-topnav [data-testid="column"]:nth-child(4){flex:0 1 145px !important;}
  .st-key-topnav [data-testid="stRadio"] [role="radiogroup"]{flex-wrap:wrap !important;}
}
@media (max-width:768px){
  .st-key-topnav [data-testid="column"]:nth-child(2){flex-basis:100% !important;}
  .st-key-topnav [data-testid="column"]:nth-child(3),.st-key-topnav [data-testid="column"]:nth-child(4){flex:1 1 130px !important;}
}

@keyframes rise{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:none;}}
@keyframes grow{from{width:0;}}
@keyframes float{0%,100%{transform:translateY(0);}50%{transform:translateY(-7px);}}
@keyframes flow{to{background-position:250% 0;}}
@keyframes shimmer{0%,100%{background-position:150% 0;}50%{background-position:-30% 0;}}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(34,197,94,.7);}70%{box-shadow:0 0 0 10px rgba(34,197,94,0);}100%{box-shadow:0 0 0 0 rgba(34,197,94,0);}}
@keyframes glow{0%,100%{box-shadow:0 0 14px rgba(101,230,160,.35);}50%{box-shadow:0 0 30px rgba(101,230,160,.7);}}
@keyframes btnglow{0%,100%{box-shadow:0 0 12px rgba(34,197,94,.45);}50%{box-shadow:0 0 28px rgba(34,197,94,.9);}}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none !important;transition:none !important;scroll-behavior:auto !important;}}
@media (max-width:768px){.main .block-container{padding:.8rem .8rem 3rem;} .kpi .kv{white-space:normal;} .hero{border-radius:18px;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

GREEN, AMBER, RED = "#65e6a0", "#f5c04a", "#ff7b72"
PALETTE = ["#65e6a0", "#4cc9f0", "#f5c04a", "#c084fc", "#ff7b72"]

# =====================================================================
# LANGUAGE, ALL 38 DISTRICTS (EXACT CITY CENTERS), 20 CROPS & GPS
# =====================================================================
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"


def is_ta():
    return st.session_state.get("lang", "en") == "ta"


def L(en_text, ta_text):
    return ta_text if is_ta() else en_text


LIVE_LOC_KEY = "📍 Live Location (GPS)"

DISTRICTS = {
    LIVE_LOC_KEY: (8.8053, 78.1453),
    "Ariyalur": (11.1401, 79.0786), "Chengalpattu": (12.6819, 79.9888), "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558), "Cuddalore": (11.7480, 79.7714), "Dharmapuri": (12.1211, 78.1582),
    "Dindigul": (10.3624, 77.9695), "Erode": (11.3410, 77.7172), "Kallakurichi": (11.7384, 78.9639),
    "Kanchipuram": (12.8342, 79.7036), "Kanniyakumari": (8.0883, 77.5385), "Karur": (10.9601, 78.0766),
    "Krishnagiri": (12.5186, 78.2137), "Madurai": (9.9252, 78.1198), "Mayiladuthurai": (11.1018, 79.6526),
    "Nagapattinam": (10.7672, 79.8449), "Namakkal": (11.2189, 78.1674), "Nilgiris": (11.4102, 76.6950),
    "Perambalur": (11.2342, 78.8807), "Pudukkottai": (10.3797, 78.8208), "Ramanathapuram": (9.3639, 78.8395),
    "Ranipet": (12.9272, 79.3330), "Salem": (11.6643, 78.1460), "Sivaganga": (9.8433, 78.4809),
    "Tenkasi": (8.9594, 77.3161), "Thanjavur": (10.7870, 79.1378), "Theni": (10.0104, 77.4768),
    "Thoothukudi": (8.8053, 78.1453), "Tiruchirappalli": (10.7905, 78.7047), "Tirunelveli": (8.7139, 77.7567),
    "Tirupathur": (12.4950, 78.5678), "Tiruppur": (11.1085, 77.3411), "Tiruvallur": (13.1430, 79.9120),
    "Tiruvannamalai": (12.2253, 79.0747), "Tiruvarur": (10.7661, 79.6344), "Vellore": (12.9165, 79.1325),
    "Viluppuram": (11.9401, 79.4861), "Virudhunagar": (9.5680, 77.9624),
}

DISTRICT_TA = {
    LIVE_LOC_KEY: "📍 நேரடி இருப்பிடம் (GPS)",
    "Ariyalur": "அரியலூர்", "Chengalpattu": "செங்கல்பட்டு", "Chennai": "சென்னை", "Coimbatore": "கோயம்புத்தூர்",
    "Cuddalore": "கடலூர்", "Dharmapuri": "தர்மபுரி", "Dindigul": "திண்டுக்கல்", "Erode": "ஈரோடு",
    "Kallakurichi": "கள்ளக்குறிச்சி", "Kanchipuram": "காஞ்சிபுரம்", "Kanniyakumari": "கன்னியாகுமரி", "Karur": "கரூர்",
    "Krishnagiri": "கிருஷ்ணகிரி", "Madurai": "மதுரை", "Mayiladuthurai": "மயிலாடுதுறை", "Nagapattinam": "நாகப்பட்டினம்",
    "Namakkal": "நாமக்கல்", "Nilgiris": "நீலகிரி", "Perambalur": "பெரம்பலூர்", "Pudukkottai": "புதுக்கோட்டை",
    "Ramanathapuram": "இராமநாதபுரம்", "Ranipet": "இராணிப்பேட்டை", "Salem": "சேலம்", "Sivaganga": "சிவகங்கை",
    "Tenkasi": "தென்காசி", "Thanjavur": "தஞ்சாவூர்", "Theni": "தேனி", "Thoothukudi": "தூத்துக்குடி",
    "Tiruchirappalli": "திருச்சிராப்பள்ளி", "Tirunelveli": "திருநெல்வேலி", "Tirupathur": "திருப்பத்தூர்",
    "Tiruppur": "திருப்பூர்", "Tiruvallur": "திருவள்ளூர்", "Tiruvannamalai": "திருவண்ணாமலை",
    "Tiruvarur": "திருவாரூர்", "Vellore": "வேலூர்", "Viluppuram": "விழுப்புரம்", "Virudhunagar": "விருதுநகர்",
}

if "district" not in st.session_state:
    st.session_state["district"] = "Thoothukudi"

CROPS = {
    "Paddy": {"rain_min": 60, "rain_ideal": 100, "temp_min": 22, "temp_ideal": 29, "waterlogging_sensitive": False},
    "Maize": {"rain_min": 25, "rain_ideal": 60, "temp_min": 18, "temp_ideal": 27, "waterlogging_sensitive": True},
    "Groundnut": {"rain_min": 20, "rain_ideal": 50, "temp_min": 22, "temp_ideal": 30, "waterlogging_sensitive": True},
    "Cotton": {"rain_min": 25, "rain_ideal": 65, "temp_min": 21, "temp_ideal": 30, "waterlogging_sensitive": True},
    "Sugarcane": {"rain_min": 50, "rain_ideal": 90, "temp_min": 20, "temp_ideal": 30, "waterlogging_sensitive": False},
    "Banana": {"rain_min": 45, "rain_ideal": 85, "temp_min": 21, "temp_ideal": 29, "waterlogging_sensitive": True},
    "Black Gram (Urad)": {"rain_min": 20, "rain_ideal": 45, "temp_min": 22, "temp_ideal": 30, "waterlogging_sensitive": True},
    "Green Gram (Moong)": {"rain_min": 20, "rain_ideal": 45, "temp_min": 22, "temp_ideal": 30, "waterlogging_sensitive": True},
    "Red Gram (Toor)": {"rain_min": 25, "rain_ideal": 55, "temp_min": 20, "temp_ideal": 29, "waterlogging_sensitive": True},
    "Pearl Millet (Cumbu)": {"rain_min": 15, "rain_ideal": 40, "temp_min": 24, "temp_ideal": 32, "waterlogging_sensitive": True},
    "Sorghum (Cholam)": {"rain_min": 15, "rain_ideal": 42, "temp_min": 23, "temp_ideal": 31, "waterlogging_sensitive": True},
    "Finger Millet (Ragi)": {"rain_min": 18, "rain_ideal": 45, "temp_min": 20, "temp_ideal": 29, "waterlogging_sensitive": True},
    "Millet": {"rain_min": 15, "rain_ideal": 40, "temp_min": 24, "temp_ideal": 32, "waterlogging_sensitive": True},
    "Sesame (Gingelly)": {"rain_min": 15, "rain_ideal": 38, "temp_min": 24, "temp_ideal": 32, "waterlogging_sensitive": True},
    "Sunflower": {"rain_min": 20, "rain_ideal": 48, "temp_min": 20, "temp_ideal": 28, "waterlogging_sensitive": True},
    "Turmeric": {"rain_min": 40, "rain_ideal": 75, "temp_min": 20, "temp_ideal": 28, "waterlogging_sensitive": True},
    "Chilli": {"rain_min": 20, "rain_ideal": 50, "temp_min": 20, "temp_ideal": 28, "waterlogging_sensitive": True},
    "Onion": {"rain_min": 18, "rain_ideal": 45, "temp_min": 19, "temp_ideal": 27, "waterlogging_sensitive": True},
    "Tomato": {"rain_min": 20, "rain_ideal": 48, "temp_min": 18, "temp_ideal": 26, "waterlogging_sensitive": True},
    "Tapioca (Cassava)": {"rain_min": 25, "rain_ideal": 60, "temp_min": 22, "temp_ideal": 30, "waterlogging_sensitive": True},
}

CROP_TA = {
    "Paddy": "நெல்", "Maize": "மக்காச்சோளம்", "Groundnut": "நிலக்கடலை", "Cotton": "பருத்தி",
    "Sugarcane": "கரும்பு", "Banana": "வாழை", "Black Gram (Urad)": "உளுந்து", "Green Gram (Moong)": "பாசிப்பயறு",
    "Red Gram (Toor)": "துவரை", "Pearl Millet (Cumbu)": "கம்பு", "Sorghum (Cholam)": "சோளம்",
    "Finger Millet (Ragi)": "கேழ்வரகு (ராகி)", "Millet": "சிறுதானியம்", "Sesame (Gingelly)": "எள்",
    "Sunflower": "சூரியகாந்தி", "Turmeric": "மஞ்சள்", "Chilli": "மிளகாய்", "Onion": "வெங்காயம்",
    "Tomato": "தக்காளி", "Tapioca (Cassava)": "மரவள்ளிக்கிழங்கு",
}

SOILS = ["Not specified", "Clay", "Loam", "Sandy", "Black soil"]
SOIL_TA = {
    "Not specified": "குறிப்பிடப்படவில்லை", "Clay": "களிமண்", "Loam": "வண்டல் மண்",
    "Sandy": "மணல் மண்", "Black soil": "கரிசல் மண்",
}

MONTH_NAMES_EN = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                  7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
MONTH_NAMES_TA = {1: "ஜன", 2: "பிப்", 3: "மார்", 4: "ஏப்", 5: "மே", 6: "ஜூன்",
                  7: "ஜூலை", 8: "ஆக", 9: "செப்", 10: "அக்", 11: "நவ", 12: "டிச"}


@st.cache_data(ttl=600, show_spinner=False)
def reverse_geocode(lat, lon, lang="en"):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "accept-language": lang},
            headers={"User-Agent": "SowingSenseAI/1.0"},
            timeout=5,
        )
        r.raise_for_status()
        addr = r.json().get("address", {})
        place = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("suburb")
            or addr.get("county")
            or addr.get("state_district")
        )
        state = addr.get("state", L("Tamil Nadu", "தமிழ்நாடு"))
        if place:
            return f"{place}, {state}"
    except Exception:
        pass
    return f"GPS ({lat:.4f}°N, {lon:.4f}°E)"


def detect_ip_location():
    for url in ["https://ipapi.co/json/", "http://ip-api.com/json/"]:
        try:
            r = requests.get(url, timeout=4)
            if r.status_code == 200:
                d = r.json()
                lat = d.get("latitude") or d.get("lat")
                lon = d.get("longitude") or d.get("lon")
                if lat and lon:
                    return float(lat), float(lon)
        except Exception:
            continue
    return 8.8053, 78.1453


def trigger_browser_gps():
    components.html(
        """<script>
        (function () {
            if (!navigator.geolocation) { return; }
            navigator.geolocation.getCurrentPosition(function (pos) {
                const lat = pos.coords.latitude, lon = pos.coords.longitude;
                const url = new URL(window.parent.location.href);
                const oldLat = parseFloat(url.searchParams.get("gps_lat"));
                const oldLon = parseFloat(url.searchParams.get("gps_lon"));
                // reload only if the position really changed (~200 m), so GPS jitter cannot cause a reload loop
                const moved = isNaN(oldLat) || isNaN(oldLon) ||
                              Math.abs(oldLat - lat) > 0.002 || Math.abs(oldLon - lon) > 0.002;
                if (moved) {
                    url.searchParams.delete("gps_err");
                    url.searchParams.set("gps_lat", lat.toFixed(5));
                    url.searchParams.set("gps_lon", lon.toFixed(5));
                    url.searchParams.set("gps_acc", Math.round(pos.coords.accuracy));
                    window.parent.location.href = url.toString();
                }
            }, function (err) {
                const url = new URL(window.parent.location.href);
                if (url.searchParams.get("gps_err") !== String(err.code)) {
                    url.searchParams.set("gps_err", err.code);
                    window.parent.location.href = url.toString();
                }
            }, {enableHighAccuracy: true, timeout: 15000, maximumAge: 0});
        })();
        </script>""",
        height=0,
    )


def _request_gps():
    # runs as a button callback, i.e. BEFORE the selectbox is created, so setting its key is allowed
    st.session_state["district"] = LIVE_LOC_KEY


qp = st.query_params
if "gps_lat" in qp and "gps_lon" in qp:
    try:
        _new_fix = (float(qp["gps_lat"]), float(qp["gps_lon"]))
        # apply the URL fix only when it is new, otherwise the user could never switch back to a district
        if st.session_state.get("_gps_qp_seen") != _new_fix:
            st.session_state["live_lat"], st.session_state["live_lon"] = _new_fix
            st.session_state["district"] = LIVE_LOC_KEY
            st.session_state["_gps_qp_seen"] = _new_fix
        st.session_state["live_acc"] = float(qp.get("gps_acc", 0) or 0)
    except ValueError:
        pass


# ---------- UI helpers ----------
def card_title(icon, title, sub=""):
    sub_html = f'<div class="cs">{sub}</div>' if sub else ""
    st.markdown(f'<div class="ctitle"><div class="ci">{icon}</div><div><div class="ct">{title}</div>{sub_html}</div></div>',
                unsafe_allow_html=True)


def page_header(kicker, title, note=""):
    st.markdown(f'<div class="kicker">{kicker}</div><h2 style="margin:.1rem 0 .2rem">{title}</h2>'
                f'<p class="small-note">{note}</p>', unsafe_allow_html=True)


def kpi_row(items):
    html = ""
    for it in items:
        sub = f'<div class="ks">{it[2]}</div>' if len(it) > 2 and it[2] else ""
        acc = " accent" if len(it) > 3 and it[3] else ""
        html += f'<div class="kpi{acc}"><div class="kl">{it[0]}</div><div class="kv">{it[1]}</div>{sub}</div>'
    st.markdown(f'<div class="kpi-grid">{html}</div>', unsafe_allow_html=True)


def chips(items, kind="warn"):
    html = "".join(f'<span class="chip {kind}">{i}</span>' for i in items)
    st.markdown(f'<div class="chip-row">{html}</div>', unsafe_allow_html=True)


def style_fig(fig, height=330, title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=44 if title else 14, b=10), colorway=PALETTE,
                      font=dict(family="Inter, Noto Sans Tamil, sans-serif", color="#d4e6df"),
                      legend=dict(orientation="h", y=-0.18, x=0, title_text=""),
                      hoverlabel=dict(bgcolor="#0b222d", font_size=13))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,190,175,.12)")
    return fig


def donut(score):
    color = GREEN if score >= 70 else AMBER if score >= 45 else RED
    lbl = L("Suitable<br>for sowing", "விதைப்புக்கு<br>உகந்தது")
    fig = go.Figure(go.Pie(values=[score, 100 - score], hole=.76, sort=False, direction="clockwise",
                           marker=dict(colors=[color, "rgba(255,255,255,.07)"], line=dict(width=0)),
                           textinfo="none", hoverinfo="skip"))
    fig.update_layout(height=210, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      annotations=[dict(text=f"<b style='font-size:30px'>{score}%</b><br><span style='font-size:12px;color:#a8c9bd'>{lbl}</span>",
                                        x=.5, y=.5, showarrow=False, font=dict(color="#fff"))])
    return fig


def bar_rows(items):
    html = ""
    for label, val in items:
        html += (f'<div class="bar-row"><div class="bar-top"><span>{label}</span><span style="color:#65e6a0">{val}%</span></div>'
                 f'<div class="bar-bg"><div class="bar-fg" style="width:{val}%"></div></div></div>')
    st.markdown(html, unsafe_allow_html=True)


# =====================================================================
# DATA / MODEL & GOOGLE-SYNCED MULTI-SOURCE LIVE WEATHER ENGINE
# =====================================================================
@st.cache_resource
def load_sowing_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


sowing_model = load_sowing_model()

SRC_LIVE, SRC_DEMO = "Live forecast (Open-Meteo)", "Demo data (offline)"

WX_CODES = {
    0: ("Clear sky", "தெளிவான வானம்", "☀️"), 1: ("Mainly clear", "பெரும்பாலும் தெளிவு", "🌤️"),
    2: ("Partly cloudy", "ஓரளவு மேகமூட்டம்", "⛅"), 3: ("Overcast", "மேகமூட்டம்", "☁️"),
    45: ("Fog", "பனிமூட்டம்", "🌫️"), 48: ("Fog", "பனிமூட்டம்", "🌫️"),
    51: ("Light drizzle", "லேசான தூறல்", "🌦️"), 53: ("Drizzle", "தூறல்", "🌦️"), 55: ("Heavy drizzle", "பலத்த தூறல்", "🌧️"),
    61: ("Light rain / showers", "லேசான மழை / சாரல்", "🌦️"), 63: ("Rain", "மழை", "🌧️"), 65: ("Heavy rain", "கனமழை", "🌧️"),
    80: ("Light rain / showers", "லேசான மழை / சாரல்", "🌦️"), 81: ("Rain showers", "பலத்த சாரல்", "🌧️"), 82: ("Violent showers", "மிக கனமழை", "⛈️"),
    95: ("Thunderstorm", "இடியுடன் மழை", "⛈️"), 96: ("Thunderstorm", "இடி மின்னல் மழை", "⛈️"), 99: ("Thunderstorm", "கடும் இடி மழை", "⛈️"),
}


def get_demo_weather(start_date, seed=42):
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


def fetch_open_meteo(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "timezone": "Asia/Kolkata",
        "cell_selection": "nearest",
        "forecast_days": 16,
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=8)
    r.raise_for_status()
    d = r.json().get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(d.get("time", [])),
        "rain_mm": d.get("precipitation_sum", []),
        "tmax_c": d.get("temperature_2m_max", []),
        "tmin_c": d.get("temperature_2m_min", []),
    }).dropna()
    if df.empty:
        raise ValueError("No forecast data returned")
    return df


@st.cache_data(ttl=60, show_spinner=False)
def fetch_current_weather(lat, lon, _v=3):
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "Asia/Kolkata",
        "cell_selection": "nearest",
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,wind_speed_10m,wind_direction_10m,weather_code",
        "hourly": "precipitation_probability,relative_humidity_2m,temperature_2m,weather_code",
        "forecast_days": 1,
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    cur = data.get("current")
    if not cur:
        raise ValueError("Current weather unavailable")

    rain_prob = None
    hourly = data.get("hourly", {})
    if hourly.get("time") and hourly.get("precipitation_probability"):
        now_hr = datetime.now().strftime("%Y-%m-%dT%H:00")
        times = hourly["time"]
        probs = hourly["precipitation_probability"]
        if now_hr in times:
            rain_prob = probs[times.index(now_hr)]
        elif probs:
            rain_prob = max(probs[:6])
    cur["rain_prob"] = int(rain_prob) if rain_prob is not None else 0

    temps = [float(cur["temperature_2m"])]
    hums = [float(cur["relative_humidity_2m"])]
    feels = [float(cur["apparent_temperature"])]

    try:
        gfs_r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "timezone": "Asia/Kolkata",
                "models": "gfs_seamless,ecmwf_ifs025",
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code",
            },
            timeout=5,
        )
        if gfs_r.status_code == 200:
            gcur = gfs_r.json().get("current", {})
            if gcur.get("temperature_2m") is not None:
                temps.append(float(gcur["temperature_2m"]))
            if gcur.get("relative_humidity_2m") is not None:
                hums.append(float(gcur["relative_humidity_2m"]))
            if gcur.get("apparent_temperature") is not None:
                feels.append(float(gcur["apparent_temperature"]))
    except Exception:
        pass

    try:
        met_r = requests.get(
            "https://api.met.no/weatherapi/locationforecast/2.0/compact",
            params={"lat": f"{lat:.4f}", "lon": f"{lon:.4f}"},
            headers={"User-Agent": "SowingSenseAI/2.0 (tamilnadu-agri-app)"},
            timeout=5,
        )
        if met_r.status_code == 200:
            ts = met_r.json().get("properties", {}).get("timeseries", [])
            if ts:
                inst = ts[0].get("data", {}).get("instant", {}).get("details", {})
                if inst.get("air_temperature") is not None:
                    temps.append(float(inst["air_temperature"]))
                if inst.get("relative_humidity") is not None:
                    hums.append(float(inst["relative_humidity"]))
                if inst.get("wind_speed") is not None:
                    cur["wind_speed_10m"] = float(inst["wind_speed"]) * 3.6
                if inst.get("wind_from_direction") is not None:
                    cur["wind_direction_10m"] = float(inst["wind_from_direction"])
                nxt1 = ts[0].get("data", {}).get("next_1_hours", {})
                sym = nxt1.get("summary", {}).get("symbol_code", "")
                p_amt = nxt1.get("details", {}).get("precipitation_amount")
                if p_amt is not None and p_amt > float(cur.get("precipitation", 0)):
                    cur["precipitation"] = p_amt
                if "rain" in sym or "sleet" in sym or (cur["rain_prob"] >= 50 and int(cur.get("weather_code", 0)) in (2, 3)):
                    cur["weather_code"] = 80
    except Exception:
        pass

    cur["temperature_2m"] = int(round(min(temps)))
    cur["apparent_temperature"] = int(round(sum(feels) / len(feels)))
    cur["relative_humidity_2m"] = int(round(max(hums)))

    if cur["rain_prob"] >= 50 and int(cur.get("weather_code", 0)) in (2, 3):
        cur["weather_code"] = 80

    precip = cur.get("precipitation")
    cur["rain"] = round(float(precip if precip is not None else cur.get("rain", 0.0)), 1)
    w_kmh = float(cur.get("wind_speed_10m", 0))
    cur["wind_speed_10m"] = int(round(w_kmh))
    cur["wind_mph"] = int(round(w_kmh * 0.621371))

    return cur


# =====================================================================
# DYNAMIC HISTORICAL CLIMATE TRENDS & MONTHLY SUMMARY PER CITY
# =====================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_city_historical_climate(lat, lon):
    """Fetches real 2024-2025 historical daily weather for the selected city from Open-Meteo Archive API."""
    try:
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": "2024-01-01",
            "end_date": "2025-12-31",
            "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "Asia/Kolkata",
        }
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=8)
        r.raise_for_status()
        d = r.json().get("daily", {})
        df = pd.DataFrame({
            "date": pd.to_datetime(d.get("time", [])),
            "avg_temp_c": d.get("temperature_2m_mean", []),
            "tmax": d.get("temperature_2m_max", []),
            "tmin": d.get("temperature_2m_min", []),
            "rain": d.get("precipitation_sum", []),
        }).dropna()
        if df.empty:
            return None, None

        df["max_temp_7day_c"] = df["tmax"].rolling(7, min_periods=1).max().round(1)
        df["min_temp_7day_c"] = df["tmin"].rolling(7, min_periods=1).min().round(1)
        df["rain_7day_mm"] = df["rain"].rolling(7, min_periods=1).sum().round(1)
        df["avg_temp_c"] = df["avg_temp_c"].round(1)

        df["month_num"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        yearly_m = df.groupby(["year", "month_num"], as_index=False).agg(
            average_temperature_c=("avg_temp_c", "mean"),
            total_rainfall_mm=("rain", "sum"),
        )
        m_df = yearly_m.groupby("month_num", as_index=False).agg(
            average_temperature_c=("average_temperature_c", "mean"),
            total_rainfall_mm=("total_rainfall_mm", "mean"),
        ).rename(columns={"month_num": "month"})
        m_df["average_temperature_c"] = m_df["average_temperature_c"].round(1)
        m_df["total_rainfall_mm"] = m_df["total_rainfall_mm"].round(1)

        return df[["date", "avg_temp_c", "max_temp_7day_c", "min_temp_7day_c", "rain_7day_mm"]], m_df
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def load_hist(lat=8.8053, lon=78.1453):
    dyn_hist, _ = fetch_city_historical_climate(lat, lon)
    if dyn_hist is not None:
        return dyn_hist
    return pd.read_csv(WEATHER_FEATURES_PATH, parse_dates=["date"]) if WEATHER_FEATURES_PATH.exists() else None


@st.cache_data(show_spinner=False)
def load_monthly(lat=8.8053, lon=78.1453):
    _, dyn_monthly = fetch_city_historical_climate(lat, lon)
    if dyn_monthly is not None:
        return dyn_monthly
    return pd.read_csv(MONTHLY_PATH) if MONTHLY_PATH.exists() else None


def get_trend_legend_map():
    return {
        "avg_temp_c": L("Avg Temperature (°C)", "சராசரி வெப்பநிலை (°C)"),
        "max_temp_7day_c": L("7-Day Max Temp (°C)", "7-நாள் அதிகபட்ச வெப்பம் (°C)"),
        "min_temp_7day_c": L("7-Day Min Temp (°C)", "7-நாள் குறைந்தபட்ச வெப்பம் (°C)"),
        "rain_7day_mm": L("7-Day Rainfall (mm)", "7-நாள் மழைப்பொழிவு (மி.மீ)"),
    }


def format_monthly_df(monthly_df):
    if monthly_df is None:
        return None
    m = monthly_df.copy()
    if "month" in m.columns and pd.api.types.is_numeric_dtype(m["month"]):
        m_map = MONTH_NAMES_TA if is_ta() else MONTH_NAMES_EN
        m["month_label"] = m["month"].map(lambda x: m_map.get(int(x), str(x)))
    else:
        m["month_label"] = m["month"].astype(str)
    return m


def predict_sowing_suitability(rain, temp, dry):
    if sowing_model is None:
        return None, None
    f = pd.DataFrame([{"rainfall_mm": rain, "temperature_c": temp, "dry_spell_days": dry}])
    pred = int(sowing_model.predict(f)[0])
    conf = float(max(sowing_model.predict_proba(f)[0]) * 100)
    label = L("Favourable (demo model)", "சாதகமானது (மாதிரி AI)") if pred == 1 else L("Less favourable (demo model)", "குறைந்த சாதகம் (மாதிரி AI)")
    return label, conf


def max_consecutive_dry(values):
    best = cur = 0
    for v in values:
        if float(v) < 1.0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best



def score_window(window, ci):
    """
    Strict crop-specific sowing suitability score.
    Rainfall thresholds are interpreted as 7-day totals.
    Scores are heuristic and require agricultural validation.
    """
    rain = max(0.0, float(window["rain_mm"].sum()))

    avg_temp = float(
        ((window["tmax_c"] + window["tmin_c"]) / 2).mean()
    )

    rainy = int((window["rain_mm"] >= 1.0).sum())
    dry_spell = max_consecutive_dry(
        window["rain_mm"].tolist()
    )

    rain_min = float(ci["rain_min"])
    rain_ideal = float(ci["rain_ideal"])
    temp_min = float(ci["temp_min"])
    temp_ideal = float(ci["temp_ideal"])

    # 1. Rainfall: insufficient rain cannot receive a high score.
    if rain_min <= 0 or rain_ideal <= rain_min:
        rain_score = 0.0
    elif rain < rain_min:
        rain_score = 50.0 * rain / rain_min
    elif rain <= rain_ideal:
        rain_score = (
            50.0
            + 50.0 * (rain - rain_min)
            / (rain_ideal - rain_min)
        )
    else:
        # Penalize rainfall exceeding the crop's ideal amount.
        rain_score = max(
            0.0,
            100.0 * (1.5 * rain_ideal - rain)
            / (0.5 * rain_ideal)
        )

    # 2. Temperature: penalize conditions outside the
    # minimum-to-ideal range and above the estimated upper limit.
    # Prefer explicit, validated crop-specific maximum temperatures.
    temp_max = temp_ideal + max(3.0, temp_ideal - temp_min)

    if temp_max <= temp_ideal or temp_ideal <= temp_min:
        temp_score = 0.0
    elif temp_min <= avg_temp <= temp_ideal:
        temp_score = (
            100.0 * (avg_temp - temp_min)
            / (temp_ideal - temp_min)
        )
    elif temp_ideal < avg_temp < temp_max:
        temp_score = (
            100.0 * (temp_max - avg_temp)
            / (temp_max - temp_ideal)
        )
    else:
        temp_score = 0.0

    # 3. Dry-spell penalty.
    dry_score = max(0.0, 100.0 - (dry_spell * 15.0))

    # 4. Excess rainfall penalty.
    excess_score = max(
        0.0,
        100.0 * (1.5 * rain_ideal - rain)
        / (0.5 * rain_ideal)
    ) if rain_ideal > 0 else 0.0

    # Weighted score: all component scores are between 0 and 100.
    score = round(
        0.40 * rain_score
        + 0.30 * temp_score
        + 0.20 * dry_score
        + 0.10 * excess_score
    )

    risks = []

    if rain < rain_min:
        risks.append(
            L("Rainfall may be insufficient",
              "மழைப்பொழிவு போதுமானதாக இல்லாமல் இருக்கலாம்")
        )

    if dry_spell >= 5:
        risks.append(
            L("Possible prolonged dry spell",
              "நீண்ட வறட்சி ஏற்பட வாய்ப்புள்ளது")
        )

    if avg_temp < temp_min:
        risks.append(
            L("Temperature below the crop's minimum threshold",
              "பயிருக்குத் தேவையான குறைந்தபட்ச வெப்பநிலைக்குக் கீழே உள்ளது")
        )
    elif avg_temp > temp_max:
        risks.append(
            L("Temperature exceeds the estimated upper threshold",
              "மதிப்பிடப்பட்ட அதிகபட்ச வெப்பநிலையை வெப்பம் கடந்துள்ளது")
        )

    if rain > rain_ideal * 1.4:
        risks.append(
            L("Excess rainfall / waterlogging risk",
              "அதிக மழை / நீர் தேங்கும் அபாயம்")
        )

    if not risks:
        risks.append(
            L("No major risk flagged by this screening method",
              "இந்த ஆரம்ப மதிப்பீட்டில் முக்கிய அபாயம் கண்டறியப்படவில்லை")
        )

    parts = [
        int(round(rain_score)),
        int(round(temp_score)),
        int(round(dry_score)),
        int(round(excess_score)),
    ]

    return {
        "score": max(0, min(100, score)),
        "rain_mm": round(rain, 1),
        "avg_temp": round(avg_temp, 1),
        "dry_spell": dry_spell,
        "rainy_days": rainy,
        "risks": risks,
        "parts": parts,
    }

def evaluate_windows(weather, ci):
    weather = weather.sort_values("date").reset_index(drop=True)
    out = []
    for s in range(max(1, len(weather) - 6)):
        w = weather.iloc[s:s + 7]
        if len(w) < 7:
            continue
        m = score_window(w, ci)
        out.append({"Start": pd.to_datetime(w.iloc[0]["date"]).date(), "End": pd.to_datetime(w.iloc[-1]["date"]).date(),
                    "Suitability score": m["score"], "Rainfall (mm)": m["rain_mm"], "Average temperature (°C)": m["avg_temp"],
                    "Longest dry spell (days)": m["dry_spell"], "Rainy days": m["rainy_days"], "Risks": "; ".join(m["risks"]),
                    "Rain score": min(100, m["parts"][0]), "Temp score": m["parts"][1], "Dry score": m["parts"][2], "Excess score": m["parts"][3]})
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values("Suitability score", ascending=False).reset_index(drop=True)


def compass(deg):
    en = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ta = ["வடக்கு", "வடகிழக்கு", "கிழக்கு", "தென்கிழக்கு", "தெற்கு", "தென்மேற்கு", "மேற்கு", "வடமேற்கு"]
    idx = int(round(float(deg) / 45)) % 8
    return ta[idx] if is_ta() else en[idx]


# =====================================================================
# SINGLE-LINE TOP NAV (4 MAIN TABS + "MORE" DROPDOWN + LANGUAGE TOGGLE)
# =====================================================================
MAIN_PAGES = {
    "Dashboard": ("🏠 Dashboard", "🏠 முகப்பு"),
    "Sowing Recommendation": ("🌱 Sowing Recommendation", "🌱 விதைப்பு பரிந்துரை"),
    "Insights & Alerts": ("🔔 Insights & Alerts", "🔔 எச்சரிக்கைகள்"),
    "About": ("ℹ️ About", "ℹ️ பற்றி"),
}

MORE_PAGES = {
    "None": ("➕ More", "➕ மேலும்"),
    "Weather & Climate": ("⛅ Weather & Climate", "⛅ வானிலை & காலநிலை"),
    "Crop Information": ("🌾 Crop Information", "🌾 பயிர் தகவல்"),
    "Soil Analysis": ("🧪 Soil Analysis", "🧪 மண் ஆய்வு"),
    "Climate Trends": ("📈 Climate Trends", "📈 காலநிலை போக்கு"),
    "Climate Twin Simulator": ("🔮 Climate Twin Simulator", "🔮 காலநிலை சிமுலேட்டர்"),
    "Export Data": ("📤 Export Data", "📤 தரவு பதிவிறக்கம்"),
}

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Dashboard"
if "nav_main" not in st.session_state:
    st.session_state["nav_main"] = "Dashboard"
if "nav_more" not in st.session_state:
    st.session_state["nav_more"] = "None"


def _on_main_nav():
    st.session_state["active_page"] = st.session_state["nav_main"]
    st.session_state["nav_more"] = "None"


def _on_more_nav():
    if st.session_state["nav_more"] != "None":
        st.session_state["active_page"] = st.session_state["nav_more"]


try:
    nav_box = st.container(key="topnav")
except TypeError:
    nav_box = st.container()
with nav_box:
    nb = st.columns([1.55, 4.55, 1.25, 1.05], gap="small")
    nb[0].markdown(f'<div class="brand"><div class="logo">🌱</div><div><div class="name">Sowing<b>Sense</b> AI</div>'
                   f'<div class="tag">{L("Smarter Sowing. Healthier Tomorrow.", "திறமையான விதைப்பு. வளமான எதிர்காலம்.")}</div></div></div>', unsafe_allow_html=True)
    nb[1].radio("Main Navigation", list(MAIN_PAGES.keys()),
                format_func=lambda p: MAIN_PAGES[p][1 if is_ta() else 0],
                horizontal=True, key="nav_main", on_change=_on_main_nav, label_visibility="collapsed")
    nb[2].selectbox("More Options", list(MORE_PAGES.keys()),
                    format_func=lambda p: MORE_PAGES[p][1 if is_ta() else 0],
                    key="nav_more", on_change=_on_more_nav, label_visibility="collapsed")
    nb[3].radio("Language", ["en", "ta"], format_func=lambda x: "🌐 EN" if x == "en" else "🇮🇳 தமிழ்",
                horizontal=True, key="lang", label_visibility="collapsed")

page = st.session_state["active_page"]

# =====================================================================
# HERO (dashboard) + FARM BAR (all pages)
# =====================================================================
is_live_sel = st.session_state.get("source", SRC_LIVE) == SRC_LIVE

if page == "Dashboard":
    st.markdown(f"""
<div class="hero"><div class="hero-top"><div>
<div class="title"><span class="grad">SowingSense</span><span class="ai-badge">AI</span></div>
<div class="sub">{L("AI-Powered Climate Intelligence for Smarter Sowing Decisions", "சிறந்த விதைப்பு முடிவுகளுக்கான AI பருவநிலை நுண்ணறிவுத் தளம்")}</div>
<div class="tags">{L("Weather insights  •  Crop analysis  •  Sustainable farming", "வானிலை தகவல்கள்  •  பயிர் ஆய்வு  •  நிலையான விவசாயம்")}</div>
</div><div class="hero-right">
<span class="pill">📅 {date.today():%a, %d %b %Y}</span>
<span class="pill {'live' if is_live_sel else 'demo'}"><span class="dot"></span>{L('Live Data', 'நேரடி தரவு') if is_live_sel else L('Demo Data', 'மாதிரி தரவு')}</span>
</div></div>
<div class="feat-row">
<div class="feat"><span class="ico">🌦️</span><div><div class="t">{L("Live Weather Data", "நேரடி வானிலை தரவு")}</div><div class="s">{L("Open-Meteo API", "Open-Meteo நிகழ்நேரம்")}</div></div></div>
<div class="feat"><span class="ico">🤖</span><div><div class="t">{L("ML-based Crop Analysis", "AI பயிர் பகுப்பாய்வு")}</div><div class="s">{L("Climate + Weather", "காலநிலை + வானிலை")}</div></div></div>
<div class="feat"><span class="ico">📊</span><div><div class="t">{L("Climate Trends", "காலநிலை போக்குகள்")}</div><div class="s">{L("Historical Analysis", "வரலாற்றுப் பகுப்பாய்வு")}</div></div></div>
<div class="feat"><span class="ico">🍃</span><div><div class="t">{L("Sustainable Farming", "நிலையான விவசாயம்")}</div><div class="s">{L("Better Future", "வளமான எதிர்காலம்")}</div></div></div>
</div></div>
<div class="jump"><a href="#sec-weather">{L("⛅ Live weather", "⛅ நேரடி வானிலை")}</a><a href="#sec-forecast">{L("📅 Forecast", "📅 முன்னறிவிப்பு")}</a><a href="#sec-trends">{L("📈 Trends", "📈 காலநிலை போக்கு")}</a><a href="#sec-suit">{L("🎯 Suitability", "🎯 உகந்த தன்மை")}</a><a href="#sec-reco">{L("🌱 AI recommendation", "🌱 AI பரிந்துரை")}</a></div>
""", unsafe_allow_html=True)

with st.container(border=True):
    card_title("📍", L("Select Farm Details", "பண்ணை விவரங்களைத் தேர்ந்தெடுக்கவும்"),
               L("Choose location, crop and analysis settings to get AI-powered sowing recommendations",
                 "இருப்பிடம், பயிர் மற்றும் ஆய்வு அமைப்புகளைத் தேர்ந்தெடுத்து AI விதைப்பு பரிந்துரைகளைப் பெறுங்கள்"))
    fc = st.columns([1.2, 1.1, 1.5, 1.1, 1.1, 1.35])
    district = fc[0].selectbox(L("District / Location", "மாவட்டம் / இருப்பிடம்"), list(DISTRICTS.keys()),
                               format_func=lambda d: DISTRICT_TA.get(d, d) if is_ta() else d, key="district")
    crop = fc[1].selectbox(L("Crop", "பயிர்"), list(CROPS.keys()),
                           format_func=lambda c: CROP_TA.get(c, c) if is_ta() else c, key="crop")
    source = fc[2].selectbox(L("Climate data source", "வானிலை தரவு மூலம்"), [SRC_LIVE, SRC_DEMO],
                             format_func=lambda s: (L("Live forecast (Open-Meteo)", "நேரடி முன்னறிவிப்பு (Open-Meteo)") if s == SRC_LIVE
                                                    else L("Demo data (offline)", "மாதிரி தரவு (ஆஃப்லைன்)")), key="source")
    start_day = fc[3].date_input(L("Start date for analysis", "ஆய்வு தொடங்கும் தேதி"), value=date.today(), key="start_day")
    soil = fc[4].selectbox(L("Soil type (optional)", "மண் வகை (விருப்பத்தேர்வு)"), SOILS, index=2,
                           format_func=lambda s: SOIL_TA.get(s, s) if is_ta() else s, key="soil")
    fc[5].markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    run = fc[5].button(L("🌱 Generate Recommendation", "🌱 பரிந்துரையை உருவாக்கு"), type="primary", width="stretch")

    bot1, bot2 = st.columns([2.2, 1.2])
    auto_refresh = bot1.toggle(L("⚡ Auto-refresh live weather every 60 s", "⚡ 60 நொடிகளுக்கு ஒருமுறை நேரடி வானிலையை புதுப்பி"), value=True, key="auto_refresh")
    bot2.button(L("📡 Use My Live Location (GPS)", "📡 என் நேரடி இருப்பிடத்தைப் பயன்படுத்து (GPS)"),
                width="stretch", on_click=_request_gps)

    if district == LIVE_LOC_KEY:
        try:
            from streamlit_js_eval import get_geolocation
            _geo = get_geolocation()          # asks the browser for permission, no page reload needed
            if isinstance(_geo, dict) and isinstance(_geo.get("coords"), dict):
                _c = _geo["coords"]
                st.session_state["live_lat"] = float(_c["latitude"])
                st.session_state["live_lon"] = float(_c["longitude"])
                st.session_state["live_acc"] = float(_c.get("accuracy") or 0)
                st.session_state.pop("live_err", None)
            elif isinstance(_geo, dict) and _geo.get("error"):
                st.session_state["live_err"] = True
        except ImportError:
            st.session_state["live_err"] = True
            st.error("Run:  pip install streamlit-js-eval   and restart the app.")

if district == LIVE_LOC_KEY and "live_lat" in st.session_state:
    lat = float(st.session_state["live_lat"])
    lon = float(st.session_state["live_lon"])
    loc_name = reverse_geocode(lat, lon, "ta" if is_ta() else "en")
    acc = st.session_state.get("live_acc", 0)
    if acc and acc > 1000:
        st.warning(L(f"GPS accuracy is low (±{acc/1000:.1f} km) - this device is probably using Wi-Fi/IP positioning. Try a phone with GPS on.",
                     f"GPS துல்லியம் குறைவு (±{acc/1000:.1f} கி.மீ). GPS உள்ள மொபைலில் முயற்சிக்கவும்."))
elif district == LIVE_LOC_KEY:
    lat, lon = DISTRICTS[LIVE_LOC_KEY]
    loc_name = L("Waiting for GPS…", "GPS காத்திருக்கிறது…")
    if st.query_params.get("gps_err") or st.session_state.get("live_err"):
        st.error(L("Location blocked or unavailable. Allow location permission in the browser (site must be HTTPS) and click the GPS button again.",
                   "இருப்பிடம் தடுக்கப்பட்டுள்ளது. உலாவியில் Location அனுமதி கொடுத்து (HTTPS தேவை) GPS பட்டனை மீண்டும் அழுத்தவும்."))
    else:
        st.info(L("Detecting your live location… allow the browser location prompt. Showing a default point until then.",
                  "உங்கள் நேரடி இருப்பிடம் கண்டறியப்படுகிறது… உலாவி அனுமதியை வழங்கவும்."))
else:
    lat, lon = DISTRICTS[district]
    loc_name = DISTRICT_TA.get(district, district) if is_ta() else district

crop_name = CROP_TA.get(crop, crop) if is_ta() else crop
soil_name = SOIL_TA.get(soil, soil) if is_ta() else soil

config_key = (district, round(lat, 4), round(lon, 4), source, start_day)
stale = source == SRC_LIVE and (time.time() - st.session_state.get("fetched_at", 0)) > 900
if run or stale or "weather_data" not in st.session_state or st.session_state.get("weather_config") != config_key:
    with st.spinner(L("Analysing climate data…", "காலநிலை தரவு ஆய்வு செய்யப்படுகிறது…")):
        if source == SRC_LIVE:
            try:
                weather = fetch_open_meteo(lat, lon)
                st.session_state["data_source_label"] = "Open-Meteo forecast"
            except Exception:
                weather = get_demo_weather(start_day)
                st.session_state["data_source_label"] = "Synthetic demo data (live forecast unavailable)"
        else:
            weather = get_demo_weather(start_day)
            st.session_state["data_source_label"] = "Synthetic demo data"
        st.session_state["weather_data"] = weather
        st.session_state["weather_config"] = config_key
        st.session_state["fetched_at"] = time.time()
    if run:
        fetch_current_weather.clear()
        st.toast(L("Recommendation updated ✨", "பரிந்துரை புதுப்பிக்கப்பட்டது ✨"), icon="🌱")

weather = st.session_state["weather_data"].copy()
weather["date"] = pd.to_datetime(weather["date"])
weather = weather.sort_values("date").reset_index(drop=True)
CI = CROPS[crop]
result = evaluate_windows(weather, CI)
weather_total_rain = float(weather["rain_mm"].sum())
avg_max_temp = float(weather["tmax_c"].mean())
rainy_days = int((weather["rain_mm"] >= 1).sum())
dry_days = int((weather["rain_mm"] < 1).sum())
raw_label = st.session_state.get("data_source_label", "Demo data")
data_label = (
    L("Open-Meteo forecast", "Open-Meteo நேரடி முன்னறிவிப்பு") if raw_label == "Open-Meteo forecast"
    else L("Synthetic demo data", "செயற்கை மாதிரி தரவு")
)
best = result.iloc[0] if not result.empty else None
second = result.iloc[1] if len(result) > 1 else best


# =====================================================================
# SECTION RENDERERS
# =====================================================================
def _weather_card(loc_label, lat, lon):
    with st.container(border=True):
        top = st.columns([3, 1])
        with top[0]:
            card_title("🌡️", f"{L('Current Live Weather', 'தற்போதைய நேரடி வானிலை')} — {loc_label}",
                       L("MET Norway + NOAA GFS + ECMWF real-time conditions", "MET Norway + NOAA GFS + ECMWF நிகழ்நேர வானிலை"))
        if top[1].button(L("🔄 Refresh", "🔄 புதுப்பி"), key="refresh_wx", width="stretch"):
            fetch_current_weather.clear()
            st.toast(L("Weather refreshed", "வானிலை புதுப்பிக்கப்பட்டது"), icon="🔄")
        try:
            c = fetch_current_weather(lat, lon)
            desc_en, desc_ta, emo = WX_CODES.get(int(c.get("weather_code", 0)), ("Conditions", "வானிலை", "🌡️"))
            desc = desc_ta if is_ta() else desc_en
            if not c.get("is_day", 1) and emo == "☀️":
                emo = "🌙"
            r_prob = c.get("rain_prob", 0)
            rain_lbl = f"{L('Rain chance', 'மழை வாய்ப்பு')}: {r_prob}%" if r_prob else L("Rainfall", "மழைப்பொழிவு")
            rain_val = f"{r_prob}% ({c.get('rain', 0.0)} mm)" if r_prob and float(c.get("rain", 0)) == 0 else f"{c.get('rain', '—')} mm"
            wind_val = f"{c.get('wind_speed_10m', '—')} km/h"
            wind_sub = f"{L('Wind', 'காற்று')} ({c.get('wind_mph', '—')} mph)"
            st.markdown(
                f'<div class="wx"><div class="wx-main"><div class="wx-emoji">{emo}</div><div>'
                f'<div class="wx-temp">{c.get("temperature_2m", "—")} °C</div><div class="wx-desc">{desc}</div>'
                f'<div class="wx-feel">{L("Feels like", "உணரப்படும் வெப்பம்")} {c.get("apparent_temperature", "—")} °C</div></div></div>'
                f'<div class="wx-grid">'
                f'<div class="tile"><div class="ti">💧</div><div class="tv">{c.get("relative_humidity_2m", "—")}%</div><div class="tl">{L("Humidity", "ஈரப்பதம்")}</div></div>'
                f'<div class="tile"><div class="ti">🌧️</div><div class="tv">{rain_val}</div><div class="tl">{rain_lbl}</div></div>'
                f'<div class="tile"><div class="ti">💨</div><div class="tv">{wind_val}</div><div class="tl">{wind_sub}</div></div>'
                f'<div class="tile"><div class="ti">🧭</div><div class="tv">{compass(c.get("wind_direction_10m", 0))}</div><div class="tl">{L("Wind direction", "காற்றின் திசை")}</div></div>'
                f'</div></div>', unsafe_allow_html=True)
            auto = L(" · 🟢 auto-updating every 60 s", " · 🟢 60 நொடிக்கு ஒருமுறை தானாக புதுப்பிக்கப்படுகிறது") if st.session_state.get("auto_refresh", True) else ""
            st.caption(f"{L('Observation', 'பதிவு நேரம்')}: {c.get('time', 'Unavailable')} · {L('Last checked', 'கடைசி சரிபார்ப்பு')} {datetime.now():%H:%M:%S}{auto}")
        except Exception:
            st.warning(L("Live weather is currently unavailable. Current-condition cards are not filled with demo values.",
                         "நேரடி வானிலை தற்போது கிடைக்கவில்லை. மாதிரி மதிப்புகள் இதில் நிரப்பப்படவில்லை."))


def render_current_weather():
    st.markdown('<div id="sec-weather"></div>', unsafe_allow_html=True)
    left, right = st.columns([2.3, 1])
    with left:
        def _card():
            _weather_card(loc_name, lat, lon)
        if hasattr(st, "fragment"):
            st.fragment(run_every=60 if st.session_state.get("auto_refresh", True) else None)(_card)()
        else:
            _card()
    with right, st.container(border=True):
        card_title("📍", L("Location Information", "இருப்பிடத் தகவல்"))
        state_lbl = "" if "," in loc_name else f", {L('Tamil Nadu', 'தமிழ்நாடு')}"
        st.markdown(f"**{loc_name}{state_lbl}**  \n<span class='small-note'>{lat:.4f}° N, {lon:.4f}° E</span>", unsafe_allow_html=True)
        try:
            mp = px.scatter_mapbox(pd.DataFrame({"lat": [lat], "lon": [lon], "n": [loc_name]}), lat="lat", lon="lon",
                                   zoom=11 if district == "Thoothukudi" or district == LIVE_LOC_KEY else 8, height=170)
            mp.update_traces(marker=dict(size=17, color=GREEN))
            mp.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(mp, width="stretch", key="loc_map")
        except Exception:
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), height=170)


def build_alerts():
    nxt = weather.head(7)
    out = []
    mx = nxt.loc[nxt["rain_mm"].idxmax()]
    if mx["rain_mm"] >= 15:
        out.append(("bad", "⛈️", L("Heavy rainfall expected", "கனமழை எதிர்பார்க்கப்படுகிறது"),
                    L(f"on {mx['date']:%d %b %Y} ({mx['rain_mm']:.1f} mm)", f"{mx['date']:%d %b %Y} அன்று ({mx['rain_mm']:.1f} மி.மீ)")))
    elif mx["rain_mm"] >= 2:
        out.append(("warn", "⚠️", L("Moderate rainfall expected", "மிதமான மழை எதிர்பார்க்கப்படுகிறது"),
                    L(f"on {mx['date']:%d %b %Y} ({mx['rain_mm']:.1f} mm)", f"{mx['date']:%d %b %Y} அன்று ({mx['rain_mm']:.1f} மி.மீ)")))
    if nxt["rain_mm"].sum() < 1:
        out.append(("warn", "🏜️", L("Dry week ahead", "வறண்ட வாரம்"),
                    L("No significant rain in next 7 days", "அடுத்த 7 நாட்களில் குறிப்பிடத்தக்க மழை இல்லை")))
    if max_consecutive_dry(nxt["rain_mm"].tolist()) >= 5:
        out.append(("warn", "☀️", L("Long dry spell", "நீண்ட வறட்சி காலம்"),
                    L("5+ consecutive dry days forecast", "தொடர்ந்து 5+ வறண்ட நாட்கள் முன்னறிவிப்பு")))
    if nxt["tmax_c"].max() >= 36:
        out.append(("bad", "🔥", L("Heat alert", "அதிக வெப்ப எச்சரிக்கை"),
                    L(f"Max temperature up to {nxt['tmax_c'].max():.0f} °C", f"அதிகபட்ச வெப்பநிலை {nxt['tmax_c'].max():.0f} °C வரை")))
    avg = float(((nxt["tmax_c"] + nxt["tmin_c"]) / 2).mean())
    if abs(avg - CI["temp_ideal"]) <= 3:
        out.append(("ok", "✅", L(f"Temperatures suitable for {crop.lower()}", f"{crop_name} பயிருக்கு ஏற்ற வெப்பநிலை"),
                    L(f"avg {avg:.1f} °C vs ideal {CI['temp_ideal']} °C", f"சராசரி {avg:.1f} °C (உகந்தது {CI['temp_ideal']} °C)")))
    else:
        out.append(("warn", "🌡️", L(f"Temperature off-ideal for {crop.lower()}", f"{crop_name} பயிருக்கு சற்று மாறுபட்ட வெப்பநிலை"),
                    L(f"avg {avg:.1f} °C vs ideal {CI['temp_ideal']} °C", f"சராசரி {avg:.1f} °C (உகந்தது {CI['temp_ideal']} °C)")))
    if nxt["tmax_c"].max() < 36:
        out.append(("ok", "🌤️", L("No extreme heat expected", "கடும் வெப்பம் எதிர்பார்க்கப்படவில்லை"),
                    L("in next 7 days", "அடுத்த 7 நாட்களில்")))
    return out


def alert_html(a):
    return (f'<div class="alert-card {a[0]}"><div class="ai">{a[1]}</div><div><div class="at">{a[2]}</div>'
            f'<div class="ad">{a[3]}</div></div></div>')


def render_forecast_and_alerts():
    st.markdown('<div id="sec-forecast"></div>', unsafe_allow_html=True)
    left, right = st.columns([2.3, 1])
    with left, st.container(border=True):
        h = st.columns([2, 1.6])
        with h[0]:
            card_title("📅", L("Weather Forecast", "வானிலை முன்னறிவிப்பு"), f"{L('Daily forecast', 'தினசரி முன்னறிவிப்பு')} · {data_label}")
        view = h[1].radio("View", ["Combined", "Temperature", "Rainfall"],
                          format_func=lambda v: {"Combined": "இணைந்தது", "Temperature": "வெப்பநிலை", "Rainfall": "மழை"}[v] if is_ta() else v,
                          horizontal=True, key="fc_view")
        days = st.radio("Range", ["7 days", "14 days"],
                        format_func=lambda d: ("7 நாட்கள்" if d == "7 days" else "14 நாட்கள்") if is_ta() else d,
                        horizontal=True, key="fc_days")
        w = weather.head(7 if days == "7 days" else 14)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if view in ("Combined", "Rainfall"):
            fig.add_bar(x=w["date"], y=w["rain_mm"], name=L("Rainfall (mm)", "மழைப்பொழிவு (மி.மீ)"), marker_color="rgba(76,201,240,.75)",
                        text=w["rain_mm"], textposition="outside", secondary_y=True)
        if view in ("Combined", "Temperature"):
            fig.add_scatter(x=w["date"], y=w["tmax_c"], name=L("Max Temp (°C)", "அதிகபட்ச வெப்பம் (°C)"), mode="lines+markers+text", line=dict(color="#ff6b6b", width=3),
                            text=w["tmax_c"].round(0).astype(int).astype(str) + "°", textposition="top center", secondary_y=False)
            fig.add_scatter(x=w["date"], y=w["tmin_c"], name=L("Min Temp (°C)", "குறைந்தபட்ச வெப்பம் (°C)"), mode="lines+markers+text", line=dict(color="#4cc9f0", width=3),
                            text=w["tmin_c"].round(0).astype(int).astype(str) + "°", textposition="bottom center", secondary_y=False)
        fig.update_yaxes(title_text=L("Temperature (°C)", "வெப்பநிலை (°C)"), secondary_y=False, range=[min(15, float(w["tmin_c"].min()) - 3), float(w["tmax_c"].max()) + 5])
        fig.update_yaxes(title_text=L("Rainfall (mm)", "மழைப்பொழிவு (மி.மீ)"), secondary_y=True, rangemode="tozero", showgrid=False)
        fig.update_xaxes(tickformat="%d %b<br>%a")
        st.plotly_chart(style_fig(fig, height=360), width="stretch", key="fc_chart")
    with right, st.container(border=True):
        card_title("🔔", L("Weather Alerts (Next 7 Days)", "வானிலை எச்சரிக்கைகள் (அடுத்த 7 நாட்கள்)"))
        alerts = build_alerts()
        st.markdown("".join(alert_html(a) for a in alerts[:4]), unsafe_allow_html=True)
        if len(alerts) > 4:
            with st.expander(L(f"View all ({len(alerts)})", f"அனைத்தையும் காண்க ({len(alerts)})")):
                st.markdown("".join(alert_html(a) for a in alerts[4:]), unsafe_allow_html=True)


def render_trends_row():
    st.markdown('<div id="sec-trends"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.15, 1.15, 1])
    hist, monthly = load_hist(lat, lon), format_monthly_df(load_monthly(lat, lon))
    legend_map = get_trend_legend_map()

    with c1, st.container(border=True):
        card_title("📈", L("Climate Trends", "காலநிலை போக்குகள்"),
                   f"{L('Historical climate data', 'வரலாற்று காலநிலை தரவு')} · {loc_name}")
        if hist is None:
            st.info(L("Historical features not found. Run build_features.py first.", "வரலாற்றுத் தரவுகள் கிடைக்கவில்லை. முதலில் build_features.py-ஐ இயக்கவும்."))
        else:
            mode = st.radio("Trend", ["Temperature", "Rainfall"],
                            format_func=lambda m: ("வெப்பநிலை" if m == "Temperature" else "மழைப்பொழிவு") if is_ta() else m,
                            horizontal=True, key="trend_mode")
            if mode == "Temperature":
                raw_cols = [c for c in ["avg_temp_c", "max_temp_7day_c", "min_temp_7day_c"] if c in hist.columns]
                if raw_cols:
                    plot_df = hist.rename(columns=legend_map)
                    nice_cols = [legend_map[c] for c in raw_cols]
                    f = px.line(plot_df, x="date", y=nice_cols, labels={"value": "°C", "date": "", "variable": ""})
                else:
                    f = go.Figure()
            else:
                if "rain_7day_mm" in hist:
                    rain_col_name = legend_map["rain_7day_mm"]
                    plot_df = hist.rename(columns={"rain_7day_mm": rain_col_name})
                    f = px.area(plot_df, x="date", y=rain_col_name, labels={rain_col_name: "mm", "date": ""})
                    f.update_traces(name=rain_col_name, showlegend=True)
                else:
                    f = go.Figure()
            st.plotly_chart(style_fig(f, height=270), width="stretch", key="trend_chart")

    with c2, st.container(border=True):
        card_title("🌧️", L("Monthly Rainfall", "மாதாந்திர மழைப்பொழிவு"),
                   f"{L('Monthly average', 'மாதாந்திர சராசரி')} · {loc_name}")
        if monthly is not None and "total_rainfall_mm" in monthly:
            f = px.bar(monthly, x="month_label", y="total_rainfall_mm",
                       labels={"total_rainfall_mm": L("Rainfall (mm)", "மழை (மி.மீ)"), "month_label": ""},
                       color="total_rainfall_mm", color_continuous_scale=["#1f6f8b", "#4cc9f0"])
            f.update_coloraxes(showscale=False)
            st.plotly_chart(style_fig(f, height=300), width="stretch", key="monthly_chart")
        else:
            st.info(L("Monthly summary not found. Run analyze_weather.py to generate it.", "மாதாந்திர சுருக்கம் கிடைக்கவில்லை. analyze_weather.py-ஐ இயக்கவும்."))

    with c3, st.container(border=True):
        st.markdown('<div id="sec-suit"></div>', unsafe_allow_html=True)
        card_title("🎯", L("Sowing Suitability", "விதைப்பு உகந்த தன்மை"))
        if best is None:
            st.warning(L("Not enough data.", "போதுமான தரவு இல்லை."))
        else:
            st.plotly_chart(donut(int(best["Suitability score"])), width="stretch", key="donut")
            bar_rows([(L("Rainfall", "மழைப்பொழிவு"), int(best["Rain score"])),
                      (L("Temperature", "வெப்பநிலை"), int(best["Temp score"])),
                      (L("Dry-spell control", "வறட்சி கட்டுப்பாடு"), int(best["Dry score"])),
                      (L("Excess-rain safety", "அதிக மழை பாதுகாப்பு"), int(best["Excess score"]))])


def reco_banner():
    if best is None:
        return
    st.markdown(f'<div class="reco"><div class="rt">{L("🌱 Recommended Sowing Window", "🌱 பரிந்துரைக்கப்படும் விதைப்பு காலம்")}</div>'
                f'<div class="rd">{best["Start"]:%d %b %Y} – {best["End"]:%d %b %Y}</div>'
                f'<div class="rs">{L("Rule-based score on current weather data · not a confirmed agronomic date", "தற்போதைய வானிலை தரவுகளின் அடிப்படையிலான மதிப்பீடு · வேளாண் அதிகாரிகளிடம் உறுதிப்படுத்தவும்")}</div></div>', unsafe_allow_html=True)


def render_recommendation():
    st.markdown('<div id="sec-reco"></div>', unsafe_allow_html=True)
    st.markdown(f"### {L('🌱 AI Sowing Recommendation', '🌱 AI விதைப்பு பரிந்துரை')}")
    if result.empty:
        st.warning(L("Not enough daily weather data to evaluate a 7-day window.", "7-நாள் காலத்தை மதிப்பிடுவதற்கு போதுமான வானிலை தரவு இல்லை."))
        return
    kpi_row([(L("Suitability score", "உகந்த மதிப்பெண்"), f"{int(best['Suitability score'])}/100", L("Top candidate", "சிறந்த தேர்வு"), True),
             (L("Recommended window", "பரிந்துரைக்கப்பட்ட காலம்"), f"{best['Start']:%d %b} – {best['End']:%d %b}", L("7-day period", "7-நாள் காலம்")),
             (L("7-day rainfall", "7-நாள் மழைப்பொழிவு"), f"{best['Rainfall (mm)']} mm", f"{int(best['Rainy days'])} {L('rainy days', 'மழை நாட்கள்')}"),
             (L("Avg temperature", "சராசரி வெப்பநிலை"), f"{best['Average temperature (°C)']} °C", f"{L('Ideal', 'உகந்தது')} {CI['temp_ideal']} °C")])
    t1, t2, t3, t4 = st.tabs([L("Summary", "சுருக்கம்"), L("Why this window", "ஏன் இந்த காலம்?"),
                              L("Compare windows", "காலங்களை ஒப்பிடு"), L("Experimental ML", "சோதனை ML மாதிரி")])
    with t1:
        st.success(L(f"Primary candidate: {best['Start']:%d %B} – {best['End']:%d %B} · Score {int(best['Suitability score'])}/100",
                     f"முதன்மை தேர்வு: {best['Start']:%d %B} – {best['End']:%d %B} · மதிப்பெண் {int(best['Suitability score'])}/100"))
        if second is not None:
            st.info(L(f"Alternative candidate: {second['Start']:%d %B} – {second['End']:%d %B} · Score {int(second['Suitability score'])}/100",
                      f"மாற்றுத் தேர்வு: {second['Start']:%d %B} – {second['End']:%d %B} · மதிப்பெண் {int(second['Suitability score'])}/100"))
        risks = best["Risks"].split("; ")
        chips(risks, "ok" if (risks[0].startswith("No major") or risks[0].startswith("முக்கிய")) else "warn")
        st.caption(L("This is the highest-ranked candidate in the available data, not a confirmed agronomic sowing date.",
                     "இது கிடைக்கக்கூடிய தரவுகளில் அதிக மதிப்பெண் பெற்ற காலமாகும்; உறுதிப்படுத்தப்பட்ட வேளாண் விதைப்பு தேதி அல்ல."))
    with t2:
        r = []
        r.append(("ok", L("✔ Estimated 7-day rainfall meets the prototype's minimum threshold.", "✔ 7-நாள் மழைப்பொழிவு குறைந்தபட்ச தேவையை பூர்த்தி செய்கிறது."))
                 if float(best["Rainfall (mm)"]) >= CI["rain_min"]
                 else ("warn", L("⚠ Rainfall is below the minimum threshold; moisture stress may be a concern.", "⚠ மழைப்பொழிவு குறைந்தபட்ச அளவை விடக் குறைவாக உள்ளது; ஈரப்பதம் பற்றாக்குறை ஏற்படலாம்.")))
        r.append(("ok", L("✔ Average temperature is near the crop's configured ideal.", "✔ சராசரி வெப்பநிலை பயிரின் உகந்த அளவிற்கு அருகில் உள்ளது."))
                 if abs(float(best["Average temperature (°C)"]) - CI["temp_ideal"]) <= 3
                 else ("warn", L("⚠ Average temperature differs from the crop's configured ideal.", "⚠ சராசரி வெப்பநிலை பயிரின் உகந்த அளவிலிருந்து மாறுபடுகிறது.")))
        r.append(("warn", L("⚠ A long dry spell is detected in this candidate window.", "⚠ இந்த காலத்தில் நீண்ட வறட்சி கண்டறியப்பட்டுள்ளது."))
                 if int(best["Longest dry spell (days)"]) >= 5
                 else ("ok", L("✔ The longest dry spell is relatively short in this window.", "✔ இந்த காலத்தில் தொடர் வறட்சி நாட்கள் குறைவாகவே உள்ளன.")))
        for k, t_txt in r:
            chips([t_txt], k)
    with t3:
        cdf = result.head(10).copy()
        cdf["Window"] = cdf.apply(lambda x: f"{x['Start']:%d %b}–{x['End']:%d %b}", axis=1)
        f = px.bar(cdf, x="Window", y="Suitability score", range_y=[0, 100], color="Suitability score",
                   color_continuous_scale=["#ff7b72", "#f5c04a", "#65e6a0"], range_color=[0, 100],
                   labels={"Window": L("Window", "காலம்"), "Suitability score": L("Suitability score", "உகந்த மதிப்பெண்")},
                   hover_data=["Rainfall (mm)", "Average temperature (°C)", "Longest dry spell (days)"])
        f.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(f, title=L("Top candidate windows", "சிறந்த விதைப்பு காலங்கள்")), width="stretch", key="cmp_chart")
    with t4:
        label, conf = predict_sowing_suitability(float(best["Rainfall (mm)"]), float(best["Average temperature (°C)"]), int(best["Longest dry spell (days)"]))
        if label is None:
            st.warning(L("The demonstration model file was not found.", "மாதிரி AI கோப்பு (model file) கிடைக்கவில்லை."))
        else:
            kpi_row([(L("Demo model output", "மாதிரி AI முடிவு"), label),
                     (L("Experimental score (not validated)", "சோதனை மதிப்பெண் (உறுதிப்படுத்தப்படாதது)"), f"{conf:.1f}%")])
        st.caption(L("The model was trained on synthetic demo data and is not validated for real-world agricultural decisions.",
                     "இந்த மாதிரி செயற்கை தரவுகளில் பயிற்சி அளிக்கப்பட்டது; நேரடி விவசாய முடிவுகளுக்கு உறுதிப்படுத்தப்படவில்லை."))


def render_simulator():
    st.markdown(f"### {L('🔮 Climate twin simulator', '🔮 காலநிலை மாற்று சிமுலேட்டர்')}")
    st.write(L("Test hypothetical changes in rainfall and temperature. This does not predict actual future climate.",
               "மழை மற்றும் வெப்பநிலையில் ஏற்படும் மாற்றங்களை மாதிரியாகக் கொண்டு சோதித்துப் பாருங்கள். இது எதிர்கால வானிலை முன்னறிவிப்பு அல்ல."))
    x, y = st.columns(2)
    rainfall_change = x.slider(L("Rainfall change (%)", "மழைப்பொழிவு மாற்றம் (%)"), -50, 50, 0, 5)
    temperature_change = y.slider(L("Temperature change (°C)", "வெப்பநிலை மாற்றம் (°C)"), -5.0, 5.0, 0.0, 0.5)
    sim = weather.copy()
    sim["rain_mm"] = (sim["rain_mm"] * (1 + rainfall_change / 100)).clip(lower=0)
    sim["tmax_c"] += temperature_change
    sim["tmin_c"] += temperature_change
    sr = evaluate_windows(sim, CI)
    sim_best = sr.iloc[0] if not sr.empty else None
    sim_score = int(sim_best["Suitability score"]) if sim_best is not None else 0
    orig = int(best["Suitability score"]) if best is not None else 0
    q1, q2, q3 = st.columns(3)
    q1.metric(L("Original score", "அசல் மதிப்பெண்"), f"{orig}/100")
    q2.metric(L("Simulated score", "சிமுலேட்டர் மதிப்பெண்"), f"{sim_score}/100")
    q3.metric(L("Score change", "மதிப்பெண் மாற்றம்"), f"{sim_score - orig:+d}", delta=f"{sim_score - orig:+d} {L('points', 'புள்ளிகள்')}")
    if sim_best is not None:
        st.info(L(f"Under this hypothetical scenario, the top-ranked window is {sim_best['Start']} to {sim_best['End']}.",
                  f"இந்த கற்பனை சூழலில், சிறந்த விதைப்பு காலம் {sim_best['Start']} முதல் {sim_best['End']} வரை ஆகும்."))
    st.caption(L("All simulator thresholds are illustrative and not validated for every crop or district.",
                 "சிமுலேட்டர் வரம்புகள் அனைத்தும் மாதிரி அளவுகளே."))


# =====================================================================
# PAGES (4 MAIN PAGES + 6 "MORE" PAGES)
# =====================================================================
if page == "Dashboard":
    kpi_row([(L("Forecast rainfall", "முன்னறிவிப்பு மழை"), f"{weather_total_rain:.1f} mm", f"{len(weather)} {L('days', 'நாட்கள்')}"),
             (L("Average max temp", "சராசரி அதிகபட்ச வெப்பம்"), f"{avg_max_temp:.1f} °C"),
             (L("Rainy days", "மழை நாட்கள்"), rainy_days),
             (L("Dry days", "வறண்ட நாட்கள்"), dry_days)])
    render_current_weather()
    render_forecast_and_alerts()
    render_trends_row()
    reco_banner()
    st.markdown("---")
    render_recommendation()
    with st.expander(L("How this prototype works / limitations", "இந்த மாதிரி எவ்வாறு செயல்படுகிறது / வரம்புகள்")):
        st.write(L(
            "It ranks rolling 7-day candidate windows using a transparent weighted score (rainfall 35%, temperature 30%, dry-spell 20%, excess-rain 15%). Forecast data may fall back to clearly labelled synthetic demo data. Crop thresholds are simplified placeholders and require local agronomic validation. Soil measurements and historical yield records are not included in the score.",
            "இது 7-நாள் வானிலை காலங்களை (மழைப்பொழிவு 35%, வெப்பநிலை 30%, வறட்சி கட்டுப்பாடு 20%, அதிக மழை பாதுகாப்பு 15%) அடிப்படையாகக் கொண்டு மதிப்பிடுகிறது. நேரடி வானிலை கிடைக்காதபோது செயற்கை மாதிரி தரவு பயன்படுத்தப்படும். பயிர் வரம்புகள் மாதிரி அளவுகளே என்பதால் உள்ளூர் வேளாண் அலுவலர்களின் ஆலோசனை அவசியம். மண் பரிசோதனை மற்றும் மகசூல் தரவுகள் இந்த மதிப்பெண்ணில் சேர்க்கப்படவில்லை."
        ))

elif page == "Sowing Recommendation":
    page_header(L("Recommendation", "பரிந்துரை"), L("Sowing recommendation", "விதைப்பு பரிந்துரை"),
                f"{crop_name} · {loc_name} · {L('Soil', 'மண்')}: {soil_name}")
    reco_banner()
    render_recommendation()
    st.markdown(f"### {L('Crop thresholds used by this prototype', 'இந்த மாதிரியில் பயன்படுத்தப்படும் பயிர் வானிலை வரம்புகள்')}")
    kpi_row([(L("Minimum rainfall", "குறைந்தபட்ச மழை"), f"{CI['rain_min']} mm"),
             (L("Ideal rainfall", "உகந்த மழை"), f"{CI['rain_ideal']} mm"),
             (L("Minimum temperature", "குறைந்தபட்ச வெப்பம்"), f"{CI['temp_min']} °C"),
             (L("Ideal temperature", "உகந்த வெப்பம்"), f"{CI['temp_ideal']} °C")])
    st.caption(L("These are illustrative thresholds, not validated local agricultural guidance.",
                 "இவை மாதிரி வரம்புகள் மட்டுமே; உள்ளூர் வேளாண் துறையால் உறுதிப்படுத்தப்பட்ட வழிகாட்டுதல்கள் அல்ல."))

elif page == "Insights & Alerts":
    page_header(L("Insights", "நுண்ணறிவு"), L("Insights & alerts", "எச்சரிக்கைகள் & நுண்ணறிவு"), "")
    st.markdown("".join(alert_html(a) for a in build_alerts()), unsafe_allow_html=True)
    if best is not None:
        st.markdown(f"### {L('Risk indicators for the top window', 'சிறந்த விதைப்பு காலத்திற்கான இடர் குறிகாட்டிகள்')}")
        kpi_row([(L("Rainfall (7 days)", "மழைப்பொழிவு (7 நாட்கள்)"), f"{best['Rainfall (mm)']} mm"),
                 (L("Longest dry spell", "நீண்ட வறட்சி நாட்கள்"), f"{int(best['Longest dry spell (days)'])} {L('day(s)', 'நாட்கள்')}"),
                 (L("Rainy days", "மழை நாட்கள்"), int(best["Rainy days"]))])
        risks = best["Risks"].split("; ")
        chips(risks, "ok" if (risks[0].startswith("No major") or risks[0].startswith("முக்கிய")) else "warn")
    st.markdown("---")
    render_simulator()

elif page == "About":
    page_header(L("About", "பற்றி"), L("About SowingSense AI", "SowingSense AI பற்றி"), "")
    st.write(L("SowingSense AI is a climate-aware crop sowing recommendation prototype designed to explore how weather conditions may support sowing-window decisions.",
               "SowingSense AI என்பது வானிலை மற்றும் காலநிலை தரவுகளின் அடிப்படையில் பயிர் விதைப்புக்கு ஏற்ற நாட்களைத் தேர்ந்தெடுக்க உதவும் ஒரு மாதிரித் தளமாகும்."))
    st.markdown(f"### {L('Current capabilities', 'தற்போதைய சிறப்பம்சங்கள்')}")
    st.write(L(
        "- Open-Meteo current conditions and short-range forecast when available\n- Clearly labelled synthetic fallback data for offline demonstrations\n- Crop-specific, rule-based ranking of rolling seven-day weather windows\n- Experimental ML model output, climate trends, risk alerts and CSV exports",
        "- Open-Meteo மூலம் நிகழ்நேர வானிலை மற்றும் குறுகிய கால முன்னறிவிப்பு\n- ஆஃப்லைன் பயன்பாட்டிற்குத் தெளிவாகக் குறிப்பிடப்பட்ட செயற்கை மாதிரி தரவு\n- ஒவ்வொரு பயிருக்கும் ஏற்ற 7-நாள் வானிலை காலங்களை தரவரிசைப்படுத்தும் முறை\n- சோதனை ML மாதிரி முடிவுகள், காலநிலை போக்குகள், வானிலை எச்சரிக்கைகள் மற்றும் CSV பதிவிறக்கம்"
    ))
    st.markdown(f"### {L('Limitations', 'வரம்புகள்')}")
    st.warning(L("Scores are illustrative and are not a guarantee of yield or a substitute for local agricultural advice. Crop thresholds are simplified placeholders. The demo model was trained on synthetic data and requires validation with real agricultural data. Soil type is not currently used in the score.",
                 "இதில் உள்ள மதிப்பெண்கள் மாதிரி அளவுகளே; இவை மகசூலுக்கான உத்தரவாதமோ அல்லது உள்ளூர் வேளாண் நிபுணர்களின் ஆலோசனைக்கு மாற்றோ அல்ல. மண் வகை தற்போது மதிப்பெண் கணக்கீட்டில் சேர்க்கப்படவில்லை."))
    st.caption(L("SowingSense AI · Smarter Sowing. Healthier Tomorrow.", "SowingSense AI · திறமையான விதைப்பு. வளமான எதிர்காலம்."))

elif page == "Weather & Climate":
    page_header(L("Weather", "வானிலை"), L("Weather & climate", "வானிலை மற்றும் காலநிலை"), f"{loc_name} · {data_label}")
    render_current_weather()
    kpi_row([(L("Total rainfall", "மொத்த மழைப்பொழிவு"), f"{weather_total_rain:.1f} mm"),
             (L("Average maximum", "சராசரி அதிகபட்ச வெப்பம்"), f"{avg_max_temp:.1f} °C"),
             (L("Rainy days", "மழை நாட்கள்"), rainy_days),
             (L("Dry days", "வறண்ட நாட்கள்"), dry_days)])
    render_forecast_and_alerts()
    st.markdown(f"### {L('Daily weather records', 'தினசரி வானிலை பதிவுகள்')}")
    disp = weather.copy()
    disp["date"] = disp["date"].dt.strftime("%d %b %Y")
    if is_ta():
        disp = disp.rename(columns={"date": "தேதி", "rain_mm": "மழை (மி.மீ)", "tmax_c": "அதிகபட்ச வெப்பம் (°C)", "tmin_c": "குறைந்தபட்ச வெப்பம் (°C)"})
    st.dataframe(disp, width="stretch", hide_index=True)

elif page == "Crop Information":
    page_header(L("Crops", "பயிர்கள்"), L("Crop information", "பயிர் தகவல்கள்"),
                L("Prototype crop settings. Confirm variety-specific guidance with the local agriculture department.",
                  "மாதிரி பயிர் அமைப்புகள். பயிர் ரகத்திற்கு ஏற்ற வழிகாட்டுதலை உள்ளூர் வேளாண் துறையிடம் உறுதிப்படுத்தவும்."))
    rows = [{
        L("Crop", "பயிர்"): CROP_TA.get(n, n) if is_ta() else n,
        L("Minimum rainfall (mm)", "குறைந்தபட்ச மழை (மி.மீ)"): i["rain_min"],
        L("Ideal rainfall (mm)", "உகந்த மழை (மி.மீ)"): i["rain_ideal"],
        L("Minimum temperature (°C)", "குறைந்தபட்ச வெப்பம் (°C)"): i["temp_min"],
        L("Ideal temperature (°C)", "உகந்த வெப்பம் (°C)"): i["temp_ideal"],
        L("Waterlogging sensitive", "நீர் தேக்கத்தால் பாதிப்பு"): (L("Yes", "ஆம்") if i["waterlogging_sensitive"] else L("No", "இல்லை")) if is_ta() else i["waterlogging_sensitive"]
    } for n, i in CROPS.items()]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.markdown(f"### {crop_name}")
    kpi_row([(L("Minimum rainfall", "குறைந்தபட்ச மழை"), f"{CI['rain_min']} mm"),
             (L("Ideal rainfall", "உகந்த மழை"), f"{CI['rain_ideal']} mm"),
             (L("Ideal temperature", "உகந்த வெப்பம்"), f"{CI['temp_ideal']} °C")])
    (st.warning if CI["waterlogging_sensitive"] else st.info)(
        L("The prototype marks this crop as waterlogging-sensitive. Check field drainage and local guidance.",
          "இந்த பயிர் நீர் தேக்கத்தால் பாதிக்கப்படக்கூடியது எனக் குறிக்கப்பட்டுள்ளது. வயல் வடிகால் வசதியை சரிபார்க்கவும்.")
        if CI["waterlogging_sensitive"]
        else L("The prototype does not flag this crop as waterlogging-sensitive.",
               "இந்த பயிர் நீர் தேக்கத்தால் எளிதில் பாதிக்கப்படாது எனக் குறிக்கப்பட்டுள்ளது.")
    )

elif page == "Soil Analysis":
    page_header(L("Soil", "மண்"), L("Soil analysis", "மண் பகுப்பாய்வு"),
                L("Soil type is captured as farm context; it does not currently change the suitability score.",
                  "மண் வகை பண்ணைத் தகவலுக்காகப் பெறப்படுகிறது; இது தற்போது விதைப்பு மதிப்பெண்ணை மாற்றுவதில்லை."))
    notes_en = {
        "Not specified": "Add a soil type or obtain a soil test for a more informed decision.",
        "Clay": "Clay soils can retain more water; check drainage and avoid field operations when soil is excessively wet.",
        "Loam": "Loam often balances water retention and drainage, but field conditions and soil tests still matter.",
        "Sandy": "Sandy soils usually drain quickly; monitor moisture and irrigation availability.",
        "Black soil": "Black soils can retain substantial moisture; check cracking, drainage and local crop guidance.",
    }
    notes_ta = {
        "Not specified": "சிறந்த முடிவுக்கு மண் வகையைத் தேர்ந்தெடுக்கவும் அல்லது மண் பரிசோதனை செய்யவும்.",
        "Clay": "களிமண் நீரை அதிகம் தக்கவைக்கும்; வடிகால் வசதியை சரிபார்க்கவும் மற்றும் மண் அதிக ஈரமாக இருக்கும்போது உழவுப் பணிகளைத் தவிர்க்கவும்.",
        "Loam": "வண்டல் மண் நீர் பிடிப்பு மற்றும் வடிகால் வசதியை சமநிலையில் வைத்திருக்கும்; எனினும் மண் பரிசோதனை அவசியம்.",
        "Sandy": "மணல் மண் நீரை விரைவாக வடியச் செய்யும்; ஈரப்பதம் மற்றும் நீர்ப்பாசன வசதியைக் கண்காணிக்கவும்.",
        "Black soil": "கரிசல் மண் அதிக ஈரப்பதத்தைத் தக்கவைக்கும்; மண் வெடிப்பு, வடிகால் மற்றும் உள்ளூர் பயிர் ஆலோசனையைப் பின்பற்றவும்.",
    }
    st.info(f"**{soil_name}:** {(notes_ta if is_ta() else notes_en)[soil]}")
    st.markdown(f"### {L('What to record before sowing', 'விதைப்புக்கு முன் கவனிக்க வேண்டியவை')}")
    st.write(L(
        "- Soil pH and electrical conductivity from a soil test\n- Available nitrogen, phosphorus and potassium\n- Drainage, field moisture and irrigation access",
        "- மண் பரிசோதனை மூலம் மண்ணின் கார-அமிலத் தன்மை (pH) மற்றும் மின் கடத்துத்திறன் (EC)\n- மண்ணில் உள்ள தழைச்சத்து (N), மணிச்சத்து (P), சாம்பல் சத்து (K) அளவு\n- வடிகால் வசதி, மண்ணின் ஈரப்பதம் மற்றும் பாசன நீர் இருப்பு"
    ))
    st.warning(L("No laboratory soil measurements are collected by this prototype. These notes are general guidance, not a soil diagnosis.",
                 "இந்த மாதிரியில் ஆய்வக மண் பரிசோதனை அளவீடுகள் சேகரிக்கப்படவில்லை. இவை பொதுவான வழிகாட்டுதல்களே."))

elif page == "Climate Trends":
    page_header(L("History", "வரலாறு"), L("Historical climate trends", "வரலாற்று காலநிலை போக்குகள்"), f"{loc_name}")
    hist, monthly = load_hist(lat, lon), format_monthly_df(load_monthly(lat, lon))
    legend_map = get_trend_legend_map()
    if hist is not None:
        st.caption(f"{L('Historical daily weather data · 2024–2025', 'வரலாற்று தினசரி வானிலை தரவு · 2024–2025')} · {loc_name}")
        items = [(L("Weather records", "வானிலை பதிவுகள்"), f"{len(hist):,}")]
        if "avg_temp_c" in hist:
            items.append((L("Average temperature", "சராசரி வெப்பநிலை"), f"{hist['avg_temp_c'].mean():.1f} °C"))
        if "rain_7day_mm" in hist:
            items.append((L("Average 7-day rainfall", "சராசரி 7-நாள் மழை"), f"{hist['rain_7day_mm'].mean():.1f} mm"))
        kpi_row(items)
        raw_cols = [c for c in ["avg_temp_c", "max_temp_7day_c", "min_temp_7day_c"] if c in hist.columns]
        if raw_cols:
            plot_df = hist.rename(columns=legend_map)
            nice_cols = [legend_map[c] for c in raw_cols]
            st.plotly_chart(style_fig(px.line(plot_df, x="date", y=nice_cols, labels={"value": "°C", "date": "", "variable": ""}),
                                      title=f"{L('Temperature trend', 'வெப்பநிலை போக்கு')} — {loc_name}"), width="stretch", key="ct1")
        if "rain_7day_mm" in hist:
            rain_col_name = legend_map["rain_7day_mm"]
            plot_df = hist.rename(columns={"rain_7day_mm": rain_col_name})
            st.plotly_chart(style_fig(px.area(plot_df, x="date", y=rain_col_name, labels={rain_col_name: "mm", "date": ""}),
                                      title=f"{L('7-day rainfall', '7-நாள் மழைப்பொழிவு')} — {loc_name}"), width="stretch", key="ct2")
        st.download_button(L("Download historical climate features", "வரலாற்று காலநிலை தரவுகளைப் பதிவிறக்கு (CSV)"),
                           hist.to_csv(index=False), file_name="sowing_window_features.csv", mime="text/csv")
    else:
        st.warning(L("Historical weather features not found. Run build_features.py first.",
                     "வரலாற்று வானிலை தரவுகள் கிடைக்கவில்லை. முதலில் build_features.py-ஐ இயக்கவும்."))
    if monthly is not None:
        st.markdown(f"### {L('Monthly climate analysis', 'மாதாந்திர காலநிலை பகுப்பாய்வு')} — {loc_name}")
        m1, m2 = st.columns(2)
        if "average_temperature_c" in monthly:
            m1.plotly_chart(style_fig(px.bar(monthly, x="month_label", y="average_temperature_c",
                                             labels={"month_label": "", "average_temperature_c": L("Avg Temp (°C)", "சராசரி வெப்பம் (°C)")}),
                                      title=L("Average temperature by month", "மாதாந்திர சராசரி வெப்பநிலை")), width="stretch", key="ct3")
        if "total_rainfall_mm" in monthly:
            m2.plotly_chart(style_fig(px.bar(monthly, x="month_label", y="total_rainfall_mm",
                                             labels={"month_label": "", "total_rainfall_mm": L("Rainfall (mm)", "மழை (மி.மீ)")}),
                                      title=L("Total rainfall by month", "மாதாந்திர மொத்த மழைப்பொழிவு")), width="stretch", key="ct4")
        disp_m = monthly.drop(columns=["month_label"], errors="ignore")
        st.dataframe(disp_m, width="stretch", hide_index=True)
    else:
        st.info(L("Monthly summary not found. Run analyze_weather.py to generate it.",
                  "மாதாந்திர சுருக்கம் கிடைக்கவில்லை. analyze_weather.py-ஐ இயக்கவும்."))

elif page == "Climate Twin Simulator":
    page_header(L("Simulator", "சிமுலேட்டர்"), L("Climate Twin Simulator — What-If Analysis", "காலநிலை மாற்று சிமுலேட்டர்"), f"{crop_name} · {loc_name}")
    render_simulator()

elif page == "Export Data":
    page_header(L("Export", "பதிவிறக்கம்"), L("Export data", "தரவு பதிவிறக்கம்"),
                L("Download the current candidate-window results and weather data.", "தற்போதைய விதைப்பு பரிந்துரை மற்றும் வானிலை தரவுகளைப் பதிவிறக்கவும்."))
    if not result.empty:
        st.markdown(f"### {L('Candidate sowing windows', 'பரிந்துரைக்கப்படும் விதைப்பு காலங்கள்')}")
        st.dataframe(result.head(20), width="stretch", hide_index=True)
        st.download_button(L("⬇️ Download recommendation report (CSV)", "⬇️ பரிந்துரை அறிக்கையைப் பதிவிறக்கு (CSV)"),
                           result.to_csv(index=False).encode("utf-8"), file_name="sowingsense_recommendations.csv", mime="text/csv", type="primary")
    st.markdown(f"### {L('Weather data', 'வானிலை தரவு')}")
    st.download_button(L("⬇️ Download weather data (CSV)", "⬇️ வானிலை தரவைப் பதிவிறக்கு (CSV)"),
                       weather.to_csv(index=False).encode("utf-8"), file_name="sowingsense_weather.csv", mime="text/csv")
    hist = load_hist(lat, lon)
    if hist is not None:
        st.download_button(L("⬇️ Download historical climate features", "⬇️ வரலாற்று காலநிலை தரவைப் பதிவிறக்கு (CSV)"),
                           hist.to_csv(index=False).encode("utf-8"), file_name="sowing_window_features.csv", mime="text/csv")
    st.info(f"{L('Active source', 'தற்போதைய தரவு மூலம்')}: {data_label}")

st.markdown(f'<div class="foot">{L("🌱 SowingSense AI · Prototype only — verify advice with local agricultural experts.", "🌱 SowingSense AI · மாதிரி அமைப்பு மட்டுமே — உள்ளூர் வேளாண் நிபுணர்களிடம் ஆலோசனை பெறவும்.")}</div>', unsafe_allow_html=True)
