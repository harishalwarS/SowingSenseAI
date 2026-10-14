import math
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).parent / "Data"
WEATHER_FEATURES_PATH = DATA_DIR / "sowing_window_features.csv"
MONTHLY_PATH = DATA_DIR / "monthly_climate_summary.csv"
MODEL_PATH = Path(__file__).parent / "sowing_model_demo.joblib"

st.set_page_config(page_title="SowingSense AI", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")

# =====================================================================
# THEME / CSS
# =====================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
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
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4{font-family:'Inter',system-ui,sans-serif;}
.main .block-container{max-width:1500px;padding:1rem 2rem 4rem;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}
h1,h2,h3,h4{color:#f0fff7 !important;letter-spacing:-.4px;}
p,label,.stMarkdown{color:#d4e6df;}
hr{border-color:var(--line);margin:1.4rem 0;}
::-webkit-scrollbar{width:10px;} ::-webkit-scrollbar-track{background:#06131c;}
::-webkit-scrollbar-thumb{background:linear-gradient(#1c8f58,#65e6a0);border-radius:10px;}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#06121b 0%,#081f1e 100%);border-right:1px solid var(--line);box-shadow:8px 0 40px rgba(34,197,94,.06);}
[data-testid="stSidebar"] [role="radiogroup"]{gap:6px;}
[data-testid="stSidebar"] [role="radiogroup"] label{
  border-radius:12px;padding:11px 14px;width:100%;border:1px solid transparent;
  background:rgba(255,255,255,.015);transition:all .22s ease;}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child{display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(101,230,160,.10);transform:translateX(5px);border-color:rgba(101,230,160,.35);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
  background:linear-gradient(135deg,#1a9a5c,#22c55e);border-color:#6ff0ae;
  box-shadow:0 0 22px rgba(34,197,94,.45), inset 0 0 12px rgba(255,255,255,.12);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:#fff !important;font-weight:800;}
.brand{display:flex;align-items:center;gap:10px;padding:6px 4px 16px;}
.brand .logo{font-size:1.9rem;filter:drop-shadow(0 0 10px rgba(101,230,160,.8));}
.brand .name{font-size:1.35rem;font-weight:900;color:#fff;letter-spacing:-.3px;}
.brand .name b{color:var(--g);text-shadow:0 0 14px rgba(101,230,160,.7);}
.brand .tag{font-size:.68rem;color:#a9c8bd;letter-spacing:.4px;}

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
.main [data-testid="stRadio"] label{border-radius:10px;padding:6px 16px;background:rgba(255,255,255,.04);border:1px solid rgba(101,230,160,.2);transition:all .2s ease;}
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


/* ---------- Top navigation (replaces sidebar) ---------- */
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none !important;}
.st-key-topnav{position:sticky;top:.6rem;z-index:999;background:rgba(6,19,28,.88);backdrop-filter:blur(16px);
  border:1px solid rgba(101,230,160,.30);border-radius:18px;padding:10px 18px;margin-bottom:16px;
  box-shadow:0 0 34px rgba(34,197,94,.16),0 10px 30px rgba(0,0,0,.35);animation:rise .5s ease both;}
.st-key-topnav [data-testid="stRadio"] [role="radiogroup"]{gap:8px;justify-content:flex-end;}
.st-key-topnav [data-testid="stRadio"] label{padding:9px 15px;border-radius:12px;}
.st-key-topnav [data-testid="stRadio"] label p{font-size:.86rem;white-space:nowrap;font-weight:700;}
.st-key-topnav [data-testid="stRadio"] label:hover{transform:translateY(-3px) scale(1.04);}
.st-key-topnav [data-testid="stRadio"] label:has(input:checked){box-shadow:0 0 26px rgba(34,197,94,.75),inset 0 0 12px rgba(255,255,255,.18);}
.st-key-topnav .brand{padding:2px 0;}
.foot{text-align:center;color:#7fa196;font-size:.78rem;margin-top:2.5rem;padding-top:1rem;border-top:1px solid rgba(101,230,160,.14);}
@media (max-width:900px){.st-key-topnav{position:relative;top:0;}}

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
                      font=dict(family="Inter, sans-serif", color="#d4e6df"),
                      legend=dict(orientation="h", y=-0.18, x=0, title_text=""),
                      hoverlabel=dict(bgcolor="#0b222d", font_size=13))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,190,175,.12)")
    return fig


def donut(score):
    color = GREEN if score >= 70 else AMBER if score >= 45 else RED
    fig = go.Figure(go.Pie(values=[score, 100 - score], hole=.76, sort=False, direction="clockwise",
                           marker=dict(colors=[color, "rgba(255,255,255,.07)"], line=dict(width=0)),
                           textinfo="none", hoverinfo="skip"))
    fig.update_layout(height=210, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      annotations=[dict(text=f"<b style='font-size:30px'>{score}%</b><br><span style='font-size:12px;color:#a8c9bd'>Suitable<br>for sowing</span>",
                                        x=.5, y=.5, showarrow=False, font=dict(color="#fff"))])
    return fig


def bar_rows(items):
    html = ""
    for label, val in items:
        html += (f'<div class="bar-row"><div class="bar-top"><span>{label}</span><span style="color:#65e6a0">{val}%</span></div>'
                 f'<div class="bar-bg"><div class="bar-fg" style="width:{val}%"></div></div></div>')
    st.markdown(html, unsafe_allow_html=True)


# =====================================================================
# DATA / MODEL
# =====================================================================
@st.cache_resource
def load_sowing_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


sowing_model = load_sowing_model()

CROPS = {
    "Paddy": {"rain_min": 60, "rain_ideal": 100, "temp_min": 22, "temp_ideal": 29, "waterlogging_sensitive": False},
    "Maize": {"rain_min": 25, "rain_ideal": 60, "temp_min": 18, "temp_ideal": 27, "waterlogging_sensitive": True},
    "Groundnut": {"rain_min": 20, "rain_ideal": 50, "temp_min": 22, "temp_ideal": 30, "waterlogging_sensitive": True},
    "Millet": {"rain_min": 15, "rain_ideal": 40, "temp_min": 24, "temp_ideal": 32, "waterlogging_sensitive": True},
    "Cotton": {"rain_min": 25, "rain_ideal": 65, "temp_min": 21, "temp_ideal": 30, "waterlogging_sensitive": True},
}
DISTRICTS = {
    "Thoothukudi": (8.7642, 78.1348), "Tirunelveli": (8.7139, 77.7567), "Madurai": (9.9252, 78.1198),
    "Chennai": (13.0827, 80.2707), "Coimbatore": (11.0168, 76.9558), "Thanjavur": (10.7870, 79.1378),
    "Salem": (11.6643, 78.1460),
}
SOILS = ["Not specified", "Clay", "Loam", "Sandy", "Black soil"]
SRC_LIVE, SRC_DEMO = "Live forecast (Open-Meteo)", "Demo data (offline)"

WX_CODES = {0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"), 3: ("Overcast", "☁️"),
            45: ("Fog", "🌫️"), 48: ("Fog", "🌫️"), 51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"), 55: ("Heavy drizzle", "🌧️"),
            61: ("Light rain", "🌧️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"), 80: ("Rain showers", "🌦️"),
            81: ("Rain showers", "🌧️"), 82: ("Violent showers", "⛈️"), 95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm", "⛈️"), 99: ("Thunderstorm", "⛈️")}


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
    import requests
    params = {"latitude": lat, "longitude": lon, "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
              "timezone": "auto", "forecast_days": 16}
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=8)
    r.raise_for_status()
    d = r.json().get("daily", {})
    df = pd.DataFrame({"date": pd.to_datetime(d.get("time", [])), "rain_mm": d.get("precipitation_sum", []),
                       "tmax_c": d.get("temperature_2m_max", []), "tmin_c": d.get("temperature_2m_min", [])}).dropna()
    if df.empty:
        raise ValueError("No forecast data returned")
    return df


@st.cache_data(ttl=60, show_spinner=False)
def fetch_current_weather(lat, lon):
    import requests
    params = {"latitude": lat, "longitude": lon, "timezone": "auto",
              "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,wind_speed_10m,wind_direction_10m,weather_code"}
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    r.raise_for_status()
    cur = r.json().get("current")
    if not cur:
        raise ValueError("Current weather unavailable")
    return cur


@st.cache_data(show_spinner=False)
def load_hist():
    return pd.read_csv(WEATHER_FEATURES_PATH, parse_dates=["date"]) if WEATHER_FEATURES_PATH.exists() else None


@st.cache_data(show_spinner=False)
def load_monthly():
    return pd.read_csv(MONTHLY_PATH) if MONTHLY_PATH.exists() else None


def predict_sowing_suitability(rain, temp, dry):
    if sowing_model is None:
        return None, None
    f = pd.DataFrame([{"rainfall_mm": rain, "temperature_c": temp, "dry_spell_days": dry}])
    pred = int(sowing_model.predict(f)[0])
    conf = float(max(sowing_model.predict_proba(f)[0]) * 100)
    return ("Favourable (demo model)" if pred == 1 else "Less favourable (demo model)"), conf


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
    rain = float(window["rain_mm"].sum())
    avg_temp = float(((window["tmax_c"] + window["tmin_c"]) / 2).mean())
    rainy = int((window["rain_mm"] >= 1.0).sum())
    dry_spell = max_consecutive_dry(window["rain_mm"].tolist())
    rain_score = min(100, rain / ci["rain_ideal"] * 100)
    if rain < ci["rain_min"]:
        rain_score *= 0.55
    temp_score = max(0, 100 - abs(avg_temp - ci["temp_ideal"]) * 8)
    if avg_temp < ci["temp_min"]:
        temp_score *= 0.6
    dry_score = max(0, 100 - dry_spell * 12)
    excess_score = max(20, 100 - max(0, rain - ci["rain_ideal"]) * 0.8)
    score = round(0.35 * rain_score + 0.30 * temp_score + 0.20 * dry_score + 0.15 * excess_score)
    risks = []
    if rain < ci["rain_min"]:
        risks.append("Rainfall may be insufficient")
    if dry_spell >= 5:
        risks.append("Possible prolonged dry spell")
    if avg_temp > ci["temp_ideal"] + 3:
        risks.append("Temperature may be high")
    if rain > ci["rain_ideal"] * 1.4:
        risks.append("Excess rainfall / waterlogging risk")
    if not risks:
        risks.append("No major risk flagged by this simple prototype")
    return {"score": int(max(0, min(100, score))), "rain_mm": round(rain, 1), "avg_temp": round(avg_temp, 1),
            "dry_spell": dry_spell, "rainy_days": rainy, "risks": risks,
            "parts": [int(round(rain_score)), int(round(temp_score)), int(round(dry_score)), int(round(excess_score))]}


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
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(round(float(deg) / 45)) % 8]


# =====================================================================
# TOP NAV
# =====================================================================
PAGES = {"Dashboard": "🏠", "Sowing Recommendation": "🌱", "Weather & Climate": "⛅", "Crop Information": "🌾",
         "Soil Analysis": "🧪", "Climate Trends": "📈", "Insights & Alerts": "🔔", "Export Data": "📤", "About": "ℹ️"}

try:
    nav_box = st.container(key="topnav")
except TypeError:  # older Streamlit without container keys
    nav_box = st.container()
with nav_box:
    nb = st.columns([1.3, 6])
    nb[0].markdown('<div class="brand"><div class="logo">🌱</div><div><div class="name">Sowing<b>Sense</b> AI</div>'
                   '<div class="tag">Smarter Sowing. Healthier Tomorrow.</div></div></div>', unsafe_allow_html=True)
    page = nb[1].radio("Navigation", list(PAGES.keys()), format_func=lambda p: f"{PAGES[p]}  {p}",
                       horizontal=True, key="nav", label_visibility="collapsed")

# =====================================================================
# HERO (dashboard) + FARM BAR (all pages)
# =====================================================================
is_live_sel = st.session_state.get("source", SRC_LIVE) == SRC_LIVE

if page == "Dashboard":
    st.markdown(f"""
<div class="hero"><div class="hero-top"><div>
<div class="title"><span class="grad">SowingSense</span><span class="ai-badge">AI</span></div>
<div class="sub">AI-Powered Climate Intelligence for Smarter Sowing Decisions</div>
<div class="tags">Weather insights &nbsp;•&nbsp; Crop analysis &nbsp;•&nbsp; Sustainable farming</div>
</div><div class="hero-right">
<span class="pill">📅 {date.today():%a, %d %b %Y}</span>
<span class="pill {'live' if is_live_sel else 'demo'}"><span class="dot"></span>{'Live Data' if is_live_sel else 'Demo Data'}</span>
</div></div>
<div class="feat-row">
<div class="feat"><span class="ico">🌦️</span><div><div class="t">Live Weather Data</div><div class="s">Open-Meteo API</div></div></div>
<div class="feat"><span class="ico">🤖</span><div><div class="t">ML-based Crop Analysis</div><div class="s">Climate + Weather</div></div></div>
<div class="feat"><span class="ico">📊</span><div><div class="t">Climate Trends</div><div class="s">Historical Analysis</div></div></div>
<div class="feat"><span class="ico">🍃</span><div><div class="t">Sustainable Farming</div><div class="s">Better Future</div></div></div>
</div></div>
<div class="jump"><a href="#sec-weather">⛅ Live weather</a><a href="#sec-forecast">📅 Forecast</a><a href="#sec-trends">📈 Trends</a><a href="#sec-suit">🎯 Suitability</a><a href="#sec-reco">🌱 AI recommendation</a></div>
""", unsafe_allow_html=True)

with st.container(border=True):
    card_title("📍", "Select Farm Details", "Choose location, crop and analysis settings to get AI-powered sowing recommendations")
    fc = st.columns([1.2, 1.1, 1.5, 1.1, 1.1, 1.35])
    district = fc[0].selectbox("District / Location", list(DISTRICTS.keys()), key="district")
    crop = fc[1].selectbox("Crop", list(CROPS.keys()), key="crop")
    source = fc[2].selectbox("Climate data source", [SRC_LIVE, SRC_DEMO], key="source")
    start_day = fc[3].date_input("Start date for analysis", value=date.today(), key="start_day")
    soil = fc[4].selectbox("Soil type (optional)", SOILS, index=2, key="soil")
    fc[5].markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    run = fc[5].button("🌱 Generate Recommendation", type="primary", use_container_width=True)
    auto_refresh = st.toggle("⚡ Auto-refresh live weather every 60 s", value=True, key="auto_refresh")

lat, lon = DISTRICTS[district]
config_key = (district, source, start_day)
stale = source == SRC_LIVE and (time.time() - st.session_state.get("fetched_at", 0)) > 900
if run or stale or "weather_data" not in st.session_state or st.session_state.get("weather_config") != config_key:
    with st.spinner("Analysing climate data…"):
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
        st.toast("Recommendation updated ✨", icon="🌱")

weather = st.session_state["weather_data"].copy()
weather["date"] = pd.to_datetime(weather["date"])
weather = weather.sort_values("date").reset_index(drop=True)
CI = CROPS[crop]
result = evaluate_windows(weather, CI)
weather_total_rain = float(weather["rain_mm"].sum())
avg_max_temp = float(weather["tmax_c"].mean())
rainy_days = int((weather["rain_mm"] >= 1).sum())
dry_days = int((weather["rain_mm"] < 1).sum())
data_label = st.session_state.get("data_source_label", "Demo data")
best = result.iloc[0] if not result.empty else None
second = result.iloc[1] if len(result) > 1 else best


# =====================================================================
# SECTION RENDERERS
# =====================================================================
def _weather_card(district, lat, lon):
    with st.container(border=True):
        top = st.columns([3, 1])
        with top[0]:
            card_title("🌡️", f"Current Live Weather — {district}", "Open-Meteo real-time conditions")
        if top[1].button("🔄 Refresh", key="refresh_wx", use_container_width=True):
            fetch_current_weather.clear()
            st.toast("Weather refreshed", icon="🔄")
        try:
            c = fetch_current_weather(lat, lon)
            desc, emo = WX_CODES.get(int(c.get("weather_code", 0)), ("Conditions", "🌡️"))
            if not c.get("is_day", 1) and emo == "☀️":
                emo = "🌙"
            st.markdown(
                f'<div class="wx"><div class="wx-main"><div class="wx-emoji">{emo}</div><div>'
                f'<div class="wx-temp">{c.get("temperature_2m", "—")} °C</div><div class="wx-desc">{desc}</div>'
                f'<div class="wx-feel">Feels like {c.get("apparent_temperature", "—")} °C</div></div></div>'
                f'<div class="wx-grid">'
                f'<div class="tile"><div class="ti">💧</div><div class="tv">{c.get("relative_humidity_2m", "—")}%</div><div class="tl">Humidity</div></div>'
                f'<div class="tile"><div class="ti">🌧️</div><div class="tv">{c.get("rain", "—")} mm</div><div class="tl">Rainfall</div></div>'
                f'<div class="tile"><div class="ti">💨</div><div class="tv">{c.get("wind_speed_10m", "—")} km/h</div><div class="tl">Wind speed</div></div>'
                f'<div class="tile"><div class="ti">🧭</div><div class="tv">{compass(c.get("wind_direction_10m", 0))}</div><div class="tl">Wind direction</div></div>'
                f'</div></div>', unsafe_allow_html=True)
            auto = " · 🟢 auto-updating every 60 s" if st.session_state.get("auto_refresh", True) else ""
            st.caption(f"Observation: {c.get('time', 'Unavailable')} · Last checked {datetime.now():%H:%M:%S}{auto}")
        except Exception:
            st.warning("Live weather is currently unavailable. Current-condition cards are not filled with demo values.")


def render_current_weather():
    st.markdown('<div id="sec-weather"></div>', unsafe_allow_html=True)
    left, right = st.columns([2.3, 1])
    with left:
        def _card():
            _weather_card(district, lat, lon)
        if hasattr(st, "fragment"):
            st.fragment(run_every=60 if st.session_state.get("auto_refresh", True) else None)(_card)()
        else:
            _card()
    with right, st.container(border=True):
        card_title("📍", "Location Information")
        st.markdown(f"**{district}, Tamil Nadu**  \n<span class='small-note'>{lat:.4f}° N, {lon:.4f}° E</span>", unsafe_allow_html=True)
        try:
            mp = px.scatter_mapbox(pd.DataFrame({"lat": [lat], "lon": [lon], "n": [district]}), lat="lat", lon="lon", zoom=7, height=170)
            mp.update_traces(marker=dict(size=17, color=GREEN))
            mp.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(mp, use_container_width=True, key="loc_map")
        except Exception:
            st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), height=170)


def build_alerts():
    nxt = weather.head(7)
    out = []
    mx = nxt.loc[nxt["rain_mm"].idxmax()]
    if mx["rain_mm"] >= 15:
        out.append(("bad", "⛈️", "Heavy rainfall expected", f"on {mx['date']:%d %b %Y} ({mx['rain_mm']:.1f} mm)"))
    elif mx["rain_mm"] >= 2:
        out.append(("warn", "⚠️", "Moderate rainfall expected", f"on {mx['date']:%d %b %Y} ({mx['rain_mm']:.1f} mm)"))
    if nxt["rain_mm"].sum() < 1:
        out.append(("warn", "🏜️", "Dry week ahead", "No significant rain in next 7 days"))
    if max_consecutive_dry(nxt["rain_mm"].tolist()) >= 5:
        out.append(("warn", "☀️", "Long dry spell", "5+ consecutive dry days forecast"))
    if nxt["tmax_c"].max() >= 36:
        out.append(("bad", "🔥", "Heat alert", f"Max temperature up to {nxt['tmax_c'].max():.0f} °C"))
    avg = float(((nxt["tmax_c"] + nxt["tmin_c"]) / 2).mean())
    if abs(avg - CI["temp_ideal"]) <= 3:
        out.append(("ok", "✅", f"Temperatures suitable for {crop.lower()}", f"avg {avg:.1f} °C vs ideal {CI['temp_ideal']} °C"))
    else:
        out.append(("warn", "🌡️", f"Temperature off-ideal for {crop.lower()}", f"avg {avg:.1f} °C vs ideal {CI['temp_ideal']} °C"))
    if nxt["tmax_c"].max() < 36:
        out.append(("ok", "🌤️", "No extreme heat expected", "in next 7 days"))
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
            card_title("📅", "Weather Forecast", f"Daily forecast · {data_label}")
        view = h[1].radio("View", ["Combined", "Temperature", "Rainfall"], horizontal=True, key="fc_view")
        days = st.radio("Range", ["7 days", "14 days"], horizontal=True, key="fc_days")
        w = weather.head(7 if days == "7 days" else 14)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if view in ("Combined", "Rainfall"):
            fig.add_bar(x=w["date"], y=w["rain_mm"], name="Rainfall (mm)", marker_color="rgba(76,201,240,.75)",
                        text=w["rain_mm"], textposition="outside", secondary_y=True)
        if view in ("Combined", "Temperature"):
            fig.add_scatter(x=w["date"], y=w["tmax_c"], name="Max Temp (°C)", mode="lines+markers+text", line=dict(color="#ff6b6b", width=3),
                            text=w["tmax_c"].round(0).astype(int).astype(str) + "°", textposition="top center", secondary_y=False)
            fig.add_scatter(x=w["date"], y=w["tmin_c"], name="Min Temp (°C)", mode="lines+markers+text", line=dict(color="#4cc9f0", width=3),
                            text=w["tmin_c"].round(0).astype(int).astype(str) + "°", textposition="bottom center", secondary_y=False)
        fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False, range=[min(15, float(w["tmin_c"].min()) - 3), float(w["tmax_c"].max()) + 5])
        fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=True, rangemode="tozero", showgrid=False)
        fig.update_xaxes(tickformat="%d %b<br>%a")
        st.plotly_chart(style_fig(fig, height=360), use_container_width=True, key="fc_chart")
    with right, st.container(border=True):
        card_title("🔔", "Weather Alerts (Next 7 Days)")
        alerts = build_alerts()
        st.markdown("".join(alert_html(a) for a in alerts[:4]), unsafe_allow_html=True)
        if len(alerts) > 4:
            with st.expander(f"View all ({len(alerts)})"):
                st.markdown("".join(alert_html(a) for a in alerts[4:]), unsafe_allow_html=True)


def render_trends_row():
    st.markdown('<div id="sec-trends"></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.15, 1.15, 1])
    hist, monthly = load_hist(), load_monthly()
    with c1, st.container(border=True):
        card_title("📈", "Climate Trends", "Historical NASA POWER data · Thoothukudi")
        if hist is None:
            st.info("Historical features not found. Run build_features.py first.")
        else:
            mode = st.radio("Trend", ["Temperature", "Rainfall"], horizontal=True, key="trend_mode")
            if mode == "Temperature":
                cols = [c for c in ["avg_temp_c", "max_temp_7day_c", "min_temp_7day_c"] if c in hist.columns]
                f = px.line(hist, x="date", y=cols, labels={"value": "°C", "date": ""}) if cols else go.Figure()
            else:
                f = px.area(hist, x="date", y="rain_7day_mm", labels={"rain_7day_mm": "mm", "date": ""}) if "rain_7day_mm" in hist else go.Figure()
            st.plotly_chart(style_fig(f, height=270), use_container_width=True, key="trend_chart")
    with c2, st.container(border=True):
        card_title("🌧️", "Monthly Rainfall", "From monthly_climate_summary.csv")
        if monthly is not None and "total_rainfall_mm" in monthly:
            f = px.bar(monthly, x="month", y="total_rainfall_mm", labels={"total_rainfall_mm": "Rainfall (mm)", "month": ""},
                       color="total_rainfall_mm", color_continuous_scale=["#1f6f8b", "#4cc9f0"])
            f.update_coloraxes(showscale=False)
            st.plotly_chart(style_fig(f, height=300), use_container_width=True, key="monthly_chart")
        else:
            st.info("Monthly summary not found. Run analyze_weather.py to generate it.")
    with c3, st.container(border=True):
        st.markdown('<div id="sec-suit"></div>', unsafe_allow_html=True)
        card_title("🎯", "Sowing Suitability")
        if best is None:
            st.warning("Not enough data.")
        else:
            st.plotly_chart(donut(int(best["Suitability score"])), use_container_width=True, key="donut")
            bar_rows([("Rainfall", int(best["Rain score"])), ("Temperature", int(best["Temp score"])),
                      ("Dry-spell control", int(best["Dry score"])), ("Excess-rain safety", int(best["Excess score"]))])


def reco_banner():
    if best is None:
        return
    st.markdown(f'<div class="reco"><div class="rt">🌱 Recommended Sowing Window</div>'
                f'<div class="rd">{best["Start"]:%d %b %Y} – {best["End"]:%d %b %Y}</div>'
                f'<div class="rs">Rule-based score on current weather data · not a confirmed agronomic date</div></div>', unsafe_allow_html=True)


def render_recommendation():
    st.markdown('<div id="sec-reco"></div>', unsafe_allow_html=True)
    st.markdown("### 🌱 AI Sowing Recommendation")
    if result.empty:
        st.warning("Not enough daily weather data to evaluate a 7-day window.")
        return
    kpi_row([("Suitability score", f"{int(best['Suitability score'])}/100", "Top candidate", True),
             ("Recommended window", f"{best['Start']:%d %b} – {best['End']:%d %b}", "7-day period"),
             ("7-day rainfall", f"{best['Rainfall (mm)']} mm", f"{int(best['Rainy days'])} rainy days"),
             ("Avg temperature", f"{best['Average temperature (°C)']} °C", f"Ideal {CI['temp_ideal']} °C")])
    t1, t2, t3, t4 = st.tabs(["Summary", "Why this window", "Compare windows", "Experimental ML"])
    with t1:
        st.success(f"Primary candidate: {best['Start']:%d %B} – {best['End']:%d %B} · Score {int(best['Suitability score'])}/100")
        if second is not None:
            st.info(f"Alternative candidate: {second['Start']:%d %B} – {second['End']:%d %B} · Score {int(second['Suitability score'])}/100")
        risks = best["Risks"].split("; ")
        chips(risks, "ok" if risks[0].startswith("No major") else "warn")
        st.caption("This is the highest-ranked candidate in the available data, not a confirmed agronomic sowing date.")
    with t2:
        r = []
        r.append(("ok", "✔ Estimated 7-day rainfall meets the prototype's minimum threshold.") if float(best["Rainfall (mm)"]) >= CI["rain_min"]
                 else ("warn", "⚠ Rainfall is below the minimum threshold; moisture stress may be a concern."))
        r.append(("ok", "✔ Average temperature is near the crop's configured ideal.") if abs(float(best["Average temperature (°C)"]) - CI["temp_ideal"]) <= 3
                 else ("warn", "⚠ Average temperature differs from the crop's configured ideal."))
        r.append(("warn", "⚠ A long dry spell is detected in this candidate window.") if int(best["Longest dry spell (days)"]) >= 5
                 else ("ok", "✔ The longest dry spell is relatively short in this window."))
        for k, t in r:
            chips([t], k)
    with t3:
        cdf = result.head(10).copy()
        cdf["Window"] = cdf.apply(lambda x: f"{x['Start']:%d %b}–{x['End']:%d %b}", axis=1)
        f = px.bar(cdf, x="Window", y="Suitability score", range_y=[0, 100], color="Suitability score",
                   color_continuous_scale=["#ff7b72", "#f5c04a", "#65e6a0"], range_color=[0, 100],
                   hover_data=["Rainfall (mm)", "Average temperature (°C)", "Longest dry spell (days)"])
        f.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(f, title="Top candidate windows"), use_container_width=True, key="cmp_chart")
    with t4:
        label, conf = predict_sowing_suitability(float(best["Rainfall (mm)"]), float(best["Average temperature (°C)"]), int(best["Longest dry spell (days)"]))
        if label is None:
            st.warning("The demonstration model file was not found.")
        else:
            kpi_row([("Demo model output", label), ("Experimental score (not validated)", f"{conf:.1f}%")])
        st.caption("The model was trained on synthetic demo data and is not validated for real-world agricultural decisions.")


# =====================================================================
# PAGES
# =====================================================================
if page == "Dashboard":
    kpi_row([("Forecast rainfall", f"{weather_total_rain:.1f} mm", f"{len(weather)} days"),
             ("Average max temp", f"{avg_max_temp:.1f} °C"), ("Rainy days", rainy_days), ("Dry days", dry_days)])
    render_current_weather()
    render_forecast_and_alerts()
    render_trends_row()
    reco_banner()
    st.markdown("---")
    render_recommendation()
    with st.expander("How this prototype works / limitations"):
        st.write("It ranks rolling 7-day candidate windows using a transparent weighted score (rainfall 35%, temperature 30%, dry-spell 20%, excess-rain 15%). Forecast data may fall back to clearly labelled synthetic demo data. Crop thresholds are simplified placeholders and require local agronomic validation. Soil measurements and historical yield records are not included in the score.")

elif page == "Sowing Recommendation":
    page_header("Recommendation", "Sowing recommendation", f"{crop} · {district} · Soil: {soil}")
    reco_banner()
    render_recommendation()
    st.markdown("### Crop thresholds used by this prototype")
    kpi_row([("Minimum rainfall", f"{CI['rain_min']} mm"), ("Ideal rainfall", f"{CI['rain_ideal']} mm"),
             ("Minimum temperature", f"{CI['temp_min']} °C"), ("Ideal temperature", f"{CI['temp_ideal']} °C")])
    st.caption("These are illustrative thresholds, not validated local agricultural guidance.")

elif page == "Weather & Climate":
    page_header("Weather", "Weather & climate", f"{district} · {data_label}")
    render_current_weather()
    kpi_row([("Total rainfall", f"{weather_total_rain:.1f} mm"), ("Average maximum", f"{avg_max_temp:.1f} °C"),
             ("Rainy days", rainy_days), ("Dry days", dry_days)])
    render_forecast_and_alerts()
    st.markdown("### Daily weather records")
    disp = weather.copy()
    disp["date"] = disp["date"].dt.strftime("%d %b %Y")
    st.dataframe(disp, use_container_width=True, hide_index=True)

elif page == "Crop Information":
    page_header("Crops", "Crop information", "Prototype crop settings. Confirm variety-specific guidance with the local agriculture department.")
    rows = [{"Crop": n, "Minimum rainfall (mm)": i["rain_min"], "Ideal rainfall (mm)": i["rain_ideal"], "Minimum temperature (°C)": i["temp_min"],
             "Ideal temperature (°C)": i["temp_ideal"], "Waterlogging sensitive": i["waterlogging_sensitive"]} for n, i in CROPS.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown(f"### {crop}")
    kpi_row([("Minimum rainfall", f"{CI['rain_min']} mm"), ("Ideal rainfall", f"{CI['rain_ideal']} mm"), ("Ideal temperature", f"{CI['temp_ideal']} °C")])
    (st.warning if CI["waterlogging_sensitive"] else st.info)(
        "The prototype marks this crop as waterlogging-sensitive. Check field drainage and local guidance." if CI["waterlogging_sensitive"]
        else "The prototype does not flag this crop as waterlogging-sensitive.")

elif page == "Soil Analysis":
    page_header("Soil", "Soil analysis", "Soil type is captured as farm context; it does not currently change the suitability score.")
    notes = {"Not specified": "Add a soil type or obtain a soil test for a more informed decision.",
             "Clay": "Clay soils can retain more water; check drainage and avoid field operations when soil is excessively wet.",
             "Loam": "Loam often balances water retention and drainage, but field conditions and soil tests still matter.",
             "Sandy": "Sandy soils usually drain quickly; monitor moisture and irrigation availability.",
             "Black soil": "Black soils can retain substantial moisture; check cracking, drainage and local crop guidance."}
    st.info(f"**{soil}:** {notes[soil]}")
    st.markdown("### What to record before sowing")
    st.write("- Soil pH and electrical conductivity from a soil test\n- Available nitrogen, phosphorus and potassium\n- Drainage, field moisture and irrigation access")
    st.warning("No laboratory soil measurements are collected by this prototype. These notes are general guidance, not a soil diagnosis.")

elif page == "Climate Trends":
    page_header("History", "Historical climate trends", "")
    hist, monthly = load_hist(), load_monthly()
    if hist is not None:
        st.caption("Source label in project data: NASA POWER daily weather data · Thoothukudi, Tamil Nadu · 2024–2025")
        items = [("Weather records", f"{len(hist):,}")]
        if "avg_temp_c" in hist:
            items.append(("Average temperature", f"{hist['avg_temp_c'].mean():.1f} °C"))
        if "rain_7day_mm" in hist:
            items.append(("Average 7-day rainfall", f"{hist['rain_7day_mm'].mean():.1f} mm"))
        kpi_row(items)
        cols = [c for c in ["avg_temp_c", "max_temp_7day_c", "min_temp_7day_c"] if c in hist.columns]
        if cols:
            st.plotly_chart(style_fig(px.line(hist, x="date", y=cols, labels={"value": "°C", "date": ""}), title="Temperature trend"), use_container_width=True, key="ct1")
        if "rain_7day_mm" in hist:
            st.plotly_chart(style_fig(px.area(hist, x="date", y="rain_7day_mm", labels={"rain_7day_mm": "mm", "date": ""}), title="7-day rainfall"), use_container_width=True, key="ct2")
        st.download_button("Download historical climate features", hist.to_csv(index=False), file_name="sowing_window_features.csv", mime="text/csv")
    else:
        st.warning("Historical weather features not found. Run build_features.py first.")
    if monthly is not None:
        st.markdown("### Monthly climate analysis")
        m1, m2 = st.columns(2)
        if "average_temperature_c" in monthly:
            m1.plotly_chart(style_fig(px.bar(monthly, x="month", y="average_temperature_c"), title="Average temperature by month"), use_container_width=True, key="ct3")
        if "total_rainfall_mm" in monthly:
            m2.plotly_chart(style_fig(px.bar(monthly, x="month", y="total_rainfall_mm"), title="Total rainfall by month"), use_container_width=True, key="ct4")
        st.dataframe(monthly, use_container_width=True, hide_index=True)
    else:
        st.info("Monthly summary not found. Run analyze_weather.py to generate it.")

elif page == "Insights & Alerts":
    page_header("Insights", "Insights & alerts", "")
    st.markdown("".join(alert_html(a) for a in build_alerts()), unsafe_allow_html=True)
    if best is not None:
        st.markdown("### Risk indicators for the top window")
        kpi_row([("Rainfall (7 days)", f"{best['Rainfall (mm)']} mm"), ("Longest dry spell", f"{int(best['Longest dry spell (days)'])} day(s)"),
                 ("Rainy days", int(best["Rainy days"]))])
        risks = best["Risks"].split("; ")
        chips(risks, "ok" if risks[0].startswith("No major") else "warn")
    st.markdown("---")
    st.markdown("### 🔮 Climate twin simulator")
    st.write("Test hypothetical changes in rainfall and temperature. This does not predict actual future climate.")
    x, y = st.columns(2)
    rainfall_change = x.slider("Rainfall change (%)", -50, 50, 0, 5)
    temperature_change = y.slider("Temperature change (°C)", -5.0, 5.0, 0.0, 0.5)
    sim = weather.copy()
    sim["rain_mm"] = (sim["rain_mm"] * (1 + rainfall_change / 100)).clip(lower=0)
    sim["tmax_c"] += temperature_change
    sim["tmin_c"] += temperature_change
    sr = evaluate_windows(sim, CI)
    sim_best = sr.iloc[0] if not sr.empty else None
    sim_score = int(sim_best["Suitability score"]) if sim_best is not None else 0
    orig = int(best["Suitability score"]) if best is not None else 0
    q1, q2, q3 = st.columns(3)
    q1.metric("Original score", f"{orig}/100")
    q2.metric("Simulated score", f"{sim_score}/100")
    q3.metric("Score change", f"{sim_score - orig:+d}", delta=f"{sim_score - orig:+d} points")
    if sim_best is not None:
        st.info(f"Under this hypothetical scenario, the top-ranked window is {sim_best['Start']} to {sim_best['End']}.")
    st.caption("All simulator thresholds are illustrative and not validated for every crop or district.")

elif page == "Export Data":
    page_header("Export", "Export data", "Download the current candidate-window results and weather data.")
    if not result.empty:
        st.markdown("### Candidate sowing windows")
        st.dataframe(result.head(20), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download recommendation report (CSV)", result.to_csv(index=False).encode("utf-8"),
                           file_name="sowingsense_recommendations.csv", mime="text/csv", type="primary")
    st.markdown("### Weather data")
    st.download_button("⬇️ Download weather data (CSV)", weather.to_csv(index=False).encode("utf-8"), file_name="sowingsense_weather.csv", mime="text/csv")
    hist = load_hist()
    if hist is not None:
        st.download_button("⬇️ Download historical climate features", hist.to_csv(index=False).encode("utf-8"), file_name="sowing_window_features.csv", mime="text/csv")
    st.info(f"Active source: {data_label}")

elif page == "About":
    page_header("About", "About SowingSense AI", "")
    st.write("SowingSense AI is a climate-aware crop sowing recommendation prototype designed to explore how weather conditions may support sowing-window decisions.")
    st.markdown("### Current capabilities")
    st.write("- Open-Meteo current conditions and short-range forecast when available\n- Clearly labelled synthetic fallback data for offline demonstrations\n"
             "- Crop-specific, rule-based ranking of rolling seven-day weather windows\n- Experimental ML model output, climate trends, risk alerts and CSV exports")
    st.markdown("### Limitations")
    st.warning("Scores are illustrative and are not a guarantee of yield or a substitute for local agricultural advice. Crop thresholds are simplified placeholders. The demo model was trained on synthetic data and requires validation with real agricultural data. Soil type is not currently used in the score.")
    st.caption("SowingSense AI · Smarter Sowing. Healthier Tomorrow.")

st.markdown('<div class="foot">🌱 SowingSense AI · Prototype only — verify advice with local agricultural experts.</div>', unsafe_allow_html=True)
