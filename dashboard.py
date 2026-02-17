import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import os
import psycopg2
from datetime import datetime, timedelta

# ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ ML libraries not installed. Install with: pip install scikit-learn joblib")

st.set_page_config(page_title="Vectora", layout="wide", initial_sidebar_state="expanded")

pio.templates["no_rangeslider"] = go.layout.Template(
    layout=dict(xaxis=dict(rangeslider=dict(visible=False)))
)
pio.templates.default = "no_rangeslider"

# Pure Minimal Orange Theme CSS (mock-aligned)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-color: #000000;
        --sidebar-bg: #111111;
        --card-bg: #111111;
        --plot-bg: #000000;
        --grid-color: #1A1A1A;
        --text-color: #FFFFFF;
        --text-secondary: #A3A3A3;
        --text-tertiary: #525252;
        --accent-primary: #FF8B3D;
        --accent-secondary: #FFB570;
        --positive: #22C55E;
        --negative: #EF4444;
    }

    html, body, [class*="st-"], .stApp {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 12% 10%, #0b0b0b 0%, #000000 55%, #000000 100%);
        color: var(--text-color);
    }

    header, [data-testid="stToolbar"] {
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--grid-color);
    }

    [data-testid="stSidebar"] .sidebar-brand {
        font-size: 22px;
        font-weight: 800;
        color: var(--accent-primary);
        letter-spacing: 0.2px;
        margin-bottom: 10px;
    }

    [data-testid="stSidebar"] .sidebar-section {
        color: var(--text-tertiary);
        font-size: 11px;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-top: 14px;
        margin-bottom: 6px;
    }

    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stCheckbox span {
        color: var(--text-secondary) !important;
    }

    /* Sidebar radio navigation */
    [data-testid="stSidebar"] .stRadio {
        padding-right: 0;
    }

    [data-testid="stSidebar"] .stRadio input[type="radio"],
    [data-testid="stSidebar"] .stRadio label > div:first-child,
    [data-testid="stSidebar"] .stRadio label > div:first-child > div {
        display: none !important;
    }

    [data-testid="stSidebar"] .stRadio label > div:last-child {
        margin-left: 0 !important;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        display: block;
        width: 100%;
        box-sizing: border-box;
        background: transparent;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 4px 0;
        border: 1px solid transparent;
        color: var(--text-secondary);
        transition: all 0.2s ease;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background: #1a1a1a;
        color: var(--text-color);
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label input:checked + div {
        color: var(--accent-primary);
        font-weight: 700;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
        background: #1a1a1a;
        border: 1px solid var(--accent-primary);
        color: var(--accent-primary);
        box-shadow: 0 0 0 1px rgba(255, 139, 61, 0.12);
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--accent-primary);
        color: #000000;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 16px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
    }

    .stButton > button:hover {
        background-color: var(--accent-secondary);
        box-shadow: 0 6px 20px rgba(255, 139, 61, 0.35);
        transform: translateY(-1px);
    }
    
    /* Secondary buttons (navigation) - sidebar grey background */
    button[kind="secondary"],
    button[data-testid$="secondary"],
    .stButton > button[kind="secondary"],
    [data-testid="stButton"] button[kind="secondary"],
    .secondary-button {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid var(--grid-color) !important;
        height: 46px !important;
        padding: 0 14px !important;
    }

    /* Force OTF pattern nav buttons to match #111111 */
    button[data-testid="baseButton-secondary"],
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-secondary"] button {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid var(--grid-color) !important;
    }
    
    button[kind="secondary"]:hover,
    button[data-testid$="secondary"]:hover,
    .stButton > button[kind="secondary"]:hover,
    [data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: #1a1a1a !important;
        border-color: var(--accent-primary) !important;
    }

    /* Inputs - Different backgrounds for different types */
    .stTextInput > div > div > input {
        background-color: var(--sidebar-bg) !important;
        border: 1px solid var(--grid-color);
        color: var(--text-color);
        border-radius: 10px;
    }
    
    /* Selectboxes (Exchange, Asset) - card background */
    .stSelectbox > div > div > select,
    .stSelectbox > div > div,
    .stSelectbox [data-baseweb="select"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--grid-color);
        color: var(--text-color);
        border-radius: 10px;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div > div {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--grid-color);
        color: var(--text-color);
        border-radius: 10px;
    }
    
    /* Number inputs - sidebar grey */
    .stNumberInput > div > div > input {
        background-color: var(--sidebar-bg) !important;
        border: 1px solid var(--grid-color);
        color: var(--text-color);
        border-radius: 10px;
    }
    
    /* Date inputs - match selectbox background (card-bg) */
    .stDateInput > div > div > input,
    .stDateInput > div > div,
    .stDateInput [data-baseweb="input"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--grid-color);
        color: var(--text-color);
        border-radius: 10px;
    }

    /* Session Returns: Total Days input */
    [data-testid="stNumberInput"] input[aria-label="Total Days"] {
        background-color: #111111 !important;
    }

    /* OTF: Consecutive Candles input */
    [data-testid="stNumberInput"] input[aria-label="Consecutive Candles"] {
        background-color: #111111 !important;
    }

    /* Gap Fills: Gap Size % input */
    [data-testid="stNumberInput"] input[aria-label="Gap Size %"] {
        background-color: #111111 !important;
    }

    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus,
    .stDateInput > div > div > input:focus {
        border-color: var(--accent-primary);
        box-shadow: 0 0 0 3px rgba(255, 139, 61, 0.15);
    }

    [data-baseweb="tag"] {
        background-color: #1a1a1a !important;
        color: var(--text-color) !important;
        border: 1px solid var(--grid-color);
        border-radius: 8px !important;
    }
    
    /* Multiselect dropdown background */
    [data-baseweb="select"] > div {
        background-color: var(--card-bg) !important;
        border-color: var(--grid-color) !important;
    }
    
    /* Multiselect options container */
    [data-baseweb="popover"] {
        background-color: var(--card-bg) !important;
    }
    
    [data-baseweb="list"] {
        background-color: var(--card-bg) !important;
    }

    [data-testid="stDownloadButton"],
    .stDownloadButton {
        display: none !important;
    }

    /* Sliders - Orange filled, grey unfilled, orange text */
    
    /* Slider container */
    .stSlider {
        padding-top: 0 !important;
    }
    
    /* Slider thumb (circle) - RED */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #EF4444 !important;
        border: 2px solid #EF4444 !important;
    }
    
    /* Slider value text (above thumb) - ORANGE, transparent background */
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"],
    .stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] {
        color: var(--accent-primary) !important;
        font-weight: 600 !important;
        background-color: transparent !important;
        background: transparent !important;
        padding: 0 !important;
    }
    
    /* Min/Max labels (on ends) - ORANGE, transparent background */
    .stSlider [data-baseweb="slider"] [data-testid="stTickBar"] > div,
    .stSlider [data-baseweb="slider"] [data-baseweb="tick-bar-item"],
    .stSlider div[class*="StyledTickBarItem"],
    .stSlider [role="presentation"] > div {
        color: var(--accent-primary) !important;
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* Base track (grey background - full width) */
    .stSlider [data-baseweb="slider"] [role="presentation"],
    .stSlider div[class*="StyledTrack"] {
        background: var(--grid-color) !important;
        background-color: var(--grid-color) !important;
        height: 4px !important;
        border-radius: 4px !important;
    }
    
    /* Inner track container */
    .stSlider [data-baseweb="slider"] > div[role="presentation"] > div {
        background: var(--grid-color) !important;
    }
    
    /* Filled portion (left of thumb) - ORANGE */
    .stSlider [data-baseweb="slider"] [data-baseweb="tick-bar"],
    .stSlider div[class*="InnerThumb"],
    .stSlider div[class*="InnerTrack"] {
        background: var(--accent-primary) !important;
        background-color: var(--accent-primary) !important;
    }
    
    /* Force override any remaining inline styles */
    .stSlider [data-baseweb="slider"] > div[role="presentation"] > div[style] {
        background: var(--accent-primary) !important;
        background-color: var(--accent-primary) !important;
    }

    /* Manual tick size input background */
    [data-testid="stNumberInput"] input[id*="tpo_manual_tick"],
    [data-testid="stNumberInput"] input[id*="daily_tpo_manual_tick"] {
        background-color: #111111 !important;
    }
    
    [data-testid="stNumberInput"] input[aria-label="Manual Tick Size ($)"] {
        background-color: #111111 !important;
    }

    /* Checkboxes - Orange styling (checkbox square only) */
    .stCheckbox > label > div[data-baseweb="checkbox"] > div[data-checked="true"] svg {
        fill: #000000 !important;
    }

    /* HHRL distance input background */
    [data-testid="stNumberInput"] input[aria-label="Distance from Price (%)"] {
        background-color: #111111 !important;
    }

    /* HHRL Hit Rate Evolution: Rolling Window input background */
    [data-testid="stNumberInput"] input[id*="hhrl_rolling_window"],
    [data-testid="stNumberInput"] input[aria-label="Rolling Window (days)"],
    .st-key-hhrl_rolling_window [data-baseweb="input"],
    .st-key-hhrl_rolling_window [data-baseweb="base-input"],
    .st-key-hhrl_rolling_window input {
        background-color: #111111 !important;
    }

    /* Quartile Opens: Rolling Window + chart bars input backgrounds */
    [data-testid="stNumberInput"] input[id*="qo_rolling_window"],
    .st-key-qo_rolling_window [data-baseweb="input"],
    .st-key-qo_rolling_window [data-baseweb="base-input"],
    .st-key-qo_rolling_window input,
    [data-testid="stNumberInput"] input[id*="qo_bars_before"],
    [data-testid="stNumberInput"] input[id*="qo_bars_after"],
    .st-key-qo_bars_before [data-baseweb="input"],
    .st-key-qo_bars_before [data-baseweb="base-input"],
    .st-key-qo_bars_before input,
    .st-key-qo_bars_after [data-baseweb="input"],
    .st-key-qo_bars_after [data-baseweb="base-input"],
    .st-key-qo_bars_after input {
        background-color: #111111 !important;
    }

    /* Round Numbers Front Run inputs */
    [data-testid="stNumberInput"] input[aria-label="Round Number Interval"],
    [data-testid="stNumberInput"] input[aria-label="Mitigation Lookback (hours)"],
    [data-testid="stNumberInput"] input[aria-label="Max Look Forward Bars"],
    [data-testid="stNumberInput"] input[aria-label="Front Run Threshold (±)"] {
        background-color: #111111 !important;
    }

    /* Weekend Breaks + Gap Fills max lookforward */
    [data-testid="stNumberInput"] input[aria-label="Max Look Forward Days"],
    [data-testid="stNumberInput"] input[aria-label="Max Look Forward Bars"],
    [data-testid="stNumberInput"] input[aria-label="Max Lookforward Bars"]
    {
        background-color: #111111 !important;
    }

    /* Gap Fills / Naked Opens / Large Wick Fills max lookforward inputs by key */
    [data-testid="stNumberInput"] input[id*="gf_lookforward"],
    [data-testid="stNumberInput"] input[id*="no_lookforward"],
    [data-testid="stNumberInput"] input[id*="wf_lookforward"] {
        background-color: #111111 !important;
    }

    /* Analyze buttons icon (black) */
    button[key^="analyze_"] > div {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
    }

    button[key^="analyze_"] > div::before {
        content: "";
        width: 16px;
        height: 16px;
        display: inline-block;
        background-size: 16px 16px;
        background-repeat: no-repeat;
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 16 16' fill='black'><rect x='1' y='7' width='3' height='8'/><rect x='6.5' y='4' width='3' height='11'/><rect x='12' y='1' width='3' height='14'/></svg>");
    }

    /* HHRL button icon: rounded bars (small, biggest, second biggest) */
    .btn-hhrl button > div {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
    }

    .btn-hhrl button > div::before {
        content: "";
        width: 16px;
        height: 16px;
        display: inline-block;
        background-size: 16px 16px;
        background-repeat: no-repeat;
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 16 16' fill='black'><rect x='1' y='9' width='3' height='6' rx='1.5' ry='1.5'/><rect x='6.5' y='2' width='3' height='13' rx='1.5' ry='1.5'/><rect x='12' y='5' width='3' height='10' rx='1.5' ry='1.5'/></svg>");
    }
    
    .stCheckbox > label > div[data-baseweb="checkbox"] > div {
        background-color: transparent !important;
        border-color: var(--grid-color) !important;
    }
    
    .stCheckbox > label > div[data-baseweb="checkbox"] > div[data-checked="true"] {
        background-color: var(--accent-primary) !important;
        border-color: var(--accent-primary) !important;
    }

    .otf-pattern-filters {
        background: transparent !important;
        border: none !important;
    }

    /* OTF pattern filters: remove orange label background */
    .otf-pattern-filters label span,
    .otf-pattern-filters label div {
        background: transparent !important;
    }

    .otf-pattern-filters [data-testid="stCheckbox"] label,
    .otf-pattern-filters [data-testid="stCheckbox"] label > div,
    .otf-pattern-filters [data-testid="stCheckbox"] label > div > div,
    .otf-pattern-filters [data-testid="stCheckbox"] label > div > div > div,
    .otf-pattern-filters [data-testid="stCheckbox"] label > div > div > div > span {
        background: transparent !important;
        box-shadow: none !important;
    }

    /* Target checkbox label text wrapper (Streamlit/BaseWeb) */
    .otf-pattern-filters [data-testid="stCheckbox"] label > span,
    .otf-pattern-filters [data-testid="stCheckbox"] label > span * {
        background-color: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
    }

    /* Remove orange pill behind checkbox label text */
    .otf-pattern-filters [data-baseweb="checkbox"] > div {
        background-color: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
    }

    .otf-pattern-filters [data-baseweb="checkbox"] > div > div,
    .otf-pattern-filters [data-baseweb="checkbox"] > div > div > div {
        background-color: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
    }

    /* OTF pattern filters: remove orange fill from checked checkbox container */
    .otf-pattern-filters [data-testid="stCheckbox"] input[type="checkbox"]:checked + div {
        background-color: transparent !important;
        border-color: var(--grid-color) !important;
    }

    /* Keep the checkbox square red while removing the pill background */
    .otf-pattern-filters [data-baseweb="checkbox"] input[type="checkbox"]:checked + div > div {
        background-color: var(--accent-primary) !important;
        border-color: var(--accent-primary) !important;
    }

    /* Hard override for the pill wrapper div in OTF filters */
    .otf-pattern-filters [data-testid="stCheckbox"] label > div {
        background-color: transparent !important;
        border-color: transparent !important;
        box-shadow: none !important;
    }

    .otf-pattern-filters [data-testid="stCheckbox"] label > div[class*="st-"] {
        background-color: transparent !important;
        border-color: transparent !important;
        box-shadow: none !important;
    }

    /* Remove orange fill on OTF filter label wrapper (Safari class combo) */
    .otf-pattern-filters .st-b8.st-f3.st-bo.st-c1.st-bq.st-br.st-f4.st-bt.st-bu.st-bv {
        background-color: transparent !important;
        border-color: transparent !important;
        box-shadow: none !important;
    }

    .otf-pattern-filters [data-testid="stWidgetLabel"],
    .otf-pattern-filters [data-testid="stWidgetLabel"] * {
        background-color: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
    }

    /* Pivot Navigation Boxes - Matching Mockup Design */
    /* Target all pivot navigation buttons */
    button[key^="pivot_btn_"] {
        background-color: transparent !important;
        border: 2px solid #FFFFFF !important;
        border-radius: 8px !important;
        padding: 20px 0 !important;
        min-height: 60px !important;
        font-size: 18px !important;
        font-weight: 400 !important;
        color: #FFFFFF !important;
        font-family: 'Courier New', Courier, monospace !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }
    
    /* Hover state */
    button[key^="pivot_btn_"]:hover {
        border-color: var(--accent-primary) !important;
        background-color: rgba(255, 139, 61, 0.05) !important;
        transform: translateY(-1px);
    }
    
    /* Selected state (primary type) */
    button[key^="pivot_btn_"][kind="primary"] {
        background-color: #111111 !important;
        border: 2px solid var(--accent-primary) !important;
        color: var(--accent-primary) !important;
        box-shadow: 0 0 0 1px rgba(255, 139, 61, 0.6), 0 0 18px rgba(255, 139, 61, 0.45) !important;
    }
    
    /* Remove default Streamlit button styling */
    button[key^="pivot_btn_"] > div {
        justify-content: center !important;
    }

    /* One Time Framing Analysis - Control Styling */
    /* Number inputs: Bars Before (Chart) and Bars After (Chart) */
    [data-testid="stNumberInput"] input[id*="otf_bars_before"],
    [data-testid="stNumberInput"] input[id*="otf_bars_after"],
    [data-testid="stNumberInput"] input[aria-label="Bars Before (Chart)"],
    [data-testid="stNumberInput"] input[aria-label="Bars After (Chart)"],
    .st-key-otf_bars_before [data-baseweb="input"],
    .st-key-otf_bars_before [data-baseweb="base-input"],
    .st-key-otf_bars_before input,
    .st-key-otf_bars_after [data-baseweb="input"],
    .st-key-otf_bars_after [data-baseweb="base-input"],
    .st-key-otf_bars_after input {
        background-color: #111111 !important;
    }

    /* OTF Selectbox: Show Pattern */
    div[data-baseweb="select"]:has(input[id*="otf_pattern_type"]),
    .st-key-otf_pattern_type [data-baseweb="select"],
    .st-key-otf_pattern_type [data-baseweb="select"] > div {
        background-color: #111111 !important;
    }

    /* OTF Navigation buttons: Previous Pattern and Next Pattern */
    button[key="prev_otf"],
    button[key="next_otf"],
    button[key="prev_hhrl_pattern"],
    button[key="next_hhrl_pattern"],
    .st-key-prev_otf button,
    .st-key-next_otf button,
    .st-key-prev_hhrl_pattern button,
    .st-key-next_hhrl_pattern button {
        background-color: #111111 !important;
        border: 1px solid var(--grid-color) !important;
        color: #FFFFFF !important;
    }

    button[key="prev_otf"]:hover,
    button[key="next_otf"]:hover,
    button[key="prev_hhrl_pattern"]:hover,
    button[key="next_hhrl_pattern"]:hover,
    .st-key-prev_otf button:hover,
    .st-key-next_otf button:hover,
    .st-key-prev_hhrl_pattern button:hover,
    .st-key-next_hhrl_pattern button:hover {
        background-color: #1a1a1a !important;
        border-color: var(--accent-primary) !important;
    }

    /* Quartile Opens chart nav buttons: enforce identical sizing */
    .st-key-prev_qo [data-testid="stButton"] > button,
    .st-key-next_qo [data-testid="stButton"] > button,
    .st-key-prev_qo button,
    .st-key-next_qo button {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid var(--grid-color) !important;
        border-radius: 12px !important;
        height: 56px !important;
        min-height: 56px !important;
        padding: 0 16px !important;
        font-size: 18px !important;
        line-height: 1 !important;
    }

    .st-key-prev_qo [data-testid="stButton"] > button:hover,
    .st-key-next_qo [data-testid="stButton"] > button:hover,
    .st-key-prev_qo button:hover,
    .st-key-next_qo button:hover {
        background-color: #1a1a1a !important;
        border-color: var(--accent-primary) !important;
    }

    /* High Hit Rate Levels: first chart nav buttons */
    .st-key-prev_pivot [data-testid="stButton"] > button,
    .st-key-next_pivot [data-testid="stButton"] > button,
    .st-key-prev_pivot button[data-testid="baseButton-secondary"],
    .st-key-next_pivot button[data-testid="baseButton-secondary"],
    .st-key-prev_pivot button,
    .st-key-next_pivot button {
        background: #111111 !important;
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid var(--grid-color) !important;
        border-radius: 12px !important;
        height: 56px !important;
        min-height: 56px !important;
        padding: 0 16px !important;
        font-size: 18px !important;
        line-height: 1 !important;
    }

    .st-key-prev_pivot [data-testid="stButton"] > button:hover,
    .st-key-next_pivot [data-testid="stButton"] > button:hover,
    .st-key-prev_pivot button[data-testid="baseButton-secondary"]:hover,
    .st-key-next_pivot button[data-testid="baseButton-secondary"]:hover,
    .st-key-prev_pivot button:hover,
    .st-key-next_pivot button:hover {
        background: #1a1a1a !important;
        background-color: #1a1a1a !important;
        border-color: var(--accent-primary) !important;
    }

    /* HHRL nav hard override */
    [data-testid="stVerticalBlock"] .st-key-prev_pivot button,
    [data-testid="stVerticalBlock"] .st-key-next_pivot button,
    [data-testid="stVerticalBlock"] .st-key-prev_pivot button[kind="secondary"],
    [data-testid="stVerticalBlock"] .st-key-next_pivot button[kind="secondary"],
    [data-testid="stVerticalBlock"] .st-key-prev_pivot button[data-testid^="baseButton"],
    [data-testid="stVerticalBlock"] .st-key-next_pivot button[data-testid^="baseButton"] {
        background: #111111 !important;
        background-color: #111111 !important;
        border-color: var(--grid-color) !important;
        border-width: 1px !important;
        border-style: solid !important;
        height: 56px !important;
        min-height: 56px !important;
        padding: 0 16px !important;
    }

    [data-testid="stVerticalBlock"] .st-key-prev_pivot button p,
    [data-testid="stVerticalBlock"] .st-key-next_pivot button p,
    [data-testid="stVerticalBlock"] .st-key-prev_pivot button span,
    [data-testid="stVerticalBlock"] .st-key-next_pivot button span {
        color: #FFFFFF !important;
        font-size: 18px !important;
        line-height: 1 !important;
        font-weight: 500 !important;
    }

    /* HHRL nav container-scoped override (reliable) */
    .st-key-hhrl_prev_wrap [data-testid="stButton"] > button,
    .st-key-hhrl_next_wrap [data-testid="stButton"] > button,
    .st-key-hhrl_prev_wrap button,
    .st-key-hhrl_next_wrap button {
        background: #111111 !important;
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid var(--grid-color) !important;
        border-radius: 12px !important;
        height: 56px !important;
        min-height: 56px !important;
        padding: 0 16px !important;
    }

    .st-key-hhrl_prev_wrap [data-testid="stButton"] > button:hover,
    .st-key-hhrl_next_wrap [data-testid="stButton"] > button:hover,
    .st-key-hhrl_prev_wrap button:hover,
    .st-key-hhrl_next_wrap button:hover {
        background: #1a1a1a !important;
        background-color: #1a1a1a !important;
        border-color: var(--accent-primary) !important;
    }

    /* Sidebar collapse control */
    [data-testid="collapsedControl"] button,
    button[title="Open sidebar"],
    button[title="Close sidebar"] {
        color: var(--text-secondary) !important;
        font-size: 16px !important;
        border: 1px solid var(--grid-color) !important;
        border-radius: 10px !important;
        background: #0f0f0f !important;
        padding: 6px 10px !important;
    }

    [data-testid="collapsedControl"] svg,
    button[title="Open sidebar"] svg,
    button[title="Close sidebar"] svg {
        display: none !important;
    }

    [data-testid="collapsedControl"] span,
    button[title="Open sidebar"] span,
    button[title="Close sidebar"] span {
        display: none !important;
    }

    button[title="Open sidebar"]::before,
    [data-testid="collapsedControl"] button[title="Open sidebar"]::before {
        content: "▶";
    }

    button[title="Close sidebar"]::before,
    [data-testid="collapsedControl"] button[title="Close sidebar"]::before {
        content: "◀";
    }

    /* Cards */
    .metric-card {
        background-color: var(--card-bg);
        border: 1px solid var(--grid-color);
        border-radius: 14px;
        padding: 16px;
        transition: all 0.2s ease;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
        height: 120px;
        margin-bottom: 16px;
        display: block;
    }

    .metric-card:hover {
        border-color: var(--accent-primary);
        border-top: 2px solid var(--accent-primary);
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(255, 139, 61, 0.2);
    }

    .metric-title {
        color: var(--text-secondary);
        font-size: 12px;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        color: var(--text-color);
    }

    .metric-subtitle {
        color: var(--text-secondary);
        font-size: 12px;
        margin-top: 4px;
    }

    /* Info boxes */
    .stInfo {
        background-color: rgba(255, 139, 61, 0.1);
        border-left: 4px solid var(--accent-primary);
    }

    .page-title {
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .page-subtitle {
        color: var(--text-tertiary);
        font-size: 14px;
        margin-bottom: 10px;
    }

    .top-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--accent-primary);
        color: #000000;
        font-weight: 800;
        border-radius: 999px;
        padding: 8px 18px;
        font-size: 13px;
        letter-spacing: 0.5px;
        box-shadow: 0 10px 30px rgba(255, 139, 61, 0.35);
    }

    /* Hide default Streamlit sidebar collapse icon */
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
        display: none !important;
    }
</style>

""", unsafe_allow_html=True)


# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Daily Returns"

if 'otf_analyzed' not in st.session_state:
    st.session_state.otf_analyzed = False

if 'otf_results' not in st.session_state:
    st.session_state.otf_results = None

if 'otf_df' not in st.session_state:
    st.session_state.otf_df = None

if 'hhrl_analyzed' not in st.session_state:
    st.session_state.hhrl_analyzed = False

if 'hhrl_results' not in st.session_state:
    st.session_state.hhrl_results = None

if 'hhrl_pivot_index' not in st.session_state:
    st.session_state.hhrl_pivot_index = -1

if 'daily_pivots_analyzed' not in st.session_state:
    st.session_state.daily_pivots_analyzed = False

if 'daily_pivots_results' not in st.session_state:
    st.session_state.daily_pivots_results = None

if 'daily_pivots_chart_index' not in st.session_state:
    st.session_state.daily_pivots_chart_index = -1

if 'tpo_analyzed' not in st.session_state:
    st.session_state.tpo_analyzed = False

if 'tpo_results' not in st.session_state:
    st.session_state.tpo_results = None

if 'tpo_profile_index' not in st.session_state:
    st.session_state.tpo_profile_index = -1

if 'weekly_pivots_chart_index' not in st.session_state:
    st.session_state.weekly_pivots_chart_index = -1

if 'monthly_pivots_chart_index' not in st.session_state:
    st.session_state.monthly_pivots_chart_index = -1

if 'session_pivots_analyzed' not in st.session_state:
    st.session_state.session_pivots_analyzed = False

if 'session_pivots_results' not in st.session_state:
    st.session_state.session_pivots_results = None

if 'session_pivots_chart_index' not in st.session_state:
    st.session_state.session_pivots_chart_index = -1

if 'wick_fills_analyzed' not in st.session_state:
    st.session_state.wick_fills_analyzed = False

if 'wick_fills_results' not in st.session_state:
    st.session_state.wick_fills_results = None

if 'wick_fills_chart_index' not in st.session_state:
    st.session_state.wick_fills_chart_index = -1

if 'quartile_opens_analyzed' not in st.session_state:
    st.session_state.quartile_opens_analyzed = False

if 'quartile_opens_results' not in st.session_state:
    st.session_state.quartile_opens_results = None

if 'quartile_opens_chart_index' not in st.session_state:
    st.session_state.quartile_opens_chart_index = -1

if 'round_numbers_analyzed' not in st.session_state:
    st.session_state.round_numbers_analyzed = False

if 'round_numbers_results' not in st.session_state:
    st.session_state.round_numbers_results = None

if 'round_numbers_chart_index' not in st.session_state:
    st.session_state.round_numbers_chart_index = -1

if 'weekend_breaks_chart_index' not in st.session_state:
    st.session_state.weekend_breaks_chart_index = -1

if 'monday_range_chart_index' not in st.session_state:
    st.session_state.monday_range_chart_index = -1

# Color scheme
bg_color = "#000000"
text_color = "#FFFFFF"
sidebar_bg = "#111111"
plot_bg = "#000000"
grid_color = "#1A1A1A"
card_bg = "#111111"
positive_color = "#22C55E"
negative_color = "#EF4444"
candle_up = "#A8A8A8"
candle_down = "#5A5A5A"
highlight_color = "#FF8B3D"
gold_highlight = "#FFD700"
selected_bg = "#262626"
blue_color = "#3B82F6"
yellow_color = "#FFB570"
light_blue = "#60A5FA"
dark_blue = "#1D4ED8"
orange_line = "#FF8B3D"


def render_page_header(title, subtitle=None, pill=None):
    col_title, col_pill = st.columns([3, 1])
    with col_title:
        st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
        if subtitle:
            st.markdown(f"<div class='page-subtitle'>{subtitle}</div>", unsafe_allow_html=True)
    with col_pill:
        if pill:
            st.markdown(
                f"<div style='text-align: right;'><span class='top-pill'>{pill}</span></div>",
                unsafe_allow_html=True
            )


def render_exchange_asset_controls(prefix):
    col_ds1, col_ds2 = st.columns(2)
    with col_ds1:
        exchange = st.selectbox(
            "Exchange",
            ["Bybit", "Binance", "Hyperliquid", "Coinbase"],
            index=0,
            key=f"{prefix}_exchange"
        )
    with col_ds2:
        asset = st.selectbox(
            "Asset",
            ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"],
            index=0,
            key=f"{prefix}_asset"
        )
    return exchange, asset


CANDLE_HOVER_TEMPLATE = (
    "Date: %{x}<br>"
    "Open: %{open:.2f}<br>"
    "High: %{high:.2f}<br>"
    "Low: %{low:.2f}<br>"
    "Close: %{close:.2f}<extra></extra>"
)


def add_ohlc_hover_trace(fig, ohlc_df):
    if ohlc_df is None or ohlc_df.empty:
        return

    hover_text = []
    for idx, row in ohlc_df.iterrows():
        hover_text.append(
            f"Date: {idx}<br>"
            f"Open: {row['Open']:.2f}<br>"
            f"High: {row['High']:.2f}<br>"
            f"Low: {row['Low']:.2f}<br>"
            f"Close: {row['Close']:.2f}"
        )

    fig.add_trace(go.Scatter(
        x=ohlc_df.index,
        y=ohlc_df['Close'],
        mode='markers',
        marker=dict(size=0.1, opacity=0),
        text=hover_text,
        hoverinfo='text',
        showlegend=False
    ))

def clamp_date(value, min_date, max_date):
    if value < min_date:
        return min_date
    if value > max_date:
        return max_date
    return value


def sync_vol_days_from_dates():
    min_d = st.session_state.vol_min_date
    max_d = st.session_state.vol_max_date
    start = clamp_date(st.session_state.vol_start_date, min_d, max_d)
    end = clamp_date(st.session_state.vol_end_date, min_d, max_d)
    if end < start:
        end = start
    st.session_state.vol_start_date = start
    st.session_state.vol_end_date = end
    st.session_state.vol_total_days = max(1, (end - start).days)


def sync_vol_end_from_days():
    min_d = st.session_state.vol_min_date
    max_d = st.session_state.vol_max_date
    start = clamp_date(st.session_state.vol_start_date, min_d, max_d)
    days = max(1, int(st.session_state.vol_total_days))
    end = start + timedelta(days=days)
    if end > max_d:
        end = max_d
    st.session_state.vol_start_date = start
    st.session_state.vol_end_date = end
    st.session_state.vol_total_days = max(1, (end - start).days)


def vol_days_minus():
    st.session_state.vol_total_days = max(1, int(st.session_state.vol_total_days) - 1)
    sync_vol_end_from_days()


def vol_days_plus():
    st.session_state.vol_total_days = min(1825, int(st.session_state.vol_total_days) + 1)
    sync_vol_end_from_days()

def sync_vol_days_from_input():
    st.session_state.vol_total_days = max(1, int(st.session_state.vol_total_days_input))
    sync_vol_end_from_days()


def winsorized_mean(values, lower_pct=5, upper_pct=95):
    if values is None or len(values) == 0:
        return None
    lower = np.percentile(values, lower_pct)
    upper = np.percentile(values, upper_pct)
    clipped = np.clip(values, lower, upper)
    return float(np.mean(clipped))


def analyze_round_number_front_runs(df_15m, start_date, end_date, day_filter, round_interval, lookback_hours, max_lookforward, timeframe, front_run_threshold):
    if timeframe == '15m':
        df_tf = df_15m.copy()
    elif timeframe == '30m':
        df_tf = df_15m.resample('30min')
    elif timeframe == '1h':
        df_tf = df_15m.resample('1H')
    elif timeframe == '4h':
        df_tf = df_15m.resample('4H')
    elif timeframe == '1D':
        df_tf = df_15m.resample('1D')
    else:
        df_tf = df_15m.resample('30min')
    
    if timeframe != '15m':
        df_tf = df_tf.agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    
    df_tf = df_tf[(df_tf.index.date >= start_date) & (df_tf.index.date <= end_date)]
    
    events = []
    if df_tf.empty:
        return events, df_tf
    
    for i, (ts, row) in enumerate(df_tf.iterrows()):
        day_name = ts.day_name()
        if day_filter and day_name not in day_filter:
            continue
        
        high = row['High']
        low = row['Low']
        close = row['Close']
        
        # Candidate round levels near price
        level_down = np.floor(close / round_interval) * round_interval
        level_up = level_down + round_interval
        candidate_levels = [level_down, level_up]
        
        for level in candidate_levels:
            if level <= 0:
                continue
            
            touched = (low <= level <= high)
            near = (abs(high - level) <= front_run_threshold) or (abs(low - level) <= front_run_threshold)
            
            if not near or touched:
                continue
            
            # Check mitigation in lookback window (exact level touch)
            lookback_start = ts - timedelta(hours=lookback_hours)
            lookback_window = df_tf[(df_tf.index >= lookback_start) & (df_tf.index < ts)]
            if not lookback_window.empty:
                mitigated = ((lookback_window['Low'] <= level) & (lookback_window['High'] >= level)).any()
                if mitigated:
                    continue
            
            direction = "up" if close < level else "down"
            entry_price = close
            
            # Look forward for hit
            bars_to_hit = None
            forward_end = min(len(df_tf), i + 1 + max_lookforward)
            forward_window = df_tf.iloc[i + 1:forward_end]
            
            if not forward_window.empty:
                for j, (f_ts, f_row) in enumerate(forward_window.iterrows(), start=1):
                    if f_row['Low'] <= level <= f_row['High']:
                        bars_to_hit = j
                        break
            
            # MFE% within lookforward window
            mfe_pct = 0.0
            if not forward_window.empty:
                if direction == "up":
                    max_high = forward_window['High'].max()
                    mfe_pct = ((max_high - entry_price) / entry_price) * 100
                else:
                    min_low = forward_window['Low'].min()
                    mfe_pct = ((entry_price - min_low) / entry_price) * 100
            
            events.append({
                'time': ts,
                'day_name': day_name,
                'level': level,
                'direction': direction,
                'entry_price': entry_price,
                'bars_to_hit': bars_to_hit,
                'hit': bars_to_hit is not None,
                'mfe_pct': mfe_pct
            })
    
    return events, df_tf

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

CONNECTION_STRING = st.secrets["DATABASE_URL"] if "DATABASE_URL" in st.secrets else os.getenv("DATABASE_URL", "postgresql://postgres.ffpspjiznmupxassxxxs:ZjjebPPo4b2U9ci@aws-1-eu-west-1.pooler.supabase.com:5432/postgres")

@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_available_pairs():
    """Get all exchange/symbol pairs available in database"""
    try:
        conn = psycopg2.connect(CONNECTION_STRING)
        query = """
        SELECT DISTINCT exchange, symbol
        FROM ohlcv_data
        ORDER BY exchange, symbol
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return pd.DataFrame(columns=['exchange', 'symbol'])

@st.cache_data(ttl=900)  # Cache for 15 minutes
def load_data(exchange='binance', symbol='BTC/USDT'):
    """Load OHLCV data from Supabase database"""
    try:
        conn = psycopg2.connect(CONNECTION_STRING)
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv_data
        WHERE exchange = %s AND symbol = %s AND timeframe = '15m'
        ORDER BY timestamp ASC
        """
        df = pd.read_sql_query(query, conn, params=(exchange, symbol))
        conn.close()

        if df.empty:
            st.error(f"No data found for {exchange} - {symbol}")
            st.stop()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.index = df.index.tz_localize(None)
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    df['hour'] = df.index.hour

    def get_session(hour):
        if 0 <= hour < 6:
            return 'Asia'
        elif 6 <= hour < 12:
            return 'London'
        elif 12 <= hour < 20:
            return 'NY'
        else:
            return 'Close'

    df['session'] = df['hour'].apply(get_session)

    daily = df.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    daily['returns'] = daily['Close'].pct_change() * 100
    daily['day_name'] = daily.index.day_name()
    daily['day_of_week'] = daily.index.dayofweek
    daily['is_weekend'] = daily['day_of_week'].isin([5, 6])

    return df, daily

# Sidebar
with st.sidebar:
    st.markdown("<div class='sidebar-brand'>Vectora</div>", unsafe_allow_html=True)

    # Exchange & Symbol Selectors
    available_pairs = get_available_pairs()

    if not available_pairs.empty:
        exchanges = available_pairs['exchange'].unique().tolist()
        selected_exchange = st.selectbox("Exchange", exchanges, key="exchange_selector")

        symbols = available_pairs[available_pairs['exchange'] == selected_exchange]['symbol'].tolist()
        selected_symbol = st.selectbox("Asset", symbols, key="symbol_selector")
    else:
        selected_exchange = 'binance'
        selected_symbol = 'BTC/USDT'

    st.markdown("<div class='sidebar-section'>Navigation</div>", unsafe_allow_html=True)
    
    pages = [
        "Daily Returns",
        "Session Returns",
        "One Time Framing Analysis",
        "High Hit Rate Levels",
        "Pivots",
        "Session TPO",
        "Monday Range",
        "Daily TPO",
        "Large Wick Fills",
        "Naked Opens",
        "Gap Fills",
        "Weekend Breaks",
        "Volume Heatmap",
        "Round Numbers Front Run",
        "Quartile Opens"
    ]
    current_index = pages.index(st.session_state.current_page) if st.session_state.current_page in pages else 0
    selected_page = st.radio("", pages, index=current_index, key="nav_radio", label_visibility="collapsed")
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()
    
    # removed sidebar info box

page = st.session_state.current_page

# Load data based on selected exchange/symbol
df_15m, df_daily = load_data(selected_exchange, selected_symbol)
last_updated = df_15m.index.max()

# Default lookback for pages that still use it (will be moved to individual pages)
lookback = 365

# ============================================================================
# PIVOTS UNIFIED PAGE HEADER
# ============================================================================

pivot_page_mode = page == "Pivots"
pivot_section = None

if pivot_page_mode:
    render_page_header(
        "Pivots",
        "Unified analysis for daily, session, weekly, and monthly pivots"
    )
    # Pivot type selection with box navigation
    # Initialize session state for pivot section
    if 'pivot_section' not in st.session_state:
        st.session_state.pivot_section = 'Session'
    
    # Create 4 columns for navigation boxes
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        if st.button('Session', key='pivot_btn_session', use_container_width=True, 
                    type='primary' if st.session_state.pivot_section == 'Session' else 'secondary'):
            st.session_state.pivot_section = 'Session'
            st.rerun()
    
    with col2:
        if st.button('Daily', key='pivot_btn_daily', use_container_width=True,
                    type='primary' if st.session_state.pivot_section == 'Daily' else 'secondary'):
            st.session_state.pivot_section = 'Daily'
            st.rerun()
    
    with col3:
        if st.button('Weekly', key='pivot_btn_weekly', use_container_width=True,
                    type='primary' if st.session_state.pivot_section == 'Weekly' else 'secondary'):
            st.session_state.pivot_section = 'Weekly'
            st.rerun()
    
    with col4:
        if st.button('Monthly', key='pivot_btn_monthly', use_container_width=True,
                    type='primary' if st.session_state.pivot_section == 'Monthly' else 'secondary'):
            st.session_state.pivot_section = 'Monthly'
            st.rerun()
    
    # Get current pivot section
    pivot_section = st.session_state.pivot_section
    st.markdown("---")

# ============================================================================
# MARKET REGIME BANNER (Top of Every Page)
# ============================================================================

if ML_AVAILABLE and page != "Market Regime":
    try:
        # Quick regime detection
        detector_banner = MarketRegimeDetector(n_regimes=4)
        if detector_banner.fit(df_15m):
            current_regime, probs, desc = detector_banner.predict_current(df_15m)
            
            if current_regime is not None:
                # Color based on regime
                if "Up" in desc:
                    banner_color = "#28a745"  # Green
                    emoji = "📈"
                elif "Down" in desc:
                    banner_color = "#dc3545"  # Red
                    emoji = "📉"
                else:
                    banner_color = "#ffc107"  # Yellow
                    emoji = "↔️"
                
                st.markdown(f"""
                <div style="background-color: {banner_color}20; border-left: 4px solid {banner_color}; padding: 10px; margin-bottom: 20px; border-radius: 5px;">
                    <span style="font-size: 14px;">
                        {emoji} <strong>Market Regime:</strong> {desc} | 
                        <strong>Confidence:</strong> {max(probs.values()):.0f}% | 
                        <a href="?page=Market Regime" style="color: {banner_color};">View Details →</a>
                    </span>
                </div>
                """, unsafe_allow_html=True)
    except:
        pass  # Silently fail if regime detection has issues

recent_daily = df_daily.last(f'{lookback}D')
recent_15m = df_15m.last(f'{lookback}D')

# ============================================================================
# DAILY PIVOTS
# ============================================================================

def analyze_daily_pivots_hourly(df_15m, df_daily, start_date, end_date, day_filter):
    """
    Analyze daily highs/lows by hour
    P1 = first pivot (high or low) of the day
    P2 = second pivot (the opposite of P1)
    """
    
    daily_filtered = df_daily[(df_daily.index.date >= start_date) & (df_daily.index.date <= end_date)].copy()
    
    if day_filter:
        daily_filtered = daily_filtered[daily_filtered['day_name'].isin(day_filter)]
    
    results = {
        'high_hours': [],
        'low_hours': [],
        'p1_hours': [],
        'p2_hours': [],
        'p1_types': [],
        'dates': []
    }
    
    for date in daily_filtered.index:
        day_data = df_15m[df_15m.index.date == date.date()]
        
        if len(day_data) == 0:
            continue
        
        high_idx = day_data['High'].idxmax()
        low_idx = day_data['Low'].idxmin()
        
        high_hour = high_idx.hour
        low_hour = low_idx.hour
        
        results['high_hours'].append(high_hour)
        results['low_hours'].append(low_hour)
        results['dates'].append(date)
        
        if high_idx < low_idx:
            results['p1_hours'].append(high_hour)
            results['p2_hours'].append(low_hour)
            results['p1_types'].append('High')
        else:
            results['p1_hours'].append(low_hour)
            results['p2_hours'].append(high_hour)
            results['p1_types'].append('Low')
    
    return results

def analyze_daily_pivots_session(df_15m, df_daily, start_date, end_date, day_filter):
    """Analyze daily highs/lows by session"""
    
    daily_filtered = df_daily[(df_daily.index.date >= start_date) & (df_daily.index.date <= end_date)].copy()
    
    if day_filter:
        daily_filtered = daily_filtered[daily_filtered['day_name'].isin(day_filter)]
    
    results = {
        'high_sessions': [],
        'low_sessions': [],
        'p1_sessions': [],
        'p2_sessions': [],
        'p1_types': [],
        'dates': [],
        'high_times': [],
        'low_times': [],
        'high_prices': [],
        'low_prices': []
    }
    
    for date in daily_filtered.index:
        day_data = df_15m[df_15m.index.date == date.date()].copy()
        
        if len(day_data) == 0:
            continue
        
        high_idx = day_data['High'].idxmax()
        low_idx = day_data['Low'].idxmin()
        
        high_session = day_data.loc[high_idx, 'session']
        low_session = day_data.loc[low_idx, 'session']
        
        high_price = day_data.loc[high_idx, 'High']
        low_price = day_data.loc[low_idx, 'Low']
        
        results['high_sessions'].append(high_session)
        results['low_sessions'].append(low_session)
        results['high_times'].append(high_idx)
        results['low_times'].append(low_idx)
        results['high_prices'].append(high_price)
        results['low_prices'].append(low_price)
        results['dates'].append(date)
        
        if high_idx < low_idx:
            results['p1_sessions'].append(high_session)
            results['p2_sessions'].append(low_session)
            results['p1_types'].append('High')
        else:
            results['p1_sessions'].append(low_session)
            results['p2_sessions'].append(high_session)
            results['p1_types'].append('Low')
    
    return results

def analyze_weekly_pivots(df_15m, df_daily, start_date, end_date, day_filter):
    """Analyze weekly highs/lows by day of week - P1 is first extreme, P2 is second"""
    
    # Resample to weekly data (week ending on Sunday, so Monday-Sunday weeks)
    df_weekly = df_daily.resample('W-SUN').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    
    df_weekly = df_weekly[(df_weekly.index.date >= start_date) & (df_weekly.index.date <= end_date)]
    
    results = {
        'high_days': [],
        'low_days': [],
        'p1_days': [],
        'p2_days': [],
        'p1_types': [],
        'dates': [],
        'week_data_list': []  # Store weekly data for chart navigation
    }
    
    for week_end in df_weekly.index:
        # Week ends on Sunday, so start is 6 days before (Monday)
        week_start = week_end - pd.Timedelta(days=6)
        week_data = df_15m[(df_15m.index >= week_start) & (df_15m.index <= week_end)]
        
        if len(week_data) == 0:
            continue
        
        # Find when high and low occurred
        high_idx = week_data['High'].idxmax()
        low_idx = week_data['Low'].idxmin()
        
        high_day = high_idx.day_name()
        low_day = low_idx.day_name()
        
        # Filter by day if specified
        if day_filter:
            if high_day not in day_filter and low_day not in day_filter:
                continue
        
        results['high_days'].append(high_day)
        results['low_days'].append(low_day)
        results['dates'].append(week_end)
        results['week_data_list'].append({
            'week_start': week_start,
            'week_end': week_end,
            'high_day': high_day,
            'low_day': low_day,
            'high_idx': high_idx,
            'low_idx': low_idx
        })
        
        # P1 is whichever comes first chronologically
        if high_idx < low_idx:
            results['p1_days'].append(high_day)
            results['p2_days'].append(low_day)
            results['p1_types'].append('High')
        else:
            results['p1_days'].append(low_day)
            results['p2_days'].append(high_day)
            results['p1_types'].append('Low')
    
    return results

def analyze_monthly_pivots(df_15m, df_daily, start_date, end_date, day_filter):
    """Analyze monthly highs/lows by day of month - P1 is first extreme, P2 is second"""
    
    # Resample to monthly data
    df_monthly = df_daily.resample('M').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()
    
    df_monthly = df_monthly[(df_monthly.index.date >= start_date) & (df_monthly.index.date <= end_date)]
    
    results = {
        'high_days': [],
        'low_days': [],
        'p1_days': [],
        'p2_days': [],
        'p1_types': [],
        'dates': [],
        'month_data_list': []  # Store monthly data for chart navigation
    }
    
    for month_end in df_monthly.index:
        month_start = month_end.replace(day=1)
        month_data = df_15m[(df_15m.index >= month_start) & (df_15m.index <= month_end)]
        
        if len(month_data) == 0:
            continue
        
        # Find when high and low occurred
        high_idx = month_data['High'].idxmax()
        low_idx = month_data['Low'].idxmin()
        
        high_day = high_idx.day  # Day of month (1-31)
        low_day = low_idx.day
        
        # Convert to day of week for filtering
        high_day_name = high_idx.day_name()
        low_day_name = low_idx.day_name()
        
        # Filter by day of week if specified
        if day_filter:
            if high_day_name not in day_filter and low_day_name not in day_filter:
                continue
        
        results['high_days'].append(high_day)
        results['low_days'].append(low_day)
        results['dates'].append(month_end)
        results['month_data_list'].append({
            'month_start': month_start,
            'month_end': month_end,
            'high_day': high_day,
            'low_day': low_day,
            'high_idx': high_idx,
            'low_idx': low_idx
        })
        
        # P1 is whichever comes first chronologically
        if high_idx < low_idx:
            results['p1_days'].append(high_day)
            results['p2_days'].append(low_day)
            results['p1_types'].append('High')
        else:
            results['p1_days'].append(low_day)
            results['p2_days'].append(high_day)
            results['p1_types'].append('Low')
    
    return results

def analyze_session_pivots(df_15m, start_date, end_date, day_filter, session, timeframe):
    """Analyze session highs/lows by time interval - P1 is first extreme, P2 is second"""
    
    # Define session times (UTC)
    sessions = {
        'Asia': (0, 6),
        'London': (6, 12),
        'New York': (12, 20),
        'Close': (20, 24)
    }
    
    start_hour, end_hour = sessions[session]
    
    # Resample data to selected timeframe
    if timeframe == '15m':
        df_resampled = df_15m.copy()
    elif timeframe == '30m':
        df_resampled = df_15m.resample('30min').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
    else:  # 1h
        df_resampled = df_15m.resample('1H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
    
    # Filter by date range
    df_resampled = df_resampled[(df_resampled.index.date >= start_date) & (df_resampled.index.date <= end_date)]
    
    results = {
        'high_intervals': [],
        'low_intervals': [],
        'p1_intervals': [],
        'p2_intervals': [],
        'p1_types': [],
        'dates': [],
        'session_data_list': []
    }
    
    # Group by date to process each day
    for date, day_data in df_resampled.groupby(df_resampled.index.date):
        day_name = pd.Timestamp(date).day_name()
        
        # Filter by day of week
        if day_filter and day_name not in day_filter:
            continue
        
        # Filter for session hours
        session_data = day_data[(day_data.index.hour >= start_hour) & (day_data.index.hour < end_hour)]
        
        if len(session_data) == 0:
            continue
        
        # Find when high and low occurred
        high_idx = session_data['High'].idxmax()
        low_idx = session_data['Low'].idxmin()
        
        # Format interval as HH:MM
        high_interval = high_idx.strftime('%H:%M')
        low_interval = low_idx.strftime('%H:%M')
        
        results['high_intervals'].append(high_interval)
        results['low_intervals'].append(low_interval)
        results['dates'].append(pd.Timestamp(date))
        results['session_data_list'].append({
            'date': pd.Timestamp(date),
            'day_name': day_name,
            'session_data': session_data,
            'high_interval': high_interval,
            'low_interval': low_interval,
            'high_idx': high_idx,
            'low_idx': low_idx
        })
        
        # P1 is whichever comes first chronologically
        if high_idx < low_idx:
            results['p1_intervals'].append(high_interval)
            results['p2_intervals'].append(low_interval)
            results['p1_types'].append('High')
        else:
            results['p1_intervals'].append(low_interval)
            results['p2_intervals'].append(high_interval)
            results['p1_types'].append('Low')
    
    return results

def analyze_wick_fills(df_15m, start_date, end_date, timeframe, method, min_wick_pct, 
                       partial_threshold, wick_direction, day_filter, session_filter, max_lookforward):
    """Analyze large wick fills and track partial/full fill statistics"""
    
    # Resample data based on timeframe
    if timeframe == '15m':
        df_resampled = df_15m.copy()
    elif timeframe == '30m':
        df_resampled = df_15m.resample('30min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == '1h':
        df_resampled = df_15m.resample('1H').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == '4h':
        df_resampled = df_15m.resample('4H').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == '1D':
        df_resampled = df_15m.resample('1D').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == '1W':
        df_resampled = df_15m.resample('W-SUN').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == '1M':
        df_resampled = df_15m.resample('M').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == 'Session':
        # Build ALL session candles (not just selected one)
        sessions = {
            'Asia': (0, 6),
            'London': (6, 12),
            'New York': (12, 20),
            'Close': (20, 24)
        }
        
        session_candles = []
        for date in pd.date_range(start_date, end_date, freq='D'):
            # Build candle for each session type
            for session_name, (start_hour, end_hour) in sessions.items():
                session_start = pd.Timestamp(date) + pd.Timedelta(hours=start_hour)
                session_end = pd.Timestamp(date) + pd.Timedelta(hours=end_hour)
                
                session_data = df_15m[(df_15m.index >= session_start) & (df_15m.index < session_end)]
                
                if len(session_data) > 0:
                    session_candles.append({
                        'Time': session_start,
                        'Open': session_data['Open'].iloc[0],
                        'High': session_data['High'].max(),
                        'Low': session_data['Low'].min(),
                        'Close': session_data['Close'].iloc[-1],
                        'SessionType': session_name  # Track which session type
                    })
        
        df_resampled = pd.DataFrame(session_candles).set_index('Time')
        # Sort by time to ensure proper ordering
        df_resampled = df_resampled.sort_index()
    
    # Filter by date range
    df_resampled = df_resampled[(df_resampled.index.date >= start_date) & (df_resampled.index.date <= end_date)]
    
    results = {
        'times': [],
        'candle_colors': [],
        'wick_types': [],
        'wick_sizes': [],
        'wick_levels': [],
        'bars_partial': [],
        'bars_full': [],
        'partial_pcts': [],
        'fill_status': [],
        # New fields for chart and enhanced raw data
        'opens': [],
        'highs': [],
        'lows': [],
        'closes': [],
        'day_names': [],
        'sessions': [],
        'partial_fill_prices': [],
        'full_fill_prices': [],
        'partial_fill_dates': [],
        'full_fill_dates': []
    }
    
    # Analyze each candle
    for i in range(len(df_resampled)):
        candle = df_resampled.iloc[i]
        candle_time = df_resampled.index[i]
        day_name = candle_time.day_name()
        
        # Filter by day of week
        if day_filter and day_name not in day_filter:
            continue
        
        # For Session timeframe, only detect wicks in selected session types
        if timeframe == 'Session':
            candle_session_type = candle['SessionType']
            if candle_session_type not in session_filter:
                continue  # Skip wick detection for non-selected sessions
        
        o, h, l, c = candle['Open'], candle['High'], candle['Low'], candle['Close']
        is_bullish = c >= o
        
        # Calculate wicks
        if is_bullish:
            top_wick = h - c
            bottom_wick = o - l
            top_wick_level = h
            bottom_wick_level = l
        else:
            top_wick = h - o
            bottom_wick = c - l
            top_wick_level = h
            bottom_wick_level = l
        
        # Calculate wick percentages based on method
        if method == '% of Price':
            top_wick_pct = (top_wick / c) * 100 if c != 0 else 0
            bottom_wick_pct = (bottom_wick / c) * 100 if c != 0 else 0
        else:  # % of Body
            body = abs(c - o)
            total_candle_top = body + top_wick
            total_candle_bottom = body + bottom_wick
            
            top_wick_pct = (top_wick / total_candle_top) * 100 if total_candle_top != 0 else 0
            bottom_wick_pct = (bottom_wick / total_candle_bottom) * 100 if total_candle_bottom != 0 else 0
        
        # Process wicks based on direction filter
        wicks_to_check = []
        
        if wick_direction in ['Top', 'Both']:
            if top_wick_pct >= min_wick_pct:
                wicks_to_check.append(('Top', top_wick_pct, top_wick_level, top_wick))
        
        if wick_direction in ['Bottom', 'Both']:
            if bottom_wick_pct >= min_wick_pct:
                wicks_to_check.append(('Bottom', bottom_wick_pct, bottom_wick_level, bottom_wick))
        
        # Check fills for each qualifying wick
        for wick_type, wick_pct, wick_level, wick_size in wicks_to_check:
            bars_partial = None
            bars_full = None
            partial_pct_filled = 0
            fill_status = 'Unfilled'
            partial_fill_price = None
            full_fill_price = None
            partial_fill_date = None
            full_fill_date = None
            
            # Get wick base (where wick starts)
            if wick_type == 'Top':
                wick_base = c if is_bullish else o
            else:  # Bottom
                wick_base = o if is_bullish else c
            
            # Calculate partial fill threshold price
            if wick_type == 'Top':
                partial_threshold_price = wick_base + (wick_size * partial_threshold / 100)
            else:
                partial_threshold_price = wick_base - (wick_size * partial_threshold / 100)
            
            # Look forward for fills
            lookforward_end = min(i + max_lookforward, len(df_resampled))
            
            for j in range(i + 1, lookforward_end):
                future_candle = df_resampled.iloc[j]
                future_h, future_l = future_candle['High'], future_candle['Low']
                future_time = df_resampled.index[j]
                bars_elapsed = j - i
                
                if wick_type == 'Top':
                    # Check if high touched partial threshold
                    if bars_partial is None and future_h >= partial_threshold_price:
                        bars_partial = bars_elapsed
                        # Calculate how much was filled
                        pct_filled = ((future_h - wick_base) / wick_size) * 100
                        partial_pct_filled = min(pct_filled, 100)
                        fill_status = 'Partial'
                        partial_fill_price = future_h
                        partial_fill_date = future_time
                    
                    # Check if high touched full wick level
                    if future_h >= wick_level:
                        bars_full = bars_elapsed
                        partial_pct_filled = 100
                        fill_status = 'Full'
                        full_fill_price = future_h
                        full_fill_date = future_time
                        break
                else:  # Bottom wick
                    # Check if low touched partial threshold
                    if bars_partial is None and future_l <= partial_threshold_price:
                        bars_partial = bars_elapsed
                        # Calculate how much was filled
                        pct_filled = ((wick_base - future_l) / wick_size) * 100
                        partial_pct_filled = min(pct_filled, 100)
                        fill_status = 'Partial'
                        partial_fill_price = future_l
                        partial_fill_date = future_time
                    
                    # Check if low touched full wick level
                    if future_l <= wick_level:
                        bars_full = bars_elapsed
                        partial_pct_filled = 100
                        fill_status = 'Full'
                        full_fill_price = future_l
                        full_fill_date = future_time
                        break
            
            # If still unfilled, count bars to current
            if fill_status == 'Unfilled':
                bars_partial = lookforward_end - i - 1
            
            # Determine session if applicable
            session_name = None
            if timeframe == 'Session':
                session_name = candle['SessionType']  # Use the actual candle's session type
            else:
                # Determine session based on hour
                hour = candle_time.hour
                if 0 <= hour < 6:
                    session_name = 'Asia'
                elif 6 <= hour < 12:
                    session_name = 'London'
                elif 12 <= hour < 20:
                    session_name = 'New York'
                else:
                    session_name = 'Close'
            
            # Store results
            results['times'].append(candle_time)
            results['candle_colors'].append('Green' if is_bullish else 'Red')
            results['wick_types'].append(wick_type)
            results['wick_sizes'].append(wick_pct)
            results['wick_levels'].append(wick_level)
            results['bars_partial'].append(bars_partial)
            results['bars_full'].append(bars_full)
            results['partial_pcts'].append(partial_pct_filled)
            results['fill_status'].append(fill_status)
            # New fields
            results['opens'].append(o)
            results['highs'].append(h)
            results['lows'].append(l)
            results['closes'].append(c)
            results['day_names'].append(day_name)
            results['sessions'].append(session_name)
            results['partial_fill_prices'].append(partial_fill_price)
            results['full_fill_prices'].append(full_fill_price)
            results['partial_fill_dates'].append(partial_fill_date)
            results['full_fill_dates'].append(full_fill_date)
    
    return results

def analyze_naked_opens(df_15m, start_date, end_date, timeframe, direction_filter, day_filter, max_lookforward):
    """
    Analyze naked opens (candles with flat tops or flat bottoms)
    Bullish candle (Close > Open) with Low = Open → flat bottom
    Bearish candle (Close < Open) with High = Open → flat top
    """
    
    # Resample based on timeframe
    if timeframe == '15m':
        df_resampled = df_15m.copy()
    elif timeframe == '30m':
        df_resampled = df_15m.resample('30min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == '1h':
        df_resampled = df_15m.resample('1H').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == '4H':
        df_resampled = df_15m.resample('4H').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == 'Daily':
        df_resampled = df_15m.resample('1D').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == 'Weekly':
        df_resampled = df_15m.resample('W-SUN').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == 'Monthly':
        df_resampled = df_15m.resample('M').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    
    # Filter by date range
    df_resampled = df_resampled[(df_resampled.index.date >= start_date) & (df_resampled.index.date <= end_date)]
    
    results = {
        'times': [],
        'day_names': [],
        'directions': [],  # 'Bullish' or 'Bearish'
        'open_prices': [],
        'high_prices': [],
        'low_prices': [],
        'close_prices': [],
        'body_sizes': [],
        'body_size_pct': [],
        'hit': [],
        'bars_to_hit': [],
        'mae_values': [],
        'mae_pct': []
    }
    
    # Detect naked opens
    for i in range(len(df_resampled)):
        candle = df_resampled.iloc[i]
        candle_time = df_resampled.index[i]
        day_name = candle_time.day_name()
        
        # Filter by day of week
        if day_filter and day_name not in day_filter:
            continue
        
        open_price = candle['Open']
        high_price = candle['High']
        low_price = candle['Low']
        close_price = candle['Close']
        
        # Detect flat opens with tolerance for floating point
        tolerance = 0.01  # Allow 1 cent tolerance
        
        is_bullish_candle = close_price > open_price
        is_bearish_candle = close_price < open_price
        
        has_flat_bottom = abs(low_price - open_price) <= tolerance and is_bullish_candle
        has_flat_top = abs(high_price - open_price) <= tolerance and is_bearish_candle
        
        # Determine if this is a naked open
        is_naked_open = False
        direction = None
        target_level = None
        
        if has_flat_bottom and (direction_filter == 'Both' or direction_filter == 'Bullish Only'):
            is_naked_open = True
            direction = 'Bullish'
            target_level = open_price  # Need to come back down to touch open
        elif has_flat_top and (direction_filter == 'Both' or direction_filter == 'Bearish Only'):
            is_naked_open = True
            direction = 'Bearish'
            target_level = open_price  # Need to come back up to touch open
        
        if not is_naked_open:
            continue
        
        # Calculate body size
        body_size = abs(close_price - open_price)
        body_size_pct = (body_size / open_price) * 100 if open_price != 0 else 0
        
        # Look forward for hit
        lookforward_end = min(i + max_lookforward + 1, len(df_resampled))
        
        hit = False
        bars_to_hit = None
        mae = 0
        mae_pct = 0
        
        for j in range(i + 1, lookforward_end):
            future_candle = df_resampled.iloc[j]
            bars_elapsed = j - i
            
            if direction == 'Bullish':
                # Flat bottom - need price to come back down to open level
                if future_candle['Low'] <= target_level:
                    hit = True
                    bars_to_hit = bars_elapsed
                    break
            else:  # Bearish
                # Flat top - need price to come back up to open level
                if future_candle['High'] >= target_level:
                    hit = True
                    bars_to_hit = bars_elapsed
                    break
        
        # Calculate MAE (Maximum Adverse Excursion)
        if direction == 'Bullish':
            # Adverse is going higher (away from target below)
            highest_reached = high_price
            lookback_range = min(i + (bars_to_hit if bars_to_hit else max_lookforward), len(df_resampled))
            for j in range(i + 1, lookback_range):
                highest_reached = max(highest_reached, df_resampled.iloc[j]['High'])
            mae = highest_reached - open_price
            mae_pct = (mae / open_price) * 100 if open_price != 0 else 0
        else:  # Bearish
            # Adverse is going lower (away from target above)
            lowest_reached = low_price
            lookback_range = min(i + (bars_to_hit if bars_to_hit else max_lookforward), len(df_resampled))
            for j in range(i + 1, lookback_range):
                lowest_reached = min(lowest_reached, df_resampled.iloc[j]['Low'])
            mae = open_price - lowest_reached
            mae_pct = (mae / open_price) * 100 if open_price != 0 else 0
        
        # Store results
        results['times'].append(candle_time)
        results['day_names'].append(day_name)
        results['directions'].append(direction)
        results['open_prices'].append(open_price)
        results['high_prices'].append(high_price)
        results['low_prices'].append(low_price)
        results['close_prices'].append(close_price)
        results['body_sizes'].append(body_size)
        results['body_size_pct'].append(body_size_pct)
        results['hit'].append(hit)
        results['bars_to_hit'].append(bars_to_hit)
        results['mae_values'].append(mae)
        results['mae_pct'].append(mae_pct)
    
    return results

# ============================================================================
# ML MODELS - GAP FILL PREDICTOR & MARKET REGIME DETECTOR
# ============================================================================

class GapFillMLPredictor:
    """
    ML model to predict gap fill probability
    Uses Random Forest trained on historical gaps
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        
    def engineer_features(self, gap_size_pct, gap_direction, hour_of_day, day_of_week, 
                         is_weekend, recent_volatility=None):
        """Create features for prediction"""
        features = {
            'gap_size_pct': abs(gap_size_pct),
            'gap_direction': 1 if gap_direction == 'Gap Up' else -1,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': 1 if is_weekend else 0,
            'is_monday': 1 if day_of_week == 0 else 0,
            'is_friday': 1 if day_of_week == 4 else 0,
            'is_asia': 1 if 0 <= hour_of_day < 6 else 0,
            'is_london': 1 if 6 <= hour_of_day < 12 else 0,
            'is_ny': 1 if 12 <= hour_of_day < 20 else 0,
            'gap_size_squared': gap_size_pct ** 2,
            'hour_squared': hour_of_day ** 2
        }
        
        if recent_volatility is not None:
            features['volatility'] = recent_volatility
        
        return features
    
    def train(self, gap_results):
        """Train model on historical gap data"""
        if not ML_AVAILABLE:
            return False
        
        features_list = []
        labels = []
        
        for i in range(len(gap_results['times'])):
            gap_time = gap_results['times'][i]
            gap_size = gap_results['gap_sizes'][i]
            gap_direction = gap_results['gap_directions'][i]
            filled = gap_results['fill_status'][i] in ['Partial', 'Full']
            
            features = self.engineer_features(
                gap_size, gap_direction, gap_time.hour, 
                gap_time.dayofweek, gap_time.dayofweek >= 5
            )
            
            features_list.append(features)
            labels.append(1 if filled else 0)
        
        if len(features_list) < 20:
            return False  # Not enough data
        
        X = pd.DataFrame(features_list)
        y = np.array(labels)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        return True
    
    def predict(self, gap_size_pct, gap_direction, hour_of_day, day_of_week, is_weekend):
        """Predict fill probability for new gap"""
        if not self.is_trained or self.model is None:
            return None, None
        
        features = self.engineer_features(
            gap_size_pct, gap_direction, hour_of_day, 
            day_of_week, is_weekend
        )
        
        X = pd.DataFrame([features])
        
        probability = self.model.predict_proba(X)[0, 1]
        
        # Confidence based on probability extremes
        if probability > 0.75 or probability < 0.25:
            confidence = "High"
        elif probability > 0.65 or probability < 0.35:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return probability, confidence


class MarketRegimeDetector:
    """
    Detect market regime using K-Means clustering
    Helps adapt strategies based on market conditions
    """
    
    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.regime_names = {}
        
    def calculate_features(self, df_15m, lookback_hours=24):
        """Calculate features for regime detection"""
        # Resample to hourly
        df_1h = df_15m.resample('1H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
        
        features_list = []
        timestamps = []
        
        for i in range(lookback_hours, len(df_1h)):
            window = df_1h.iloc[i-lookback_hours:i]
            current_time = df_1h.index[i]
            
            returns = window['Close'].pct_change()
            
            # Features
            features = {
                'volatility': returns.std(),
                'trend': (window['Close'].iloc[-1] - window['Close'].iloc[0]) / window['Close'].iloc[0],
                'range': (window['High'].max() - window['Low'].min()) / window['Close'].mean(),
                'momentum': (window['Close'].iloc[-1] - window['Close'].iloc[-6]) / window['Close'].iloc[-6],
                'direction_consistency': (returns > 0).sum() / len(returns)
            }
            
            features_list.append(features)
            timestamps.append(current_time)
        
        return pd.DataFrame(features_list, index=timestamps)
    
    def fit(self, df_15m):
        """Train regime detector"""
        if not ML_AVAILABLE:
            return False
        
        features_df = self.calculate_features(df_15m)
        
        if len(features_df) < 100:
            return False  # Not enough data
        
        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features_df)
        
        # Fit K-Means
        self.model = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        labels = self.model.fit_predict(X_scaled)
        
        # Name regimes
        features_df['regime'] = labels
        
        for regime in range(self.n_regimes):
            regime_data = features_df[features_df['regime'] == regime]
            vol = regime_data['volatility'].mean()
            trend = regime_data['trend'].mean()
            
            if vol < 0.015:
                vol_name = "Low Vol"
            elif vol < 0.025:
                vol_name = "Med Vol"
            else:
                vol_name = "High Vol"
            
            if abs(trend) < 0.01:
                trend_name = "Ranging"
            elif trend > 0.02:
                trend_name = "Strong Up"
            elif trend > 0:
                trend_name = "Mild Up"
            elif trend < -0.02:
                trend_name = "Strong Down"
            else:
                trend_name = "Mild Down"
            
            self.regime_names[regime] = f"{vol_name} - {trend_name}"
        
        self.is_trained = True
        return True
    
    def predict_current(self, df_15m):
        """Predict current market regime"""
        if not self.is_trained or self.model is None:
            return None, None, "Not trained"
        
        # Get last 24 hours
        window = df_15m.tail(96)  # 96 * 15min = 24 hours
        
        if len(window) < 24:
            return None, None, "Insufficient data"
        
        # Resample
        df_1h = window.resample('1H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()
        
        if len(df_1h) < 24:
            return None, None, "Insufficient data"
        
        returns = df_1h['Close'].pct_change()
        
        features = {
            'volatility': returns.std(),
            'trend': (df_1h['Close'].iloc[-1] - df_1h['Close'].iloc[0]) / df_1h['Close'].iloc[0],
            'range': (df_1h['High'].max() - df_1h['Low'].min()) / df_1h['Close'].mean(),
            'momentum': (df_1h['Close'].iloc[-1] - df_1h['Close'].iloc[-6]) / df_1h['Close'].iloc[-6],
            'direction_consistency': (returns > 0).sum() / len(returns)
        }
        
        X = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X)
        
        regime = self.model.predict(X_scaled)[0]
        
        # Calculate distances (proxy for probability)
        distances = self.model.transform(X_scaled)[0]
        inv_distances = 1 / (distances + 1e-10)
        probabilities = inv_distances / inv_distances.sum()
        
        prob_dict = {self.regime_names[i]: probabilities[i] * 100 for i in range(self.n_regimes)}
        
        description = self.regime_names[regime]
        
        return regime, prob_dict, description


class PivotHitMLPredictor:
    """
    ML model to predict if a pivot will be hit
    Trained on historical pivot data from High Hit Rate Levels
    """
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.feature_names = None
        
    def engineer_features(self, pivot_data):
        """
        Create features for pivot hit prediction
        
        pivot_data = {
            'hour': 0-23,
            'day_of_week': 0-6,
            'position': 'Above' or 'Below',
            'distance_pct': % distance from current price,
            'atr_pct': ATR as % of price,
            'historical_hit_rate': hit rate for this hour/position
        }
        """
        features = {
            'hour': pivot_data['hour'],
            'day_of_week': pivot_data.get('day_of_week', 2),  # Default Wednesday
            'position_encoded': 1 if pivot_data['position'] == 'Above' else -1,
            'distance_pct': abs(pivot_data.get('distance_pct', 0)),
            'atr_pct': pivot_data.get('atr_pct', 1.0),
            'historical_hit_rate': pivot_data.get('historical_hit_rate', 50.0),
            
            # Session encoding
            'is_asia': 1 if 0 <= pivot_data['hour'] < 6 else 0,
            'is_london': 1 if 6 <= pivot_data['hour'] < 12 else 0,
            'is_ny': 1 if 12 <= pivot_data['hour'] < 20 else 0,
            'is_close': 1 if 20 <= pivot_data['hour'] < 24 else 0,
            
            # Day encoding
            'is_monday': 1 if pivot_data.get('day_of_week', 2) == 0 else 0,
            'is_friday': 1 if pivot_data.get('day_of_week', 2) == 4 else 0,
            'is_weekend': 1 if pivot_data.get('day_of_week', 2) >= 5 else 0,
            
            # Interaction features
            'distance_x_atr': abs(pivot_data.get('distance_pct', 0)) * pivot_data.get('atr_pct', 1.0),
            'hour_squared': pivot_data['hour'] ** 2
        }
        
        return features
    
    def train(self, pivots_df):
        """
        Train model on historical pivot data
        
        pivots_df must have columns:
        - timestamp, hour, day_name, pivot, current_price, position, hit, atr, ...
        """
        if not ML_AVAILABLE:
            return False
        
        if len(pivots_df) < 50:
            return False  # Need at least 50 pivots
        
        features_list = []
        labels = []
        
        for _, pivot_row in pivots_df.iterrows():
            # Calculate features
            distance_pct = abs((pivot_row['pivot'] - pivot_row['current_price']) / pivot_row['current_price']) * 100
            atr_pct = (pivot_row.get('atr', 100) / pivot_row['current_price']) * 100 if 'atr' in pivot_row else 1.0
            
            # Get historical hit rate for this hour/position
            hour_position_df = pivots_df[
                (pivots_df['hour'] == pivot_row['hour']) &
                (pivots_df['position'] == pivot_row['position'])
            ]
            hist_hit_rate = (hour_position_df['hit'].sum() / len(hour_position_df)) * 100 if len(hour_position_df) > 0 else 50.0
            
            pivot_data = {
                'hour': pivot_row['hour'],
                'day_of_week': pivot_row['timestamp'].dayofweek if hasattr(pivot_row['timestamp'], 'dayofweek') else 2,
                'position': pivot_row['position'],
                'distance_pct': distance_pct,
                'atr_pct': atr_pct,
                'historical_hit_rate': hist_hit_rate
            }
            
            features = self.engineer_features(pivot_data)
            features_list.append(features)
            labels.append(1 if pivot_row['hit'] else 0)
        
        X = pd.DataFrame(features_list)
        y = np.array(labels)
        
        self.feature_names = X.columns.tolist()
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        return True
    
    def predict(self, pivot_data):
        """
        Predict hit probability for a pivot
        
        Returns: (probability, confidence)
        """
        if not self.is_trained or self.model is None:
            return None, None
        
        features = self.engineer_features(pivot_data)
        X = pd.DataFrame([features])
        
        # Ensure features match training
        X = X[self.feature_names]
        
        probability = self.model.predict_proba(X)[0, 1]
        
        # Confidence based on probability extremes
        if probability > 0.75 or probability < 0.25:
            confidence = "High"
        elif probability > 0.65 or probability < 0.35:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return probability, confidence
    
    def get_feature_importance(self):
        """Get feature importance for interpretation"""
        if not self.is_trained:
            return None
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance


def analyze_gap_fills(df_15m, start_date, end_date, gap_end_time, gap_start_time, 
                     min_gap_pct, partial_threshold, day_filter, max_lookforward):
    """
    Analyze gap fills by creating custom gaps between two time periods
    Gap Close = price at gap_end_time (e.g., 20:00)
    Gap Open = price at gap_start_time next period (e.g., 06:00 next day)
    """
    
    df_filtered = df_15m[(df_15m.index.date >= start_date) & (df_15m.index.date <= end_date)].copy()
    
    results = {
        'times': [],
        'gap_directions': [],
        'gap_sizes': [],
        'gap_close_prices': [],
        'gap_open_prices': [],
        'bars_partial': [],
        'bars_full': [],
        'partial_pcts': [],
        'fill_status': [],
        'mae_values': [],
        'mae_pct': [],
        'day_names': []
    }
    
    # Generate list of gap close times
    dates = pd.date_range(start_date, end_date, freq='D')
    
    for date in dates:
        day_name = date.day_name()
        
        # Filter by day of week
        if day_filter and day_name not in day_filter:
            continue
        
        # Parse gap end time (gap close)
        gap_end_hour = int(gap_end_time.split(':')[0])
        gap_end_minute = int(gap_end_time.split(':')[1])
        gap_close_timestamp = pd.Timestamp(date) + pd.Timedelta(hours=gap_end_hour, minutes=gap_end_minute)
        
        # Parse gap start time (gap open) - next day if after midnight
        gap_start_hour = int(gap_start_time.split(':')[0])
        gap_start_minute = int(gap_start_time.split(':')[1])
        
        # If gap start time is before gap end time, it means next day
        if (gap_start_hour < gap_end_hour) or (gap_start_hour == gap_end_hour and gap_start_minute < gap_end_minute):
            gap_open_timestamp = pd.Timestamp(date) + pd.Timedelta(days=1, hours=gap_start_hour, minutes=gap_start_minute)
        else:
            gap_open_timestamp = pd.Timestamp(date) + pd.Timedelta(hours=gap_start_hour, minutes=gap_start_minute)
        
        # Check if both timestamps exist in data
        if gap_close_timestamp not in df_15m.index or gap_open_timestamp not in df_15m.index:
            continue
        
        gap_close_price = df_15m.loc[gap_close_timestamp, 'Close']
        gap_open_price = df_15m.loc[gap_open_timestamp, 'Open']
        
        # Calculate gap size
        gap_size = abs(gap_open_price - gap_close_price)
        gap_size_pct = (gap_size / gap_close_price) * 100 if gap_close_price != 0 else 0
        
        # Check minimum gap size
        if gap_size_pct < min_gap_pct:
            continue
        
        # Determine gap direction
        if gap_open_price > gap_close_price:
            gap_direction = 'Gap Up'
            target_price = gap_close_price
            is_gap_up = True
        else:
            gap_direction = 'Gap Down'
            target_price = gap_close_price
            is_gap_up = False
        
        # Calculate partial fill threshold
        partial_fill_price = gap_open_price + (target_price - gap_open_price) * (partial_threshold / 100)
        
        # Look forward for fills
        gap_open_idx = df_15m.index.get_loc(gap_open_timestamp)
        lookforward_end = min(gap_open_idx + max_lookforward, len(df_15m))
        
        bars_partial = None
        bars_full = None
        partial_pct_filled = 0
        fill_status = 'Unfilled'
        mae = 0
        mae_pct = 0
        
        # Check fills
        for j in range(gap_open_idx + 1, lookforward_end):
            future_candle = df_15m.iloc[j]
            bars_elapsed = j - gap_open_idx
            
            if is_gap_up:
                # Gap up needs to fill down
                if bars_partial is None and future_candle['Low'] <= partial_fill_price:
                    bars_partial = bars_elapsed
                    pct_filled = ((gap_open_price - future_candle['Low']) / gap_size) * 100
                    partial_pct_filled = min(pct_filled, 100)
                    fill_status = 'Partial'
                
                if future_candle['Low'] <= target_price:
                    bars_full = bars_elapsed
                    partial_pct_filled = 100
                    fill_status = 'Full'
                    break
            else:
                # Gap down needs to fill up
                if bars_partial is None and future_candle['High'] >= partial_fill_price:
                    bars_partial = bars_elapsed
                    pct_filled = ((future_candle['High'] - gap_open_price) / gap_size) * 100
                    partial_pct_filled = min(pct_filled, 100)
                    fill_status = 'Partial'
                
                if future_candle['High'] >= target_price:
                    bars_full = bars_elapsed
                    partial_pct_filled = 100
                    fill_status = 'Full'
                    break
        
        # Calculate MAE
        if is_gap_up:
            # Adverse is going higher (away from target)
            highest_reached = gap_open_price
            lookback_range = gap_open_idx + (bars_full if bars_full else bars_partial if bars_partial else max_lookforward)
            lookback_range = min(lookback_range, len(df_15m))
            for j in range(gap_open_idx + 1, lookback_range):
                highest_reached = max(highest_reached, df_15m.iloc[j]['High'])
            mae = highest_reached - gap_open_price
            mae_pct = (mae / gap_open_price) * 100 if gap_open_price != 0 else 0
        else:
            # Adverse is going lower (away from target)
            lowest_reached = gap_open_price
            lookback_range = gap_open_idx + (bars_full if bars_full else bars_partial if bars_partial else max_lookforward)
            lookback_range = min(lookback_range, len(df_15m))
            for j in range(gap_open_idx + 1, lookback_range):
                lowest_reached = min(lowest_reached, df_15m.iloc[j]['Low'])
            mae = gap_open_price - lowest_reached
            mae_pct = (mae / gap_open_price) * 100 if gap_open_price != 0 else 0
        
        # If unfilled, set bars_partial to max lookforward
        if fill_status == 'Unfilled':
            bars_partial = lookforward_end - gap_open_idx - 1
        
        # Store results
        results['times'].append(gap_open_timestamp)
        results['gap_directions'].append(gap_direction)
        results['gap_sizes'].append(gap_size_pct)
        results['gap_close_prices'].append(gap_close_price)
        results['gap_open_prices'].append(gap_open_price)
        results['bars_partial'].append(bars_partial)
        results['bars_full'].append(bars_full)
        results['partial_pcts'].append(partial_pct_filled)
        results['fill_status'].append(fill_status)
        results['mae_values'].append(mae)
        results['mae_pct'].append(mae_pct)
        results['day_names'].append(day_name)
    
    return results

def analyze_quartile_opens(df_15m, start_date, end_date, timeframe, day_filter, session_filter):
    """Analyze quartile opens - track if X candle sweeps X-1 high/low when opening in upper/lower quartile"""
    
    # Resample data based on timeframe
    if timeframe == '15m':
        df_resampled = df_15m.copy()
    elif timeframe == '30m':
        df_resampled = df_15m.resample('30min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == '1h':
        df_resampled = df_15m.resample('1H').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == 'Daily':
        df_resampled = df_15m.resample('1D').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == 'Weekly':
        df_resampled = df_15m.resample('W-SUN').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == 'Monthly':
        df_resampled = df_15m.resample('M').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
        }).dropna()
    elif timeframe == 'Session':
        # Build ALL session candles
        sessions = {
            'Asia': (0, 6),
            'London': (6, 12),
            'New York': (12, 20),
            'Close': (20, 24)
        }
        
        session_candles = []
        for date in pd.date_range(start_date, end_date, freq='D'):
            for session_name, (start_hour, end_hour) in sessions.items():
                session_start = pd.Timestamp(date) + pd.Timedelta(hours=start_hour)
                session_end = pd.Timestamp(date) + pd.Timedelta(hours=end_hour)
                
                session_data = df_15m[(df_15m.index >= session_start) & (df_15m.index < session_end)]
                
                if len(session_data) > 0:
                    session_candles.append({
                        'Time': session_start,
                        'Open': session_data['Open'].iloc[0],
                        'High': session_data['High'].max(),
                        'Low': session_data['Low'].min(),
                        'Close': session_data['Close'].iloc[-1],
                        'SessionType': session_name
                    })
        
        df_resampled = pd.DataFrame(session_candles).set_index('Time')
        df_resampled = df_resampled.sort_index()
    
    # Filter by date range
    df_resampled = df_resampled[(df_resampled.index.date >= start_date) & (df_resampled.index.date <= end_date)]
    
    # CRITICAL: Keep unfiltered version for sweep checking
    df_resampled_full = df_resampled.copy()
    
    results = {
        'times': [],
        'day_names': [],
        'sessions': [],
        'quartile_types': [],  # 'Upper', 'Lower', or 'None'
        'x_minus_1_high': [],
        'x_minus_1_low': [],
        'x_minus_1_open': [],
        'x_minus_1_close': [],
        'x_open': [],
        'x_high': [],
        'x_low': [],
        'x_close': [],
        'swept': [],  # True/False
        'bars_to_sweep': [],
        'mae_values': [],  # Maximum adverse excursion
        'mae_pct': [],  # MAE as percentage
        'sweep_times': [],  # When sweep occurred
        'sweep_hours': []  # Hour of day when sweep occurred
    }
    
    # Analyze each candle (starting from index 1, need X-1 candle)
    for i in range(1, len(df_resampled_full)):
        x_minus_1 = df_resampled_full.iloc[i-1]
        x_candle = df_resampled_full.iloc[i]
        
        x_time = df_resampled_full.index[i]
        day_name = x_time.day_name()
        
        # Filter by day of week
        if day_filter and day_name not in day_filter:
            continue
        
        # Filter by session (for Session timeframe only)
        if timeframe == 'Session':
            x_session_type = x_candle['SessionType']
            if session_filter and x_session_type not in session_filter:
                continue
        
        # Get X-1 candle range
        x_minus_1_high = x_minus_1['High']
        x_minus_1_low = x_minus_1['Low']
        x_minus_1_range = x_minus_1_high - x_minus_1_low
        
        if x_minus_1_range == 0:
            continue  # Skip if no range
        
        # Calculate quartiles
        upper_quartile_start = x_minus_1_low + (x_minus_1_range * 0.75)  # Top 25%
        lower_quartile_end = x_minus_1_low + (x_minus_1_range * 0.25)    # Bottom 25%
        
        # Check X candle open
        x_open = x_candle['Open']
        
        quartile_type = 'None'
        should_track = False
        target_level = None
        is_upper = False
        
        if x_open >= upper_quartile_start:
            # Opened in upper quartile - track if sweeps X-1 high
            quartile_type = 'Upper'
            should_track = True
            target_level = x_minus_1_high
            is_upper = True
        elif x_open <= lower_quartile_end:
            # Opened in lower quartile - track if sweeps X-1 low
            quartile_type = 'Lower'
            should_track = True
            target_level = x_minus_1_low
            is_upper = False
        
        if not should_track:
            continue  # Only track quartile opens
        
        # Determine session
        session_name = None
        if timeframe == 'Session':
            session_name = x_candle['SessionType']
        else:
            hour = x_time.hour
            if 0 <= hour < 6:
                session_name = 'Asia'
            elif 6 <= hour < 12:
                session_name = 'London'
            elif 12 <= hour < 20:
                session_name = 'New York'
            else:
                session_name = 'Close'
        
        # Define lookforward limit based on timeframe
        lookforward_limits = {
            '15m': 96,      # 1 day (24 hours / 15min = 96 bars)
            '30m': 48,      # 1 day (24 hours / 30min = 48 bars)
            '1h': 24,       # 1 day (24 hours / 1h = 24 bars)
            'Session': 4,   # 4 sessions (1 day)
            'Daily': 1,     # Next 1 day only
            'Weekly': 1,    # Next 1 week only
            'Monthly': 1    # Next 1 month only
        }
        
        max_lookforward = lookforward_limits.get(timeframe, 1)

        # Check for first sweep in current and future candles
        # IMPORTANT: Use df_resampled_full (unfiltered) for sweep checking
        swept = False
        bars_to_sweep = None
        mae = 0
        mae_pct = 0
        sweep_time = None
        sweep_hour = None

        lookforward_end = min(i + max_lookforward + 1, len(df_resampled_full))
        for j in range(i, lookforward_end):
            future_candle = df_resampled_full.iloc[j]

            if is_upper:
                if future_candle['High'] >= target_level:
                    swept = True
                    bars_to_sweep = j - i
                    sweep_time = df_resampled_full.index[j]
                    sweep_hour = sweep_time.hour
                    break
            else:
                if future_candle['Low'] <= target_level:
                    swept = True
                    bars_to_sweep = j - i
                    sweep_time = df_resampled_full.index[j]
                    sweep_hour = sweep_time.hour
                    break
        # Refine sweep timestamp/hour on raw 15m data so distribution reflects actual UTC sweep hour
        if swept:
            raw_end = sweep_time if sweep_time is not None else df_resampled_full.index[lookforward_end - 1]
            raw_window = df_15m[(df_15m.index >= x_time) & (df_15m.index <= raw_end)]

            if len(raw_window) > 0:
                if is_upper:
                    raw_hits = raw_window[raw_window["High"] >= target_level]
                else:
                    raw_hits = raw_window[raw_window["Low"] <= target_level]

                if len(raw_hits) > 0:
                    first_hit_time = raw_hits.index[0]
                    sweep_time = first_hit_time
                    sweep_hour = first_hit_time.hour

        # Calculate MAE (Maximum Adverse Excursion) within X candle only
        # For upper quartile: How far did price go DOWN in X candle
        # For lower quartile: How far did price go UP in X candle
        if is_upper:
            # Entry is x_open, adverse is going lower
            lowest_in_x = x_candle['Low']
            mae = x_open - lowest_in_x
            mae_pct = (mae / x_open) * 100 if x_open != 0 else 0
        else:
            # Entry is x_open, adverse is going higher
            highest_in_x = x_candle['High']
            mae = highest_in_x - x_open
            mae_pct = (mae / x_open) * 100 if x_open != 0 else 0
        
        # Store results
        results['times'].append(x_time)
        results['day_names'].append(day_name)
        results['sessions'].append(session_name)
        results['quartile_types'].append(quartile_type)
        results['x_minus_1_high'].append(x_minus_1_high)
        results['x_minus_1_low'].append(x_minus_1_low)
        results['x_minus_1_open'].append(x_minus_1['Open'])
        results['x_minus_1_close'].append(x_minus_1['Close'])
        results['x_open'].append(x_open)
        results['x_high'].append(x_candle['High'])
        results['x_low'].append(x_candle['Low'])
        results['x_close'].append(x_candle['Close'])
        results['swept'].append(swept)
        results['bars_to_sweep'].append(bars_to_sweep)
        results['mae_values'].append(mae)
        results['mae_pct'].append(mae_pct)
        results['sweep_times'].append(sweep_time)
        results['sweep_hours'].append(sweep_hour)
    
    return results

# ============================================================================
# SESSION TPO FUNCTIONS (WITH FIXES)
# ============================================================================

def calculate_atr(df, period=30):
    """Calculate ATR for tick size"""
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Handle NaN cases
    if len(atr) == 0 or pd.isna(atr.iloc[-1]):
        return 100.0  # Default fallback
    
    return atr.iloc[-1]

def round_to_tick(price, tick_size):
    """Round price to nearest tick"""
    # Validate inputs
    if pd.isna(price) or pd.isna(tick_size) or tick_size <= 0:
        return price
    
    return round(price / tick_size) * tick_size

def build_tpo_profile(session_data, tick_size):
    """Build TPO profile for a session"""
    
    profile = {}
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for idx, (timestamp, row) in enumerate(session_data.iterrows()):
        if idx >= len(letters):
            break
        
        letter = letters[idx]
        high = row['High']
        low = row['Low']
        
        low_tick = round_to_tick(low, tick_size)
        high_tick = round_to_tick(high, tick_size)
        
        current_price = low_tick
        while current_price <= high_tick:
            if current_price not in profile:
                profile[current_price] = []
            profile[current_price].append(letter)
            current_price += tick_size
    
    return profile

def analyze_tpo_profile(profile):
    """Analyze TPO profile for poor highs/lows - POC/VAH/VAL removed per user request"""
    
    if not profile:
        return None
    
    sorted_prices = sorted(profile.keys(), reverse=True)
    
    session_high = sorted_prices[0]
    session_low = sorted_prices[-1]
    
    tpos_at_high = len(profile[session_high])
    tpos_at_low = len(profile[session_low])
    
    # Poor high/low: 2+ TPO blocks at the extreme price level only
    poor_high_level = session_high if tpos_at_high >= 2 else None
    poor_low_level = session_low if tpos_at_low >= 2 else None
    is_poor_high = poor_high_level is not None
    is_poor_low = poor_low_level is not None
    
    # Calculate range
    price_range = session_high - session_low
    
    return {
        'profile': profile,
        'sorted_prices': sorted_prices,
        'session_high': session_high,
        'session_low': session_low,
        'tpos_at_high': tpos_at_high,
        'tpos_at_low': tpos_at_low,
        'poor_high_level': poor_high_level,
        'poor_low_level': poor_low_level,
        'is_poor_high': is_poor_high,
        'is_poor_low': is_poor_low,
        'range': price_range
    }

def check_sweep(poor_price, is_high, future_sessions, tick_size):
    """Check if poor extreme was swept in future sessions"""
    
    swept = False
    sweep_session_idx = None
    mae = 0
    mae_pct = 0
    
    for idx, session_data in enumerate(future_sessions):
        if len(session_data) == 0:
            continue
        
        if is_high:
            session_high = session_data['High'].max()
            if session_high >= poor_price:
                swept = True
                sweep_session_idx = idx
                
                before_sweep = session_data[session_data['High'] < poor_price]
                if len(before_sweep) > 0:
                    lowest = before_sweep['Low'].min()
                    mae = abs(poor_price - lowest)
                    mae_pct = (mae / poor_price) * 100
                break
        else:
            session_low = session_data['Low'].min()
            if session_low <= poor_price:
                swept = True
                sweep_session_idx = idx
                
                before_sweep = session_data[session_data['Low'] > poor_price]
                if len(before_sweep) > 0:
                    highest = before_sweep['High'].max()
                    mae = abs(highest - poor_price)
                    mae_pct = (mae / poor_price) * 100
                break
    
    return swept, sweep_session_idx, mae, mae_pct



def generate_tpo_letters(num_bars):
    """
    Generate TPO labels for any number of bars.
    A-Z (26), then a-z (26), then A1-Z1, a1-z1, A2-Z2, a2-z2, etc.
    """
    letters = []
    base_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    base_len = len(base_alphabet)

    for i in range(num_bars):
        if i < base_len:
            letters.append(base_alphabet[i])
        else:
            offset = i - base_len
            suffix_num = (offset // base_len) + 1
            letter_idx = offset % base_len
            letters.append(f"{base_alphabet[letter_idx]}{suffix_num}")

    return letters

def resample_to_30m(df_15m):
    """
    Resample 15m OHLC data to 30m
    """
    if len(df_15m) == 0:
        return df_15m
    
    # Resample to 30m
    df_30m = df_15m.resample('30min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return df_30m

def build_tpo_profile_with_letters(session_data, tick_size, tpo_letters=None):
    """
    Build TPO profile with custom letters (supports extended alphabet)
    """
    
    profile = {}
    
    if tpo_letters is None:
        # Generate standard A-Z letters
        tpo_letters = generate_tpo_letters(len(session_data))
    
    for idx, (timestamp, row) in enumerate(session_data.iterrows()):
        if idx >= len(tpo_letters):
            break
        
        letter = tpo_letters[idx]
        high = row['High']
        low = row['Low']
        
        # Validate inputs
        if pd.isna(high) or pd.isna(low):
            continue
        
        low_tick = round_to_tick(low, tick_size)
        high_tick = round_to_tick(high, tick_size)
        
        current_price = low_tick
        while current_price <= high_tick:
            if current_price not in profile:
                profile[current_price] = []
            
            profile[current_price].append(letter)
            
            current_price += tick_size
    
    return profile

def compute_value_area(profile_counts, value_area_pct=0.68):
    """
    Compute value area prices around POC using cumulative TPO counts.
    profile_counts: dict {price: count}
    Returns set of prices in value area.
    """
    if not profile_counts:
        return set()
    total_tpos = sum(profile_counts.values())
    if total_tpos == 0:
        return set()

    poc_price = max(profile_counts.keys(), key=lambda p: profile_counts[p])
    prices_sorted = sorted(profile_counts.keys())
    poc_idx = prices_sorted.index(poc_price)

    va_prices = {poc_price}
    cum = profile_counts[poc_price]
    left = poc_idx - 1
    right = poc_idx + 1

    while cum / total_tpos < value_area_pct and (left >= 0 or right < len(prices_sorted)):
        left_price = prices_sorted[left] if left >= 0 else None
        right_price = prices_sorted[right] if right < len(prices_sorted) else None

        left_count = profile_counts[left_price] if left_price is not None else -1
        right_count = profile_counts[right_price] if right_price is not None else -1

        if right_count >= left_count:
            if right_price is not None:
                va_prices.add(right_price)
                cum += right_count
            right += 1
        else:
            if left_price is not None:
                va_prices.add(left_price)
                cum += left_count
            left -= 1

    return va_prices

def blend_to_black(hex_color, factor):
    """
    Blend hex_color toward black by factor (0..1).
    """
    factor = max(0.0, min(1.0, factor))
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r * (1 - factor))
    g = int(g * (1 - factor))
    b = int(b * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"

def create_tpo_letter_annotations(profile, data_with_letters, letter_sequence):
    """
    Create TPO letter annotations overlaid on candlestick chart
    Similar to Session TPO display style
    """
    
    annotations = []
    
    if not profile or len(data_with_letters) == 0:
        return annotations
    
    # For each price level in the profile
    for price, tpos in profile.items():
        # For each letter at this price
        for tpo_letter in tpos:
            # Find the time position for this letter
            if tpo_letter in letter_sequence:
                letter_idx = letter_sequence.index(tpo_letter)
                
                if letter_idx < len(data_with_letters):
                    time_position = data_with_letters.index[letter_idx]
                    
                    annotations.append(
                        dict(
                            x=time_position,
                            y=price,
                            text=tpo_letter,
                            showarrow=False,
                            font=dict(size=7, color='white', family='Courier New, monospace'),
                            bgcolor='rgba(0,0,0,0.6)',
                            borderpad=1,
                            bordercolor='white',
                            borderwidth=0.5
                        )
                    )
    
    return annotations

def create_tpo_histogram_shapes(profile, chart_start_time, chart_width_minutes=1440):
    """
    Create TPO profile as histogram on LEFT side of chart (traditional market profile view)
    Matches ExoCharts display style - histogram bars on left with letters
    """
    
    shapes = []
    annotations = []
    
    if not profile:
        return shapes, annotations
    
    # Get TPO counts per price level
    tpo_counts = {price: len(tpos) for price, tpos in profile.items()}
    max_tpos = max(tpo_counts.values()) if tpo_counts else 1
    
    # Sort prices from high to low for proper display
    sorted_prices = sorted(tpo_counts.keys(), reverse=True)
    
    # Calculate histogram parameters
    # Use first 10% of chart width for TPO histogram
    histogram_width_minutes = chart_width_minutes * 0.10
    
    for price in sorted_prices:
        tpo_count = tpo_counts[price]
        tpo_letters = ''.join(sorted(profile[price]))
        
        # Normalize bar length (0 to 1)
        normalized_length = tpo_count / max_tpos
        
        # Calculate bar width in time
        bar_width_minutes = histogram_width_minutes * normalized_length
        bar_end_time = chart_start_time + timedelta(minutes=bar_width_minutes)
        
        # Create horizontal bar for this price level
        shapes.append(
            dict(
                type="rect",
                x0=chart_start_time,
                x1=bar_end_time,
                y0=price - 20,  # Bar thickness
                y1=price + 20,
                fillcolor='rgba(100, 149, 237, 0.6)',  # Cornflower blue
                line=dict(color='rgba(100, 149, 237, 0.9)', width=0.5),
                layer='below'
            )
        )
        
        # Add TPO letters inside the bar (on the left)
        annotations.append(
            dict(
                x=chart_start_time + timedelta(minutes=2),
                y=price,
                text=tpo_letters,
                showarrow=False,
                font=dict(size=7, color='white', family='Courier New, monospace', weight='bold'),
                xanchor='left',
                yanchor='middle',
                bgcolor='rgba(0,0,0,0.3)',
                borderpad=1
            )
        )
    
    return shapes, annotations

def analyze_session_tpo(df_15m, start_date, end_date, day_filter, session_filter, tpo_block_size, tick_mode, manual_tick, atr_multiplier, sessions_to_track):
    """Main TPO analysis function - FIXED with adjustable ATR multiplier"""
    
    df_tpo_30m = df_15m.resample('30min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    atr_period = 20

    if tpo_block_size == "30m":
        df_tpo = df_tpo_30m.copy()
    else:
        df_tpo = df_15m.copy()
    
    df_tpo = df_tpo[(df_tpo.index.date >= start_date) & (df_tpo.index.date <= end_date)]
    
    results = []
    
    all_sessions = []
    for date in pd.date_range(start_date, end_date, freq='D'):
        for session_name in ['Asia', 'London', 'NY', 'Close']:
            if session_name == 'Asia':
                start_hour, end_hour = 0, 6
            elif session_name == 'London':
                start_hour, end_hour = 6, 12
            elif session_name == 'NY':
                start_hour, end_hour = 12, 20
            else:
                start_hour, end_hour = 20, 24
            
            session_data = df_tpo[
                (df_tpo.index.date == date.date()) &
                (df_tpo.index.hour >= start_hour) &
                (df_tpo.index.hour < end_hour)
            ]

            session_data_30m = df_tpo_30m[
                (df_tpo_30m.index.date == date.date()) &
                (df_tpo_30m.index.hour >= start_hour) &
                (df_tpo_30m.index.hour < end_hour)
            ]
            
            if len(session_data) > 0:
                day_name = date.day_name()
                all_sessions.append({
                    'date': date,
                    'session': session_name,
                    'day_name': day_name,
                    'data': session_data
                })
    
    for i, sess_info in enumerate(all_sessions):
        date = sess_info['date']
        session_name = sess_info['session']
        day_name = sess_info['day_name']
        session_data = sess_info['data']
        
        if day_filter and day_name not in day_filter:
            continue
        
        if session_filter and session_name not in session_filter:
            continue
        
        if len(session_data) < 3:
            continue
        
        # FIXED: Use adjustable ATR multiplier (not hardcoded 1.5)
        if tick_mode == "Auto":
            atr_source = session_data_30m if len(session_data_30m) > 0 else session_data
            atr = calculate_atr(atr_source, atr_period)
            tick_size = atr * atr_multiplier
            
            # Validate tick_size
            if pd.isna(tick_size) or tick_size <= 0:
                tick_size = 100.0  # Default fallback
        else:
            tick_size = manual_tick
        
        tick_size = max(float(tick_size), 1.0)
        
        profile_analysis = analyze_tpo_profile(build_tpo_profile(session_data, tick_size))
        
        if not profile_analysis:
            continue
        
        future_sessions_data = []
        for j in range(i + 1, min(i + 1 + sessions_to_track, len(all_sessions))):
            future_sessions_data.append(all_sessions[j]['data'])
        
        poor_high_swept = False
        poor_high_sweep_session = None
        poor_high_mae = 0
        poor_high_mae_pct = 0
        
        poor_low_swept = False
        poor_low_sweep_session = None
        poor_low_mae = 0
        poor_low_mae_pct = 0
        
        if profile_analysis['is_poor_high']:
            poor_high_swept, sweep_idx, mae, mae_pct = check_sweep(
                profile_analysis['poor_high_level'],
                True,
                future_sessions_data,
                tick_size
            )
            poor_high_sweep_session = sweep_idx
            poor_high_mae = mae
            poor_high_mae_pct = mae_pct
        
        if profile_analysis['is_poor_low']:
            poor_low_swept, sweep_idx, mae, mae_pct = check_sweep(
                profile_analysis['poor_low_level'],
                False,
                future_sessions_data,
                tick_size
            )
            poor_low_sweep_session = sweep_idx
            poor_low_mae = mae
            poor_low_mae_pct = mae_pct
        
        results.append({
            'date': date,
            'session': session_name,
            'day_name': day_name,
            'tick_size': tick_size,
            'atr': atr if tick_mode == "Auto" else None,
            'session_high': profile_analysis['session_high'],
            'session_low': profile_analysis['session_low'],
            'tpos_at_high': profile_analysis['tpos_at_high'],
            'tpos_at_low': profile_analysis['tpos_at_low'],
            'is_poor_high': profile_analysis['is_poor_high'],
            'is_poor_low': profile_analysis['is_poor_low'],
            'poor_high_swept': poor_high_swept,
            'poor_high_sweep_session': poor_high_sweep_session,
            'poor_high_mae': poor_high_mae,
            'poor_high_mae_pct': poor_high_mae_pct,
            'poor_low_swept': poor_low_swept,
            'poor_low_sweep_session': poor_low_sweep_session,
            'poor_low_mae': poor_low_mae,
            'poor_low_mae_pct': poor_low_mae_pct,
            'profile_analysis': profile_analysis,
            'session_data': session_data
        })
    
    return results

# ============================================================================
# DAILY PIVOTS PAGE  
# ============================================================================

if page == "Daily Pivots" or (pivot_page_mode and pivot_section == "Daily"):
    if not pivot_page_mode:
        render_page_header(
            "Daily Pivots Analysis",
            "Analyze when daily highs and lows occur"
        )
    else:
        st.subheader("Daily Pivots Analysis")
    
    # Exchange & Asset at top
    render_exchange_asset_controls("daily_pivots")
    
    # Analysis type selection
    pivot_category = st.selectbox(
        "Select Analysis Type",
        options=["Hourly", "Session"],
        index=0,
        key="dp_pivot_category"
    )
    
    st.markdown("---")
    
    # ========================================================================
    # HOURLY SECTION
    # ========================================================================
    
    if pivot_category == "Hourly":
        col1, col2 = st.columns(2)
        
        with col1:
            day_filter_dp = st.multiselect(
                "Day of Week Filter",
                options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                key="dp_hourly_day_filter"
            )
        
        with col2:
            hour_select_dp = st.selectbox(
                "Hour to Analyze",
                options=list(range(24)),
                format_func=lambda x: f"{x:02d}:00",
                key="dp_hourly_hour"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            min_date = df_15m.index.min().date()
            max_date = df_15m.index.max().date()
            
            start_date_dp = st.date_input(
                "Start Date",
                value=max_date - timedelta(days=365),
                min_value=min_date,
                max_value=max_date,
                key="dp_hourly_start"
            )
        
        with col4:
            end_date_dp = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="dp_hourly_end"
            )
        
        st.markdown("---")
        analyze_dp = st.button(
            "Analyze Daily Pivots",
            use_container_width=True,
            type="primary",
            key="analyze_dp_hourly"
        )
    
    # ========================================================================
    # SESSION SECTION  
    # ========================================================================
    
    else:  # Session
        col1, col2 = st.columns(2)
        
        with col1:
            day_filter_dp = st.multiselect(
                "Day of Week Filter",
                options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                key="dp_session_day_filter"
            )
        
        with col2:
            session_filter_dp = st.multiselect(
                "Session Filter",
                options=['Asia', 'London', 'NY', 'Close'],
                default=['Asia', 'London', 'NY', 'Close'],
                key="dp_session_filter"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            min_date = df_15m.index.min().date()
            max_date = df_15m.index.max().date()
            
            start_date_dp = st.date_input(
                "Start Date",
                value=max_date - timedelta(days=365),
                min_value=min_date,
                max_value=max_date,
                key="dp_session_start"
            )
        
        with col4:
            end_date_dp = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="dp_session_end"
            )
        
        st.markdown("---")
        analyze_dp = st.button(
            "Analyze Daily Pivots",
            use_container_width=True,
            type="primary",
            key="analyze_dp_session"
        )
    
    # ========================================================================
    # ANALYSIS RESULTS
    # ========================================================================
    
    if pivot_category == "Hourly":
        if analyze_dp:
            with st.spinner("Analyzing daily pivots..."):
                results = analyze_daily_pivots_hourly(df_15m, df_daily, start_date_dp, end_date_dp, day_filter_dp)
                
                st.session_state.daily_pivots_analyzed = True
                st.session_state.daily_pivots_results = results
                st.session_state.daily_pivots_chart_index = -1
        
        if st.session_state.daily_pivots_analyzed and st.session_state.daily_pivots_results:
            results = st.session_state.daily_pivots_results
            
            high_hours = pd.Series(results['high_hours'])
            low_hours = pd.Series(results['low_hours'])
            p1_hours = pd.Series(results['p1_hours'])
            p2_hours = pd.Series(results['p2_hours'])
            
            total_days = len(results['dates'])
            
            st.markdown("---")
            st.subheader("Daily High/Low Distribution by Hour")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Daily High Distribution**")
                
                high_dist = []
                for hour in range(24):
                    count = (high_hours == hour).sum()
                    pct = (count / total_days * 100) if total_days > 0 else 0
                    
                    last_occurrences = [i for i, h in enumerate(results['high_hours']) if h == hour]
                    days_since = len(results['high_hours']) - last_occurrences[-1] if last_occurrences else total_days
                    
                    high_dist.append({
                        'Hour': f"{hour:02d}:00",
                        'Percentage': pct,
                        'Count': count,
                        'Days Since': days_since
                    })
                
                high_df = pd.DataFrame(high_dist)
                
                def color_high(val):
                    if val > 8:
                        return 'background-color: #28a745; color: white'
                    elif val > 5:
                        return 'background-color: #ffc107; color: black'
                    elif val > 2:
                        return 'background-color: #fd7e14; color: white'
                    else:
                        return 'background-color: #dc3545; color: white'
                
                # Format percentage column
                high_df_styled = high_df.copy()
                high_df_styled['Percentage'] = high_df_styled['Percentage'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(
                    high_df.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                    use_container_width=True,
                    height=600
                )
            
            with col2:
                st.write("**Daily Low Distribution**")
                
                low_dist = []
                for hour in range(24):
                    count = (low_hours == hour).sum()
                    pct = (count / total_days * 100) if total_days > 0 else 0
                    
                    last_occurrences = [i for i, h in enumerate(results['low_hours']) if h == hour]
                    days_since = len(results['low_hours']) - last_occurrences[-1] if last_occurrences else total_days
                    
                    low_dist.append({
                        'Hour': f"{hour:02d}:00",
                        'Percentage': pct,
                        'Count': count,
                        'Days Since': days_since
                    })
                
                low_df = pd.DataFrame(low_dist)
                
                st.dataframe(
                    low_df.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                    use_container_width=True,
                    height=600
                )
            
            st.markdown("---")
            st.subheader("P1/P2 Distribution by Hour")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**P1 Distribution**")
                
                p1_dist = []
                for hour in range(24):
                    count = (p1_hours == hour).sum()
                    pct = (count / total_days * 100) if total_days > 0 else 0
                    
                    last_occurrences = [i for i, h in enumerate(results['p1_hours']) if h == hour]
                    days_since = len(results['p1_hours']) - last_occurrences[-1] if last_occurrences else total_days
                    
                    p1_dist.append({
                        'Hour': f"{hour:02d}:00",
                        'Percentage': pct,
                        'Count': count,
                        'Days Since': days_since
                    })
                
                p1_df = pd.DataFrame(p1_dist)
                
                st.dataframe(
                    p1_df.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                    use_container_width=True,
                    height=600
                )
            
            with col2:
                st.write("**P2 Distribution**")
                
                p2_dist = []
                for hour in range(24):
                    count = (p2_hours == hour).sum()
                    pct = (count / total_days * 100) if total_days > 0 else 0
                    
                    last_occurrences = [i for i, h in enumerate(results['p2_hours']) if h == hour]
                    days_since = len(results['p2_hours']) - last_occurrences[-1] if last_occurrences else total_days
                    
                    p2_dist.append({
                        'Hour': f"{hour:02d}:00",
                        'Percentage': pct,
                        'Count': count,
                        'Days Since': days_since
                    })
                
                p2_df = pd.DataFrame(p2_dist)
                
                st.dataframe(
                    p2_df.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                    use_container_width=True,
                    height=600
                )
            
            st.markdown("---")
            st.subheader("P1/P2 Probability by Hour")
            
            p1_pcts = [p1_df.iloc[i]['Percentage'] for i in range(24)]
            p2_pcts = [p2_df.iloc[i]['Percentage'] for i in range(24)]
            total_pcts = [p1_pcts[i] + p2_pcts[i] for i in range(24)]
            hours_labels = [f"{i:02d}" for i in range(24)]
            
            fig_p1p2 = go.Figure()
            
            fig_p1p2.add_trace(go.Bar(
                x=hours_labels,
                y=p1_pcts,
                name='Pivot 1 %',
                marker_color=light_blue,
                text=[f"{v:.1f}" for v in p1_pcts],
                textposition='inside'
            ))
            
            fig_p1p2.add_trace(go.Bar(
                x=hours_labels,
                y=p2_pcts,
                name='Pivot 2 %',
                marker_color=dark_blue,
                text=[f"{v:.1f}" for v in p2_pcts],
                textposition='inside'
            ))
            
            fig_p1p2.add_trace(go.Scatter(
                x=hours_labels,
                y=total_pcts,
                name='Total %',
                mode='lines+markers',
                line=dict(color=orange_line, width=3),
                marker=dict(size=8)
            ))
            
            fig_p1p2.update_layout(
                barmode='group',
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(title="Hour", showgrid=False, color=text_color),
                yaxis=dict(title="Percentage (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )
            
            st.plotly_chart(fig_p1p2, use_container_width=True)
            
            st.markdown("---")
            st.subheader(f"Hit Rate Evolution: {hour_select_dp:02d}:00")
            
            rolling_window_dp = st.number_input(
                "Rolling Window (days)",
                min_value=7,
                max_value=365,
                value=56,
                step=7,
                key="dp_rolling_window"
            )
            
            selected_hour_highs = [(i, h) for i, h in enumerate(results['high_hours']) if h == hour_select_dp]
            selected_hour_lows = [(i, h) for i, h in enumerate(results['low_hours']) if h == hour_select_dp]
            
            if len(selected_hour_highs) > 0 or len(selected_hour_lows) > 0:
                dates = results['dates']
                high_hits = [1 if h == hour_select_dp else 0 for h in results['high_hours']]
                low_hits = [1 if h == hour_select_dp else 0 for h in results['low_hours']]
                
                df_evolution = pd.DataFrame({
                    'date': dates,
                    'high_hit': high_hits,
                    'low_hit': low_hits
                })
                
                df_evolution['cumulative_high'] = (df_evolution['high_hit'].expanding().sum() / df_evolution['high_hit'].expanding().count()) * 100
                df_evolution['cumulative_low'] = (df_evolution['low_hit'].expanding().sum() / df_evolution['low_hit'].expanding().count()) * 100
                
                df_evolution['rolling_high'] = df_evolution['high_hit'].rolling(window=min(rolling_window_dp, len(df_evolution)), min_periods=1).mean() * 100
                df_evolution['rolling_low'] = df_evolution['low_hit'].rolling(window=min(rolling_window_dp, len(df_evolution)), min_periods=1).mean() * 100
                
                # Separate chart for Daily Highs
                st.write("**Daily Highs Hit Rate Evolution**")
                
                fig_evolution_high = go.Figure()
                
                fig_evolution_high.add_trace(go.Scatter(
                    x=df_evolution['date'],
                    y=df_evolution['cumulative_high'],
                    mode='lines',
                    name='Cumulative High Rate',
                    line=dict(color=blue_color, width=2)
                ))
                
                fig_evolution_high.add_trace(go.Scatter(
                    x=df_evolution['date'],
                    y=df_evolution['rolling_high'],
                    mode='lines',
                    name=f'Rolling High Rate ({rolling_window_dp}d)',
                    line=dict(color=yellow_color, width=2, dash='dash')
                ))
                
                fig_evolution_high.update_layout(
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=350
                )
                
                st.plotly_chart(fig_evolution_high, use_container_width=True)
                
                # Separate chart for Daily Lows
                st.write("**Daily Lows Hit Rate Evolution**")
                
                fig_evolution_low = go.Figure()
                
                fig_evolution_low.add_trace(go.Scatter(
                    x=df_evolution['date'],
                    y=df_evolution['cumulative_low'],
                    mode='lines',
                    name='Cumulative Low Rate',
                    line=dict(color=dark_blue, width=2)
                ))
                
                fig_evolution_low.add_trace(go.Scatter(
                    x=df_evolution['date'],
                    y=df_evolution['rolling_low'],
                    mode='lines',
                    name=f'Rolling Low Rate ({rolling_window_dp}d)',
                    line=dict(color=orange_line, width=2, dash='dash')
                ))
                
                fig_evolution_low.update_layout(
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=350
                )
                
                st.plotly_chart(fig_evolution_low, use_container_width=True)
            
            st.markdown("---")
            st.subheader(f"Historical Highs/Lows at {hour_select_dp:02d}:00")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                bars_before_dp = st.number_input(
                    "Bars Before",
                    min_value=1,
                    max_value=200,
                    value=30,
                    step=1,
                    key="dp_bars_before"
                )
            
            with col2:
                bars_after_dp = st.number_input(
                    "Bars After",
                    min_value=1,
                    max_value=200,
                    value=30,
                    step=1,
                    key="dp_bars_after"
                )
            
            with col3:
                high_low_filter = st.selectbox(
                    "Show",
                    options=["Highs", "Lows"],
                    key="dp_high_low_filter"
                )
            
            with col4:
                pass  # Spacer
            
            if high_low_filter == "Highs":
                instances = [(results['dates'][i], results['high_hours'][i]) for i in range(len(results['dates'])) if results['high_hours'][i] == hour_select_dp]
            else:
                instances = [(results['dates'][i], results['low_hours'][i]) for i in range(len(results['dates'])) if results['low_hours'][i] == hour_select_dp]
            
            if len(instances) > 0:
                col5, col6 = st.columns([1, 1])
                
                with col5:
                    if st.button("◀ Previous", use_container_width=True, key="dp_prev", type="secondary"):
                        if st.session_state.daily_pivots_chart_index > 0:
                            st.session_state.daily_pivots_chart_index -= 1
                        else:
                            st.session_state.daily_pivots_chart_index = len(instances) - 1
                        st.rerun()
                
                with col6:
                    if st.button("Next ▶", use_container_width=True, key="dp_next"):
                        if st.session_state.daily_pivots_chart_index < len(instances) - 1:
                            st.session_state.daily_pivots_chart_index += 1
                        else:
                            st.session_state.daily_pivots_chart_index = 0
                        st.rerun()
                
                current_idx = st.session_state.daily_pivots_chart_index
                if current_idx == -1:
                    current_idx = len(instances) - 1
                
                selected_date = instances[current_idx][0]
                
                st.caption(f"Showing instance {current_idx + 1} of {len(instances)} | {selected_date.strftime('%Y-%m-%d')}")
                
                df_30m = df_15m.resample('30min').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()
                
                day_30m = df_30m[df_30m.index.date == selected_date.date()]
                target_hour_data = day_30m[day_30m.index.hour == hour_select_dp]
                
                if len(target_hour_data) > 0:
                    target_time = target_hour_data.index[0]
                    
                    target_idx = df_30m.index.get_indexer([target_time], method='nearest')[0]
                    
                    start_idx = max(0, target_idx - bars_before_dp)
                    end_idx = min(len(df_30m) - 1, target_idx + bars_after_dp)
                    
                    chart_data = df_30m.iloc[start_idx:end_idx+1]
                    
                    fig_candles = go.Figure()
                    
                    # Build hover text for OHLC data
                    hover_text = []
                    
                    for idx, row in chart_data.iterrows():
                        is_up = row['Close'] > row['Open']
                        # Highlight target candle with gold color
                        if idx == target_time:
                            color = highlight_color
                        else:
                            color = candle_up if is_up else candle_down
                        
                        # Collect hover text
                        hover_text.append(
                            f"Date: {idx}<br>"
                            f"Open: {row['Open']:.2f}<br>"
                            f"High: {row['High']:.2f}<br>"
                            f"Low: {row['Low']:.2f}<br>"
                            f"Close: {row['Close']:.2f}"
                        )
                        
                        fig_candles.add_trace(go.Bar(
                            x=[idx],
                            y=[abs(row['Close'] - row['Open'])],
                            base=[min(row['Open'], row['Close'])],
                            marker_color=color,
                            marker_line_width=0,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        fig_candles.add_trace(go.Scatter(
                            x=[idx, idx],
                            y=[row['Low'], row['High']],
                            mode='lines',
                            line=dict(color=color, width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # Add invisible hover trace
                    fig_candles.add_trace(go.Scatter(
                        x=chart_data.index,
                        y=chart_data['Close'],
                        mode='markers',
                        marker=dict(size=0.1, opacity=0),
                        text=hover_text,
                        hoverinfo='text',
                        showlegend=False
                    ))
                    
                    fig_candles.update_layout(
                        title=f"{high_low_filter[:-1]} at {hour_select_dp:02d}:00 - {selected_date.strftime('%Y-%m-%d')}",
                        plot_bgcolor=plot_bg,
                        paper_bgcolor=plot_bg,
                        font=dict(color=text_color),
                        xaxis=dict(showgrid=False, color=text_color),
                        yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_candles, use_container_width=True)
            
            st.markdown("---")
    
    # ========================================================================
    # SESSION SECTION
    # ========================================================================
    
    elif pivot_category == "Session":
        st.subheader("Session Analysis Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            day_filter_session = st.multiselect(
                "Day of Week Filter",
                options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                key="dp_session_day_filter_session"
            )
        
        with col2:
            session_select = st.selectbox(
                "Session to Analyze",
                options=['Asia', 'London', 'NY', 'Close'],
                key="dp_session_select"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            min_date = df_15m.index.min().date()
            max_date = df_15m.index.max().date()
            
            start_date_session = st.date_input(
                "Start Date",
                value=max_date - timedelta(days=365),
                min_value=min_date,
                max_value=max_date,
                key="dp_session_start_session"
            )
        
        with col4:
            end_date_session = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="dp_session_end_session"
            )
        
        st.markdown("---")
        analyze_session = st.button(
            "Analyze Daily Pivots",
            use_container_width=True,
            type="primary",
            key="analyze_dp_session_alt"
        )
        
        if analyze_session:
            with st.spinner("Analyzing session pivots..."):
                results_session = analyze_daily_pivots_session(df_15m, df_daily, start_date_session, end_date_session, day_filter_session)
                
                st.session_state.daily_pivots_analyzed = True
                st.session_state.daily_pivots_results = results_session
                st.session_state.daily_pivots_chart_index = -1
        
        if st.session_state.daily_pivots_analyzed and st.session_state.daily_pivots_results:
            results_session = st.session_state.daily_pivots_results
            
            high_sessions = pd.Series(results_session['high_sessions'])
            low_sessions = pd.Series(results_session['low_sessions'])
            p1_sessions = pd.Series(results_session['p1_sessions'])
            p2_sessions = pd.Series(results_session['p2_sessions'])
            
            total_days = len(results_session['dates'])
            
            st.markdown("---")
            st.subheader("Session Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Daily High/Low by Session**")
                
                session_dist = []
                for session in ['Asia', 'London', 'NY', 'Close']:
                    high_count = (high_sessions == session).sum()
                    low_count = (low_sessions == session).sum()
                    high_pct = (high_count / total_days * 100) if total_days > 0 else 0
                    low_pct = (low_count / total_days * 100) if total_days > 0 else 0
                    
                    session_dist.append({
                        'Session': session,
                        'High %': high_pct,
                        'Low %': low_pct
                    })
                
                session_df = pd.DataFrame(session_dist)
                
                def color_percentage(val):
                    if val > 30:
                        return 'background-color: #28a745; color: white'
                    elif val > 25:
                        return 'background-color: #ffc107; color: black'
                    elif val > 20:
                        return 'background-color: #fd7e14; color: white'
                    else:
                        return 'background-color: #dc3545; color: white'
                
                st.dataframe(
                    session_df.style.applymap(color_percentage, subset=['High %', 'Low %']).format({'High %': '{:.2f}%', 'Low %': '{:.2f}%'}),
                    use_container_width=True,
                    hide_index=True
                )
            
            with col2:
                st.write("**P1/P2 by Session**")
                
                p1p2_dist = []
                for session in ['Asia', 'London', 'NY', 'Close']:
                    p1_count = (p1_sessions == session).sum()
                    p2_count = (p2_sessions == session).sum()
                    p1_pct = (p1_count / total_days * 100) if total_days > 0 else 0
                    p2_pct = (p2_count / total_days * 100) if total_days > 0 else 0
                    
                    p1p2_dist.append({
                        'Session': session,
                        'P1 %': p1_pct,
                        'P2 %': p2_pct
                    })
                
                p1p2_df = pd.DataFrame(p1p2_dist)
                
                def color_percentage(val):
                    if val > 30:
                        return 'background-color: #28a745; color: white'
                    elif val > 25:
                        return 'background-color: #ffc107; color: black'
                    elif val > 20:
                        return 'background-color: #fd7e14; color: white'
                    else:
                        return 'background-color: #dc3545; color: white'
                
                st.dataframe(
                    p1p2_df.style.applymap(color_percentage, subset=['P1 %', 'P2 %']).format({'P1 %': '{:.2f}%', 'P2 %': '{:.2f}%'}),
                    use_container_width=True,
                    hide_index=True
                )
            
            st.markdown("---")
            st.subheader("P1/P2 Probability by Session")
            
            sessions_list = ['Asia', 'London', 'NY', 'Close']
            # Recalculate numeric percentages for chart
            p1_pcts_session = []
            p2_pcts_session = []
            for session in sessions_list:
                p1_count = (p1_sessions == session).sum()
                p2_count = (p2_sessions == session).sum()
                p1_pcts_session.append((p1_count / total_days * 100) if total_days > 0 else 0)
                p2_pcts_session.append((p2_count / total_days * 100) if total_days > 0 else 0)
            
            total_pcts_session = [p1_pcts_session[i] + p2_pcts_session[i] for i in range(len(sessions_list))]
            
            fig_session = go.Figure()
            
            fig_session.add_trace(go.Bar(
                x=sessions_list,
                y=p1_pcts_session,
                name='Pivot 1 %',
                marker_color=light_blue
            ))
            
            fig_session.add_trace(go.Bar(
                x=sessions_list,
                y=p2_pcts_session,
                name='Pivot 2 %',
                marker_color=dark_blue
            ))
            
            fig_session.add_trace(go.Scatter(
                x=sessions_list,
                y=total_pcts_session,
                name='Total %',
                mode='lines+markers',
                line=dict(color=orange_line, width=3),
                marker=dict(size=10)
            ))
            
            fig_session.update_layout(
                barmode='group',  # Changed from 'stack' to 'group'
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(title="Session", showgrid=False, color=text_color),
                yaxis=dict(title="Percentage (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            
            st.plotly_chart(fig_session, use_container_width=True)
            
            st.markdown("---")
            st.subheader(f"Hit Rate Evolution: {session_select} Session")
            
            rolling_window_session = st.number_input(
                "Rolling Window (days)",
                min_value=7,
                max_value=365,
                value=56,
                step=7,
                key="dp_session_rolling"
            )
            
            dates = results_session['dates']
            # For composite: count if ANY pivot hit (high OR low in selected session)
            composite_hits = []
            for i in range(len(dates)):
                hit = 0
                if results_session['high_sessions'][i] == session_select or results_session['low_sessions'][i] == session_select:
                    hit = 1
                composite_hits.append(hit)
            
            # For high only: count if high in selected session
            high_hits = [1 if s == session_select else 0 for s in results_session['high_sessions']]
            
            # For low only: count if low in selected session
            low_hits = [1 if s == session_select else 0 for s in results_session['low_sessions']]
            
            df_evolution_session = pd.DataFrame({
                'date': dates,
                'composite_hit': composite_hits,
                'high_hit': high_hits,
                'low_hit': low_hits
            })
            
            # CHART 1: Composite Hit Rate
            st.write("**Composite Hit Rate (High or Low in Selected Session)**")
            
            df_evolution_session['cumulative_composite'] = (df_evolution_session['composite_hit'].expanding().sum() / df_evolution_session['composite_hit'].expanding().count()) * 100
            df_evolution_session['rolling_composite'] = df_evolution_session['composite_hit'].rolling(window=min(rolling_window_session, len(df_evolution_session)), min_periods=1).mean() * 100
            
            fig_composite = go.Figure()
            
            fig_composite.add_trace(go.Scatter(
                x=df_evolution_session['date'],
                y=df_evolution_session['cumulative_composite'],
                mode='lines',
                name='Cumulative Composite',
                line=dict(color=blue_color, width=2)
            ))
            
            fig_composite.add_trace(go.Scatter(
                x=df_evolution_session['date'],
                y=df_evolution_session['rolling_composite'],
                mode='lines',
                name=f'Rolling Composite ({rolling_window_session}d)',
                line=dict(color=yellow_color, width=2, dash='dash')
            ))
            
            fig_composite.update_layout(
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(showgrid=False, color=text_color),
                yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=350
            )
            
            st.plotly_chart(fig_composite, use_container_width=True)
            
            # CHART 2: High Hit Rate Only
            st.write("**High Hit Rate (Daily High in Selected Session)**")
            
            df_evolution_session['cumulative_high'] = (df_evolution_session['high_hit'].expanding().sum() / df_evolution_session['high_hit'].expanding().count()) * 100
            df_evolution_session['rolling_high'] = df_evolution_session['high_hit'].rolling(window=min(rolling_window_session, len(df_evolution_session)), min_periods=1).mean() * 100
            
            fig_high = go.Figure()
            
            fig_high.add_trace(go.Scatter(
                x=df_evolution_session['date'],
                y=df_evolution_session['cumulative_high'],
                mode='lines',
                name='Cumulative High',
                line=dict(color=light_blue, width=2)
            ))
            
            fig_high.add_trace(go.Scatter(
                x=df_evolution_session['date'],
                y=df_evolution_session['rolling_high'],
                mode='lines',
                name=f'Rolling High ({rolling_window_session}d)',
                line=dict(color='#FFA500', width=2, dash='dash')
            ))
            
            fig_high.update_layout(
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(showgrid=False, color=text_color),
                yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=350
            )
            
            st.plotly_chart(fig_high, use_container_width=True)
            
            # CHART 3: Low Hit Rate Only
            st.write("**Low Hit Rate (Daily Low in Selected Session)**")
            
            df_evolution_session['cumulative_low'] = (df_evolution_session['low_hit'].expanding().sum() / df_evolution_session['low_hit'].expanding().count()) * 100
            df_evolution_session['rolling_low'] = df_evolution_session['low_hit'].rolling(window=min(rolling_window_session, len(df_evolution_session)), min_periods=1).mean() * 100
            
            fig_low = go.Figure()
            
            fig_low.add_trace(go.Scatter(
                x=df_evolution_session['date'],
                y=df_evolution_session['cumulative_low'],
                mode='lines',
                name='Cumulative Low',
                line=dict(color=dark_blue, width=2)
            ))
            
            fig_low.add_trace(go.Scatter(
                x=df_evolution_session['date'],
                y=df_evolution_session['rolling_low'],
                mode='lines',
                name=f'Rolling Low ({rolling_window_session}d)',
                line=dict(color=orange_line, width=2, dash='dash')
            ))
            
            fig_low.update_layout(
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(showgrid=False, color=text_color),
                yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=350
            )
            
            st.plotly_chart(fig_low, use_container_width=True)
            
            # 30M CANDLESTICK CHART WITH NAVIGATION
            st.markdown("---")
            st.subheader("30m Candlestick Chart - Daily High/Low Candles")
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                bars_before_dp = st.number_input(
                    "Bars Before",
                    min_value=1,
                    max_value=200,
                    value=20,
                    step=5,
                    key="dp_bars_before"
                )
            
            with col2:
                bars_after_dp = st.number_input(
                    "Bars After",
                    min_value=1,
                    max_value=200,
                    value=30,
                    step=5,
                    key="dp_bars_after"
                )
            
            with col3:
                st.write("")
                st.write("")
                if st.button("◀ Previous", use_container_width=True, key="prev_dp", type="secondary"):
                    if st.session_state.daily_pivots_chart_index > 0:
                        st.session_state.daily_pivots_chart_index -= 1
                    else:
                        st.session_state.daily_pivots_chart_index = len(dates) - 1
                    st.rerun()
            
            with col4:
                st.write("")
                st.write("")
                if st.button("Next ▶", use_container_width=True, key="next_dp", type="secondary"):
                    if st.session_state.daily_pivots_chart_index < len(dates) - 1:
                        st.session_state.daily_pivots_chart_index += 1
                    else:
                        st.session_state.daily_pivots_chart_index = 0
                    st.rerun()
            
            # Get current day
            current_dp_idx = st.session_state.daily_pivots_chart_index
            if current_dp_idx == -1 or current_dp_idx >= len(dates):
                current_dp_idx = len(dates) - 1
            
            # Ensure index is within bounds (in case of stale session state)
            current_dp_idx = max(0, min(current_dp_idx, len(dates) - 1))
            
            current_date = dates[current_dp_idx]
            high_session = results_session['high_sessions'][current_dp_idx]
            low_session = results_session['low_sessions'][current_dp_idx]
            high_time = results_session['high_times'][current_dp_idx]
            low_time = results_session['low_times'][current_dp_idx]
            high_price = results_session['high_prices'][current_dp_idx]
            low_price = results_session['low_prices'][current_dp_idx]
            
            st.caption(f"Day {current_dp_idx + 1} of {len(dates)} | {current_date.strftime('%Y-%m-%d')} | High: {high_session} ({high_time.strftime('%H:%M')}) | Low: {low_session} ({low_time.strftime('%H:%M')})")
            
            # Resample to 30m
            df_30m = df_15m.resample('30min').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            
            # Get day's 30m candles
            day_30m = df_30m[df_30m.index.date == current_date.date()]
            
            if len(day_30m) > 0:
                # Find which 30m candles contain the high and low
                high_candle_idx = None
                low_candle_idx = None
                
                for idx, candle in day_30m.iterrows():
                    if high_time >= idx and high_time < idx + pd.Timedelta(minutes=30):
                        high_candle_idx = idx
                    if low_time >= idx and low_time < idx + pd.Timedelta(minutes=30):
                        low_candle_idx = idx
                
                # Build chart
                fig_dp = go.Figure()
                
                # Get index positions
                high_pos = day_30m.index.get_loc(high_candle_idx) if high_candle_idx is not None and high_candle_idx in day_30m.index else -1
                low_pos = day_30m.index.get_loc(low_candle_idx) if low_candle_idx is not None and low_candle_idx in day_30m.index else -1
                
                # Get visible range
                start_pos = max(0, min(high_pos, low_pos) - bars_before_dp) if high_pos >= 0 or low_pos >= 0 else 0
                end_pos = min(len(day_30m), max(high_pos, low_pos) + bars_after_dp + 1) if high_pos >= 0 or low_pos >= 0 else len(day_30m)
                
                chart_data_dp = day_30m.iloc[start_pos:end_pos]
                
                # Add candles
                for idx, row in chart_data_dp.iterrows():
                    is_high_candle = (idx == high_candle_idx)
                    is_low_candle = (idx == low_candle_idx)
                    is_up = row['Close'] > row['Open']
                    
                    if is_high_candle or is_low_candle:
                        color = highlight_color  # Gold
                    else:
                        color = candle_up if is_up else candle_down
                    
                    # Body
                    fig_dp.add_trace(go.Bar(
                        x=[idx],
                        y=[abs(row['Close'] - row['Open'])],
                        base=[min(row['Open'], row['Close'])],
                        marker_color=color,
                        marker_line_width=0,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # Wicks
                    fig_dp.add_trace(go.Scatter(
                        x=[idx, idx],
                        y=[row['Low'], row['High']],
                        mode='lines',
                        line=dict(color=color, width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                add_ohlc_hover_trace(fig_dp, chart_data_dp)
                
                fig_dp.update_layout(
                    title=f"Daily High/Low - {current_date.strftime('%Y-%m-%d')} (30m candles)",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_dp, use_container_width=True)
            else:
                st.warning("No 30m data available for this day")
            
            st.markdown("---")
            

# Continue with all other pages in next message due to length...
# ============================================================================
# HIGH HIT RATE LEVELS
# ============================================================================

def calculate_hourly_pivots(df_15m, start_date, end_date, day_filter=None):
    df_hourly = df_15m.resample('1H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    df_hourly = df_hourly[(df_hourly.index.date >= start_date) & (df_hourly.index.date <= end_date)]
    
    pivots_data = []
    
    for i in range(24, len(df_hourly)):
        current_time = df_hourly.index[i]
        current_hour = current_time.hour
        current_day = current_time.day_name()
        
        if day_filter and current_day not in day_filter:
            continue
        
        lookback_data = df_hourly.iloc[i-24:i]
        
        high_24h = lookback_data['High'].max()
        low_24h = lookback_data['Low'].min()
        close_24h = lookback_data['Close'].iloc[-1]
        
        pivot = (high_24h + low_24h + close_24h + close_24h) / 4
        
        current_price = df_hourly.iloc[i]['Open']
        
        position = 'Above' if pivot > current_price else 'Below'
        
        next_24h_idx = min(i + 24, len(df_hourly))
        future_data = df_hourly.iloc[i:next_24h_idx]
        
        hit = False
        hit_hour = None
        bars_to_hit = None
        mae = 0
        
        if position == 'Above':
            if (future_data['High'] >= pivot).any():
                hit = True
                hit_idx = future_data[future_data['High'] >= pivot].index[0]
                hit_hour = hit_idx.hour
                # Calculate bars to hit
                hit_position = future_data.index.get_loc(hit_idx)
                bars_to_hit = hit_position
                
                data_before_hit = future_data.loc[:hit_idx]
                min_price = data_before_hit['Low'].min()
                mae = abs(pivot - min_price)
        else:
            if (future_data['Low'] <= pivot).any():
                hit = True
                hit_idx = future_data[future_data['Low'] <= pivot].index[0]
                hit_hour = hit_idx.hour
                # Calculate bars to hit
                hit_position = future_data.index.get_loc(hit_idx)
                bars_to_hit = hit_position
                
                data_before_hit = future_data.loc[:hit_idx]
                max_price = data_before_hit['High'].max()
                mae = abs(max_price - pivot)
        
        pivots_data.append({
            'timestamp': current_time,
            'hour': current_hour,
            'day_name': current_day,
            'pivot': pivot,
            'current_price': current_price,
            'position': position,
            'hit': hit,
            'hit_hour': hit_hour,
            'bars_to_hit': bars_to_hit,
            'mae': mae,
            'mae_pct': (mae / pivot * 100) if pivot > 0 else 0
        })
    
    return pd.DataFrame(pivots_data)

def calculate_hit_rates(pivots_df):
    hit_rates = []
    
    for hour in range(24):
        for position in ['Above', 'Below']:
            subset = pivots_df[(pivots_df['hour'] == hour) & (pivots_df['position'] == position)]
            
            if len(subset) > 0:
                hit_count = subset['hit'].sum()
                total_count = len(subset)
                hit_rate = (hit_count / total_count * 100) if total_count > 0 else 0
                
                hit_rates.append({
                    'hour': hour,
                    'position': position,
                    'hit_rate': hit_rate,
                    'hit_count': int(hit_count),
                    'total_count': int(total_count),
                    'miss_count': int(total_count - hit_count)
                })
    
    return pd.DataFrame(hit_rates)

def create_mae_buckets(mae_values):
    buckets = []
    
    for mae in mae_values:
        if mae < 5:
            bucket = int(mae / 0.5) * 0.5
            bucket_label = f"{bucket:.1f}-{bucket+0.5:.1f}%"
        elif mae < 10:
            bucket = int(mae)
            bucket_label = f"{bucket}-{bucket+1}%"
        else:
            bucket_label = "10+%"
        
        buckets.append(bucket_label)
    
    return buckets

if page == "High Hit Rate Levels":
    st.markdown("<div class='page-title'>High Hit Rate Levels</div>", unsafe_allow_html=True)
    # Removed subtitle per request

    render_exchange_asset_controls("hhrl")

    col1, col2 = st.columns(2)

    with col1:
        day_filter_hhrl = st.multiselect(
            "Day of Week Filter",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            help="Only analyze pivots plotted on these days"
        )

    with col2:
        hour_select = st.selectbox(
            "Hour to Analyze",
            options=list(range(24)),
            format_func=lambda x: f"{x:02d}:00",
            help="Select specific hour for detailed analysis"
        )

    col3, col4 = st.columns(2)

    with col3:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()

        start_date_hhrl = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=1095),
            min_value=min_date,
            max_value=max_date,
            key="hhrl_start"
        )

    with col4:
        end_date_hhrl = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="hhrl_end"
        )

    st.markdown("<div class='btn-hhrl'>", unsafe_allow_html=True)
    analyze_hhrl = st.button(
        " Analyze High Hit Rate Levels",
        use_container_width=True,
        type="primary",
        key="analyze_hhrl"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    if analyze_hhrl:
        with st.spinner("Calculating pivot levels and hit rates..."):
            pivots_df_all = calculate_hourly_pivots(df_15m, start_date_hhrl, end_date_hhrl, None)
            pivots_df_selected = calculate_hourly_pivots(df_15m, start_date_hhrl, end_date_hhrl, day_filter_hhrl)
            hit_rates_df = calculate_hit_rates(pivots_df_selected)
            
            st.session_state.hhrl_analyzed = True
            st.session_state.hhrl_results = {
                'pivots_df_all': pivots_df_all,
                'pivots_df_selected': pivots_df_selected,
                'hit_rates_df': hit_rates_df
            }
            st.session_state.hhrl_pivot_index = -1
    
    if st.session_state.hhrl_analyzed and st.session_state.hhrl_results:
        results = st.session_state.hhrl_results
        pivots_df_all = results['pivots_df_all']
        pivots_df_selected = results['pivots_df_selected']
        hit_rates_df = results['hit_rates_df']
        
        st.markdown("---")
        st.subheader("Hit Rate Heatmap (Filtered Days)")
        
        heatmap_data = hit_rates_df.pivot(index='position', columns='hour', values='hit_rate')
        heatmap_data = heatmap_data.reindex(['Below', 'Above'])
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[f"{h:02d}:00" for h in heatmap_data.columns],
            y=heatmap_data.index,
            colorscale='RdYlGn',
            text=heatmap_data.values.round(1),
            texttemplate='%{text}%',
            textfont={"size": 10},
            colorbar=dict(title="Hit Rate %")
        ))
        
        fig_heatmap.update_layout(
            plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg,
            font=dict(color=text_color),
            xaxis=dict(title="Hour", color=text_color, showgrid=False),
            yaxis=dict(title="Position", color=text_color),
            height=250
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.markdown("---")
        st.subheader(f"Detailed Analysis: {hour_select:02d}:00")
        
        hour_data_all = pivots_df_all[pivots_df_all['hour'] == hour_select]
        hour_above_all = hour_data_all[hour_data_all['position'] == 'Above']
        hour_below_all = hour_data_all[hour_data_all['position'] == 'Below']
        
        above_all_hit_rate = (hour_above_all['hit'].sum() / len(hour_above_all) * 100) if len(hour_above_all) > 0 else 0
        below_all_hit_rate = (hour_below_all['hit'].sum() / len(hour_below_all) * 100) if len(hour_below_all) > 0 else 0
        
        hour_data_selected = pivots_df_selected[pivots_df_selected['hour'] == hour_select]
        hour_above_selected = hour_data_selected[hour_data_selected['position'] == 'Above']
        hour_below_selected = hour_data_selected[hour_data_selected['position'] == 'Below']
        
        above_selected_hit_rate = (hour_above_selected['hit'].sum() / len(hour_above_selected) * 100) if len(hour_above_selected) > 0 else 0
        below_selected_hit_rate = (hour_below_selected['hit'].sum() / len(hour_below_selected) * 100) if len(hour_below_selected) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Above Hit Rate (All Days)</div>
                <div class="metric-value">{above_all_hit_rate:.1f}%</div>
                <div class="metric-subtitle">{int(hour_above_all['hit'].sum())}/{len(hour_above_all)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Below Hit Rate (All Days)</div>
                <div class="metric-value">{below_all_hit_rate:.1f}%</div>
                <div class="metric-subtitle">{int(hour_below_all['hit'].sum())}/{len(hour_below_all)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Above Hit Rate (Selected Days)</div>
                <div class="metric-value">{above_selected_hit_rate:.1f}%</div>
                <div class="metric-subtitle">{int(hour_above_selected['hit'].sum())}/{len(hour_above_selected)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Below Hit Rate (Selected Days)</div>
                <div class="metric-value">{below_selected_hit_rate:.1f}%</div>
                <div class="metric-subtitle">{int(hour_below_selected['hit'].sum())}/{len(hour_below_selected)}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("Pattern Visualization")

        col_spacer1, col1, col2, col_spacer2 = st.columns([1, 2, 2, 1])

        with col1:
            bars_before_hhrl = st.number_input(
                "Bars Before (Chart)",
                min_value=1,
                max_value=200,
                value=24,
                step=1,
                key="hhrl_bars_before_chart"
            )

        with col2:
            bars_after_hhrl = st.number_input(
                "Bars After (Chart)",
                min_value=1,
                max_value=200,
                value=48,
                step=1,
                key="hhrl_bars_after_chart"
            )

        col5, col6 = st.columns(2)

        with col5:
            if st.button("◀ Previous Pattern", use_container_width=True, key="prev_hhrl_pattern", type="secondary"):
                if len(hour_data_selected) > 0:
                    if st.session_state.hhrl_pivot_index > 0:
                        st.session_state.hhrl_pivot_index -= 1
                    else:
                        st.session_state.hhrl_pivot_index = len(hour_data_selected) - 1
                st.rerun()

        with col6:
            if st.button("Next Pattern ▶", use_container_width=True, key="next_hhrl_pattern", type="secondary"):
                if len(hour_data_selected) > 0:
                    if st.session_state.hhrl_pivot_index < len(hour_data_selected) - 1:
                        st.session_state.hhrl_pivot_index += 1
                    else:
                        st.session_state.hhrl_pivot_index = 0
                st.rerun()

        if len(hour_data_selected) > 0:
            current_idx = st.session_state.hhrl_pivot_index
            if current_idx == -1 or current_idx >= len(hour_data_selected):
                current_idx = len(hour_data_selected) - 1

            current_idx = max(0, min(current_idx, len(hour_data_selected) - 1))

            current_pivot = hour_data_selected.iloc[current_idx]
            pivot_time = current_pivot["timestamp"]
            pivot_level = current_pivot["pivot"]
            pivot_end_time = pivot_time + pd.Timedelta(hours=24)

            st.caption(f"Showing pivot {current_idx + 1} of {len(hour_data_selected)} | {pivot_time.strftime('%Y-%m-%d %H:%M')}")

            df_30m = df_15m.resample("30min").agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last"
            }).dropna()

            pivot_idx = df_30m.index.get_indexer([pivot_time], method="nearest")[0]

            start_idx = max(0, pivot_idx - bars_before_hhrl)
            end_idx = min(len(df_30m) - 1, pivot_idx + bars_after_hhrl)
            chart_data = df_30m.iloc[start_idx:end_idx+1]

            fig_level = go.Figure()

            for idx, row in chart_data.iterrows():
                is_up = row["Close"] > row["Open"]
                color = candle_up if is_up else candle_down

                fig_level.add_trace(go.Bar(
                    x=[idx],
                    y=[abs(row["Close"] - row["Open"])],
                    base=[min(row["Open"], row["Close"])],
                    marker_color=color,
                    marker_line_width=0,
                    showlegend=False,
                    hoverinfo="skip"
                ))

                fig_level.add_trace(go.Scatter(
                    x=[idx, idx],
                    y=[row["Low"], row["High"]],
                    mode="lines",
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hoverinfo="skip"
                ))

            add_ohlc_hover_trace(fig_level, chart_data)

            fig_level.add_shape(
                type="line",
                x0=pivot_time,
                x1=min(pivot_end_time, chart_data.index[-1]),
                y0=pivot_level,
                y1=pivot_level,
                line=dict(color=highlight_color, width=2)
            )

            fig_level.add_annotation(
                x=pivot_time,
                y=pivot_level,
                text=f"HHR Level: {pivot_level:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=highlight_color,
                font=dict(color=highlight_color)
            )

            fig_level.add_shape(
                type="line",
                x0=pivot_time,
                x1=pivot_time,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color=text_color, width=1, dash="dot")
            )

            fig_level.update_layout(
                title=f"{hour_select:02d}:00 HHR Level - {pivot_time.strftime('%Y-%m-%d')}",
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(showgrid=False, color=text_color),
                yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                height=500,
                hovermode="x unified"
            )

            st.plotly_chart(fig_level, use_container_width=True)

        st.markdown("---")
        st.subheader("Hit Rate Evolution")
        
        rolling_window_hhrl = st.number_input(
            "Rolling Window (days)",
            min_value=7,
            max_value=365,
            value=56,
            step=7,
            key="hhrl_rolling_window"
        )
        
        if len(hour_data_selected) > 0:
            hour_data_sorted = hour_data_selected.sort_values('timestamp').copy()
            
            hour_data_sorted['cumulative_all'] = (hour_data_sorted['hit'].expanding().sum() / hour_data_sorted['hit'].expanding().count()) * 100
            hour_data_sorted['rolling_all'] = hour_data_sorted['hit'].rolling(window=min(rolling_window_hhrl, len(hour_data_sorted)), min_periods=1).mean() * 100
            
            below_data = hour_data_sorted[hour_data_sorted['position'] == 'Below'].copy()
            if len(below_data) > 0:
                below_data['cumulative_below'] = (below_data['hit'].expanding().sum() / below_data['hit'].expanding().count()) * 100
                below_data['rolling_below'] = below_data['hit'].rolling(window=min(rolling_window_hhrl, len(below_data)), min_periods=1).mean() * 100
            
            above_data = hour_data_sorted[hour_data_sorted['position'] == 'Above'].copy()
            if len(above_data) > 0:
                above_data['cumulative_above'] = (above_data['hit'].expanding().sum() / above_data['hit'].expanding().count()) * 100
                above_data['rolling_above'] = above_data['hit'].rolling(window=min(rolling_window_hhrl, len(above_data)), min_periods=1).mean() * 100
            
            st.write("**Composite (All Levels)**")
            
            fig_composite = go.Figure()
            
            fig_composite.add_trace(go.Scatter(
                x=hour_data_sorted['timestamp'],
                y=hour_data_sorted['cumulative_all'],
                mode='lines',
                name='Cumulative',
                line=dict(color=blue_color, width=2)
            ))
            
            fig_composite.add_trace(go.Scatter(
                x=hour_data_sorted['timestamp'],
                y=hour_data_sorted['rolling_all'],
                mode='lines',
                name=f'Rolling ({rolling_window_hhrl}d)',
                line=dict(color=gold_highlight, width=2, dash='dash')
            ))
            
            fig_composite.update_layout(
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(showgrid=False, color=text_color),
                yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=300
            )
            
            st.plotly_chart(fig_composite, use_container_width=True)
            
            if len(below_data) > 0:
                st.write("**Below Levels (Support)**")
                
                fig_below = go.Figure()
                
                fig_below.add_trace(go.Scatter(
                    x=below_data['timestamp'],
                    y=below_data['cumulative_below'],
                    mode='lines',
                    name='Cumulative',
                    line=dict(color=blue_color, width=2)
                ))
                
                fig_below.add_trace(go.Scatter(
                    x=below_data['timestamp'],
                    y=below_data['rolling_below'],
                    mode='lines',
                    name=f'Rolling ({rolling_window_hhrl}d)',
                    line=dict(color=gold_highlight, width=2, dash='dash')
                ))
                
                fig_below.update_layout(
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=300
                )
                
                st.plotly_chart(fig_below, use_container_width=True)
            
            if len(above_data) > 0:
                st.write("**Above Levels (Resistance)**")
                
                fig_above = go.Figure()
                
                fig_above.add_trace(go.Scatter(
                    x=above_data['timestamp'],
                    y=above_data['cumulative_above'],
                    mode='lines',
                    name='Cumulative',
                    line=dict(color=blue_color, width=2)
                ))
                
                fig_above.add_trace(go.Scatter(
                    x=above_data['timestamp'],
                    y=above_data['rolling_above'],
                    mode='lines',
                    name=f'Rolling ({rolling_window_hhrl}d)',
                    line=dict(color=gold_highlight, width=2, dash='dash')
                ))
                
                fig_above.update_layout(
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=300
                )
                
                st.plotly_chart(fig_above, use_container_width=True)
        
        # ML PIVOT HIT PREDICTOR
        if ML_AVAILABLE:
            st.markdown("---")
            st.subheader("🤖 ML Pivot Hit Predictor")
            # Removed model-trained caption per request
            
            # Train ML model
            ml_pivot_predictor = PivotHitMLPredictor()
            training_success = ml_pivot_predictor.train(hour_data_selected)
            
            if training_success:
                
                # Calculate model accuracy on training data
                correct_predictions = 0
                total_predictions = 0
                
                for _, pivot_row in hour_data_selected.iterrows():
                    distance_pct = abs((pivot_row['pivot'] - pivot_row['current_price']) / pivot_row['current_price']) * 100
                    atr_pct = (pivot_row.get('atr', 100) / pivot_row['current_price']) * 100 if 'atr' in pivot_row else 1.0
                    
                    hour_position_df = hour_data_selected[
                        (hour_data_selected['hour'] == pivot_row['hour']) &
                        (hour_data_selected['position'] == pivot_row['position'])
                    ]
                    hist_hit_rate = (hour_position_df['hit'].sum() / len(hour_position_df)) * 100 if len(hour_position_df) > 0 else 50.0
                    
                    pivot_data = {
                        'hour': pivot_row['hour'],
                        'position': pivot_row['position'],
                        'distance_pct': distance_pct,
                        'atr_pct': atr_pct,
                        'historical_hit_rate': hist_hit_rate
                    }
                    
                    ml_prob, _ = ml_pivot_predictor.predict(pivot_data)
                    
                    if ml_prob is not None:
                        ml_prediction = 1 if ml_prob > 0.5 else 0
                        actual = 1 if pivot_row['hit'] else 0
                        
                        if ml_prediction == actual:
                            correct_predictions += 1
                        total_predictions += 1
                
                model_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
                
                # Removed training data accuracy display per request
                
                # Use analysis filters directly
                st.markdown("---")
                
                # Filter using analysis controls
                hour_data_filtered = hour_data_selected[
                    (hour_data_selected['day_name'].isin(day_filter_hhrl)) &
                    (hour_data_selected['hour'] == hour_select)
                ]
                
                st.write(f"**Pivots matching filters:** {len(hour_data_filtered)}")
                
                if len(hour_data_filtered) < 20:
                    st.warning("⚠️ Too few pivots matching these filters. Broadening to all selected days...")
                    hour_data_filtered = hour_data_selected[hour_data_selected['day_name'].isin(day_filter_hhrl)]
                    st.write(f"**Pivots after broadening:** {len(hour_data_filtered)}")
                
                # Removed training data visualization per request
                
                # Example prediction interface
                st.markdown("---")
                st.write("**Test Pivot Prediction:**")
                
                col_ml1, col_ml2, col_ml3 = st.columns(3)
                
                with col_ml1:
                    test_hour = st.selectbox(
                        "Hour",
                        options=list(range(24)),
                        index=12,  # NY open
                        key="ml_pivot_hour"
                    )
                
                with col_ml2:
                    test_position = st.selectbox(
                        "Position",
                        options=['Above', 'Below'],
                        key="ml_pivot_position"
                    )
                
                with col_ml3:
                    test_distance = st.number_input(
                        "Distance from Price (%)",
                        min_value=0.1,
                        max_value=5.0,
                        value=0.5,
                        step=0.1,
                        key="ml_pivot_distance"
                    )
                
                # Calculate historical hit rate for this hour/position (using filtered data)
                test_hour_df = hour_data_filtered[
                    (hour_data_filtered['hour'] == test_hour) &
                    (hour_data_filtered['position'] == test_position)
                ]
                test_hist_rate = (test_hour_df['hit'].sum() / len(test_hour_df)) * 100 if len(test_hour_df) > 0 else 50.0
                
                # Make prediction
                test_pivot_data = {
                    'hour': test_hour,
                    'position': test_position,
                    'distance_pct': test_distance,
                    'atr_pct': 1.0,  # Normalized
                    'historical_hit_rate': test_hist_rate
                }
                
                ml_prob, ml_conf = ml_pivot_predictor.predict(test_pivot_data)
                
                if ml_prob is not None:
                    st.markdown("---")
                    col_pred1, col_pred2, col_pred3 = st.columns(3)
                    
                    with col_pred1:
                        st.metric("Historical Hit Rate", f"{test_hist_rate:.1f}%")
                    
                    with col_pred2:
                        st.metric("ML Predicted Probability", f"{ml_prob*100:.1f}%")
                    
                    with col_pred3:
                        st.metric("Confidence", f"{ml_conf}")
                    
                    # Interpretation
                if ml_prob > 0.7:
                    pass
                elif ml_prob > 0.55:
                    st.info(f"ℹ️ **Moderate Hit Signal** - ML predicts {ml_prob*100:.0f}% probability")
                else:
                    st.warning(f"⚠️ **Weak Hit Signal** - ML predicts {ml_prob*100:.0f}% probability")
                    
                    # Feature importance
                    st.markdown("---")
                    st.write("** Feature Importance:**")
                    
                    importance_df = ml_pivot_predictor.get_feature_importance()
                    
                    if importance_df is not None:
                        # Show top 5 features
                        top_features = importance_df.head(5)
                        
                        fig_importance = go.Figure(data=[
                            go.Bar(
                                x=top_features['importance'],
                                y=top_features['feature'],
                                orientation='h',
                                marker_color='#3498db',
                                text=[f'{x:.3f}' for x in top_features['importance']],
                                textposition='outside',
                                textfont=dict(color=text_color)
                            )
                        ])
                        
                        fig_importance.update_layout(
                            plot_bgcolor=plot_bg,
                            paper_bgcolor=plot_bg,
                            font=dict(color=text_color),
                            xaxis=dict(title='Importance', showgrid=True, gridcolor=grid_color, color=text_color),
                            yaxis=dict(showgrid=False, color=text_color),
                            height=300,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_importance, use_container_width=True)
                
                # ==========================================
                # PIVOT BROWSER WITH CANDLESTICK CHARTS
                # ==========================================
                
                st.markdown("---")
                st.subheader(" Browse Individual Pivots")
                st.caption("View pivots with ML predictions on candlestick charts")
                
                # Initialize session state for pivot browser
                if 'pivot_browser_index' not in st.session_state:
                    st.session_state.pivot_browser_index = 0
                
                # Get all pivots with ML predictions
                pivots_with_ml = []
                
                for _, pivot_row in hour_data_filtered.iterrows():
                    distance_pct = abs((pivot_row['pivot'] - pivot_row['current_price']) / pivot_row['current_price']) * 100
                    atr_pct = (pivot_row.get('atr', 100) / pivot_row['current_price']) * 100 if 'atr' in pivot_row else 1.0
                    
                    hour_position_df = hour_data_filtered[
                        (hour_data_filtered['hour'] == pivot_row['hour']) &
                        (hour_data_filtered['position'] == pivot_row['position'])
                    ]
                    hist_hit_rate = (hour_position_df['hit'].sum() / len(hour_position_df)) * 100 if len(hour_position_df) > 0 else 50.0
                    
                    pivot_data = {
                        'hour': pivot_row['hour'],
                        'day_of_week': pivot_row['timestamp'].dayofweek if hasattr(pivot_row['timestamp'], 'dayofweek') else 2,
                        'position': pivot_row['position'],
                        'distance_pct': distance_pct,
                        'atr_pct': atr_pct,
                        'historical_hit_rate': hist_hit_rate
                    }
                    
                    ml_prob, ml_conf = ml_pivot_predictor.predict(pivot_data)
                    
                    if ml_prob is not None:
                        pivots_with_ml.append({
                            'pivot_row': pivot_row,
                            'ml_prob': ml_prob,
                            'ml_conf': ml_conf,
                            'distance_pct': distance_pct
                        })
                
                if len(pivots_with_ml) == 0:
                    st.warning("No pivots available for browsing.")
                else:
                    # Filter options
                    col_filter1, col_filter2 = st.columns(2)
                    
                    with col_filter1:
                        confidence_filter = st.selectbox(
                            "Filter by Confidence",
                            options=['All', 'High Only', 'Medium+', 'Low Only'],
                            key="pivot_browser_confidence_filter"
                        )
                    
                    with col_filter2:
                        outcome_filter = st.selectbox(
                            "Filter by Outcome",
                            options=['All', 'Hits Only', 'Misses Only'],
                            key="pivot_browser_outcome_filter"
                        )
                    
                    # Apply filters
                    filtered_pivots = pivots_with_ml.copy()
                    
                    if confidence_filter == 'High Only':
                        filtered_pivots = [p for p in filtered_pivots if p['ml_conf'] == 'High']
                    elif confidence_filter == 'Medium+':
                        filtered_pivots = [p for p in filtered_pivots if p['ml_conf'] in ['High', 'Medium']]
                    elif confidence_filter == 'Low Only':
                        filtered_pivots = [p for p in filtered_pivots if p['ml_conf'] == 'Low']
                    
                    if outcome_filter == 'Hits Only':
                        filtered_pivots = [p for p in filtered_pivots if p['pivot_row']['hit']]
                    elif outcome_filter == 'Misses Only':
                        filtered_pivots = [p for p in filtered_pivots if not p['pivot_row']['hit']]
                    
                    if len(filtered_pivots) == 0:
                        st.warning("No pivots match the selected filters.")
                    else:
                        # Navigation
                        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
                        
                        with col_nav1:
                            if st.button("◀ Previous Pivot", key="pivot_browser_prev", type="secondary"):
                                if st.session_state.pivot_browser_index > 0:
                                    st.session_state.pivot_browser_index -= 1
                                st.rerun()
                        
                        with col_nav2:
                            st.write(f"**Pivot {st.session_state.pivot_browser_index + 1} of {len(filtered_pivots)}**")
                        
                        with col_nav3:
                            if st.button("Next Pivot ▶", key="pivot_browser_next", type="secondary"):
                                if st.session_state.pivot_browser_index < len(filtered_pivots) - 1:
                                    st.session_state.pivot_browser_index += 1
                                st.rerun()
                        
                        # Ensure index is valid
                        if st.session_state.pivot_browser_index >= len(filtered_pivots):
                            st.session_state.pivot_browser_index = len(filtered_pivots) - 1
                        
                        current_pivot_data = filtered_pivots[st.session_state.pivot_browser_index]
                        current_pivot = current_pivot_data['pivot_row']
                        ml_prob = current_pivot_data['ml_prob']
                        ml_conf = current_pivot_data['ml_conf']
                        
                        # Get bars before/after
                        pivot_time = current_pivot['timestamp']
                        bars_before = 50
                        bars_after = 50
                        
                        chart_start = pivot_time - pd.Timedelta(minutes=15 * bars_before)
                        chart_end = pivot_time + pd.Timedelta(minutes=15 * bars_after)
                        
                        chart_data = df_15m[
                            (df_15m.index >= chart_start) &
                            (df_15m.index <= chart_end)
                        ]
                        
                        if len(chart_data) > 0:
                            # Create candlestick chart
                            fig_pivot = go.Figure()
                            
                            fig_pivot.add_trace(go.Candlestick(
                                x=chart_data.index,
                                open=chart_data['Open'],
                                high=chart_data['High'],
                                low=chart_data['Low'],
                                close=chart_data['Close'],
                                name='Price',
                                increasing_line_color='#A8A8A8',
                                decreasing_line_color='#5A5A5A',
                                increasing_fillcolor='#A8A8A8',
                                decreasing_fillcolor='#5A5A5A',
                                hovertemplate=CANDLE_HOVER_TEMPLATE
                            ))
                            
                            # Add pivot line
                            pivot_color = 'yellow' if current_pivot['position'] == 'Above' else 'cyan'
                            fig_pivot.add_hline(
                                y=current_pivot['pivot'],
                                line_dash="dash",
                                line_color=pivot_color,
                                line_width=2,
                                annotation_text=f"Pivot: ${current_pivot['pivot']:.0f}",
                                annotation_position="right"
                            )
                            
                            # Add ML confidence annotation
                            conf_color = 'green' if ml_conf == 'High' else 'yellow' if ml_conf == 'Medium' else 'red'
                            
                            fig_pivot.add_annotation(
                                x=pivot_time,
                                y=current_pivot['pivot'],
                                text=f"ML: {ml_conf.upper()}<br>{ml_prob*100:.0f}%",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor=conf_color,
                                bgcolor=conf_color,
                                font=dict(color='white', size=12),
                                opacity=0.9,
                                ax=0,
                                ay=-40 if current_pivot['position'] == 'Above' else 40
                            )
                            
                            # If hit, mark where
                            if current_pivot['hit'] and current_pivot.get('bars_to_hit') is not None:
                                hit_time = pivot_time + pd.Timedelta(minutes=15 * current_pivot['bars_to_hit'])
                                fig_pivot.add_vline(
                                    x=hit_time,
                                    line_dash="dot",
                                    line_color='green',
                                    line_width=2
                                )
                                fig_pivot.add_annotation(
                                    x=hit_time,
                                    y=1.02,
                                    xref="x",
                                    yref="paper",
                                    text=f"Hit @ {current_pivot.get('hit_hour', 0)}:00",
                                    showarrow=False,
                                    font=dict(color='green', size=12)
                                )
                            
                            fig_pivot.update_layout(
                                title=f"Pivot at {pivot_time.strftime('%Y-%m-%d %H:%M')} - {current_pivot['day_name']}",
                                xaxis_title="Time",
                                yaxis_title="Price",
                                plot_bgcolor=plot_bg,
                                paper_bgcolor=plot_bg,
                                font=dict(color=text_color),
                                xaxis=dict(showgrid=False, color=text_color),
                                yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                                height=600,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_pivot, use_container_width=True)
                            
                            # Pivot details
                            st.markdown("---")
                            st.subheader("Pivot Details")
                            
                            col_d1, col_d2, col_d3 = st.columns(3)
                            
                            with col_d1:
                                st.write("**Timing:**")
                                st.write(f"Date: {pivot_time.strftime('%Y-%m-%d')}")
                                st.write(f"Time: {pivot_time.strftime('%H:%M')}")
                                st.write(f"Day: {current_pivot['day_name']}")
                            
                            with col_d2:
                                st.write("**Pivot Info:**")
                                st.write(f"Price: ${current_pivot['pivot']:.2f}")
                                st.write(f"Position: {current_pivot['position']}")
                                st.write(f"Distance: {current_pivot_data['distance_pct']:.2f}%")
                            
                            with col_d3:
                                st.write("**ML Prediction:**")
                                st.write(f"Probability: {ml_prob*100:.1f}%")
                                st.write(f"Confidence: {ml_conf}")
                                outcome_text = "✅ HIT" if current_pivot['hit'] else "❌ MISS"
                                st.write(f"Actual: {outcome_text}")
                                if current_pivot['hit']:
                                    st.write(f"Hit Hour: {current_pivot.get('hit_hour', 'N/A')}:00")
                                    st.write(f"Bars to Hit: {current_pivot.get('bars_to_hit', 'N/A')}")
                        
                        else:
                            st.error("No chart data available for this pivot.")
            
            else:
                st.info("ℹ️ Need at least 50 pivots to train ML model. Adjust filters to get more data.")
        
        # CUMULATIVE PROBABILITY OF PIVOT HIT
        st.markdown("---")
        st.subheader("Cumulative Probability of Pivot Hit")
        
        if len(hour_data_selected) > 0:
            hits_only = hour_data_selected[hour_data_selected['hit']].copy()
            
            if len(hits_only) > 0:
                bars_to_hit = hits_only['bars_to_hit'].dropna().values
                max_bars = int(max(bars_to_hit)) if len(bars_to_hit) > 0 else 24
                
                if max_bars > 0:
                    bars_range = range(0, min(max_bars + 1, 25))
                    
                    probabilities = []
                    for bar_count in bars_range:
                        hit_count = sum(1 for b in bars_to_hit if b <= bar_count)
                        prob = (hit_count / len(hour_data_selected)) * 100
                        probabilities.append(prob)
                    
                    fig_cum_prob = go.Figure()
                    
                    fig_cum_prob.add_trace(go.Scatter(
                        x=list(bars_range),
                        y=probabilities,
                        mode='lines+markers',
                        name='Cumulative Probability',
                        line=dict(color='#3498DB', width=3),
                        marker=dict(size=6)
                    ))
                    
                    fig_cum_prob.update_layout(
                        plot_bgcolor=plot_bg,
                        paper_bgcolor=plot_bg,
                        font=dict(color=text_color),
                        xaxis=dict(title="Number of Hours", showgrid=False, color=text_color),
                        yaxis=dict(title="Probability (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_cum_prob, use_container_width=True)
                else:
                    st.info("No hits found to calculate cumulative probability.")
            else:
                st.info("No hits found to calculate cumulative probability.")
        
        hour_hits = hour_data_selected[hour_data_selected['hit']]
        
        if len(hour_hits) > 0:
            st.markdown("---")
            st.subheader("MAE Distribution (Maximum Adverse Excursion)")
            
            mae_values = hour_hits['mae_pct'].dropna().values
            mae_buckets = create_mae_buckets(mae_values)
            
            bucket_counts = pd.Series(mae_buckets).value_counts()
            
            bucket_order = []
            for i in range(0, 10):
                bucket_order.append(f"{i*0.5:.1f}-{(i+1)*0.5:.1f}%")
            for i in range(5, 10):
                bucket_order.append(f"{i}-{i+1}%")
            bucket_order.append("10+%")
            
            bucket_counts = bucket_counts.reindex([b for b in bucket_order if b in bucket_counts.index], fill_value=0)
            
            colors_gradient = []
            for i in range(len(bucket_counts)):
                ratio = i / max(len(bucket_counts) - 1, 1)
                if ratio < 0.5:
                    r = int(0 + (255 * ratio * 2))
                    g = 255
                else:
                    r = 255
                    g = int(255 - (255 * (ratio - 0.5) * 2))
                color_hex = f"#{r:02x}{g:02x}00"
                colors_gradient.append(color_hex)
            
            fig_mae = go.Figure(data=[
                go.Bar(
                    x=bucket_counts.index,
                    y=bucket_counts.values,
                    marker=dict(
                        color=colors_gradient,
                        line=dict(color=grid_color, width=1)
                    ),
                    text=bucket_counts.values,
                    textposition='outside',
                    showlegend=False
                )
            ])
            
            fig_mae.update_layout(
                title="MAE Before Level Hit",
                xaxis_title="MAE Range",
                yaxis_title="Count",
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(showgrid=False, color=text_color, tickangle=-45),
                yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                height=400
            )
            
            st.plotly_chart(fig_mae, use_container_width=True)
        
        if len(hour_hits) > 0 and hour_hits['hit_hour'].notna().any():
            st.markdown("---")
            st.subheader("Hour Distribution of Hits")
            st.caption("Which hours of the day do levels get hit?")
            
            hour_hit_counts = hour_hits['hit_hour'].value_counts().sort_index()
            
            fig_hour_dist = go.Figure(data=[
                go.Bar(
                    x=[f"{int(h):02d}:00" for h in hour_hit_counts.index],
                    y=hour_hit_counts.values,
                    marker_color=positive_color,
                    text=hour_hit_counts.values,
                    textposition='outside'
                )
            ])
            
            fig_hour_dist.update_layout(
                title="When Do Levels Get Hit?",
                xaxis_title="Hour of Day",
                yaxis_title="Count",
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(showgrid=False, color=text_color),
                yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                height=400
            )
            
            st.plotly_chart(fig_hour_dist, use_container_width=True)
        
        st.markdown("---")
        with st.expander("Raw Data", expanded=False):
            display_data = hour_data_selected[['timestamp', 'position', 'pivot', 'current_price', 'hit', 'mae', 'mae_pct', 'hit_hour']].copy()
            display_data['timestamp'] = display_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            display_data['pivot'] = display_data['pivot'].round(2)
            display_data['current_price'] = display_data['current_price'].round(2)
            display_data['mae'] = display_data['mae'].round(2)
            display_data['mae_pct'] = display_data['mae_pct'].round(2)
            display_data['hit'] = display_data['hit'].map({True: 'Y', False: 'N'})
            display_data['hit_hour'] = display_data['hit_hour'].apply(lambda x: f"{int(x):02d}:00" if pd.notna(x) else '-')
            
            display_data = display_data.rename(columns={
                'timestamp': 'Time',
                'position': 'A/B',
                'pivot': 'Pivot Level',
                'current_price': 'Price at Spawn',
                'hit': 'Hit',
                'mae': 'MAE',
                'mae_pct': 'MAE %',
                'hit_hour': 'Hour Hit'
            })
            
            st.dataframe(display_data, use_container_width=True, height=400)
        
        st.markdown("---")
        st.subheader("Hit Rate Summary (All Hours)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Above Levels (Resistance - Pivot Above Price)**")
            above_df = hit_rates_df[hit_rates_df['position'] == 'Above'][['hour', 'hit_rate', 'hit_count', 'miss_count', 'total_count']].copy()
            above_df = above_df.rename(columns={'hit_rate': 'Hit Rate %', 'hit_count': 'Y', 'miss_count': 'N', 'total_count': 'Total'})
            above_df['Hit Rate %'] = above_df['Hit Rate %'].round(1)
            above_df['hour'] = above_df['hour'].apply(lambda x: f"{x:02d}:00")
            st.dataframe(above_df.set_index('hour'), use_container_width=True, height=400)
        
        with col2:
            st.write("**Below Levels (Support - Pivot Below Price)**")
            below_df = hit_rates_df[hit_rates_df['position'] == 'Below'][['hour', 'hit_rate', 'hit_count', 'miss_count', 'total_count']].copy()
            below_df = below_df.rename(columns={'hit_rate': 'Hit Rate %', 'hit_count': 'Y', 'miss_count': 'N', 'total_count': 'Total'})
            below_df['Hit Rate %'] = below_df['Hit Rate %'].round(1)
            below_df['hour'] = below_df['hour'].apply(lambda x: f"{x:02d}:00")
            st.dataframe(below_df.set_index('hour'), use_container_width=True, height=400)

# ============================================================================
# ONE TIME FRAMING ANALYSIS
# ============================================================================

def resample_to_timeframe(df, timeframe):
    return df.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

def detect_otf_patterns(df, n_candles=3, check_low_high_intact=True, check_close_above_below=True, check_direction=True, exclude_weekends=False):
    results = {
        'bullish_patterns': 0,
        'bearish_patterns': 0,
        'bullish_success': 0,
        'bearish_success': 0,
        'next_changes': [],
        'green_changes': [],  # Track green pattern changes separately
        'red_changes': [],    # Track red pattern changes separately
        'bullish_indices': [],
        'bearish_indices': []
    }
    
    for i in range(len(df) - n_candles):
        candles = df.iloc[i:i+n_candles+1]
        
        if len(candles) < n_candles + 1:
            continue
        
        if exclude_weekends:
            if candles.index[0].dayofweek >= 5:
                continue
        
        pattern_candles = candles.iloc[:n_candles]
        next_candle = candles.iloc[n_candles]
        
        bullish_otf = True
        for j in range(n_candles):
            current = pattern_candles.iloc[j]
            
            if check_direction and current['Close'] <= current['Open']:
                bullish_otf = False
                break
            
            if j > 0:
                prev = pattern_candles.iloc[j-1]
                
                if check_low_high_intact and current['Low'] < prev['Low']:
                    bullish_otf = False
                    break
                
                if check_close_above_below and current['Close'] <= prev['High']:
                    bullish_otf = False
                    break
        
        bearish_otf = True
        for j in range(n_candles):
            current = pattern_candles.iloc[j]
            
            if check_direction and current['Close'] >= current['Open']:
                bearish_otf = False
                break
            
            if j > 0:
                prev = pattern_candles.iloc[j-1]
                
                if check_low_high_intact and current['High'] > prev['High']:
                    bearish_otf = False
                    break
                
                if check_close_above_below and current['Close'] >= prev['Low']:
                    bearish_otf = False
                    break
        
        if bullish_otf:
            results['bullish_patterns'] += 1
            results['bullish_indices'].append(i)
            next_is_green = next_candle['Close'] > next_candle['Open']
            if next_is_green:
                results['bullish_success'] += 1
            
            last_pattern_close = pattern_candles.iloc[-1]['Close']
            next_change = ((next_candle['Close'] - last_pattern_close) / last_pattern_close) * 100
            results['next_changes'].append(next_change)
            results['green_changes'].append(next_change)  # Track green separately
        
        elif bearish_otf:
            results['bearish_patterns'] += 1
            results['bearish_indices'].append(i)
            next_is_red = next_candle['Close'] < next_candle['Open']
            if next_is_red:
                results['bearish_success'] += 1
            
            last_pattern_close = pattern_candles.iloc[-1]['Close']
            next_change = ((next_candle['Close'] - last_pattern_close) / last_pattern_close) * 100
            results['next_changes'].append(next_change)
            results['red_changes'].append(next_change)  # Track red separately
    
    return results

def create_pattern_chart(df, pattern_idx, n_candles, bars_before, bars_after, pattern_type):
    start_idx = max(0, pattern_idx - bars_before)
    end_idx = min(len(df) - 1, pattern_idx + n_candles + bars_after)
    
    chart_data = df.iloc[start_idx:end_idx+1].copy()
    
    highlight_start = pattern_idx - start_idx
    highlight_end = highlight_start + n_candles
    
    fig = go.Figure()
    
    for i, (idx, row) in enumerate(chart_data.iterrows()):
        is_up = row['Close'] > row['Open']
        is_highlighted = highlight_start <= i < highlight_end
        
        if is_highlighted:
            color = gold_highlight
            line_color = gold_highlight
        else:
            color = candle_up if is_up else candle_down
            line_color = color
        
        fig.add_trace(go.Bar(
            x=[idx],
            y=[abs(row['Close'] - row['Open'])],
            base=[min(row['Open'], row['Close'])],
            marker_color=color,
            marker_line_width=0,
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=[idx, idx],
            y=[row['Low'], row['High']],
            mode='lines',
            line=dict(color=line_color, width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    hover_text = []
    for idx, row in chart_data.iterrows():
        hover_text.append(
            f"Date: {idx}<br>"
            f"Open: {row['Open']:.2f}<br>"
            f"High: {row['High']:.2f}<br>"
            f"Low: {row['Low']:.2f}<br>"
            f"Close: {row['Close']:.2f}"
        )
    
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data['Close'],
        mode='markers',
        marker=dict(size=0.1, opacity=0),
        text=hover_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Last {pattern_type.capitalize()} OTF Pattern ({n_candles} candles)",
        plot_bgcolor=plot_bg,
        paper_bgcolor=plot_bg,
        font=dict(color=text_color),
        xaxis=dict(showgrid=False, color=text_color, rangeslider_visible=False),
        yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
        height=500,
        hovermode='x unified'
    )
    
    return fig

if page == "One Time Framing Analysis":
    render_page_header(
        "One Time-Framing Analysis",
        "Analyze consecutive green/red candle patterns and predict next candle probability"
    )
    # Exchange & Asset at top
    render_exchange_asset_controls("otf")
    
    st.caption("Configure your consecutive pattern analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        timeframe = st.selectbox(
            "Timeframe",
            options=["30min", "1H", "4H", "1D"],
            index=3
        )
        
        timeframe_map = {
            "30min": "30min",
            "1H": "1H",
            "4H": "4H",
            "1D": "1D"
        }
        resample_code = timeframe_map[timeframe]
    
    with col2:
        consecutive_candles = st.number_input(
            "Consecutive Candles",
            min_value=1,
            max_value=30,
            value=3,
            step=1,
            help="Number of consecutive green/red candles to analyze (1-30)"
        )
        st.caption("Number of consecutive green/red candles to analyze (1-30)")
    
    col3, col4 = st.columns(2)
    
    with col3:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()
        
        start_date = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=365),
            min_value=min_date,
            max_value=max_date
        )
    
    with col4:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    
    st.markdown("---")
    st.markdown("<div class='otf-pattern-filters'>", unsafe_allow_html=True)
    st.caption("Pattern Definition Filters - Enable/disable specific rules for pattern detection")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        check_low_high = st.checkbox(
            "Previous candle high/low is intact",
            value=True,
            help="Bullish: current low >= previous low | Bearish: current high <= previous high"
        )
    
    with col2:
        check_close = st.checkbox(
            "Close is above/below previous high/low",
            value=True,
            help="Bullish: closes above previous high | Bearish: closes below previous low"
        )
    
    with col3:
        check_direction = st.checkbox(
            "Close is bullish/bearish in direction",
            value=True,
            help="Bullish: close > open (green) | Bearish: close < open (red)"
        )
    
    with col4:
        exclude_weekends = st.checkbox(
            "Exclude weekends",
            value=False,
            help="Exclude Saturday and Sunday from analysis"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    analyze_button = st.button(
        "Analyze One Time-Framing Patterns",
        use_container_width=True,
        type="primary",
        key="analyze_otf"
    )
    if analyze_button:
        st.session_state.otf_analyzed = True
        
        filtered_15m = df_15m[(df_15m.index.date >= start_date) & (df_15m.index.date <= end_date)]
        resampled_df = resample_to_timeframe(filtered_15m, resample_code)
        
        results = detect_otf_patterns(
            resampled_df, 
            consecutive_candles,
            check_low_high,
            check_close,
            check_direction,
            exclude_weekends
        )
        
        st.session_state.otf_results = results
        st.session_state.otf_df = resampled_df
    
    if st.session_state.otf_analyzed and st.session_state.otf_results is not None:
        st.markdown("---")
        
        results = st.session_state.otf_results
        resampled_df = st.session_state.otf_df
        
        total_patterns = results['bullish_patterns'] + results['bearish_patterns']
        
        if total_patterns == 0:
            st.warning("No One Time-Framing patterns found with the selected filters and date range.")
        else:
            st.subheader("Pattern Analysis Results")
            st.caption(f"Analysis of {consecutive_candles} consecutive candle patterns")
            
            green_success_rate = (results['bullish_success'] / results['bullish_patterns'] * 100) if results['bullish_patterns'] > 0 else 0
            red_success_rate = (results['bearish_success'] / results['bearish_patterns'] * 100) if results['bearish_patterns'] > 0 else 0
            avg_green_change = np.mean(results['green_changes']) if results['green_changes'] else 0
            avg_red_change = np.mean(results['red_changes']) if results['red_changes'] else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="height: 150px;">
                    <div class="metric-title">Total Patterns</div>
                    <div class="metric-value">{total_patterns}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="height: 150px;">
                    <div class="metric-title">Green Success Rate</div>
                    <div class="metric-value" style="color: {positive_color};">{green_success_rate:.1f}%</div>
                    <div class="metric-subtitle">{results['bullish_success']}/{results['bullish_patterns']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="height: 150px;">
                    <div class="metric-title">Red Success Rate</div>
                    <div class="metric-value" style="color: {negative_color};">{red_success_rate:.1f}%</div>
                    <div class="metric-subtitle">{results['bearish_success']}/{results['bearish_patterns']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card" style="height: 150px;">
                    <div class="metric-title">Avg Net Green Change</div>
                    <div class="metric-value" style="color: {positive_color};">{avg_green_change:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card" style="height: 150px;">
                    <div class="metric-title">Avg Net Red Change</div>
                    <div class="metric-value" style="color: {negative_color};">{avg_red_change:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Pattern Visualization")
            
            # Center the 3 control buttons
            col_spacer1, col1, col2, col3, col_spacer2 = st.columns([1, 2, 2, 2, 1])
            
            with col1:
                bars_before = st.number_input(
                    "Bars Before (Chart)",
                    min_value=1,
                    max_value=200,
                    value=50,
                    step=1,
                    key="otf_bars_before"
                )
            
            with col2:
                bars_after = st.number_input(
                    "Bars After (Chart)",
                    min_value=1,
                    max_value=200,
                    value=50,
                    step=1,
                    key="otf_bars_after"
                )
            
            with col3:
                pattern_type = st.selectbox(
                    "Show Pattern",
                    options=["Bullish", "Bearish"],
                    index=0,
                    key="otf_pattern_type"
                )
            
            # Initialize session state for chart index
            if 'otf_chart_index' not in st.session_state:
                st.session_state.otf_chart_index = -1
            
            # Navigation buttons
            col5, col6 = st.columns(2)
            
            with col5:
                if st.button("◀ Previous Pattern", use_container_width=True, key="prev_otf", type="secondary"):
                    if pattern_type == "Bullish" and len(results['bullish_indices']) > 0:
                        if st.session_state.otf_chart_index > 0:
                            st.session_state.otf_chart_index -= 1
                        else:
                            st.session_state.otf_chart_index = len(results['bullish_indices']) - 1
                    elif pattern_type == "Bearish" and len(results['bearish_indices']) > 0:
                        if st.session_state.otf_chart_index > 0:
                            st.session_state.otf_chart_index -= 1
                        else:
                            st.session_state.otf_chart_index = len(results['bearish_indices']) - 1
                    st.rerun()
            
            with col6:
                if st.button("Next Pattern ▶", use_container_width=True, key="next_otf", type="secondary"):
                    if pattern_type == "Bullish" and len(results['bullish_indices']) > 0:
                        if st.session_state.otf_chart_index < len(results['bullish_indices']) - 1:
                            st.session_state.otf_chart_index += 1
                        else:
                            st.session_state.otf_chart_index = 0
                    elif pattern_type == "Bearish" and len(results['bearish_indices']) > 0:
                        if st.session_state.otf_chart_index < len(results['bearish_indices']) - 1:
                            st.session_state.otf_chart_index += 1
                        else:
                            st.session_state.otf_chart_index = 0
                    st.rerun()
            
            if pattern_type == "Bullish" and len(results['bullish_indices']) > 0:
                current_idx = st.session_state.otf_chart_index
                if current_idx == -1 or current_idx >= len(results['bullish_indices']):
                    current_idx = len(results['bullish_indices']) - 1
                
                # Ensure index is within bounds
                current_idx = max(0, min(current_idx, len(results['bullish_indices']) - 1))
                
                pattern_idx = results['bullish_indices'][current_idx]
                st.caption(f"Showing Bullish Pattern {current_idx + 1} of {len(results['bullish_indices'])}")
                fig = create_pattern_chart(resampled_df, pattern_idx, consecutive_candles, bars_before, bars_after, "bullish")
                st.plotly_chart(fig, use_container_width=True)
            
            elif pattern_type == "Bearish" and len(results['bearish_indices']) > 0:
                current_idx = st.session_state.otf_chart_index
                if current_idx == -1 or current_idx >= len(results['bearish_indices']):
                    current_idx = len(results['bearish_indices']) - 1
                
                # Ensure index is within bounds
                current_idx = max(0, min(current_idx, len(results['bearish_indices']) - 1))
                
                pattern_idx = results['bearish_indices'][current_idx]
                st.caption(f"Showing Bearish Pattern {current_idx + 1} of {len(results['bearish_indices'])}")
                fig = create_pattern_chart(resampled_df, pattern_idx, consecutive_candles, bars_before, bars_after, "bearish")
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info(f"No {pattern_type.lower()} patterns found with the selected filters.")

# ============================================================================
# DAILY RETURNS PAGE
# ============================================================================

elif page == "Daily Returns":
    render_page_header("Daily Returns Analysis")
    # Exchange & Asset first
    render_exchange_asset_controls("dr")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date_dr = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            key="dr_start"
        )
    with col2:
        end_date_dr = st.date_input(
            "End Date",
            value=datetime.now(),
            key="dr_end"
        )
    
    col3, col4 = st.columns(2)
    with col3:
        lookback_days_dr = st.slider(
            "Days Lookback",
            min_value=7,
            max_value=1825,
            value=365,
            step=1,
            key="dr_lookback"
        )
    with col4:
        total_days_dr = st.number_input(
            "Total Days",
            min_value=7,
            max_value=1825,
            value=365,
            step=1,
            key="dr_total"
        )

    st.markdown(
        """
        <script>
        (function() {
          const updateSliderTrack = () => {
            const slider = document.querySelector('[aria-label="Days Lookback"]');
            if (!slider) return;
            const base = slider.closest('[data-baseweb="slider"]');
            if (!base) return;
            const track = base.querySelector('div[style*="height: 0.25rem"]');
            if (!track) return;
            const min = parseFloat(slider.getAttribute('aria-valuemin') || '0');
            const max = parseFloat(slider.getAttribute('aria-valuemax') || '100');
            const val = parseFloat(slider.getAttribute('aria-valuenow') || '0');
            const pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
            track.style.backgroundImage = `linear-gradient(to right, var(--accent-primary) 0%, var(--accent-primary) ${pct}%, #1a1a1a ${pct}%, #1a1a1a 100%)`;
            track.style.backgroundColor = 'transparent';
          };
          updateSliderTrack();
          setInterval(updateSliderTrack, 300);
        })();
        </script>
        """,
        unsafe_allow_html=True
    )
    
    # Use the most recent value (slider or input)
    lookback_dr = total_days_dr if total_days_dr != 365 else lookback_days_dr
    
    # Recalculate recent_daily based on controls
    recent_daily = df_daily[
        (df_daily.index.date >= start_date_dr) & 
        (df_daily.index.date <= end_date_dr)
    ].last(f'{lookback_dr}D')
    
    st.markdown("---")
    
    day_stats = recent_daily.groupby('day_name')['returns'].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_stats = day_stats.reindex([d for d in day_order if d in day_stats.index])
    
    weekday_avg = recent_daily[~recent_daily['is_weekend']]['returns'].mean()
    weekend_avg = recent_daily[recent_daily['is_weekend']]['returns'].mean()
    day_stats_extended = day_stats.copy()
    day_stats_extended['W/d'] = weekday_avg
    day_stats_extended['W/e'] = weekend_avg
    
    colors = [positive_color if val > 0 else negative_color for val in day_stats_extended.values]
    
    fig1 = go.Figure(data=[
        go.Bar(
            x=day_stats_extended.index,
            y=day_stats_extended.values,
            marker_color=colors,
            text=[f'{val:.2f}%' for val in day_stats_extended.values],
            textposition='outside',
            textfont=dict(color=text_color),
            showlegend=False
        )
    ])
    
    fig1.update_layout(
        plot_bgcolor=plot_bg,
        paper_bgcolor=plot_bg,
        font=dict(size=14, color=text_color),
        xaxis=dict(
            title='', 
            showgrid=False, 
            showline=True, 
            linewidth=1, 
            linecolor=grid_color, 
            color=text_color,
            tickfont=dict(color=text_color)
        ),
        yaxis=dict(
            title='', 
            showgrid=True, 
            gridwidth=1, 
            gridcolor=grid_color, 
            zeroline=True, 
            zerolinewidth=2, 
            zerolinecolor=grid_color, 
            ticksuffix='%', 
            color=text_color,
            tickfont=dict(color=text_color)
        ),
        height=500,
        margin=dict(l=50, r=50, t=30, b=50)
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Statistics")
    
    stats_table = recent_daily.groupby('day_name')['returns'].agg([
        ('Mean %', 'mean'),
        ('Std Dev %', 'std'),
        ('Count', 'count')
    ]).round(3)
    
    stats_table = stats_table.reindex([d for d in day_order if d in stats_table.index])
    
    weekday_data = recent_daily[~recent_daily['is_weekend']]['returns']
    weekend_data = recent_daily[recent_daily['is_weekend']]['returns']
    
    stats_table.loc['Weekday'] = [weekday_data.mean(), weekday_data.std(), len(weekday_data)]
    stats_table.loc['Weekend'] = [weekend_data.mean(), weekend_data.std(), len(weekend_data)]
    
    st.dataframe(stats_table, use_container_width=True)

# ============================================================================
# SESSION RETURNS PAGE
# ============================================================================

elif page == "Session Returns":
    render_page_header("Session Returns Analysis")
    render_exchange_asset_controls("sr")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date_sr = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            key="sr_start"
        )
    with col2:
        end_date_sr = st.date_input(
            "End Date",
            value=datetime.now(),
            key="sr_end"
        )
    
    col3, col4 = st.columns(2)
    with col3:
        lookback_days_sr = st.slider(
            "Days Lookback",
            min_value=7,
            max_value=1825,
            value=365,
            step=1,
            key="sr_lookback"
        )
    with col4:
        total_days_sr = st.number_input(
            "Total Days",
            min_value=7,
            max_value=1825,
            value=365,
            step=1,
            key="sr_total"
        )
    
    # Use the most recent value (slider or input)
    lookback_sr = total_days_sr if total_days_sr != 365 else lookback_days_sr
    
    # Recalculate recent_15m based on controls
    recent_15m = df_15m[
        (df_15m.index.date >= start_date_sr) & 
        (df_15m.index.date <= end_date_sr)
    ].last(f'{lookback_sr}D')
    
    day_filter = st.multiselect(
        "Filter by Day of Week",
        options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        key="sr_day_filter"
    )
    
    st.markdown("---")
    
    session_data = []
    
    for session_name in ['Asia', 'London', 'NY', 'Close']:
        session_bars = recent_15m[recent_15m['session'] == session_name].copy()
        session_bars['day_name'] = session_bars.index.day_name()
        session_bars_filtered = session_bars[session_bars['day_name'].isin(day_filter)]
        
        session_returns = []
        for date, group in session_bars_filtered.groupby(session_bars_filtered.index.date):
            if len(group) > 0:
                session_open = group['Open'].iloc[0]
                session_close = group['Close'].iloc[-1]
                ret = ((session_close - session_open) / session_open) * 100
                session_returns.append(ret)
        
        avg_return = sum(session_returns) / len(session_returns) if session_returns else 0
        session_data.append({'Session': session_name, 'Avg Return %': avg_return})
    
    session_df = pd.DataFrame(session_data)
    colors_session = [positive_color if val > 0 else negative_color for val in session_df['Avg Return %']]
    
    fig2 = go.Figure(data=[
        go.Bar(
            x=session_df['Session'],
            y=session_df['Avg Return %'],
            marker_color=colors_session,
            text=[f'{val:.2f}%' for val in session_df['Avg Return %']],
            textposition='outside',
            textfont=dict(color=text_color),
            showlegend=False
        )
    ])
    
    fig2.update_layout(
        plot_bgcolor=plot_bg,
        paper_bgcolor=plot_bg,
        font=dict(size=14, color=text_color),
        xaxis=dict(
            title='', 
            showgrid=False, 
            showline=True, 
            linewidth=1, 
            linecolor=grid_color, 
            color=text_color,
            tickfont=dict(color=text_color)
        ),
        yaxis=dict(
            title='', 
            showgrid=True, 
            gridwidth=1, 
            gridcolor=grid_color,
            zeroline=True, 
            zerolinewidth=2, 
            zerolinecolor=grid_color, 
            ticksuffix='%', 
            color=text_color,
            tickfont=dict(color=text_color)
        ),
        height=500,
        margin=dict(l=50, r=50, t=30, b=50)
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Statistics")
    
    session_stats = []
    for session_name in ['Asia', 'London', 'NY', 'Close']:
        session_bars = recent_15m[recent_15m['session'] == session_name].copy()
        session_bars['day_name'] = session_bars.index.day_name()
        session_bars_filtered = session_bars[session_bars['day_name'].isin(day_filter)]
        
        session_returns = []
        for date, group in session_bars_filtered.groupby(session_bars_filtered.index.date):
            if len(group) > 0:
                session_open = group['Open'].iloc[0]
                session_close = group['Close'].iloc[-1]
                ret = ((session_close - session_open) / session_open) * 100
                session_returns.append(ret)
        
        session_stats.append({
            'Session': session_name,
            'Mean %': np.mean(session_returns) if session_returns else 0,
            'Std Dev %': np.std(session_returns) if session_returns else 0,
            'Count': len(session_returns)
        })
    
    session_stats_df = pd.DataFrame(session_stats).set_index('Session')
    st.dataframe(session_stats_df.round(3), use_container_width=True)


# ============================================================================
# SESSION TPO PAGE (WITH FIXES)
# ============================================================================

elif page == "Session TPO":
    render_page_header(
        "Session TPO Analysis",
        "Analyze poor highs/lows in session profiles and their sweep rates"
    )
    render_exchange_asset_controls("tpo")
    
    tpo_block_size = st.selectbox(
        "TPO Block Size",
        options=["15m", "30m"],
        index=0,
        key="tpo_block_size",
        help="15m = Each 15-minute bar is one TPO block | 30m = Each 30-minute bar is one TPO block"
    )

    
    col1, col2 = st.columns(2)
    
    with col1:
        tick_mode = st.selectbox(
            "Tick Size Mode",
            options=["Auto (ATR × Multiplier)", "Manual"],
            index=0,
            key="tpo_tick_mode"
        )
    
    with col2:
        if tick_mode == "Manual":
            manual_tick_size = st.number_input(
                "Manual Tick Size ($)",
                min_value=1.0,
                max_value=1000.0,
                value=100.0,
                step=10.0,
                key="tpo_manual_tick"
            )
            atr_multiplier = 0.15  # Not used in manual mode
        else:
            manual_tick_size = 100.0  # Not used in auto mode
            atr_multiplier = st.slider(
                "ATR Multiplier",
                min_value=0.05,
                max_value=0.50,
                value=0.15,
                step=0.01,
                help="Tick size = ATR(30m) × this multiplier",
                key="tpo_atr_multiplier"
            )
    
    col3, col4 = st.columns(2)
    
    with col3:
        day_filter_tpo = st.multiselect(
            "Day of Week Filter",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            key="tpo_day_filter"
        )
    
    with col4:
        session_filter_tpo = st.multiselect(
            "Session Filter",
            options=['Asia', 'London', 'NY', 'Close'],
            default=['Asia', 'London', 'NY', 'Close'],
            key="tpo_session_filter"
        )
    
    col5, col6 = st.columns(2)
    
    with col5:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()
        
        start_date_tpo = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=180),
            min_value=min_date,
            max_value=max_date,
            key="tpo_start"
        )
    
    with col6:
        end_date_tpo = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="tpo_end"
        )
    
    sessions_to_track = st.slider(
        "Sessions to Track for Sweeps",
        min_value=1,
        max_value=10,
        value=6,
        step=1,
        help="How many future sessions to check for sweeps",
        key="tpo_sessions_track"
    )
    
    analyze_tpo = st.button(
        "Analyze Session TPO",
        use_container_width=True,
        type="primary",
        key="analyze_tpo"
    )
    if analyze_tpo:
        with st.spinner("Analyzing session TPO profiles..."):
            results = analyze_session_tpo(
                df_15m,
                start_date_tpo,
                end_date_tpo,
                day_filter_tpo,
                session_filter_tpo,
                tpo_block_size,
                tick_mode.split()[0],  # Extract "Auto" or "Manual"
                manual_tick_size,
                atr_multiplier,
                sessions_to_track
            )
            
            st.session_state.tpo_analyzed = True
            st.session_state.tpo_results = results
            st.session_state.tpo_profile_index = -1
    
    if st.session_state.tpo_analyzed and st.session_state.tpo_results:
        results = st.session_state.tpo_results
        
        if len(results) == 0:
            st.warning("No profiles found with the selected filters.")
        else:
            # Calculate statistics
            total_profiles = len(results)
            poor_high_count = sum(1 for r in results if r['is_poor_high'])
            poor_low_count = sum(1 for r in results if r['is_poor_low'])
            
            poor_high_swept_count = sum(1 for r in results if r['poor_high_swept'])
            poor_low_swept_count = sum(1 for r in results if r['poor_low_swept'])
            
            poor_high_sweep_rate = (poor_high_swept_count / poor_high_count * 100) if poor_high_count > 0 else 0
            poor_low_sweep_rate = (poor_low_swept_count / poor_low_count * 100) if poor_low_count > 0 else 0
            
            # Sessions to sweep statistics
            poor_high_sessions_to_sweep = [r['poor_high_sweep_session'] + 1 for r in results if r['poor_high_swept']]
            poor_low_sessions_to_sweep = [r['poor_low_sweep_session'] + 1 for r in results if r['poor_low_swept']]
            
            avg_sessions_high = np.mean(poor_high_sessions_to_sweep) if poor_high_sessions_to_sweep else 0
            median_sessions_high = np.median(poor_high_sessions_to_sweep) if poor_high_sessions_to_sweep else 0
            
            avg_sessions_low = np.mean(poor_low_sessions_to_sweep) if poor_low_sessions_to_sweep else 0
            median_sessions_low = np.median(poor_low_sessions_to_sweep) if poor_low_sessions_to_sweep else 0
            
            st.markdown("---")
            st.subheader("Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Profiles</div>
                    <div class="metric-value">{total_profiles}</div>
                    <div class="metric-subtitle">{tpo_block_size} blocks</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Poor Highs</div>
                    <div class="metric-value">{poor_high_count}</div>
                    <div class="metric-subtitle">{poor_high_count/total_profiles*100:.1f}% of profiles</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Poor Lows</div>
                    <div class="metric-value">{poor_low_count}</div>
                    <div class="metric-subtitle">{poor_low_count/total_profiles*100:.1f}% of profiles</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_tick = np.mean([r['tick_size'] for r in results])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Tick Size</div>
                    <div class="metric-value">${avg_tick:.0f}</div>
                    <div class="metric-subtitle">{tick_mode.split()[0]} mode</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Poor High Sweep Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Sweep Rate</div>
                    <div class="metric-value">{poor_high_sweep_rate:.1f}%</div>
                    <div class="metric-subtitle">{poor_high_swept_count}/{poor_high_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Sessions</div>
                    <div class="metric-value">{avg_sessions_high:.1f}</div>
                    <div class="metric-subtitle">to sweep</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Median Sessions</div>
                    <div class="metric-value">{median_sessions_high:.0f}</div>
                    <div class="metric-subtitle">to sweep</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_mae_high = np.mean([r['poor_high_mae_pct'] for r in results if r['poor_high_swept']]) if poor_high_swept_count > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg MAE</div>
                    <div class="metric-value">{avg_mae_high:.2f}%</div>
                    <div class="metric-subtitle">before sweep</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Poor Low Sweep Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Sweep Rate</div>
                    <div class="metric-value">{poor_low_sweep_rate:.1f}%</div>
                    <div class="metric-subtitle">{poor_low_swept_count}/{poor_low_count}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg Sessions</div>
                    <div class="metric-value">{avg_sessions_low:.1f}</div>
                    <div class="metric-subtitle">to sweep</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Median Sessions</div>
                    <div class="metric-value">{median_sessions_low:.0f}</div>
                    <div class="metric-subtitle">to sweep</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_mae_low = np.mean([r['poor_low_mae_pct'] for r in results if r['poor_low_swept']]) if poor_low_swept_count > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg MAE</div>
                    <div class="metric-value">{avg_mae_low:.2f}%</div>
                    <div class="metric-subtitle">before sweep</div>
                </div>
                """, unsafe_allow_html=True)
            
            # MAE Distribution charts
            if poor_high_swept_count > 0:
                st.markdown("---")
                st.subheader("Poor High MAE Distribution")
                
                mae_high_values = [r['poor_high_mae_pct'] for r in results if r['poor_high_swept']]
                mae_buckets_high = []
                for mae in mae_high_values:
                    if mae <= 1:
                        mae_buckets_high.append("0-1%")
                    elif mae <= 2:
                        mae_buckets_high.append("1-2%")
                    elif mae <= 3:
                        mae_buckets_high.append("2-3%")
                    elif mae <= 5:
                        mae_buckets_high.append("3-5%")
                    elif mae <= 10:
                        mae_buckets_high.append("5-10%")
                    else:
                        mae_buckets_high.append("10%+")
                
                mae_counts_high = pd.Series(mae_buckets_high).value_counts()
                bucket_order = ["0-1%", "1-2%", "2-3%", "3-5%", "5-10%", "10%+"]
                mae_counts_high = mae_counts_high.reindex([b for b in bucket_order if b in mae_counts_high.index], fill_value=0)
                
                fig_mae_high = go.Figure(data=[
                    go.Bar(
                        x=mae_counts_high.index,
                        y=mae_counts_high.values,
                        marker_color=positive_color,
                        text=mae_counts_high.values,
                        textposition='outside'
                    )
                ])
                
                fig_mae_high.update_layout(
                    xaxis_title="MAE Range",
                    yaxis_title="Count",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    height=400
                )
                
                st.plotly_chart(fig_mae_high, use_container_width=True)
            
            if poor_low_swept_count > 0:
                st.markdown("---")
                st.subheader("Poor Low MAE Distribution")
                
                mae_low_values = [r['poor_low_mae_pct'] for r in results if r['poor_low_swept']]
                mae_buckets_low = []
                for mae in mae_low_values:
                    if mae <= 1:
                        mae_buckets_low.append("0-1%")
                    elif mae <= 2:
                        mae_buckets_low.append("1-2%")
                    elif mae <= 3:
                        mae_buckets_low.append("2-3%")
                    elif mae <= 5:
                        mae_buckets_low.append("3-5%")
                    elif mae <= 10:
                        mae_buckets_low.append("5-10%")
                    else:
                        mae_buckets_low.append("10%+")
                
                mae_counts_low = pd.Series(mae_buckets_low).value_counts()
                mae_counts_low = mae_counts_low.reindex([b for b in bucket_order if b in mae_counts_low.index], fill_value=0)
                
                fig_mae_low = go.Figure(data=[
                    go.Bar(
                        x=mae_counts_low.index,
                        y=mae_counts_low.values,
                        marker_color=negative_color,
                        text=mae_counts_low.values,
                        textposition='outside'
                    )
                ])
                
                fig_mae_low.update_layout(
                    xaxis_title="MAE Range",
                    yaxis_title="Count",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    height=400
                )
                
                st.plotly_chart(fig_mae_low, use_container_width=True)
            
            # Sessions to sweep distribution
            if poor_high_swept_count > 0:
                st.markdown("---")
                st.subheader("Poor High - Sessions to Sweep Distribution")
                
                sessions_counts_high = pd.Series(poor_high_sessions_to_sweep).value_counts().sort_index()
                
                fig_sessions_high = go.Figure(data=[
                    go.Bar(
                        x=sessions_counts_high.index,
                        y=sessions_counts_high.values,
                        marker_color=positive_color,
                        text=sessions_counts_high.values,
                        textposition='outside'
                    )
                ])
                
                fig_sessions_high.update_layout(
                    xaxis_title="Sessions to Sweep",
                    yaxis_title="Count",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    height=400
                )
                
                st.plotly_chart(fig_sessions_high, use_container_width=True)
            
            if poor_low_swept_count > 0:
                st.markdown("---")
                st.subheader("Poor Low - Sessions to Sweep Distribution")
                
                sessions_counts_low = pd.Series(poor_low_sessions_to_sweep).value_counts().sort_index()
                
                fig_sessions_low = go.Figure(data=[
                    go.Bar(
                        x=sessions_counts_low.index,
                        y=sessions_counts_low.values,
                        marker_color=negative_color,
                        text=sessions_counts_low.values,
                        textposition='outside'
                    )
                ])
                
                fig_sessions_low.update_layout(
                    xaxis_title="Sessions to Sweep",
                    yaxis_title="Count",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    height=400
                )
                
                st.plotly_chart(fig_sessions_low, use_container_width=True)
            
            # Profile visualization
            st.markdown("---")
            st.subheader("TPO Profile Visualization")
            st.subheader("Chart Display Options")
            col_bars1, col_bars2 = st.columns(2)
            with col_bars1:
                bars_before_session = st.number_input(
                    "Bars Before Day",
                    min_value=10,
                    max_value=200,
                    value=12,
                    step=10,
                    help="Number of bars to show before day start",
                    key="session_tpo_bars_before"
                )
            with col_bars2:
                bars_after_session = st.number_input(
                    "Bars After Day",
                    min_value=10,
                    max_value=200,
                    value=12,
                    step=10,
                    help="Number of bars to show after day end",
                    key="session_tpo_bars_after"
                )
            
            # Build day list from results
            day_list = sorted({r['date'].date() for r in results})
            if 'tpo_day_index' not in st.session_state:
                st.session_state.tpo_day_index = -1
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("◀ Previous Day", use_container_width=True, key="tpo_prev_day", type="secondary"):
                    if st.session_state.tpo_day_index > 0:
                        st.session_state.tpo_day_index -= 1
                    else:
                        st.session_state.tpo_day_index = len(day_list) - 1
                    st.rerun()
            with col2:
                if st.button("Next Day ▶", use_container_width=True, key="tpo_next_day"):
                    if st.session_state.tpo_day_index < len(day_list) - 1:
                        st.session_state.tpo_day_index += 1
                    else:
                        st.session_state.tpo_day_index = 0
                    st.rerun()
            
            day_idx = st.session_state.tpo_day_index
            if day_idx == -1:
                day_idx = len(day_list) - 1
            day_idx = max(0, min(day_idx, len(day_list) - 1))
            current_day = day_list[day_idx]
            day_sessions = [r for r in results if r['date'].date() == current_day]
            
            # Chart data for full day with bars before/after
            day_start = pd.Timestamp(current_day)
            day_end = day_start + timedelta(days=1)
            if tpo_block_size == "15m":
                chart_df = df_15m
                block_minutes = 15
            else:
                chart_df = df_15m.resample('30min').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                block_minutes = 30
            
            chart_start = day_start - timedelta(minutes=block_minutes * bars_before_session)
            chart_end = day_end + timedelta(minutes=block_minutes * bars_after_session)
            chart_data = chart_df[(chart_df.index >= chart_start) & (chart_df.index <= chart_end)]
            
            st.caption(f"Day {current_day.strftime('%Y-%m-%d')} | Sessions shown: {', '.join([r['session'] for r in day_sessions])}")
            
            fig_profile = go.Figure()
            
            # Candlesticks
            for idx, row in chart_data.iterrows():
                is_up = row['Close'] > row['Open']
                color = '#A8A8A8' if is_up else '#5A5A5A'
                opacity = 1.0
                fig_profile.add_trace(go.Bar(
                    x=[idx],
                    y=[abs(row['Close'] - row['Open'])],
                    base=[min(row['Open'], row['Close'])],
                    marker_color=color,
                    marker_line_width=0,
                    showlegend=False,
                    hoverinfo='skip',
                    opacity=opacity
                ))
                fig_profile.add_trace(go.Scatter(
                    x=[idx, idx],
                    y=[row['Low'], row['High']],
                    mode='lines',
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hoverinfo='skip',
                    opacity=opacity
                ))
            
            add_ohlc_hover_trace(fig_profile, chart_data)
            
            # Session color mapping
            session_colors = {
                'Asia': '#00E65C',
                'London': '#004C6C',
                'NY': '#FFC906',
                'Close': '#223971'
            }
            
            # Draw blocks per session
            for r in day_sessions:
                profile = r['profile_analysis']['profile']
                sorted_prices = sorted(profile.keys(), reverse=True)
            else:
                profile_counts = {price: len(tpos) for price, tpos in profile.items()}
                total_tpos = sum(profile_counts.values()) if profile_counts else 1
                value_area = compute_value_area(profile_counts, value_area_pct=0.68)
                tick_size = r['tick_size']
                base_color_session = session_colors.get(r['session'], '#B5B5B5')
                
                session_data = r['session_data']
                session_start = session_data.index[0]
                
                gap = tick_size * 0.08
                block_width = timedelta(minutes=block_minutes)
                
                for price in sorted_prices:
                    count = profile_counts.get(price, 0)
                    if count <= 0:
                        continue
                    base_color = "#B5B5B5" if price in value_area else base_color_session
                    factor = min(0.6, count / total_tpos)
                    block_color = blend_to_black(base_color, factor)
                    
                    fig_profile.add_shape(
                        type="rect",
                        x0=session_start,
                        x1=session_start + (block_width * count),
                        y0=price - (tick_size / 2) + gap,
                        y1=price + (tick_size / 2) - gap,
                        line=dict(color=grid_color, width=0.5),
                        fillcolor=block_color
                    )
                
                # Poor high/low rays for session
                if r['is_poor_high']:
                    fig_profile.add_shape(
                        type="line",
                        x0=session_start,
                        x1=session_start + timedelta(minutes=block_minutes * len(session_data)),
                        y0=r.get('poor_high_level', r['session_high']),
                        y1=r.get('poor_high_level', r['session_high']),
                        line=dict(color=negative_color, width=2)
                    )
                if r['is_poor_low']:
                    fig_profile.add_shape(
                        type="line",
                        x0=session_start,
                        x1=session_start + timedelta(minutes=block_minutes * len(session_data)),
                        y0=r.get('poor_low_level', r['session_low']),
                        y1=r.get('poor_low_level', r['session_low']),
                        line=dict(color=negative_color, width=2)
                    )
            
            # Column grid lines (block width)
            current_x = day_start
            while current_x <= day_end:
                fig_profile.add_shape(
                    type="line",
                    x0=current_x,
                    x1=current_x,
                    y0=chart_data['Low'].min() if len(chart_data) else 0,
                    y1=chart_data['High'].max() if len(chart_data) else 1,
                    line=dict(color=grid_color, width=0.5)
                )
                current_x += timedelta(minutes=block_minutes)
            
            fig_profile.update_layout(
                title=f"Session TPO Profiles - {current_day.strftime('%Y-%m-%d')}",
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(showgrid=False, color=text_color),
                yaxis=dict(showgrid=False, gridcolor=grid_color, color=text_color),
                height=700,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_profile, use_container_width=True)
            
            # Profile details per session (table)
            profile_rows = []
            for r in day_sessions:
                pa = r['profile_analysis']
                profile_rows.append({
                    'Session': r['session'],
                    'Range': f"{pa['range']:.2f}",
                    'High': f"{pa['session_high']:.2f}",
                    'Low': f"{pa['session_low']:.2f}",
                    'TPOs High': pa['tpos_at_high'],
                    'TPOs Low': pa['tpos_at_low'],
                    'Poor High': '✓' if r['is_poor_high'] else '',
                    'Poor Low': '✓' if r['is_poor_low'] else ''
                })
            
            st.markdown("---")
            st.subheader("Profile Details (Sessions)")
            st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, height=220)
            
            # Raw data table
            st.markdown("---")
            with st.expander("Raw Data - All Profiles", expanded=False):
                raw_data_tpo = []
                for r in results:
                    raw_data_tpo.append({
                        'Date': r['date'].strftime('%Y-%m-%d'),
                        'Session': r['session'],
                        'Day': r['day_name'],
                        'Tick': f"${r['tick_size']:.2f}",
                        'High': f"{r['session_high']:.2f}",
                        'Low': f"{r['session_low']:.2f}",
                        'TPOs High': r['tpos_at_high'],
                        'TPOs Low': r['tpos_at_low'],
                        'Poor H?': '✓' if r['is_poor_high'] else '',
                        'Poor L?': '✓' if r['is_poor_low'] else '',
                        'H Swept?': '✓' if r['poor_high_swept'] else '',
                        'L Swept?': '✓' if r['poor_low_swept'] else '',
                        'H Sessions': r['poor_high_sweep_session'] + 1 if r['poor_high_swept'] else '',
                        'L Sessions': r['poor_low_sweep_session'] + 1 if r['poor_low_swept'] else '',
                        'H MAE%': f"{r['poor_high_mae_pct']:.2f}%" if r['poor_high_swept'] else '',
                        'L MAE%': f"{r['poor_low_mae_pct']:.2f}%" if r['poor_low_swept'] else ''
                    })
                
                raw_df_tpo = pd.DataFrame(raw_data_tpo)
                st.dataframe(raw_df_tpo, use_container_width=True, height=400)

# ============================================================================
# PLACEHOLDER PAGES
# ============================================================================

elif page == "Monday Range":
    render_page_header(
        "Monday Range",
        "Probability of breaking Monday range and rotating across the range"
    )
    render_exchange_asset_controls("mr")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()
        start_date_mr = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=365),
            min_value=min_date,
            max_value=max_date,
            key="mr_start"
        )
    
    with col2:
        end_date_mr = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="mr_end"
        )
    
    with col3:
        max_lookforward_days_mr = st.number_input(
            "Max Look Forward Days",
            min_value=1,
            max_value=7,
            value=3,
            step=1,
            key="mr_lookforward_days"
        )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        mr_timeframe = st.selectbox(
            "Chart Timeframe",
            options=['15m', '30m', '1h', '4h', '1D'],
            index=1,
            key="mr_timeframe"
        )
    
    with col5:
        bars_before_mr = st.number_input(
            "Bars Before (Chart)",
            min_value=1,
            max_value=200,
            value=40,
            step=1,
            key="mr_bars_before"
        )
    
    with col6:
        bars_after_mr = st.number_input(
            "Bars After (Chart)",
            min_value=1,
            max_value=200,
            value=80,
            step=1,
            key="mr_bars_after"
        )
    
    analyze_mr = st.button(
        "Analyze Monday Range",
        use_container_width=True,
        type="primary",
        key="analyze_mr"
    )
    if analyze_mr:
        with st.spinner("Analyzing Monday range breaks..."):
            mondays = []
            df_range = df_15m[(df_15m.index.date >= start_date_mr) & (df_15m.index.date <= end_date_mr)]
            
            for date in pd.date_range(start_date_mr, end_date_mr, freq='D'):
                if date.dayofweek != 0:
                    continue
                monday_start = pd.Timestamp(date)  # Monday 00:00
                monday_end = monday_start + timedelta(days=1) - timedelta(minutes=1)
                
                monday_data = df_range[(df_range.index >= monday_start) & (df_range.index <= monday_end)]
                if monday_data.empty:
                    continue
                
                monday_high = monday_data['High'].max()
                monday_low = monday_data['Low'].min()
                monday_end_time = monday_data.index[-1]
                
                forward_end = monday_end_time + timedelta(days=max_lookforward_days_mr)
                forward_data = df_15m[(df_15m.index > monday_end_time) & (df_15m.index <= forward_end)]
                
                break_high = False
                break_low = False
                break_high_time = None
                break_low_time = None
                
                if not forward_data.empty:
                    high_break_idx = forward_data.index[(forward_data['High'] >= monday_high)]
                    low_break_idx = forward_data.index[(forward_data['Low'] <= monday_low)]
                    
                    if len(high_break_idx) > 0:
                        break_high = True
                        break_high_time = high_break_idx[0]
                    if len(low_break_idx) > 0:
                        break_low = True
                        break_low_time = low_break_idx[0]
                
                rotate_to_high = False
                rotate_to_low = False
                mfe_pct = None
                
                if break_low and break_low_time is not None:
                    after_break = forward_data[forward_data.index >= break_low_time]
                    if not after_break.empty:
                        rotate_to_high = (after_break['High'] >= monday_high).any()
                        if rotate_to_high:
                            sweep_time = after_break.index[after_break['High'] >= monday_high][0]
                            interim = after_break[after_break.index <= sweep_time]
                        else:
                            interim = after_break
                        min_low = interim['Low'].min() if not interim.empty else None
                        if min_low is not None and monday_low != 0:
                            mfe_pct = ((monday_low - min_low) / monday_low) * 100
                
                if break_high and break_high_time is not None:
                    after_break = forward_data[forward_data.index >= break_high_time]
                    if not after_break.empty:
                        rotate_to_low = (after_break['Low'] <= monday_low).any()
                        if rotate_to_low:
                            sweep_time = after_break.index[after_break['Low'] <= monday_low][0]
                            interim = after_break[after_break.index <= sweep_time]
                        else:
                            interim = after_break
                        max_high = interim['High'].max() if not interim.empty else None
                        if max_high is not None and monday_high != 0:
                            mfe_pct = ((max_high - monday_high) / monday_high) * 100
                
                mondays.append({
                    'monday_start': monday_start,
                    'monday_end': monday_end_time,
                    'high': monday_high,
                    'low': monday_low,
                    'break_high': break_high,
                    'break_low': break_low,
                    'rotate_to_high': rotate_to_high,
                    'rotate_to_low': rotate_to_low,
                    'mfe_pct': mfe_pct
                })
            
            st.session_state.monday_range_results = mondays
            st.session_state.monday_range_chart_index = -1
    
    if 'monday_range_results' in st.session_state and st.session_state.monday_range_results:
        mondays = st.session_state.monday_range_results
        total_mondays = len(mondays)
        
        if total_mondays == 0:
            st.warning("No Monday ranges found for the selected date range.")
        else:
            break_high_count = sum(1 for w in mondays if w['break_high'])
            break_low_count = sum(1 for w in mondays if w['break_low'])
            
            break_high_pct = (break_high_count / total_mondays) * 100
            break_low_pct = (break_low_count / total_mondays) * 100
            any_break_pct = (sum(1 for w in mondays if w['break_high'] or w['break_low']) / total_mondays) * 100
            
            low_rotations = [w for w in mondays if w['break_low']]
            high_rotations = [w for w in mondays if w['break_high']]
            
            rotate_to_high_pct = (sum(1 for w in low_rotations if w['rotate_to_high']) / len(low_rotations) * 100) if low_rotations else 0
            rotate_to_low_pct = (sum(1 for w in high_rotations if w['rotate_to_low']) / len(high_rotations) * 100) if high_rotations else 0
            
            mfe_values = [w['mfe_pct'] for w in mondays if w['mfe_pct'] is not None]
            avg_mfe = np.mean(mfe_values) if mfe_values else 0
            
            st.subheader("Summary Statistics")
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Break High %</div>
                    <div class="metric-value">{break_high_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Break Low %</div>
                    <div class="metric-value">{break_low_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Any Break %</div>
                    <div class="metric-value">{any_break_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            col_s4, col_s5, col_s6 = st.columns(3)
            with col_s4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Rotate to High % (after Low Break)</div>
                    <div class="metric-value">{rotate_to_high_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Rotate to Low % (after High Break)</div>
                    <div class="metric-value">{rotate_to_low_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg MFE %</div>
                    <div class="metric-value">{avg_mfe:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
            
            with col_nav1:
                if st.button("◀ Previous", key="mr_prev", type="secondary"):
                    if st.session_state.monday_range_chart_index > 0:
                        st.session_state.monday_range_chart_index -= 1
                    else:
                        st.session_state.monday_range_chart_index = len(mondays) - 1
                    st.rerun()
            
            with col_nav2:
                st.write("")
            
            with col_nav3:
                if st.button("Next ▶", key="mr_next"):
                    if st.session_state.monday_range_chart_index < len(mondays) - 1:
                        st.session_state.monday_range_chart_index += 1
                    else:
                        st.session_state.monday_range_chart_index = 0
                    st.rerun()
            
            current_idx = st.session_state.monday_range_chart_index
            if current_idx == -1 or current_idx >= len(mondays):
                current_idx = len(mondays) - 1
            
            current = mondays[current_idx]
            monday_end_time = current['monday_end']
            
            if mr_timeframe == '15m':
                df_chart = df_15m.copy()
            elif mr_timeframe == '30m':
                df_chart = df_15m.resample('30min').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()
            elif mr_timeframe == '1h':
                df_chart = df_15m.resample('1H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()
            elif mr_timeframe == '4h':
                df_chart = df_15m.resample('4H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()
            else:
                df_chart = df_15m.resample('1D').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()
            
            if monday_end_time in df_chart.index or len(df_chart) > 0:
                idx = df_chart.index.get_indexer([monday_end_time], method='nearest')[0]
                start_idx = max(0, idx - bars_before_mr)
                end_idx = min(len(df_chart) - 1, idx + bars_after_mr)
                chart_data = df_chart.iloc[start_idx:end_idx+1]
                
                fig_mr = go.Figure()
                fig_mr.add_trace(go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name='Price',
                    increasing_line_color=candle_up,
                    decreasing_line_color=candle_down,
                    increasing_fillcolor=candle_up,
                    decreasing_fillcolor=candle_down,
                    hovertemplate=CANDLE_HOVER_TEMPLATE
                ))
                
                x0 = current['monday_start']
                x1 = chart_data.index[-1]
                
                fig_mr.add_shape(
                    type="line",
                    x0=x0,
                    x1=x1,
                    y0=current['high'],
                    y1=current['high'],
                    line=dict(color=highlight_color, width=2, dash='solid')
                )
                
                fig_mr.add_shape(
                    type="line",
                    x0=x0,
                    x1=x1,
                    y0=current['low'],
                    y1=current['low'],
                    line=dict(color=highlight_color, width=2, dash='solid')
                )
                
                fig_mr.update_layout(
                    title=f"Monday Range - {current['monday_start'].strftime('%Y-%m-%d')}",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_mr, use_container_width=True)
            
            st.markdown("---")
            with st.expander("Raw Data", expanded=False):
                raw_mr = pd.DataFrame(mondays)
                raw_mr['monday_start'] = raw_mr['monday_start'].dt.strftime('%Y-%m-%d')
                raw_mr['monday_end'] = raw_mr['monday_end'].dt.strftime('%Y-%m-%d %H:%M')
                raw_mr = raw_mr.rename(columns={
                    'monday_start': 'Monday Start',
                    'monday_end': 'Monday End',
                    'high': 'Monday High',
                    'low': 'Monday Low',
                    'break_high': 'Break High',
                    'break_low': 'Break Low',
                    'rotate_to_high': 'Rotate to High',
                    'rotate_to_low': 'Rotate to Low',
                    'mfe_pct': 'MFE %'
                })
                st.dataframe(raw_mr, use_container_width=True, height=400)

elif page == "Session Pivots" or (pivot_page_mode and pivot_section == "Session"):
    if not pivot_page_mode:
        render_page_header(
            "Session Pivots Analysis",
            "Analyze when session highs and lows occur by time interval"
        )
    else:
        st.subheader("Session Pivots Analysis")
    render_exchange_asset_controls("sp")
    st.info("**Sessions (UTC):** Asia (00:00-06:00) | London (06:00-12:00) | New York (12:00-20:00) | Close (20:00-00:00)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        day_filter_sp = st.multiselect(
            "Day of Week Filter",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            key="sp_day_filter"
        )
    
    with col2:
        session_select_sp = st.selectbox(
            "Session",
            options=['Asia', 'London', 'New York', 'Close'],
            index=0,
            key="sp_session"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        timeframe_sp = st.selectbox(
            "Timeframe",
            options=['15m', '30m', '1h'],
            index=0,
            key="sp_timeframe"
        )
    
    with col4:
        # Generate interval options based on session and timeframe
        sessions_times = {
            'Asia': (0, 6),
            'London': (6, 12),
            'New York': (12, 20),
            'Close': (20, 24)
        }
        
        start_h, end_h = sessions_times[session_select_sp]
        
        intervals_list = []
        if timeframe_sp == '15m':
            for h in range(start_h, end_h):
                for m in [0, 15, 30, 45]:
                    intervals_list.append(f"{h:02d}:{m:02d}")
        elif timeframe_sp == '30m':
            for h in range(start_h, end_h):
                for m in [0, 30]:
                    intervals_list.append(f"{h:02d}:{m:02d}")
        else:  # 1h
            for h in range(start_h, end_h):
                intervals_list.append(f"{h:02d}:00")
        
        interval_select_sp = st.selectbox(
            "Select Interval",
            options=intervals_list,
            index=0,
            key="sp_interval"
        )
    
    col5, col6 = st.columns(2)
    
    with col5:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()
        
        start_date_sp = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=365),
            min_value=min_date,
            max_value=max_date,
            key="sp_start"
        )
    
    with col6:
        end_date_sp = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="sp_end"
        )
    
    analyze_sp = st.button(
        "Analyze Session Pivots",
        use_container_width=True,
        type="primary",
        key="analyze_sp"
    )
    if analyze_sp:
        with st.spinner(f"Analyzing {session_select_sp} session pivots..."):
            results = analyze_session_pivots(df_15m, start_date_sp, end_date_sp, day_filter_sp, session_select_sp, timeframe_sp)
            
            st.session_state.session_pivots_analyzed = True
            st.session_state.session_pivots_results = results
    
    if 'session_pivots_analyzed' in st.session_state and st.session_state.session_pivots_analyzed and 'session_pivots_results' in st.session_state:
        results = st.session_state.session_pivots_results
        
        high_intervals = pd.Series(results['high_intervals'])
        low_intervals = pd.Series(results['low_intervals'])
        p1_intervals = pd.Series(results['p1_intervals'])
        p2_intervals = pd.Series(results['p2_intervals'])
        
        total_sessions = len(results['dates'])
        
        st.markdown("---")
        st.subheader("Session High/Low Distribution by Interval")
        
        col1, col2 = st.columns(2)
        
        # Get all unique intervals
        all_intervals = sorted(set(results['high_intervals'] + results['low_intervals']))
        
        with col1:
            st.write("**Session High Distribution**")
            
            high_dist = []
            for interval in all_intervals:
                count = (high_intervals == interval).sum()
                pct = (count / total_sessions * 100) if total_sessions > 0 else 0
                
                high_dist.append({
                    'Interval': interval,
                    'Percentage': pct,
                    'Count': count
                })
            
            high_df_sp = pd.DataFrame(high_dist)
            
            def color_high(val):
                if val > 8:
                    return 'background-color: #28a745; color: white'
                elif val > 5:
                    return 'background-color: #ffc107; color: black'
                elif val > 2:
                    return 'background-color: #fd7e14; color: white'
                else:
                    return 'background-color: #dc3545; color: white'
            
            st.dataframe(
                high_df_sp.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                use_container_width=True,
                height=400
            )
        
        with col2:
            st.write("**Session Low Distribution**")
            
            low_dist = []
            for interval in all_intervals:
                count = (low_intervals == interval).sum()
                pct = (count / total_sessions * 100) if total_sessions > 0 else 0
                
                low_dist.append({
                    'Interval': interval,
                    'Percentage': pct,
                    'Count': count
                })
            
            low_df_sp = pd.DataFrame(low_dist)
            
            st.dataframe(
                low_df_sp.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                use_container_width=True,
                height=400
            )
        
        st.markdown("---")
        st.subheader("P1/P2 Distribution by Interval")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**P1 Distribution**")
            
            p1_dist = []
            for interval in all_intervals:
                count = (p1_intervals == interval).sum()
                pct = (count / total_sessions * 100) if total_sessions > 0 else 0
                
                p1_dist.append({
                    'Interval': interval,
                    'Percentage': pct,
                    'Count': count
                })
            
            p1_df_sp = pd.DataFrame(p1_dist)
            
            st.dataframe(
                p1_df_sp.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                use_container_width=True,
                height=400
            )
        
        with col2:
            st.write("**P2 Distribution**")
            
            p2_dist = []
            for interval in all_intervals:
                count = (p2_intervals == interval).sum()
                pct = (count / total_sessions * 100) if total_sessions > 0 else 0
                
                p2_dist.append({
                    'Interval': interval,
                    'Percentage': pct,
                    'Count': count
                })
            
            p2_df_sp = pd.DataFrame(p2_dist)
            
            st.dataframe(
                p2_df_sp.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                use_container_width=True,
                height=400
            )
        
        st.markdown("---")
        st.subheader("P1/P2 Probability by Interval")
        
        # Build percentage lists for chart
        p1_pcts_sp = []
        p2_pcts_sp = []
        
        for interval in all_intervals:
            p1_count = (p1_intervals == interval).sum()
            p2_count = (p2_intervals == interval).sum()
            p1_pct = (p1_count / total_sessions * 100) if total_sessions > 0 else 0
            p2_pct = (p2_count / total_sessions * 100) if total_sessions > 0 else 0
            
            p1_pcts_sp.append(p1_pct)
            p2_pcts_sp.append(p2_pct)
        
        fig_p1p2_sp = go.Figure()
        
        fig_p1p2_sp.add_trace(go.Bar(
            x=all_intervals,
            y=p1_pcts_sp,
            name='Pivot 1 %',
            marker_color=light_blue,
            text=[f"{v:.1f}" for v in p1_pcts_sp],
            textposition='inside'
        ))
        
        fig_p1p2_sp.add_trace(go.Bar(
            x=all_intervals,
            y=p2_pcts_sp,
            name='Pivot 2 %',
            marker_color=dark_blue,
            text=[f"{v:.1f}" for v in p2_pcts_sp],
            textposition='inside'
        ))
        
        # Add Total % line
        total_pcts_sp = [p1_pcts_sp[i] + p2_pcts_sp[i] for i in range(len(all_intervals))]
        
        fig_p1p2_sp.add_trace(go.Scatter(
            x=all_intervals,
            y=total_pcts_sp,
            name='Total %',
            mode='lines+markers',
            line=dict(color=orange_line, width=3),
            marker=dict(size=8)
        ))
        
        fig_p1p2_sp.update_layout(
            barmode='group',
            plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg,
            font=dict(color=text_color),
            xaxis=dict(title="Time Interval", showgrid=False, color=text_color),
            yaxis=dict(title="Percentage (%)", showgrid=True, gridcolor=grid_color, color=text_color),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig_p1p2_sp, use_container_width=True)
        
        st.markdown("---")
        st.subheader(f"Hit Rate Evolution: {interval_select_sp}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pass  # Spacer
        
        with col2:
            rolling_window_sp = st.number_input(
                "Rolling Window (sessions)",
                min_value=1,
                max_value=100,
                value=20,
                step=1,
                key="sp_rolling_window"
            )
        
        # Calculate hit rates for selected interval
        high_hits_sp = [1 if interval == interval_select_sp else 0 for interval in results['high_intervals']]
        low_hits_sp = [1 if interval == interval_select_sp else 0 for interval in results['low_intervals']]
        
        df_evolution_sp = pd.DataFrame({
            'date': results['dates'],
            'high_hit': high_hits_sp,
            'low_hit': low_hits_sp
        })
        
        df_evolution_sp['cumulative_high'] = (df_evolution_sp['high_hit'].expanding().sum() / df_evolution_sp['high_hit'].expanding().count()) * 100
        df_evolution_sp['cumulative_low'] = (df_evolution_sp['low_hit'].expanding().sum() / df_evolution_sp['low_hit'].expanding().count()) * 100
        
        df_evolution_sp['rolling_high'] = df_evolution_sp['high_hit'].rolling(window=min(rolling_window_sp, len(df_evolution_sp)), min_periods=1).mean() * 100
        df_evolution_sp['rolling_low'] = df_evolution_sp['low_hit'].rolling(window=min(rolling_window_sp, len(df_evolution_sp)), min_periods=1).mean() * 100
        
        # Chart for High hit rate
        st.write("**Session Highs Hit Rate Evolution**")
        
        fig_evolution_high_sp = go.Figure()
        
        fig_evolution_high_sp.add_trace(go.Scatter(
            x=df_evolution_sp['date'],
            y=df_evolution_sp['cumulative_high'],
            mode='lines',
            name='Cumulative High Rate',
            line=dict(color=blue_color, width=2)
        ))
        
        fig_evolution_high_sp.add_trace(go.Scatter(
            x=df_evolution_sp['date'],
            y=df_evolution_sp['rolling_high'],
            mode='lines',
            name=f'Rolling High Rate ({rolling_window_sp})',
            line=dict(color=yellow_color, width=2, dash='dash')
        ))
        
        fig_evolution_high_sp.update_layout(
            plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg,
            font=dict(color=text_color),
            xaxis=dict(showgrid=False, color=text_color),
            yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350
        )
        
        st.plotly_chart(fig_evolution_high_sp, use_container_width=True)
        
        # Chart for Low hit rate
        st.write("**Session Lows Hit Rate Evolution**")
        
        fig_evolution_low_sp = go.Figure()
        
        fig_evolution_low_sp.add_trace(go.Scatter(
            x=df_evolution_sp['date'],
            y=df_evolution_sp['cumulative_low'],
            mode='lines',
            name='Cumulative Low Rate',
            line=dict(color=dark_blue, width=2)
        ))
        
        fig_evolution_low_sp.add_trace(go.Scatter(
            x=df_evolution_sp['date'],
            y=df_evolution_sp['rolling_low'],
            mode='lines',
            name=f'Rolling Low Rate ({rolling_window_sp})',
            line=dict(color=orange_line, width=2, dash='dash')
        ))
        
        fig_evolution_low_sp.update_layout(
            plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg,
            font=dict(color=text_color),
            xaxis=dict(showgrid=False, color=text_color),
            yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350
        )
        
        st.plotly_chart(fig_evolution_low_sp, use_container_width=True)
        
        st.markdown("---")
        st.subheader(f"{session_select_sp} Session Chart ({timeframe_sp} Candles)")
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            bars_before_sp = st.number_input(
                "Bars Before (Chart)",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                key="sp_bars_before"
            )
        
        with col2:
            bars_after_sp = st.number_input(
                "Bars After (Chart)",
                min_value=1,
                max_value=100,
                value=20,
                step=1,
                key="sp_bars_after"
            )
        
        with col3:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("◀ Previous", use_container_width=True, key="prev_session", type="secondary"):
                if st.session_state.session_pivots_chart_index > 0:
                    st.session_state.session_pivots_chart_index -= 1
                else:
                    st.session_state.session_pivots_chart_index = len(results['session_data_list']) - 1
                st.rerun()
        
        with col4:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("Next ▶", use_container_width=True, key="next_session", type="secondary"):
                if st.session_state.session_pivots_chart_index < len(results['session_data_list']) - 1:
                    st.session_state.session_pivots_chart_index += 1
                else:
                    st.session_state.session_pivots_chart_index = 0
                st.rerun()
        
        # Get current session data
        current_session_idx = st.session_state.session_pivots_chart_index
        if current_session_idx == -1:
            current_session_idx = len(results['session_data_list']) - 1
        
        session_info = results['session_data_list'][current_session_idx]
        session_date = session_info['date']
        high_interval = session_info['high_interval']
        low_interval = session_info['low_interval']
        session_data_raw = session_info['session_data']
        
        st.caption(f"Session {current_session_idx + 1} of {len(results['session_data_list'])} | {session_date.strftime('%Y-%m-%d')} ({session_info['day_name']}) | High: {high_interval} | Low: {low_interval}")
        
        if len(session_data_raw) > 0:
            # Find target interval indices
            target_time = pd.Timestamp(f"{session_date.strftime('%Y-%m-%d')} {interval_select_sp}")
            
            # Create chart with bars before/after
            if target_time in session_data_raw.index:
                target_idx = session_data_raw.index.get_loc(target_time)
                
                # Extend to include bars before/after from full df_15m
                chart_start = target_time - pd.Timedelta(minutes=15*bars_before_sp if timeframe_sp == '15m' else 30*bars_before_sp if timeframe_sp == '30m' else 60*bars_before_sp)
                chart_end = target_time + pd.Timedelta(minutes=15*bars_after_sp if timeframe_sp == '15m' else 30*bars_after_sp if timeframe_sp == '30m' else 60*bars_after_sp)
                
                # Resample based on timeframe
                if timeframe_sp == '15m':
                    chart_data = df_15m[(df_15m.index >= chart_start) & (df_15m.index <= chart_end)]
                elif timeframe_sp == '30m':
                    chart_data_full = df_15m[(df_15m.index >= chart_start) & (df_15m.index <= chart_end)]
                    chart_data = chart_data_full.resample('30min').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last'
                    }).dropna()
                else:  # 1h
                    chart_data_full = df_15m[(df_15m.index >= chart_start) & (df_15m.index <= chart_end)]
                    chart_data = chart_data_full.resample('1H').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last'
                    }).dropna()
                
                fig_session = go.Figure()
                
                # Add candles, highlighting intervals with high/low
                for idx, row in chart_data.iterrows():
                    is_up = row['Close'] > row['Open']
                    current_interval = idx.strftime('%H:%M')
                    
                    # Highlight if this interval had high or low
                    if current_interval == high_interval or current_interval == low_interval:
                        color = highlight_color
                    else:
                        color = candle_up if is_up else candle_down
                    
                    fig_session.add_trace(go.Bar(
                        x=[idx],
                        y=[abs(row['Close'] - row['Open'])],
                        base=[min(row['Open'], row['Close'])],
                        marker_color=color,
                        marker_line_width=0,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig_session.add_trace(go.Scatter(
                        x=[idx, idx],
                        y=[row['Low'], row['High']],
                        mode='lines',
                        line=dict(color=color, width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                add_ohlc_hover_trace(fig_session, chart_data)
                
                fig_session.update_layout(
                    title=f"{session_select_sp} Session - {session_date.strftime('%Y-%m-%d')}",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_session, use_container_width=True)
            else:
                st.warning(f"Selected interval {interval_select_sp} not found in this session")
        else:
            st.warning("No data available for this session")
        
        st.markdown("---")


# ============================================================================
# WEEKLY PIVOTS PAGE
# ============================================================================

elif page == "Weekly Pivots" or (pivot_page_mode and pivot_section == "Weekly"):
    if not pivot_page_mode:
        render_page_header(
            "Weekly Pivots Analysis",
            "Analyze when weekly highs and lows occur by day of week"
        )
    else:
        st.subheader("Weekly Pivots Analysis")
    render_exchange_asset_controls("wp")
    st.info("**Days of Week:** Monday (1) | Tuesday (2) | Wednesday (3) | Thursday (4) | Friday (5) | Saturday (6) | Sunday (7)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        day_filter_wp = st.multiselect(
            "Day of Week Filter",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            key="wp_day_filter"
        )
    
    with col2:
        pass  # Spacer
    
    col3, col4 = st.columns(2)
    
    with col3:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()
        
        start_date_wp = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=365),
            min_value=min_date,
            max_value=max_date,
            key="wp_start"
        )
    
    with col4:
        end_date_wp = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="wp_end"
        )
    
    analyze_wp = st.button(
        "Analyze Weekly Pivots",
        use_container_width=True,
        type="primary",
        key="analyze_wp"
    )
    if analyze_wp:
        with st.spinner("Analyzing weekly pivots..."):
            results = analyze_weekly_pivots(df_15m, df_daily, start_date_wp, end_date_wp, day_filter_wp)
            
            st.session_state.weekly_pivots_analyzed = True
            st.session_state.weekly_pivots_results = results
    
    if 'weekly_pivots_analyzed' in st.session_state and st.session_state.weekly_pivots_analyzed and 'weekly_pivots_results' in st.session_state:
        results = st.session_state.weekly_pivots_results
        
        high_days = pd.Series(results['high_days'])
        low_days = pd.Series(results['low_days'])
        p1_days = pd.Series(results['p1_days'])
        p2_days = pd.Series(results['p2_days'])
        
        total_weeks = len(results['dates'])
        
        st.markdown("---")
        st.subheader("Weekly High/Low Distribution by Day")
        
        col1, col2 = st.columns(2)
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        with col1:
            st.write("**Weekly High Distribution**")
            
            high_dist = []
            for day in day_order:
                count = (high_days == day).sum()
                pct = (count / total_weeks * 100) if total_weeks > 0 else 0
                
                high_dist.append({
                    'Day': day,
                    'Percentage': pct,
                    'Count': count
                })
            
            high_df_wp = pd.DataFrame(high_dist)
            
            def color_high(val):
                if val > 20:
                    return 'background-color: #28a745; color: white'
                elif val > 15:
                    return 'background-color: #ffc107; color: black'
                elif val > 10:
                    return 'background-color: #fd7e14; color: white'
                else:
                    return 'background-color: #dc3545; color: white'
            
            st.dataframe(
                high_df_wp.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                use_container_width=True
            )
        
        with col2:
            st.write("**Weekly Low Distribution**")
            
            low_dist = []
            for day in day_order:
                count = (low_days == day).sum()
                pct = (count / total_weeks * 100) if total_weeks > 0 else 0
                
                low_dist.append({
                    'Day': day,
                    'Percentage': pct,
                    'Count': count
                })
            
            low_df_wp = pd.DataFrame(low_dist)
            
            st.dataframe(
                low_df_wp.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                use_container_width=True
            )
        
        st.markdown("---")
        st.subheader("P1/P2 Distribution by Day")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**P1 Distribution**")
            
            p1_dist = []
            for day in day_order:
                count = (p1_days == day).sum()
                pct = (count / total_weeks * 100) if total_weeks > 0 else 0
                
                p1_dist.append({
                    'Day': day,
                    'Percentage': pct,
                    'Count': count
                })
            
            p1_df_wp = pd.DataFrame(p1_dist)
            
            st.dataframe(
                p1_df_wp.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                use_container_width=True
            )
        
        with col2:
            st.write("**P2 Distribution**")
            
            p2_dist = []
            for day in day_order:
                count = (p2_days == day).sum()
                pct = (count / total_weeks * 100) if total_weeks > 0 else 0
                
                p2_dist.append({
                    'Day': day,
                    'Percentage': pct,
                    'Count': count
                })
            
            p2_df_wp = pd.DataFrame(p2_dist)
            
            st.dataframe(
                p2_df_wp.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                use_container_width=True
            )
        
        st.markdown("---")
        st.subheader("P1/P2 Probability by Day")
        
        p1_pcts_wp = [p1_df_wp.iloc[i]['Percentage'] for i in range(len(day_order))]
        p2_pcts_wp = [p2_df_wp.iloc[i]['Percentage'] for i in range(len(day_order))]
        
        fig_p1p2_wp = go.Figure()
        
        fig_p1p2_wp.add_trace(go.Bar(
            x=day_order,
            y=p1_pcts_wp,
            name='Pivot 1 %',
            marker_color=light_blue,
            text=[f"{v:.1f}" for v in p1_pcts_wp],
            textposition='inside'
        ))
        
        fig_p1p2_wp.add_trace(go.Bar(
            x=day_order,
            y=p2_pcts_wp,
            name='Pivot 2 %',
            marker_color=dark_blue,
            text=[f"{v:.1f}" for v in p2_pcts_wp],
            textposition='inside'
        ))
        
        # Add Total % line
        total_pcts_wp = [p1_pcts_wp[i] + p2_pcts_wp[i] for i in range(len(day_order))]
        
        fig_p1p2_wp.add_trace(go.Scatter(
            x=day_order,
            y=total_pcts_wp,
            name='Total %',
            mode='lines+markers',
            line=dict(color=orange_line, width=3),
            marker=dict(size=8)
        ))
        
        fig_p1p2_wp.update_layout(
            barmode='group',
            plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg,
            font=dict(color=text_color),
            xaxis=dict(title="Day of Week", showgrid=False, color=text_color),
            yaxis=dict(title="Percentage (%)", showgrid=True, gridcolor=grid_color, color=text_color),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig_p1p2_wp, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Hit Rate Evolution by Day")
        
        col1, col2 = st.columns(2)
        
        with col1:
            day_select_wp = st.selectbox(
                "Select Day of Week",
                options=day_order,
                index=0,
                key="wp_day_select"
            )
        
        with col2:
            rolling_window_wp = st.number_input(
                "Rolling Window (weeks)",
                min_value=1,
                max_value=52,
                value=10,
                step=1,
                key="wp_rolling_window"
            )
        
        # Calculate hit rates for selected day
        high_hits_wp = [1 if d == day_select_wp else 0 for d in results['high_days']]
        low_hits_wp = [1 if d == day_select_wp else 0 for d in results['low_days']]
        
        df_evolution_wp = pd.DataFrame({
            'date': results['dates'],
            'high_hit': high_hits_wp,
            'low_hit': low_hits_wp
        })
        
        df_evolution_wp['cumulative_high'] = (df_evolution_wp['high_hit'].expanding().sum() / df_evolution_wp['high_hit'].expanding().count()) * 100
        df_evolution_wp['cumulative_low'] = (df_evolution_wp['low_hit'].expanding().sum() / df_evolution_wp['low_hit'].expanding().count()) * 100
        
        df_evolution_wp['rolling_high'] = df_evolution_wp['high_hit'].rolling(window=min(rolling_window_wp, len(df_evolution_wp)), min_periods=1).mean() * 100
        df_evolution_wp['rolling_low'] = df_evolution_wp['low_hit'].rolling(window=min(rolling_window_wp, len(df_evolution_wp)), min_periods=1).mean() * 100
        
        # Chart for High hit rate
        st.write("**Weekly Highs Hit Rate Evolution**")
        
        fig_evolution_high_wp = go.Figure()
        
        fig_evolution_high_wp.add_trace(go.Scatter(
            x=df_evolution_wp['date'],
            y=df_evolution_wp['cumulative_high'],
            mode='lines',
            name='Cumulative High Rate',
            line=dict(color=blue_color, width=2)
        ))
        
        fig_evolution_high_wp.add_trace(go.Scatter(
            x=df_evolution_wp['date'],
            y=df_evolution_wp['rolling_high'],
            mode='lines',
            name=f'Rolling High Rate ({rolling_window_wp}w)',
            line=dict(color=yellow_color, width=2, dash='dash')
        ))
        
        fig_evolution_high_wp.update_layout(
            plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg,
            font=dict(color=text_color),
            xaxis=dict(showgrid=False, color=text_color),
            yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350
        )
        
        st.plotly_chart(fig_evolution_high_wp, use_container_width=True)
        
        # Chart for Low hit rate
        st.write("**Weekly Lows Hit Rate Evolution**")
        
        fig_evolution_low_wp = go.Figure()
        
        fig_evolution_low_wp.add_trace(go.Scatter(
            x=df_evolution_wp['date'],
            y=df_evolution_wp['cumulative_low'],
            mode='lines',
            name='Cumulative Low Rate',
            line=dict(color=dark_blue, width=2)
        ))
        
        fig_evolution_low_wp.add_trace(go.Scatter(
            x=df_evolution_wp['date'],
            y=df_evolution_wp['rolling_low'],
            mode='lines',
            name=f'Rolling Low Rate ({rolling_window_wp}w)',
            line=dict(color=orange_line, width=2, dash='dash')
        ))
        
        fig_evolution_low_wp.update_layout(
            plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg,
            font=dict(color=text_color),
            xaxis=dict(showgrid=False, color=text_color),
            yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350
        )
        
        st.plotly_chart(fig_evolution_low_wp, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Weekly Chart (4H Candles)")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("◀ Previous Week", use_container_width=True, key="prev_week", type="secondary"):
                if st.session_state.weekly_pivots_chart_index > 0:
                    st.session_state.weekly_pivots_chart_index -= 1
                else:
                    st.session_state.weekly_pivots_chart_index = len(results['week_data_list']) - 1
                st.rerun()
        
        with col2:
            if st.button("Next Week ▶", use_container_width=True, key="next_week", type="secondary"):
                if st.session_state.weekly_pivots_chart_index < len(results['week_data_list']) - 1:
                    st.session_state.weekly_pivots_chart_index += 1
                else:
                    st.session_state.weekly_pivots_chart_index = 0
                st.rerun()
        
        # Get current week data
        current_week_idx = st.session_state.weekly_pivots_chart_index
        if current_week_idx == -1:
            current_week_idx = len(results['week_data_list']) - 1
        
        week_info = results['week_data_list'][current_week_idx]
        week_start = week_info['week_start']
        week_end = week_info['week_end']
        high_day = week_info['high_day']
        low_day = week_info['low_day']
        
        st.caption(f"Week {current_week_idx + 1} of {len(results['week_data_list'])} | {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')} | High: {high_day} | Low: {low_day}")
        
        # Resample to 4H candles for the week
        week_data_15m = df_15m[(df_15m.index >= week_start) & (df_15m.index <= week_end)]
        
        if len(week_data_15m) > 0:
            week_data_4h = week_data_15m.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).dropna()
            
            fig_week = go.Figure()
            
            # Add candles, highlighting the day that put in high/low
            for idx, row in week_data_4h.iterrows():
                is_up = row['Close'] > row['Open']
                current_day = idx.day_name()
                
                # Highlight if this candle is from the day that put in high or low
                if current_day == high_day or current_day == low_day:
                    color = highlight_color
                else:
                    color = candle_up if is_up else candle_down
                
                fig_week.add_trace(go.Bar(
                    x=[idx],
                    y=[abs(row['Close'] - row['Open'])],
                    base=[min(row['Open'], row['Close'])],
                    marker_color=color,
                    marker_line_width=0,
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig_week.add_trace(go.Scatter(
                    x=[idx, idx],
                    y=[row['Low'], row['High']],
                    mode='lines',
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            add_ohlc_hover_trace(fig_week, week_data_4h)
            
            fig_week.update_layout(
                title=f"Week of {week_start.strftime('%Y-%m-%d')} (4H Candles)",
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(showgrid=False, color=text_color),
                yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_week, use_container_width=True)
        else:
            st.warning("No data available for this week")
        
        st.markdown("---")


# ============================================================================
# MONTHLY PIVOTS PAGE
# ============================================================================

elif page == "Monthly Pivots" or (pivot_page_mode and pivot_section == "Monthly"):
    if not pivot_page_mode:
        render_page_header(
            "Monthly Pivots Analysis",
            "Analyze when monthly highs and lows occur by day of month"
        )
    else:
        st.subheader("Monthly Pivots Analysis")
    render_exchange_asset_controls("mp")
    st.info("**Days of Month:** 1st, 2nd, 3rd... through 28th/29th/30th/31st depending on month")
    
    col1, col2 = st.columns(2)
    
    with col1:
        day_filter_mp = st.multiselect(
            "Day of Week Filter (optional)",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            key="mp_day_filter",
            help="Filter by day of week that the high/low occurred on"
        )
    
    with col2:
        pass  # Spacer
    
    col3, col4 = st.columns(2)
    
    with col3:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()
        
        start_date_mp = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=730),
            min_value=min_date,
            max_value=max_date,
            key="mp_start"
        )
    
    with col4:
        end_date_mp = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="mp_end"
        )
    
    analyze_mp = st.button(
        "Analyze Monthly Pivots",
        use_container_width=True,
        type="primary",
        key="analyze_mp"
    )
    if analyze_mp:
        with st.spinner("Analyzing monthly pivots..."):
            results = analyze_monthly_pivots(df_15m, df_daily, start_date_mp, end_date_mp, day_filter_mp)
            
            st.session_state.monthly_pivots_analyzed = True
            st.session_state.monthly_pivots_results = results
    
    if 'monthly_pivots_analyzed' in st.session_state and st.session_state.monthly_pivots_analyzed and 'monthly_pivots_results' in st.session_state:
        results = st.session_state.monthly_pivots_results
        
        high_days = pd.Series(results['high_days'])
        low_days = pd.Series(results['low_days'])
        p1_days = pd.Series(results['p1_days'])
        p2_days = pd.Series(results['p2_days'])
        
        total_months = len(results['dates'])
        
        st.markdown("---")
        st.subheader("Monthly High/Low Distribution by Day of Month")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Monthly High Distribution**")
            
            high_dist = []
            for day in range(1, 32):
                count = (high_days == day).sum()
                pct = (count / total_months * 100) if total_months > 0 else 0
                
                if count > 0:  # Only show days that have data
                    high_dist.append({
                        'Day': f"{day}",
                        'Percentage': pct,
                        'Count': count
                    })
            
            high_df_mp = pd.DataFrame(high_dist)
            
            def color_high(val):
                if val > 8:
                    return 'background-color: #28a745; color: white'
                elif val > 5:
                    return 'background-color: #ffc107; color: black'
                elif val > 3:
                    return 'background-color: #fd7e14; color: white'
                else:
                    return 'background-color: #dc3545; color: white'
            
            st.dataframe(
                high_df_mp.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                use_container_width=True,
                height=600
            )
        
        with col2:
            st.write("**Monthly Low Distribution**")
            
            low_dist = []
            for day in range(1, 32):
                count = (low_days == day).sum()
                pct = (count / total_months * 100) if total_months > 0 else 0
                
                if count > 0:  # Only show days that have data
                    low_dist.append({
                        'Day': f"{day}",
                        'Percentage': pct,
                        'Count': count
                    })
            
            low_df_mp = pd.DataFrame(low_dist)
            
            st.dataframe(
                low_df_mp.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                use_container_width=True,
                height=600
            )
        
        st.markdown("---")
        st.subheader("P1/P2 Distribution by Day of Month")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**P1 Distribution**")
            
            p1_dist = []
            for day in range(1, 32):
                count = (p1_days == day).sum()
                pct = (count / total_months * 100) if total_months > 0 else 0
                
                if count > 0:
                    p1_dist.append({
                        'Day': f"{day}",
                        'Percentage': pct,
                        'Count': count
                    })
            
            p1_df_mp = pd.DataFrame(p1_dist)
            
            st.dataframe(
                p1_df_mp.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                use_container_width=True,
                height=600
            )
        
        with col2:
            st.write("**P2 Distribution**")
            
            p2_dist = []
            for day in range(1, 32):
                count = (p2_days == day).sum()
                pct = (count / total_months * 100) if total_months > 0 else 0
                
                if count > 0:
                    p2_dist.append({
                        'Day': f"{day}",
                        'Percentage': pct,
                        'Count': count
                    })
            
            p2_df_mp = pd.DataFrame(p2_dist)
            
            st.dataframe(
                p2_df_mp.style.applymap(color_high, subset=['Percentage']).format({'Percentage': '{:.2f}%'}),
                use_container_width=True,
                height=600
            )
        
        st.markdown("---")
        st.subheader("P1/P2 Probability by Day of Month")
        
        # Get unique days that have data
        all_days_mp = sorted(set(p1_days.tolist() + p2_days.tolist()))
        
        # Build percentage lists for chart
        p1_pcts_mp = []
        p2_pcts_mp = []
        day_labels_mp = []
        
        for day in all_days_mp:
            p1_count = (p1_days == day).sum()
            p2_count = (p2_days == day).sum()
            p1_pct = (p1_count / total_months * 100) if total_months > 0 else 0
            p2_pct = (p2_count / total_months * 100) if total_months > 0 else 0
            
            p1_pcts_mp.append(p1_pct)
            p2_pcts_mp.append(p2_pct)
            day_labels_mp.append(str(day))
        
        fig_p1p2_mp = go.Figure()
        
        fig_p1p2_mp.add_trace(go.Bar(
            x=day_labels_mp,
            y=p1_pcts_mp,
            name='Pivot 1 %',
            marker_color=light_blue,
            text=[f"{v:.1f}" for v in p1_pcts_mp],
            textposition='inside'
        ))
        
        fig_p1p2_mp.add_trace(go.Bar(
            x=day_labels_mp,
            y=p2_pcts_mp,
            name='Pivot 2 %',
            marker_color=dark_blue,
            text=[f"{v:.1f}" for v in p2_pcts_mp],
            textposition='inside'
        ))
        
        # Add Total % line
        total_pcts_mp = [p1_pcts_mp[i] + p2_pcts_mp[i] for i in range(len(day_labels_mp))]
        
        fig_p1p2_mp.add_trace(go.Scatter(
            x=day_labels_mp,
            y=total_pcts_mp,
            name='Total %',
            mode='lines+markers',
            line=dict(color=orange_line, width=3),
            marker=dict(size=8)
        ))
        
        fig_p1p2_mp.update_layout(
            barmode='group',
            plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg,
            font=dict(color=text_color),
            xaxis=dict(title="Day of Month", showgrid=False, color=text_color),
            yaxis=dict(title="Percentage (%)", showgrid=True, gridcolor=grid_color, color=text_color),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig_p1p2_mp, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Hit Rate Evolution by Day of Month")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get unique days that have data
            available_days_mp = sorted(set(results['high_days'] + results['low_days']))
            day_select_mp = st.selectbox(
                "Select Day of Month",
                options=available_days_mp,
                index=0,
                key="mp_day_select"
            )
        
        with col2:
            rolling_window_mp = st.number_input(
                "Rolling Window (months)",
                min_value=1,
                max_value=24,
                value=6,
                step=1,
                key="mp_rolling_window"
            )
        
        # Calculate hit rates for selected day
        high_hits_mp = [1 if d == day_select_mp else 0 for d in results['high_days']]
        low_hits_mp = [1 if d == day_select_mp else 0 for d in results['low_days']]
        
        df_evolution_mp = pd.DataFrame({
            'date': results['dates'],
            'high_hit': high_hits_mp,
            'low_hit': low_hits_mp
        })
        
        df_evolution_mp['cumulative_high'] = (df_evolution_mp['high_hit'].expanding().sum() / df_evolution_mp['high_hit'].expanding().count()) * 100
        df_evolution_mp['cumulative_low'] = (df_evolution_mp['low_hit'].expanding().sum() / df_evolution_mp['low_hit'].expanding().count()) * 100
        
        df_evolution_mp['rolling_high'] = df_evolution_mp['high_hit'].rolling(window=min(rolling_window_mp, len(df_evolution_mp)), min_periods=1).mean() * 100
        df_evolution_mp['rolling_low'] = df_evolution_mp['low_hit'].rolling(window=min(rolling_window_mp, len(df_evolution_mp)), min_periods=1).mean() * 100
        
        # Chart for High hit rate
        st.write("**Monthly Highs Hit Rate Evolution**")
        
        fig_evolution_high_mp = go.Figure()
        
        fig_evolution_high_mp.add_trace(go.Scatter(
            x=df_evolution_mp['date'],
            y=df_evolution_mp['cumulative_high'],
            mode='lines',
            name='Cumulative High Rate',
            line=dict(color=blue_color, width=2)
        ))
        
        fig_evolution_high_mp.add_trace(go.Scatter(
            x=df_evolution_mp['date'],
            y=df_evolution_mp['rolling_high'],
            mode='lines',
            name=f'Rolling High Rate ({rolling_window_mp}m)',
            line=dict(color=yellow_color, width=2, dash='dash')
        ))
        
        fig_evolution_high_mp.update_layout(
            plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg,
            font=dict(color=text_color),
            xaxis=dict(showgrid=False, color=text_color),
            yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350
        )
        
        st.plotly_chart(fig_evolution_high_mp, use_container_width=True)
        
        # Chart for Low hit rate
        st.write("**Monthly Lows Hit Rate Evolution**")
        
        fig_evolution_low_mp = go.Figure()
        
        fig_evolution_low_mp.add_trace(go.Scatter(
            x=df_evolution_mp['date'],
            y=df_evolution_mp['cumulative_low'],
            mode='lines',
            name='Cumulative Low Rate',
            line=dict(color=dark_blue, width=2)
        ))
        
        fig_evolution_low_mp.add_trace(go.Scatter(
            x=df_evolution_mp['date'],
            y=df_evolution_mp['rolling_low'],
            mode='lines',
            name=f'Rolling Low Rate ({rolling_window_mp}m)',
            line=dict(color=orange_line, width=2, dash='dash')
        ))
        
        fig_evolution_low_mp.update_layout(
            plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg,
            font=dict(color=text_color),
            xaxis=dict(showgrid=False, color=text_color),
            yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350
        )
        
        st.plotly_chart(fig_evolution_low_mp, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Monthly Chart (Daily Candles)")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("◀ Previous Month", use_container_width=True, key="prev_month", type="secondary"):
                if st.session_state.monthly_pivots_chart_index > 0:
                    st.session_state.monthly_pivots_chart_index -= 1
                else:
                    st.session_state.monthly_pivots_chart_index = len(results['month_data_list']) - 1
                st.rerun()
        
        with col2:
            if st.button("Next Month ▶", use_container_width=True, key="next_month", type="secondary"):
                if st.session_state.monthly_pivots_chart_index < len(results['month_data_list']) - 1:
                    st.session_state.monthly_pivots_chart_index += 1
                else:
                    st.session_state.monthly_pivots_chart_index = 0
                st.rerun()
        
        # Get current month data
        current_month_idx = st.session_state.monthly_pivots_chart_index
        if current_month_idx == -1:
            current_month_idx = len(results['month_data_list']) - 1
        
        month_info = results['month_data_list'][current_month_idx]
        month_start = month_info['month_start']
        month_end = month_info['month_end']
        high_day = month_info['high_day']
        low_day = month_info['low_day']
        
        st.caption(f"Month {current_month_idx + 1} of {len(results['month_data_list'])} | {month_start.strftime('%Y-%m-%d')} to {month_end.strftime('%Y-%m-%d')} | High: Day {high_day} | Low: Day {low_day}")
        
        # Get daily candles for the month
        month_data_daily = df_daily[(df_daily.index >= month_start) & (df_daily.index <= month_end)]
        
        if len(month_data_daily) > 0:
            fig_month = go.Figure()
            
            # Add candles, highlighting the day that put in high/low
            for idx, row in month_data_daily.iterrows():
                is_up = row['Close'] > row['Open']
                current_day = idx.day
                
                # Highlight if this is the day that put in high or low
                if current_day == high_day or current_day == low_day:
                    color = highlight_color
                else:
                    color = candle_up if is_up else candle_down
                
                fig_month.add_trace(go.Bar(
                    x=[idx],
                    y=[abs(row['Close'] - row['Open'])],
                    base=[min(row['Open'], row['Close'])],
                    marker_color=color,
                    marker_line_width=0,
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig_month.add_trace(go.Scatter(
                    x=[idx, idx],
                    y=[row['Low'], row['High']],
                    mode='lines',
                    line=dict(color=color, width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            add_ohlc_hover_trace(fig_month, month_data_daily)
            
            fig_month.update_layout(
                title=f"Month of {month_start.strftime('%B %Y')} (Daily Candles)",
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(showgrid=False, color=text_color),
                yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_month, use_container_width=True)
        else:
            st.warning("No data available for this month")
        
        st.markdown("---")

# ============================================================================
# DAILY TPO ANALYSIS FUNCTION
# ============================================================================

def analyze_daily_tpo(df_15m, start_date, end_date, day_filter, tick_mode, manual_tick, atr_multiplier, days_to_track):
    """
    Analyze daily TPO profiles with ATR-based tick sizing
    Similar to session TPO but for full trading days
    """
    
    # Calculate ATR from last 60 15-min bars (15 hours)
    atr_period = 60
    
    results = []
    
    # Get all trading days
    all_days = []
    for date in pd.date_range(start_date, end_date, freq='D'):
        daily_data = df_15m[df_15m.index.date == date.date()]
        
        if len(daily_data) > 0:
            day_name = date.day_name()
            all_days.append({
                'date': date,
                'day_name': day_name,
                'data': daily_data
            })
    
    for i, day_info in enumerate(all_days):
        date = day_info['date']
        day_name = day_info['day_name']
        daily_data = day_info['data']
        
        # Filter by day of week
        if day_filter and day_name not in day_filter:
            continue
        
        if len(daily_data) < 10:  # Need at least 10 bars for meaningful profile
            continue
        
        # Calculate tick size using ATR
        if tick_mode == "Auto":
            atr = calculate_atr(daily_data, atr_period)
            tick_size = atr * atr_multiplier
            
            # Validate tick_size
            if pd.isna(tick_size) or tick_size <= 0:
                tick_size = 100.0  # Default fallback
        else:
            tick_size = manual_tick
        
        tick_size = max(float(tick_size), 1.0)
        
        # Resample to 30m bars for Daily TPO
        daily_data_30m = resample_to_30m(daily_data)
        
        if len(daily_data_30m) < 5:  # Need at least 5 30m bars
            continue
        
        # Generate extended alphabet letters
        tpo_letters = generate_tpo_letters(len(daily_data_30m))
        
        # Build TPO profile from 30m data
        profile = build_tpo_profile_with_letters(daily_data_30m, tick_size, tpo_letters)
        
        if not profile:
            continue
        
        # Analyze profile for poor highs/lows
        profile_analysis = analyze_tpo_profile(profile)
        
        if not profile_analysis:
            continue
        
        # Get future days for sweep tracking
        future_days_data = []
        for j in range(i + 1, min(i + 1 + days_to_track, len(all_days))):
            future_days_data.append(all_days[j]['data'])
        
        # Track poor high sweeps
        poor_high_swept = False
        poor_high_sweep_day = None
        poor_high_mae = 0
        poor_high_mae_pct = 0
        
        if profile_analysis['is_poor_high']:
            poor_high_swept, sweep_idx, mae, mae_pct = check_sweep(
                profile_analysis['poor_high_level'],
                True,
                future_days_data,
                tick_size
            )
            poor_high_sweep_day = sweep_idx
            poor_high_mae = mae
            poor_high_mae_pct = mae_pct
        
        # Track poor low sweeps
        poor_low_swept = False
        poor_low_sweep_day = None
        poor_low_mae = 0
        poor_low_mae_pct = 0
        
        if profile_analysis['is_poor_low']:
            poor_low_swept, sweep_idx, mae, mae_pct = check_sweep(
                profile_analysis['poor_low_level'],
                False,
                future_days_data,
                tick_size
            )
            poor_low_sweep_day = sweep_idx
            poor_low_mae = mae
            poor_low_mae_pct = mae_pct
        
        # Store results
        results.append({
            'date': date,
            'day_name': day_name,
            'data': daily_data,  # Keep original 15m for display
            'data_30m': daily_data_30m,  # 30m data used for TPO
            'tpo_letters': tpo_letters,  # Letter sequence
            'profile': profile,
            'tick_size': tick_size,
            'atr': atr if tick_mode == "Auto" else None,
            'is_poor_high': profile_analysis['is_poor_high'],
            'is_poor_low': profile_analysis['is_poor_low'],
            'poor_high_swept': poor_high_swept,
            'poor_high_sweep_day': poor_high_sweep_day,
            'poor_high_mae': poor_high_mae,
            'poor_high_mae_pct': poor_high_mae_pct,
            'poor_low_swept': poor_low_swept,
            'poor_low_sweep_day': poor_low_sweep_day,
            'poor_low_mae': poor_low_mae,
            'poor_low_mae_pct': poor_low_mae_pct,
            'session_high': profile_analysis['session_high'],
            'session_low': profile_analysis['session_low'],
            'range': profile_analysis['range']
        })
    
    return results

if page == "Daily TPO":
    render_page_header(
        "Daily TPO Analysis",
        "Analyze poor highs/lows in daily profiles and their sweep rates"
    )
    render_exchange_asset_controls("daily_tpo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tick_mode_daily = st.selectbox(
            "Tick Size Mode",
            options=["Auto (ATR × Multiplier)", "Manual"],
            index=0,
            key="daily_tpo_tick_mode"
        )
    
    with col2:
        if tick_mode_daily == "Manual":
            manual_tick_daily = st.number_input(
                "Manual Tick Size ($)",
                min_value=1.0,
                max_value=1000.0,
                value=100.0,
                step=10.0,
                key="daily_tpo_manual_tick"
            )
            atr_multiplier_daily = 0.15  # Not used in manual mode
        else:
            manual_tick_daily = 100.0  # Not used in auto mode
            atr_multiplier_daily = st.slider(
                "ATR Multiplier",
                min_value=0.05,
                max_value=0.50,
                value=0.15,
                step=0.01,
                help="Tick size = ATR × this multiplier (calculated from last 15 hours)",
                key="daily_tpo_atr_multiplier"
            )
    
    col3, col4 = st.columns(2)
    
    with col3:
        day_filter_daily_tpo = st.multiselect(
            "Day of Week Filter",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            key="daily_tpo_day_filter"
        )
    
    with col4:
        days_to_track = st.slider(
            "Days to Track for Sweeps",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="How many future days to check for sweeps",
            key="daily_tpo_days_track"
        )
    
    col5, col6 = st.columns(2)
    
    with col5:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()
        
        start_date_daily_tpo = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=180),
            min_value=min_date,
            max_value=max_date,
            key="daily_tpo_start"
        )
    
    with col6:
        end_date_daily_tpo = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="daily_tpo_end"
        )
    
    analyze_daily_tpo_btn = st.button(
        " Analyze Daily TPO",
        use_container_width=True,
        type="primary",
        key="analyze_daily_tpo"
    )
    if analyze_daily_tpo_btn:
        with st.spinner("Analyzing daily TPO profiles..."):
            results = analyze_daily_tpo(
                df_15m,
                start_date_daily_tpo,
                end_date_daily_tpo,
                day_filter_daily_tpo,
                tick_mode_daily.split()[0],  # Extract "Auto" or "Manual"
                manual_tick_daily,
                atr_multiplier_daily,
                days_to_track
            )
            
            st.session_state.daily_tpo_analyzed = True
            st.session_state.daily_tpo_results = results
            st.session_state.daily_tpo_profile_index = -1
    
    if 'daily_tpo_analyzed' in st.session_state and st.session_state.daily_tpo_analyzed and 'daily_tpo_results' in st.session_state:
        results = st.session_state.daily_tpo_results
        
        if len(results) == 0:
            st.warning("No profiles found with the selected filters.")
        else:
            st.markdown("---")
            st.subheader("Summary Statistics")
            
            # Calculate statistics
            total_profiles = len(results)
            poor_high_count = sum(1 for r in results if r['is_poor_high'])
            poor_low_count = sum(1 for r in results if r['is_poor_low'])
            
            poor_high_swept_count = sum(1 for r in results if r['poor_high_swept'])
            poor_low_swept_count = sum(1 for r in results if r['poor_low_swept'])
            
            poor_high_sweep_rate = (poor_high_swept_count / poor_high_count * 100) if poor_high_count > 0 else 0
            poor_low_sweep_rate = (poor_low_swept_count / poor_low_count * 100) if poor_low_count > 0 else 0
            
            # MAE statistics
            poor_high_maes = [r['poor_high_mae_pct'] for r in results if r['is_poor_high']]
            poor_low_maes = [r['poor_low_mae_pct'] for r in results if r['is_poor_low']]
            
            avg_high_mae = np.mean(poor_high_maes) if poor_high_maes else 0
            avg_low_mae = np.mean(poor_low_maes) if poor_low_maes else 0
            
            # Display statistics (even 2x3 grid)
            row1_c1, row1_c2, row1_c3 = st.columns(3)
            with row1_c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Profiles</div>
                    <div class="metric-value">{total_profiles}</div>
                </div>
                """, unsafe_allow_html=True)
            with row1_c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Poor Highs</div>
                    <div class="metric-value">{poor_high_count}</div>
                    <div class="metric-subtitle">{poor_high_count/total_profiles*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with row1_c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Poor Lows</div>
                    <div class="metric-value">{poor_low_count}</div>
                    <div class="metric-subtitle">{poor_low_count/total_profiles*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            row2_c1, row2_c2, row2_c3 = st.columns(3)
            with row2_c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Poor High Sweep Rate</div>
                    <div class="metric-value">{poor_high_sweep_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with row2_c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Poor Low Sweep Rate</div>
                    <div class="metric-value">{poor_low_sweep_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with row2_c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg MAE (High / Low)</div>
                    <div class="metric-value">{avg_high_mae:.2f}% / {avg_low_mae:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Profile navigation
            st.markdown("---")
            st.subheader("Chart Display Options")
            
            col_bars1, col_bars2 = st.columns(2)
            with col_bars1:
                bars_before_daily = st.number_input(
                    "Bars Before Day",
                    min_value=10,
                    max_value=200,
                    value=12,
                    step=10,
                    help="Number of 15m bars to show before day start",
                    key="daily_tpo_bars_before"
                )
            with col_bars2:
                bars_after_daily = st.number_input(
                    "Bars After Day",
                    min_value=10,
                    max_value=200,
                    value=20,
                    step=10,
                    help="Number of 15m bars to show after day end",
                    key="daily_tpo_bars_after"
                )
            
            profile_display_mode = st.radio(
                "Profile Display",
                options=["Letters", "Blocks"],
                horizontal=True,
                key="daily_tpo_display_mode"
            )
            
            st.markdown("---")
            st.subheader("Profile Viewer")
            
            if 'daily_tpo_profile_index' not in st.session_state:
                st.session_state.daily_tpo_profile_index = -1
            
            col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
            
            with col_nav1:
                if st.button("◀ Previous Profile", key="prev_daily_tpo", type="secondary"):
                    if st.session_state.daily_tpo_profile_index > 0:
                        st.session_state.daily_tpo_profile_index -= 1
                    else:
                        st.session_state.daily_tpo_profile_index = len(results) - 1
                    st.rerun()
            
            with col_nav3:
                if st.button("Next Profile ▶", key="next_daily_tpo"):
                    if st.session_state.daily_tpo_profile_index < len(results) - 1:
                        st.session_state.daily_tpo_profile_index += 1
                    else:
                        st.session_state.daily_tpo_profile_index = 0
                    st.rerun()
            
            # Get current profile
            current_idx = st.session_state.daily_tpo_profile_index
            if current_idx == -1 or current_idx >= len(results):
                current_idx = len(results) - 1
            
            # Ensure index is within bounds
            current_idx = max(0, min(current_idx, len(results) - 1))
            
            current_profile = results[current_idx]
            
            # Profile info
            st.caption(f"Profile {current_idx + 1} of {len(results)} | {current_profile['date'].strftime('%Y-%m-%d')} ({current_profile['day_name']}) | Tick Size: ${current_profile['tick_size']:.2f}" + 
                      (f" | ATR: ${current_profile['atr']:.2f}" if current_profile['atr'] else ""))
            
            # Create TPO visualization
            fig = go.Figure()
            
            # Add candlesticks with grey colors
            # Extend chart range by bars before/after
            daily_30m = current_profile['data_30m']
            day_start = daily_30m.index[0]
            day_end = daily_30m.index[-1]
            chart_start = day_start - timedelta(minutes=30 * bars_before_daily)
            chart_end = day_end + timedelta(minutes=30 * bars_after_daily)
            extended_30m = df_15m.resample('30min').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            chart_data = extended_30m[(extended_30m.index >= chart_start) & (extended_30m.index <= chart_end)]

            fig.add_trace(go.Candlestick(
                x=chart_data.index,
                open=chart_data['Open'],
                high=chart_data['High'],
                low=chart_data['Low'],
                close=chart_data['Close'],
                name='Price',
                increasing_line_color='#A8A8A8',
                decreasing_line_color='#5A5A5A',
                increasing_fillcolor='#A8A8A8',
                decreasing_fillcolor='#5A5A5A',
                hovertemplate=CANDLE_HOVER_TEMPLATE
            ))
            
            # Add poor high line
            if current_profile.get('is_poor_high'):
                color_high = 'green' if current_profile['poor_high_swept'] else 'red'
                fig.add_shape(
                    type="line",
                    x0=chart_data.index[0],
                    x1=chart_data.index[-1],
                    y0=current_profile.get('poor_high_level', current_profile['session_high']),
                    y1=current_profile.get('poor_high_level', current_profile['session_high']),
                    line=dict(color=color_high, width=2, dash='solid')
                )
            
            # Add poor low line
            if current_profile.get('is_poor_low'):
                color_low = 'green' if current_profile['poor_low_swept'] else 'red'
                fig.add_shape(
                    type="line",
                    x0=chart_data.index[0],
                    x1=chart_data.index[-1],
                    y0=current_profile.get('poor_low_level', current_profile['session_low']),
                    y1=current_profile.get('poor_low_level', current_profile['session_low']),
                    line=dict(color=color_low, width=2, dash='solid')
                )
            
            # Add TPO profile (vertical text on left, same as Session TPO)
            profile = current_profile['profile']

            # Sort prices from high to low for proper display
            sorted_prices = sorted(profile.keys(), reverse=True)
            
            # Position profile at chart start (left edge)
            first_timestamp = daily_30m.index[0]
            
            if profile_display_mode == "Letters":
                letter_sequence = current_profile.get('tpo_letters', [])

                for price in sorted_prices:
                    letters_at_price = profile[price]

                    # Keep letter order aligned to bar sequence (same build order used by blocks)
                    if letter_sequence:
                        letter_set = set(letters_at_price)
                        ordered_letters = [ltr for ltr in letter_sequence if ltr in letter_set]
                        tpo_letters = ''.join(ordered_letters)
                    else:
                        tpo_letters = ''.join(letters_at_price)

                    # Determine color - red for poor extremes
                    if price == current_profile['session_high'] and current_profile['is_poor_high']:
                        text_color_tpo = negative_color
                    elif price == current_profile['session_low'] and current_profile['is_poor_low']:
                        text_color_tpo = negative_color
                    else:
                        text_color_tpo = "#FFFFFF"

                    fig.add_annotation(
                        x=first_timestamp,
                        y=price,
                        text=tpo_letters,
                        showarrow=False,
                        font=dict(family="monospace", size=10, color=text_color_tpo),
                        xanchor='left',
                        yanchor='middle'
                    )
            else:
                profile_counts = {price: len(tpos) for price, tpos in profile.items()}
                total_tpos = sum(profile_counts.values()) if profile_counts else 1
                value_area = compute_value_area(profile_counts, value_area_pct=0.68)
                block_width = timedelta(minutes=30)
                tick_size = current_profile['tick_size']
                
                max_count = max(profile_counts.values()) if profile_counts else 1
                gap = tick_size * 0.08  # small gap between rows
                column_step = timedelta(minutes=30)

                for price in sorted_prices:
                    count = profile_counts.get(price, 0)
                    if count <= 0:
                        continue
                    base_color = "#B5B5B5" if price in value_area else "#6495ED"
                    factor = min(0.6, count / total_tpos)
                    block_color = blend_to_black(base_color, factor)

                    # Highlight poor high/low blocks by sweep status
                    if price == current_profile.get('poor_high_level') and current_profile['is_poor_high']:
                        block_color = "#22C55E" if current_profile['poor_high_swept'] else "#EF4444"
                    if price == current_profile.get('poor_low_level') and current_profile['is_poor_low']:
                        block_color = "#22C55E" if current_profile['poor_low_swept'] else "#EF4444"
                    
                    fig.add_shape(
                        type="rect",
                        x0=first_timestamp,
                        x1=first_timestamp + (block_width * count),
                        y0=price - (tick_size / 2) + gap,
                        y1=price + (tick_size / 2) - gap,
                        line=dict(color=grid_color, width=0.5),
                        fillcolor=block_color
                    )

                # Column grid lines (30m columns)
                column_end = first_timestamp + (block_width * max_count)
                current_x = first_timestamp
                while current_x <= column_end:
                    fig.add_shape(
                        type="line",
                        x0=current_x,
                        x1=current_x,
                        y0=min(sorted_prices) - (tick_size / 2),
                        y1=max(sorted_prices) + (tick_size / 2),
                        line=dict(color=grid_color, width=0.5)
                    )
                    current_x += column_step
            
            fig.update_layout(
                title=f"Daily TPO - {current_profile['date'].strftime('%Y-%m-%d')} ({current_profile['day_name']}) | 30m Bars",
                xaxis_title="Time (UTC)",
                yaxis_title="Price",
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(showgrid=False, color=text_color),
                yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                height=600,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Profile details
            st.markdown("---")
            st.subheader("Profile Details")
            
            col_d1, col_d2, col_d3 = st.columns(3)
            
            with col_d1:
                st.write("**Profile Levels:**")
                st.write(f"High: ${current_profile['session_high']:.2f}")
                st.write(f"Low: ${current_profile['session_low']:.2f}")
                st.write(f"Range: ${current_profile['range']:.2f}")
            
            with col_d2:
                st.write("**Profile Info:**")
                st.write(f"Range: ${current_profile['range']:.2f}")
                st.write(f"TPOs at High: {current_profile.get('tpos_at_high', 'N/A')}")
                st.write(f"TPOs at Low: {current_profile.get('tpos_at_low', 'N/A')}")
            
            with col_d3:
                st.write("**Sweep Info:**")
                if current_profile['is_poor_high']:
                    sweep_status_h = "✅ Swept" if current_profile['poor_high_swept'] else "❌ Not Swept"
                    st.write(f"Poor High: {sweep_status_h}")
                    if current_profile['poor_high_swept']:
                        st.write(f"Day {current_profile['poor_high_sweep_day'] + 1} | MAE: {current_profile['poor_high_mae_pct']:.2f}%")
                else:
                    st.write("Poor High: None")
                
                if current_profile['is_poor_low']:
                    sweep_status_l = "✅ Swept" if current_profile['poor_low_swept'] else "❌ Not Swept"
                    st.write(f"Poor Low: {sweep_status_l}")
                    if current_profile['poor_low_swept']:
                        st.write(f"Day {current_profile['poor_low_sweep_day'] + 1} | MAE: {current_profile['poor_low_mae_pct']:.2f}%")
                else:
                    st.write("Poor Low: None")

            # Raw data table
            st.markdown("---")
            with st.expander("Raw Data - All Profiles", expanded=False):
                raw_daily = []
                for r in results:
                    raw_daily.append({
                        'Date': r['date'].strftime('%Y-%m-%d'),
                        'Day': r['day_name'],
                        'Tick': f"${r['tick_size']:.2f}",
                        'High': f"{r['session_high']:.2f}",
                        'Low': f"{r['session_low']:.2f}",
                        'Poor H?': '✓' if r['is_poor_high'] else '',
                        'Poor L?': '✓' if r['is_poor_low'] else '',
                        'H Swept?': '✓' if r['poor_high_swept'] else '',
                        'L Swept?': '✓' if r['poor_low_swept'] else '',
                        'H Day': r['poor_high_sweep_day'] + 1 if r['poor_high_swept'] else '',
                        'L Day': r['poor_low_sweep_day'] + 1 if r['poor_low_swept'] else '',
                        'H MAE%': f"{r['poor_high_mae_pct']:.2f}%" if r['poor_high_swept'] else '',
                        'L MAE%': f"{r['poor_low_mae_pct']:.2f}%" if r['poor_low_swept'] else ''
                    })
                
                st.dataframe(pd.DataFrame(raw_daily), use_container_width=True, height=400)
            
elif page == "Large Wick Fills":
    render_page_header(
        "Large Wick Fills Analysis",
        "Track and analyze large wick fills with partial and full fill statistics"
    )
    render_exchange_asset_controls("wf")
    
    col1, col2, col3 = st.columns(3)

    
    with col1:
        timeframe_wf = st.selectbox(
            "Timeframe",
            options=['15m', '30m', '1h', '4h', 'Session', '1D', '1W', '1M'],
            index=1,  # Default 30m
            key="wf_timeframe"
        )
    
    with col2:
        method_wf = st.selectbox(
            "Wick Measurement Method",
            options=['% of Price', '% of Body'],
            index=0,
            key="wf_method"
        )
    
    with col3:
            wick_direction_wf = st.selectbox(
                "Wick Direction",
                options=['Both', 'Top', 'Bottom'],
                index=0,
                key="wf_direction"
        )
    col4, col5, col6 = st.columns(3)
    
    with col4:
        min_wick_size_wf = st.slider(
            "Minimum Wick Size (%)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="wf_min_wick"
        )
    
    with col5:
        partial_threshold_wf = st.slider(
            "Partial Fill Threshold (%)",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            key="wf_partial_threshold"
        )
    
    with col6:
        wf_lookforward = st.number_input(
            "Max Lookforward Bars",
            min_value=10,
            max_value=5000,
            value=1000,
            step=50,
            key="wf_lookforward"
        )
    
    # Session filter (only show if timeframe is Session)
    if timeframe_wf == 'Session':
        col7, col8 = st.columns(2)
        
        with col7:
            session_wf = st.multiselect(
                "Sessions (select one or more)",
                options=['Asia', 'London', 'New York', 'Close'],
                default=['Asia'],
                key="wf_session"
            )
        
        with col8:
            pass  # Spacer
    else:
        session_wf = None
    
    day_filter_wf = st.multiselect(
        "Day of Week Filter",
        options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        key="wf_day_filter"
    )

    col11, col12 = st.columns(2)

    with col11:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()

        start_date_wf = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=365),
            min_value=min_date,
            max_value=max_date,
            key="wf_start"
        )

    with col12:
        end_date_wf = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="wf_end"
        )
    
    analyze_wf = st.button(
        " Analyze Wick Fills",
        use_container_width=True,
        type="primary",
        key="analyze_wf"
    )
    if analyze_wf:
        # Validate session selection for Session timeframe
        if timeframe_wf == 'Session' and (not session_wf or len(session_wf) == 0):
            st.error("Please select at least one session when using Session timeframe.")
        else:
            with st.spinner("Analyzing wick fills..."):
                results = analyze_wick_fills(
                    df_15m, start_date_wf, end_date_wf, timeframe_wf, method_wf,
                    min_wick_size_wf, partial_threshold_wf, wick_direction_wf,
                    day_filter_wf, session_wf, wf_lookforward
                )
                
                st.session_state.wick_fills_analyzed = True
                st.session_state.wick_fills_results = results
    
    if 'wick_fills_analyzed' in st.session_state and st.session_state.wick_fills_analyzed and 'wick_fills_results' in st.session_state:
        results = st.session_state.wick_fills_results
        
        if len(results['times']) == 0:
            st.warning("No wicks found matching the criteria. Try adjusting the filters.")
        else:
            st.markdown("---")
            st.subheader("Summary Statistics")
            
            # Calculate statistics
            wick_sizes = np.array(results['wick_sizes'])
            bars_partial = np.array([b for b in results['bars_partial'] if b is not None])
            bars_full = np.array([b for b in results['bars_full'] if b is not None])
            
            avg_wick_size = np.mean(wick_sizes)
            median_partial = np.median(bars_partial) if len(bars_partial) > 0 else 0
            median_full = np.median(bars_full) if len(bars_full) > 0 else 0
            
            # Winsorized averages (cap at 5th and 95th percentile)
            if len(bars_partial) > 0:
                p5_partial = np.percentile(bars_partial, 5)
                p95_partial = np.percentile(bars_partial, 95)
                bars_partial_winsor = np.clip(bars_partial, p5_partial, p95_partial)
                winsor_avg_partial = np.mean(bars_partial_winsor)
            else:
                winsor_avg_partial = 0
            
            if len(bars_full) > 0:
                p5_full = np.percentile(bars_full, 5)
                p95_full = np.percentile(bars_full, 95)
                bars_full_winsor = np.clip(bars_full, p5_full, p95_full)
                winsor_avg_full = np.mean(bars_full_winsor)
            else:
                winsor_avg_full = 0
            
            # Unfilled percentage
            unfilled_count = sum(1 for status in results['fill_status'] if status == 'Unfilled')
            unfilled_pct = (unfilled_count / len(results['times'])) * 100
            
            # Display statistics in cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Average Wick Size</div>
                    <div class="metric-value">{avg_wick_size:.2f} %</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Median Partial Bars</div>
                    <div class="metric-value">{median_partial:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Winsorised Avg Partial Bars</div>
                    <div class="metric-value">{winsor_avg_partial:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Unfilled Wick %</div>
                    <div class="metric-value">{unfilled_pct:.1f} %</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Median Full Bars</div>
                    <div class="metric-value">{median_full:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Winsorised Avg Full Bars</div>
                    <div class="metric-value">{winsor_avg_full:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Cumulative Probability of Wick Fill")
            
            # Calculate cumulative probabilities
            max_bars = max(max([b for b in results['bars_partial'] if b is not None], default=0),
                          max([b for b in results['bars_full'] if b is not None], default=0))
            
            if max_bars > 0:
                bars_range = range(1, min(max_bars + 1, 101))  # Cap at 100 for readability
                
                partial_probs = []
                full_probs = []
                
                for bar_count in bars_range:
                    # Partial fill probability
                    partial_filled = sum(1 for b in results['bars_partial'] if b is not None and b <= bar_count)
                    partial_prob = (partial_filled / len(results['times'])) * 100
                    partial_probs.append(partial_prob)
                    
                    # Full fill probability
                    full_filled = sum(1 for b in results['bars_full'] if b is not None and b <= bar_count)
                    full_prob = (full_filled / len(results['times'])) * 100
                    full_probs.append(full_prob)
                
                fig_prob = go.Figure()
                
                fig_prob.add_trace(go.Scatter(
                    x=list(bars_range),
                    y=partial_probs,
                    mode='lines+markers',
                    name=f'Bars Partial (≥{partial_threshold_wf}% fill)',
                    line=dict(color=blue_color, width=3),
                    marker=dict(size=6)
                ))
                
                fig_prob.add_trace(go.Scatter(
                    x=list(bars_range),
                    y=full_probs,
                    mode='lines+markers',
                    name='Bars Full (100% fill)',
                    line=dict(color=dark_blue, width=3),
                    marker=dict(size=6)
                ))
                
                fig_prob.update_layout(
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(title="Number of Bars", showgrid=False, color=text_color),
                    yaxis=dict(title="Probability (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Price Chart - Individual Wick Navigation")
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                bars_before_wf = st.number_input(
                    "Bars Before",
                    min_value=1,
                    max_value=200,
                    value=20,
                    step=5,
                    key="wf_bars_before"
                )
            
            with col2:
                bars_after_wf = st.number_input(
                    "Bars After",
                    min_value=1,
                    max_value=200,
                    value=30,
                    step=5,
                    key="wf_bars_after"
                )
            
            with col3:
                st.write("")  # Spacer
                st.write("")  # Spacer
                if st.button("◀ Previous", use_container_width=True, key="prev_wick", type="secondary"):
                    if st.session_state.wick_fills_chart_index > 0:
                        st.session_state.wick_fills_chart_index -= 1
                    else:
                        st.session_state.wick_fills_chart_index = len(results['times']) - 1
                    st.rerun()
            
            with col4:
                st.write("")  # Spacer
                st.write("")  # Spacer
                if st.button("Next ▶", use_container_width=True, key="next_wick", type="secondary"):
                    if st.session_state.wick_fills_chart_index < len(results['times']) - 1:
                        st.session_state.wick_fills_chart_index += 1
                    else:
                        st.session_state.wick_fills_chart_index = 0
                    st.rerun()
            
            # Get current wick
            current_wick_idx = st.session_state.wick_fills_chart_index
            if current_wick_idx == -1:
                current_wick_idx = len(results['times']) - 1
            
            wick_time = results['times'][current_wick_idx]
            wick_type = results['wick_types'][current_wick_idx]
            wick_level = results['wick_levels'][current_wick_idx]
            wick_size_pct = results['wick_sizes'][current_wick_idx]
            bars_partial_wf = results['bars_partial'][current_wick_idx]
            bars_full_wf = results['bars_full'][current_wick_idx]
            fill_status_wf = results['fill_status'][current_wick_idx]
            partial_fill_date = results['partial_fill_dates'][current_wick_idx]
            full_fill_date = results['full_fill_dates'][current_wick_idx]
            partial_fill_price = results['partial_fill_prices'][current_wick_idx]
            full_fill_price = results['full_fill_prices'][current_wick_idx]
            
            # Build caption
            caption_parts = [
                f"Wick {current_wick_idx + 1} of {len(results['times'])}",
                f"{wick_time.strftime('%Y-%m-%d %H:%M')}",
                f"{wick_type} Wick: {wick_size_pct:.2f}%",
                f"Status: {fill_status_wf}"
            ]
            
            if bars_partial_wf:
                caption_parts.append(f"Partial: {bars_partial_wf} bars")
            if bars_full_wf:
                caption_parts.append(f"Full: {bars_full_wf} bars")
            
            st.caption(" | ".join(caption_parts))
            
            # Get chart data range
            # Find wick time in df_resampled
            if timeframe_wf == '15m':
                chart_df = df_15m.copy()
            elif timeframe_wf == '30m':
                chart_df = df_15m.resample('30min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe_wf == '1h':
                chart_df = df_15m.resample('1H').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe_wf == '4h':
                chart_df = df_15m.resample('4H').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe_wf == '1D':
                chart_df = df_15m.resample('1D').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe_wf == '1W':
                chart_df = df_15m.resample('W-SUN').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe_wf == '1M':
                chart_df = df_15m.resample('M').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe_wf == 'Session':
                # Rebuild ALL session candles for chart (not just selected session)
                sessions = {
                    'Asia': (0, 6),
                    'London': (6, 12),
                    'New York': (12, 20),
                    'Close': (20, 24)
                }
                session_candles = []
                for date in pd.date_range(start_date_wf, end_date_wf, freq='D'):
                    # Build candle for each session type
                    for session_name, (start_hour, end_hour) in sessions.items():
                        session_start = pd.Timestamp(date) + pd.Timedelta(hours=start_hour)
                        session_end = pd.Timestamp(date) + pd.Timedelta(hours=end_hour)
                        session_data = df_15m[(df_15m.index >= session_start) & (df_15m.index < session_end)]
                        if len(session_data) > 0:
                            session_candles.append({
                                'Time': session_start,
                                'Open': session_data['Open'].iloc[0],
                                'High': session_data['High'].max(),
                                'Low': session_data['Low'].min(),
                                'Close': session_data['Close'].iloc[-1],
                                'SessionType': session_name
                            })
                chart_df = pd.DataFrame(session_candles).set_index('Time')
                # Sort by time to ensure proper ordering
                chart_df = chart_df.sort_index()
            
            # Find index of wick time in chart_df
            if wick_time in chart_df.index:
                wick_idx = chart_df.index.get_loc(wick_time)
                
                start_idx = max(0, wick_idx - bars_before_wf)
                end_idx = min(len(chart_df), wick_idx + bars_after_wf + 1)
                
                chart_data = chart_df.iloc[start_idx:end_idx]
                
                fig_chart = go.Figure()
                
                # Add candles
                for idx, row in chart_data.iterrows():
                    is_wick_candle = (idx == wick_time)
                    is_up = row['Close'] > row['Open']
                    
                    if is_wick_candle:
                        color = gold_highlight  # Gold for wick candle
                    else:
                        color = candle_up if is_up else candle_down
                    
                    # Body
                    fig_chart.add_trace(go.Bar(
                        x=[idx],
                        y=[abs(row['Close'] - row['Open'])],
                        base=[min(row['Open'], row['Close'])],
                        marker_color=color,
                        marker_line_width=0,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # Wicks
                    fig_chart.add_trace(go.Scatter(
                        x=[idx, idx],
                        y=[row['Low'], row['High']],
                        mode='lines',
                        line=dict(color=color, width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                add_ohlc_hover_trace(fig_chart, chart_data)
                
                # Add wick level line
                fig_chart.add_trace(go.Scatter(
                    x=[chart_data.index[0], chart_data.index[-1]],
                    y=[wick_level, wick_level],
                    mode='lines',
                    name=f'{wick_type} Wick Level',
                    line=dict(color=orange_line, width=2, dash='solid'),
                    hoverinfo='skip'
                ))
                
                # Add partial fill marker if exists
                if partial_fill_date and partial_fill_date in chart_data.index:
                    fig_chart.add_trace(go.Scatter(
                        x=[partial_fill_date],
                        y=[partial_fill_price],
                        mode='markers+text',
                        name='Partial Fill',
                        marker=dict(size=12, color=yellow_color, symbol='star'),
                        text=['P'],
                        textposition='top center',
                        textfont=dict(size=10, color=text_color)
                    ))
                
                # Add full fill marker if exists
                if full_fill_date and full_fill_date in chart_data.index:
                    fig_chart.add_trace(go.Scatter(
                        x=[full_fill_date],
                        y=[full_fill_price],
                        mode='markers+text',
                        name='Full Fill',
                        marker=dict(size=12, color=blue_color, symbol='star'),
                        text=['F'],
                        textposition='top center',
                        textfont=dict(size=10, color=text_color)
                    ))
                
                fig_chart.update_layout(
                    title=f"{wick_type} Wick - {wick_time.strftime('%Y-%m-%d %H:%M')} ({timeframe_wf})",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_chart, use_container_width=True)
            else:
                st.warning("Selected wick time not found in chart data")
            
            st.markdown("---")
            st.subheader("Wick Details")
            
            # Create enhanced details table with all fields
            details_df = pd.DataFrame({
                'Time': results['times'],
                'Day': results['day_names'],
                'Session': results['sessions'],
                'Candle Color': results['candle_colors'],
                'Wick': results['wick_types'],
                'Wick Size %': [f"{w:.2f}" for w in results['wick_sizes']],
                'Open': [f"{o:.2f}" for o in results['opens']],
                'High': [f"{h:.2f}" for h in results['highs']],
                'Low': [f"{l:.2f}" for l in results['lows']],
                'Close': [f"{c:.2f}" for c in results['closes']],
                'Wick Level': [f"{wl:.2f}" for wl in results['wick_levels']],
                'Bars Partial': results['bars_partial'],
                'Bars Full': results['bars_full'],
                'Partial %': [f"{p:.1f}" for p in results['partial_pcts']],
                'Partial Fill Price': [f"{pfp:.2f}" if pfp is not None else "N/A" for pfp in results['partial_fill_prices']],
                'Partial Fill Date': [pfd.strftime('%Y-%m-%d %H:%M') if pfd is not None else "N/A" for pfd in results['partial_fill_dates']],
                'Full Fill Price': [f"{ffp:.2f}" if ffp is not None else "N/A" for ffp in results['full_fill_prices']],
                'Full Fill Date': [ffd.strftime('%Y-%m-%d %H:%M') if ffd is not None else "N/A" for ffd in results['full_fill_dates']],
                'Status': results['fill_status']
            })
            
            st.dataframe(details_df, use_container_width=True, height=500)

elif page == "Naked Opens":
    render_page_header(
        "Naked Opens Analysis",
        "Analyze candles with flat tops/bottoms and track if the open level gets revisited"
    )
    render_exchange_asset_controls("no")
    
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        timeframe_no = st.selectbox(
            "Timeframe",
            options=['15m', '30m', '1h', '4H', 'Daily', 'Weekly', 'Monthly'],
            index=3,  # Default 4H
            key="no_timeframe"
        )
    
    with col2:
        direction_filter_no = st.selectbox(
            "Direction Filter",
            options=['Both', 'Bullish Only', 'Bearish Only'],
            index=0,
            key="no_direction"
        )
    
    with col3:
        no_lookforward = st.number_input(
            "Max Lookforward Bars",
            min_value=10,
            max_value=5000,
            value=100,
            step=10,
            key="no_lookforward"
        )
    
    day_filter_no = st.multiselect(
        "Day of Week Filter",
        options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        key="no_day_filter"
    )
    col6, col7 = st.columns(2)
    
    with col6:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()
        
        start_date_no = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=365),
            min_value=min_date,
            max_value=max_date,
            key="no_start"
        )
    
    with col7:
        end_date_no = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="no_end"
        )
    
    analyze_no = st.button(
        " Analyze Naked Opens",
        use_container_width=True,
        type="primary",
        key="analyze_no"
    )
    if analyze_no:
        with st.spinner("Analyzing naked opens..."):
            results = analyze_naked_opens(
                df_15m, start_date_no, end_date_no, timeframe_no,
                direction_filter_no, day_filter_no, no_lookforward
            )
            
            st.session_state.naked_opens_analyzed = True
            st.session_state.naked_opens_results = results
            st.session_state.naked_opens_chart_index = -1
    
    if 'naked_opens_analyzed' in st.session_state and st.session_state.naked_opens_analyzed and 'naked_opens_results' in st.session_state:
        results = st.session_state.naked_opens_results
        
        if len(results['times']) == 0:
            st.warning("No naked opens found matching the criteria. Try adjusting the filters.")
        else:
            st.markdown("---")
            st.subheader("Summary Statistics")
            
            # Calculate statistics
            bars_to_hit = np.array([b for b in results['bars_to_hit'] if b is not None])
            
            # Direction-specific statistics
            bullish_indices = [i for i, d in enumerate(results['directions']) if d == 'Bullish']
            bearish_indices = [i for i, d in enumerate(results['directions']) if d == 'Bearish']
            
            bullish_hit = sum([1 for i in bullish_indices if results['hit'][i]])
            bearish_hit = sum([1 for i in bearish_indices if results['hit'][i]])
            
            bullish_hit_rate = (bullish_hit / len(bullish_indices) * 100) if len(bullish_indices) > 0 else 0
            bearish_hit_rate = (bearish_hit / len(bearish_indices) * 100) if len(bearish_indices) > 0 else 0
            
            composite_hit = sum(results['hit'])
            composite_hit_rate = (composite_hit / len(results['times']) * 100) if len(results['times']) > 0 else 0
            
            median_bars = np.median(bars_to_hit) if len(bars_to_hit) > 0 else 0
            
            # Winsorized average
            if len(bars_to_hit) > 0:
                p5 = np.percentile(bars_to_hit, 5)
                p95 = np.percentile(bars_to_hit, 95)
                bars_winsor = np.clip(bars_to_hit, p5, p95)
                winsor_avg = np.mean(bars_winsor)
            else:
                winsor_avg = 0
            
            unfilled_count = sum(1 for h in results['hit'] if not h)
            unfilled_pct = (unfilled_count / len(results['times'])) * 100
            
            # Display statistics in cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Naked Opens</div>
                    <div class="metric-value">{len(results['times'])}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Bullish Naked Opens</div>
                    <div class="metric-value">{len(bullish_indices)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Bullish Hit Rate</div>
                    <div class="metric-value">{bullish_hit_rate:.1f}%</div>
                    <div class="metric-subtitle">{bullish_hit}/{len(bullish_indices)}</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Bearish Hit Rate</div>
                    <div class="metric-value">{bearish_hit_rate:.1f}%</div>
                    <div class="metric-subtitle">{bearish_hit}/{len(bearish_indices)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Composite Hit Rate</div>
                    <div class="metric-value">{composite_hit_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Unfilled %</div>
                    <div class="metric-value">{unfilled_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Median Bars to Hit</div>
                    <div class="metric-value">{median_bars:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Winsorized Avg</div>
                    <div class="metric-value">{winsor_avg:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Bearish Naked Opens</div>
                    <div class="metric-value">{len(bearish_indices)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Cumulative Probability of Open Hit")
            
            # Calculate cumulative probabilities
            max_bars = max(bars_to_hit) if len(bars_to_hit) > 0 else 0
            
            if max_bars > 0:
                bars_range = range(1, min(int(max_bars) + 1, 101))
                
                probabilities = []
                
                for bar_count in bars_range:
                    filled = sum(1 for b in results['bars_to_hit'] if b is not None and b <= bar_count)
                    prob = (filled / len(results['times'])) * 100
                    probabilities.append(prob)
                
                fig_prob = go.Figure()
                
                fig_prob.add_trace(go.Scatter(
                    x=list(bars_range),
                    y=probabilities,
                    mode='lines+markers',
                    name='Cumulative Probability',
                    line=dict(color='#3498DB', width=3),
                    marker=dict(size=6)
                ))
                
                fig_prob.update_layout(
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(title="Number of Bars", showgrid=False, color=text_color),
                    yaxis=dict(title="Probability (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Price Chart - Individual Naked Opens")
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                bars_before_no = st.number_input(
                    "Bars Before",
                    min_value=1,
                    max_value=200,
                    value=20,
                    step=5,
                    key="no_bars_before"
                )
            
            with col2:
                bars_after_no = st.number_input(
                    "Bars After",
                    min_value=1,
                    max_value=200,
                    value=30,
                    step=5,
                    key="no_bars_after"
                )
            
            with col3:
                st.write("")
                st.write("")
                if st.button("◀ Previous", use_container_width=True, key="prev_no", type="secondary"):
                    if st.session_state.naked_opens_chart_index > 0:
                        st.session_state.naked_opens_chart_index -= 1
                    else:
                        st.session_state.naked_opens_chart_index = len(results['times']) - 1
                    st.rerun()
            
            with col4:
                st.write("")
                st.write("")
                if st.button("Next ▶", use_container_width=True, key="next_no", type="secondary"):
                    if st.session_state.naked_opens_chart_index < len(results['times']) - 1:
                        st.session_state.naked_opens_chart_index += 1
                    else:
                        st.session_state.naked_opens_chart_index = 0
                    st.rerun()
            
            # Get current naked open
            current_no_idx = st.session_state.naked_opens_chart_index
            if current_no_idx == -1 or current_no_idx >= len(results['times']):
                current_no_idx = len(results['times']) - 1
            
            # Ensure index is within bounds (in case of stale session state)
            current_no_idx = max(0, min(current_no_idx, len(results['times']) - 1))
            
            no_time = results['times'][current_no_idx]
            no_direction = results['directions'][current_no_idx]
            no_open = results['open_prices'][current_no_idx]
            no_hit = results['hit'][current_no_idx]
            bars_to_hit_no = results['bars_to_hit'][current_no_idx]
            
            # Build caption
            hit_status = "Yes" if no_hit else "No"
            bars_text = f"{bars_to_hit_no} bars" if bars_to_hit_no is not None else "N/A"
            
            st.caption(f"Naked Open {current_no_idx + 1} of {len(results['times'])} | {no_time.strftime('%Y-%m-%d %H:%M')} | {no_direction} | Hit: {hit_status} | Bars: {bars_text}")
            
            # Resample based on timeframe for chart
            if timeframe_no == '15m':
                chart_df = df_15m.copy()
            elif timeframe_no == '30m':
                chart_df = df_15m.resample('30min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe_no == '1h':
                chart_df = df_15m.resample('1H').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe_no == '4H':
                chart_df = df_15m.resample('4H').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe_no == 'Daily':
                chart_df = df_15m.resample('1D').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe_no == 'Weekly':
                chart_df = df_15m.resample('W-SUN').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif timeframe_no == 'Monthly':
                chart_df = df_15m.resample('M').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            
            # Find no_time in chart_df
            if no_time in chart_df.index or any(abs((chart_df.index - no_time).total_seconds()) < 3600):
                # Find nearest candle
                no_idx = chart_df.index.get_indexer([no_time], method='nearest')[0]
                
                start_idx = max(0, no_idx - bars_before_no)
                end_idx = min(len(chart_df) - 1, no_idx + bars_after_no)
                
                chart_data = chart_df.iloc[start_idx:end_idx+1]
                
                fig_no_chart = go.Figure()
                
                # Add candles
                for idx, row in chart_data.iterrows():
                    is_up = row['Close'] > row['Open']
                    color = candle_up if is_up else candle_down
                    
                    fig_no_chart.add_trace(go.Bar(
                        x=[idx],
                        y=[abs(row['Close'] - row['Open'])],
                        base=[min(row['Open'], row['Close'])],
                        marker_color=color,
                        marker_line_width=0,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig_no_chart.add_trace(go.Scatter(
                        x=[idx, idx],
                        y=[row['Low'], row['High']],
                        mode='lines',
                        line=dict(color=color, width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                add_ohlc_hover_trace(fig_no_chart, chart_data)
                
                # Add open level line
                fig_no_chart.add_trace(go.Scatter(
                    x=[chart_data.index[0], chart_data.index[-1]],
                    y=[no_open, no_open],
                    mode='lines',
                    name='Open Level',
                    line=dict(color=highlight_color, width=2, dash='solid')
                ))
                
                fig_no_chart.update_layout(
                    title=f"{no_direction} Naked Open - {no_time.strftime('%Y-%m-%d %H:%M')} ({timeframe_no})",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_no_chart, use_container_width=True)
            else:
                st.warning("Selected naked open time not found in chart data")
            
            st.markdown("---")
            st.subheader("MAE Distribution")
            st.caption("Maximum Adverse Excursion - how far price moved against the position")
            
            mae_values = [results['mae_pct'][i] for i in range(len(results['times']))]
            
            fig_mae = go.Figure()
            
            fig_mae.add_trace(go.Histogram(
                x=mae_values,
                name='MAE',
                marker_color=positive_color,
                opacity=0.7,
                nbinsx=30
            ))
            
            fig_mae.update_layout(
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(title="MAE (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                yaxis=dict(title="Frequency", showgrid=True, gridcolor=grid_color, color=text_color),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_mae, use_container_width=True)
            
            st.markdown("---")


elif page == "Gap Fills":
    render_page_header(
        "Gap Fills Analysis",
        "Analyze custom gap fills with time-based gap generation"
    )
    render_exchange_asset_controls("gf")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Generate time options (15m intervals)
        time_options = []
        for h in range(24):
            for m in [0, 15, 30, 45]:
                time_options.append(f"{h:02d}:{m:02d}")
        
        gap_end_time = st.selectbox(
            "End of Gap (Close)",
            options=time_options,
            index=time_options.index("20:00"),
            key="gf_gap_end"
        )
    
    with col2:
        gap_start_time = st.selectbox(
            "Start of Gap (Open)",
            options=time_options,
            index=time_options.index("06:00"),
            key="gf_gap_start"
        )
    
    with col3:
        min_gap_size_gf = st.slider(
            "Minimum Gap Size (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            key="gf_min_gap"
        )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        partial_threshold_gf = st.slider(
            "Partial Fill Threshold (%)",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            key="gf_partial_threshold"
        )
    
    with col5:
        gf_lookforward = st.number_input(
            "Max Lookforward Bars",
            min_value=10,
            max_value=5000,
            value=1000,
            step=50,
            key="gf_lookforward"
        )
    
    with col6:
        day_filter_gf = st.multiselect(
            "Day of Week Filter",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            key="gf_day_filter"
        )
    
    col9, col10 = st.columns(2)
    
    with col9:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()
        
        start_date_gf = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=365),
            min_value=min_date,
            max_value=max_date,
            key="gf_start"
        )
    
    with col10:
        end_date_gf = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="gf_end"
        )
    
    analyze_gf = st.button(
        " Analyze Gap Fills",
        use_container_width=True,
        type="primary",
        key="analyze_gf"
    )
    if analyze_gf:
        with st.spinner("Analyzing gap fills..."):
            results = analyze_gap_fills(
                df_15m, start_date_gf, end_date_gf, gap_end_time, gap_start_time,
                min_gap_size_gf, partial_threshold_gf, day_filter_gf, gf_lookforward
            )
            
            st.session_state.gap_fills_analyzed = True
            st.session_state.gap_fills_results = results
            st.session_state.gap_fills_chart_index = -1
    
    if 'gap_fills_analyzed' in st.session_state and st.session_state.gap_fills_analyzed and 'gap_fills_results' in st.session_state:
        results = st.session_state.gap_fills_results
        
        if len(results['times']) == 0:
            st.warning("No gaps found matching the criteria. Try adjusting the filters.")
        else:
            st.markdown("---")
            st.subheader("Summary Statistics")
            
            # Calculate statistics
            gap_sizes = np.array(results['gap_sizes'])
            bars_partial = np.array([b for b in results['bars_partial'] if b is not None])
            bars_full = np.array([b for b in results['bars_full'] if b is not None])
            
            # Gap direction statistics
            gap_up_indices = [i for i, d in enumerate(results['gap_directions']) if d == 'Gap Up']
            gap_down_indices = [i for i, d in enumerate(results['gap_directions']) if d == 'Gap Down']
            
            gap_up_filled = sum([1 for i in gap_up_indices if results['fill_status'][i] in ['Partial', 'Full']])
            gap_down_filled = sum([1 for i in gap_down_indices if results['fill_status'][i] in ['Partial', 'Full']])
            
            gap_up_hit_rate = (gap_up_filled / len(gap_up_indices) * 100) if len(gap_up_indices) > 0 else 0
            gap_down_hit_rate = (gap_down_filled / len(gap_down_indices) * 100) if len(gap_down_indices) > 0 else 0
            
            composite_filled = sum([1 for status in results['fill_status'] if status in ['Partial', 'Full']])
            composite_hit_rate = (composite_filled / len(results['times']) * 100) if len(results['times']) > 0 else 0
            
            avg_gap_size = np.mean(gap_sizes)
            median_partial = np.median(bars_partial) if len(bars_partial) > 0 else 0
            median_full = np.median(bars_full) if len(bars_full) > 0 else 0
            
            # Winsorized averages
            if len(bars_partial) > 0:
                p5_partial = np.percentile(bars_partial, 5)
                p95_partial = np.percentile(bars_partial, 95)
                bars_partial_winsor = np.clip(bars_partial, p5_partial, p95_partial)
                winsor_avg_partial = np.mean(bars_partial_winsor)
            else:
                winsor_avg_partial = 0
            
            if len(bars_full) > 0:
                p5_full = np.percentile(bars_full, 5)
                p95_full = np.percentile(bars_full, 95)
                bars_full_winsor = np.clip(bars_full, p5_full, p95_full)
                winsor_avg_full = np.mean(bars_full_winsor)
            else:
                winsor_avg_full = 0
            
            unfilled_count = sum(1 for status in results['fill_status'] if status == 'Unfilled')
            unfilled_pct = (unfilled_count / len(results['times'])) * 100
            
            # Display statistics in cards (3x3 grid)
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Total Gaps</div>
                    <div class="metric-value">{len(results['times'])}</div>
                </div>
                """, unsafe_allow_html=True)
            with r1c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Average Gap Size</div>
                    <div class="metric-value">{avg_gap_size:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with r1c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Composite Hit Rate</div>
                    <div class="metric-value">{composite_hit_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Gap Up Hit Rate</div>
                    <div class="metric-value">{gap_up_hit_rate:.1f}%</div>
                    <div class="metric-subtitle">{len(gap_up_indices)} gaps</div>
                </div>
                """, unsafe_allow_html=True)
            with r2c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Gap Down Hit Rate</div>
                    <div class="metric-value">{gap_down_hit_rate:.1f}%</div>
                    <div class="metric-subtitle">{len(gap_down_indices)} gaps</div>
                </div>
                """, unsafe_allow_html=True)
            with r2c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Median Partial Bars</div>
                    <div class="metric-value">{median_partial:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            r3c1, r3c2, r3c3 = st.columns(3)
            with r3c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Winsorised Avg Partial</div>
                    <div class="metric-value">{winsor_avg_partial:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with r3c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Median Full Bars</div>
                    <div class="metric-value">{median_full:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            with r3c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Winsorised Avg Full</div>
                    <div class="metric-value">{winsor_avg_full:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # ML PREDICTIONS SECTION
            if ML_AVAILABLE:
                st.markdown("---")
                st.subheader("ML Gap Fill Predictor")
                
                # Train ML model
                ml_predictor = GapFillMLPredictor()
                training_success = ml_predictor.train(results)
                
                if training_success:
                    # Show example prediction for next gap
                    st.write("**Example: Predict Next Gap Fill**")
                    
                    col_ml1, col_ml2, col_ml3 = st.columns(3)
                    
                    with col_ml1:
                        test_gap_size = st.number_input(
                            "Gap Size %",
                            min_value=0.1,
                            max_value=5.0,
                            value=0.8,
                            step=0.1,
                            key="ml_gap_size"
                        )
                    
                    with col_ml2:
                        test_direction = st.selectbox(
                            "Direction",
                            options=['Gap Up', 'Gap Down'],
                            key="ml_direction"
                        )
                    
                    with col_ml3:
                        test_time = st.selectbox(
                            "Gap Start Time",
                            options=[f"{h:02d}:00" for h in range(24)],
                            index=6,  # 06:00
                            key="ml_time"
                        )
                    
                    # Predict
                    test_hour = int(test_time.split(':')[0])
                    test_day = 2  # Wednesday as example
                    
                    ml_prob, ml_conf = ml_predictor.predict(
                        test_gap_size, test_direction, test_hour, test_day, False
                    )
                    
                    if ml_prob is not None:
                        # Display prediction
                        st.markdown("---")
                        col_pred1, col_pred2, col_pred3 = st.columns(3)
                        
                        with col_pred1:
                            st.metric("Historical Fill Rate", f"{composite_hit_rate:.1f}%")
                        
                        with col_pred2:
                            st.metric("ML Predicted Probability", f"{ml_prob*100:.1f}%")
                        
                        with col_pred3:
                            st.metric("Confidence", f"{ml_conf}")
                        
                    
                else:
                    st.info("ℹ️ Need at least 20 gaps to train ML model. Keep collecting data!")
            
            st.markdown("---")
            st.subheader("Cumulative Probability of Gap Fill")
            
            # Calculate cumulative probabilities
            max_bars = max(max([b for b in results['bars_partial'] if b is not None], default=0),
                          max([b for b in results['bars_full'] if b is not None], default=0))
            
            if max_bars > 0:
                bars_range = range(1, min(max_bars + 1, 101))
                
                partial_probs = []
                full_probs = []
                
                for bar_count in bars_range:
                    # Partial fill probability
                    partial_filled = sum(1 for b in results['bars_partial'] if b is not None and b <= bar_count)
                    partial_prob = (partial_filled / len(results['times'])) * 100
                    partial_probs.append(partial_prob)
                    
                    # Full fill probability
                    full_filled = sum(1 for b in results['bars_full'] if b is not None and b <= bar_count)
                    full_prob = (full_filled / len(results['times'])) * 100
                    full_probs.append(full_prob)
                
                fig_prob = go.Figure()
                
                fig_prob.add_trace(go.Scatter(
                    x=list(bars_range),
                    y=partial_probs,
                    mode='lines+markers',
                    name=f'Partial Fill (≥{partial_threshold_gf}%)',
                    line=dict(color='#FF8C00', width=3),
                    marker=dict(size=6)
                ))
                
                fig_prob.add_trace(go.Scatter(
                    x=list(bars_range),
                    y=full_probs,
                    mode='lines+markers',
                    name='Full Fill (100%)',
                    line=dict(color='#3498DB', width=3),
                    marker=dict(size=6)
                ))
                
                fig_prob.update_layout(
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(title="Number of Bars", showgrid=False, color=text_color),
                    yaxis=dict(title="Probability (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_prob, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Price Chart - Individual Gap Navigation")
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                bars_before_gf = st.number_input(
                    "Bars Before",
                    min_value=1,
                    max_value=200,
                    value=20,
                    step=5,
                    key="gf_bars_before"
                )
            
            with col2:
                bars_after_gf = st.number_input(
                    "Bars After",
                    min_value=1,
                    max_value=200,
                    value=30,
                    step=5,
                    key="gf_bars_after"
                )
            
            with col3:
                st.write("")
                st.write("")
                if st.button("◀ Previous", use_container_width=True, key="prev_gap", type="secondary"):
                    if st.session_state.gap_fills_chart_index > 0:
                        st.session_state.gap_fills_chart_index -= 1
                    else:
                        st.session_state.gap_fills_chart_index = len(results['times']) - 1
                    st.rerun()
            
            with col4:
                st.write("")
                st.write("")
                if st.button("Next ▶", use_container_width=True, key="next_gap", type="secondary"):
                    if st.session_state.gap_fills_chart_index < len(results['times']) - 1:
                        st.session_state.gap_fills_chart_index += 1
                    else:
                        st.session_state.gap_fills_chart_index = 0
                    st.rerun()
            
            # Get current gap
            current_gap_idx = st.session_state.gap_fills_chart_index
            if current_gap_idx == -1 or current_gap_idx >= len(results['times']):
                current_gap_idx = len(results['times']) - 1
            
            # Ensure index is within bounds (in case of stale session state)
            current_gap_idx = max(0, min(current_gap_idx, len(results['times']) - 1))
            
            gap_time = results['times'][current_gap_idx]
            gap_direction = results['gap_directions'][current_gap_idx]
            gap_size_pct = results['gap_sizes'][current_gap_idx]
            gap_close_price = results['gap_close_prices'][current_gap_idx]
            gap_open_price = results['gap_open_prices'][current_gap_idx]
            bars_partial_gf = results['bars_partial'][current_gap_idx]
            bars_full_gf = results['bars_full'][current_gap_idx]
            fill_status_gf = results['fill_status'][current_gap_idx]
            
            # Build caption
            caption_parts = [
                f"Gap {current_gap_idx + 1} of {len(results['times'])}",
                f"{gap_time.strftime('%Y-%m-%d %H:%M')}",
                f"{gap_direction}: {gap_size_pct:.2f}%",
                f"Status: {fill_status_gf}"
            ]
            
            if bars_partial_gf:
                caption_parts.append(f"Partial: {bars_partial_gf} bars")
            if bars_full_gf:
                caption_parts.append(f"Full: {bars_full_gf} bars")
            
            st.caption(" | ".join(caption_parts))
            
            # Resample to 30m
            df_30m = df_15m.resample('30min').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
            }).dropna()
            
            # Calculate gap close and gap open timestamps for this specific gap
            gap_date = gap_time.date()
            
            # Parse gap end time (gap close)
            gap_end_hour = int(gap_end_time.split(':')[0])
            gap_end_minute = int(gap_end_time.split(':')[1])
            gap_close_timestamp = pd.Timestamp(gap_date) + pd.Timedelta(hours=gap_end_hour, minutes=gap_end_minute)
            
            # Parse gap start time (gap open) - adjust for next day if needed
            gap_start_hour = int(gap_start_time.split(':')[0])
            gap_start_minute = int(gap_start_time.split(':')[1])
            
            # If gap start time is before gap end time, it means next day
            if (gap_start_hour < gap_end_hour) or (gap_start_hour == gap_end_hour and gap_start_minute < gap_end_minute):
                gap_open_timestamp = pd.Timestamp(gap_date) + pd.Timedelta(days=1, hours=gap_start_hour, minutes=gap_start_minute)
            else:
                gap_open_timestamp = pd.Timestamp(gap_date) + pd.Timedelta(hours=gap_start_hour, minutes=gap_start_minute)
            
            # Find gap close and gap open in 30m data
            if gap_close_timestamp in df_30m.index or any(abs((df_30m.index - gap_close_timestamp).total_seconds()) < 1800):
                # Find nearest 30m candles for gap close and gap open
                gap_close_idx = df_30m.index.get_indexer([gap_close_timestamp], method='nearest')[0]
                gap_open_idx = df_30m.index.get_indexer([gap_open_timestamp], method='nearest')[0]
                
                # Get candles BEFORE gap close (excluding gap period)
                before_start_idx = max(0, gap_close_idx - bars_before_gf)
                before_end_idx = gap_close_idx + 1  # Include gap close candle
                chart_data_before = df_30m.iloc[before_start_idx:before_end_idx]
                
                # Get candles AFTER gap open (excluding gap period)
                after_start_idx = gap_open_idx
                after_end_idx = min(len(df_30m), gap_open_idx + bars_after_gf + 1)
                chart_data_after = df_30m.iloc[after_start_idx:after_end_idx]
                
                fig_gap_chart = go.Figure()
                
                # Add candles BEFORE gap
                for idx, row in chart_data_before.iterrows():
                    is_up = row['Close'] > row['Open']
                    color = candle_up if is_up else candle_down
                    
                    fig_gap_chart.add_trace(go.Bar(
                        x=[idx],
                        y=[abs(row['Close'] - row['Open'])],
                        base=[min(row['Open'], row['Close'])],
                        marker_color=color,
                        marker_line_width=0,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig_gap_chart.add_trace(go.Scatter(
                        x=[idx, idx],
                        y=[row['Low'], row['High']],
                        mode='lines',
                        line=dict(color=color, width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Add candles AFTER gap
                for idx, row in chart_data_after.iterrows():
                    is_up = row['Close'] > row['Open']
                    color = candle_up if is_up else candle_down
                    
                    fig_gap_chart.add_trace(go.Bar(
                        x=[idx],
                        y=[abs(row['Close'] - row['Open'])],
                        base=[min(row['Open'], row['Close'])],
                        marker_color=color,
                        marker_line_width=0,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig_gap_chart.add_trace(go.Scatter(
                        x=[idx, idx],
                        y=[row['Low'], row['High']],
                        mode='lines',
                        line=dict(color=color, width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Add gap close level (across entire chart)
                all_timestamps = list(chart_data_before.index) + list(chart_data_after.index)
                fig_gap_chart.add_trace(go.Scatter(
                    x=[all_timestamps[0], all_timestamps[-1]],
                    y=[gap_close_price, gap_close_price],
                    mode='lines',
                    name='Gap Close Level',
                    line=dict(color=orange_line, width=2, dash='solid')
                ))
                
                # Add gap open level (across entire chart)
                fig_gap_chart.add_trace(go.Scatter(
                    x=[all_timestamps[0], all_timestamps[-1]],
                    y=[gap_open_price, gap_open_price],
                    mode='lines',
                    name='Gap Open Level',
                    line=dict(color=blue_color, width=2, dash='dash')
                ))
                
                chart_data_gap = pd.concat([chart_data_before, chart_data_after]).sort_index()
                add_ohlc_hover_trace(fig_gap_chart, chart_data_gap)
                
                fig_gap_chart.update_layout(
                    title=f"{gap_direction} - {gap_time.strftime('%Y-%m-%d %H:%M')} (30m with Visual Gap)",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_gap_chart, use_container_width=True)
            else:
                st.warning("Selected gap time not found in 30m chart data")
            
            st.markdown("---")
            st.subheader("MAE Distribution")
            st.caption("Maximum Adverse Excursion - how far price moved against the position")
            
            mae_values = [results['mae_pct'][i] for i in range(len(results['times']))]
            
            fig_mae = go.Figure()
            
            fig_mae.add_trace(go.Histogram(
                x=mae_values,
                name='MAE',
                marker_color=positive_color,
                opacity=0.7,
                nbinsx=30
            ))
            
            fig_mae.update_layout(
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(title="MAE (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                yaxis=dict(title="Frequency", showgrid=True, gridcolor=grid_color, color=text_color),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_mae, use_container_width=True)


elif page == "Weekend Breaks":
    render_page_header(
        "Weekend Breaks",
        "Probability of breaking weekend range and rotating across the range"
    )
    render_exchange_asset_controls("wb")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()
        start_date_wb = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=365),
            min_value=min_date,
            max_value=max_date,
            key="wb_start"
        )
    
    with col2:
        end_date_wb = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="wb_end"
        )
    
    with col3:
        max_lookforward_days = st.number_input(
            "Max Look Forward Days",
            min_value=1,
            max_value=7,
            value=3,
            step=1,
            key="wb_lookforward_days"
        )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        wb_timeframe = st.selectbox(
            "Chart Timeframe",
            options=['15m', '30m', '1h', '4h', '1D'],
            index=1,
            key="wb_timeframe"
        )
    
    with col5:
        bars_before_wb = st.number_input(
            "Bars Before (Chart)",
            min_value=1,
            max_value=200,
            value=60,
            step=1,
            key="wb_bars_before"
        )
    
    with col6:
        bars_after_wb = st.number_input(
            "Bars After (Chart)",
            min_value=1,
            max_value=200,
            value=60,
            step=1,
            key="wb_bars_after"
        )
    
    analyze_wb = st.button(
        "Analyze Weekend Breaks",
        use_container_width=True,
        type="primary",
        key="analyze_wb"
    )
    if analyze_wb:
        with st.spinner("Analyzing weekend breaks..."):
            weekends = []
            df_range = df_15m[(df_15m.index.date >= start_date_wb) & (df_15m.index.date <= end_date_wb)]
            
            for date in pd.date_range(start_date_wb, end_date_wb, freq='D'):
                if date.dayofweek != 5:
                    continue
                weekend_start = pd.Timestamp(date)  # Saturday 00:00
                weekend_end = weekend_start + timedelta(days=2) - timedelta(minutes=1)
                
                weekend_data = df_range[(df_range.index >= weekend_start) & (df_range.index <= weekend_end)]
                if weekend_data.empty:
                    continue
                
                weekend_high = weekend_data['High'].max()
                weekend_low = weekend_data['Low'].min()
                weekend_end_time = weekend_data.index[-1]
                
                forward_end = weekend_end_time + timedelta(days=max_lookforward_days)
                forward_data = df_15m[(df_15m.index > weekend_end_time) & (df_15m.index <= forward_end)]
                
                break_high = False
                break_low = False
                break_high_time = None
                break_low_time = None
                
                if not forward_data.empty:
                    high_break_idx = forward_data.index[(forward_data['High'] >= weekend_high)]
                    low_break_idx = forward_data.index[(forward_data['Low'] <= weekend_low)]
                    
                    if len(high_break_idx) > 0:
                        break_high = True
                        break_high_time = high_break_idx[0]
                    if len(low_break_idx) > 0:
                        break_low = True
                        break_low_time = low_break_idx[0]
                
                # Rotation logic
                rotate_to_high = False
                rotate_to_low = False
                mfe_pct = None
                
                if break_low and break_low_time is not None:
                    after_break = forward_data[forward_data.index >= break_low_time]
                    if not after_break.empty:
                        rotate_to_high = (after_break['High'] >= weekend_high).any()
                        if rotate_to_high:
                            sweep_time = after_break.index[after_break['High'] >= weekend_high][0]
                            interim = after_break[after_break.index <= sweep_time]
                        else:
                            interim = after_break
                        min_low = interim['Low'].min() if not interim.empty else None
                        if min_low is not None and weekend_low != 0:
                            mfe_pct = ((weekend_low - min_low) / weekend_low) * 100
                
                if break_high and break_high_time is not None:
                    after_break = forward_data[forward_data.index >= break_high_time]
                    if not after_break.empty:
                        rotate_to_low = (after_break['Low'] <= weekend_low).any()
                        if rotate_to_low:
                            sweep_time = after_break.index[after_break['Low'] <= weekend_low][0]
                            interim = after_break[after_break.index <= sweep_time]
                        else:
                            interim = after_break
                        max_high = interim['High'].max() if not interim.empty else None
                        if max_high is not None and weekend_high != 0:
                            mfe_pct = ((max_high - weekend_high) / weekend_high) * 100
                
                weekends.append({
                    'weekend_start': weekend_start,
                    'weekend_end': weekend_end_time,
                    'high': weekend_high,
                    'low': weekend_low,
                    'break_high': break_high,
                    'break_low': break_low,
                    'rotate_to_high': rotate_to_high,
                    'rotate_to_low': rotate_to_low,
                    'mfe_pct': mfe_pct
                })
            
            st.session_state.weekend_breaks_results = weekends
            st.session_state.weekend_breaks_chart_index = -1
    
    if 'weekend_breaks_results' in st.session_state and st.session_state.weekend_breaks_results:
        weekends = st.session_state.weekend_breaks_results
        total_weekends = len(weekends)
        
        if total_weekends == 0:
            st.warning("No weekend ranges found for the selected date range.")
        else:
            break_high_count = sum(1 for w in weekends if w['break_high'])
            break_low_count = sum(1 for w in weekends if w['break_low'])
            
            break_high_pct = (break_high_count / total_weekends) * 100
            break_low_pct = (break_low_count / total_weekends) * 100
            any_break_pct = (sum(1 for w in weekends if w['break_high'] or w['break_low']) / total_weekends) * 100
            
            low_rotations = [w for w in weekends if w['break_low']]
            high_rotations = [w for w in weekends if w['break_high']]
            
            rotate_to_high_pct = (sum(1 for w in low_rotations if w['rotate_to_high']) / len(low_rotations) * 100) if low_rotations else 0
            rotate_to_low_pct = (sum(1 for w in high_rotations if w['rotate_to_low']) / len(high_rotations) * 100) if high_rotations else 0
            
            mfe_values = [w['mfe_pct'] for w in weekends if w['mfe_pct'] is not None]
            avg_mfe = np.mean(mfe_values) if mfe_values else 0
            
            st.subheader("Summary Statistics")
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Break High %</div>
                    <div class="metric-value">{break_high_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Break Low %</div>
                    <div class="metric-value">{break_low_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Any Break %</div>
                    <div class="metric-value">{any_break_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            col_s4, col_s5, col_s6 = st.columns(3)
            with col_s4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Rotate to High % (after Low Break)</div>
                    <div class="metric-value">{rotate_to_high_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Rotate to Low % (after High Break)</div>
                    <div class="metric-value">{rotate_to_low_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg MFE %</div>
                    <div class="metric-value">{avg_mfe:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Chart navigation
            col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 1])
            
            with col_nav1:
                if st.button("◀ Previous", key="wb_prev", type="secondary"):
                    if st.session_state.weekend_breaks_chart_index > 0:
                        st.session_state.weekend_breaks_chart_index -= 1
                    else:
                        st.session_state.weekend_breaks_chart_index = len(weekends) - 1
                    st.rerun()
            
            with col_nav2:
                st.write("")
            
            with col_nav3:
                if st.button("Next ▶", key="wb_next"):
                    if st.session_state.weekend_breaks_chart_index < len(weekends) - 1:
                        st.session_state.weekend_breaks_chart_index += 1
                    else:
                        st.session_state.weekend_breaks_chart_index = 0
                    st.rerun()
            
            current_idx = st.session_state.weekend_breaks_chart_index
            if current_idx == -1 or current_idx >= len(weekends):
                current_idx = len(weekends) - 1
            
            current = weekends[current_idx]
            weekend_end_time = current['weekend_end']
            
            if wb_timeframe == '15m':
                df_chart = df_15m.copy()
            elif wb_timeframe == '30m':
                df_chart = df_15m.resample('30min').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()
            elif wb_timeframe == '1h':
                df_chart = df_15m.resample('1H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()
            elif wb_timeframe == '4h':
                df_chart = df_15m.resample('4H').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()
            else:
                df_chart = df_15m.resample('1D').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()
            
            if weekend_end_time in df_chart.index or len(df_chart) > 0:
                idx = df_chart.index.get_indexer([weekend_end_time], method='nearest')[0]
                start_idx = max(0, idx - bars_before_wb)
                end_idx = min(len(df_chart) - 1, idx + bars_after_wb)
                chart_data = df_chart.iloc[start_idx:end_idx+1]
                
                fig_wb = go.Figure()
                fig_wb.add_trace(go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name='Price',
                    increasing_line_color=candle_up,
                    decreasing_line_color=candle_down,
                    increasing_fillcolor=candle_up,
                    decreasing_fillcolor=candle_down,
                    hovertemplate=CANDLE_HOVER_TEMPLATE
                ))
                
                x0 = current['weekend_start']
                x1 = chart_data.index[-1]
                
                fig_wb.add_shape(
                    type="line",
                    x0=x0,
                    x1=x1,
                    y0=current['high'],
                    y1=current['high'],
                    line=dict(color=highlight_color, width=2, dash='solid')
                )
                
                fig_wb.add_shape(
                    type="line",
                    x0=x0,
                    x1=x1,
                    y0=current['low'],
                    y1=current['low'],
                    line=dict(color=highlight_color, width=2, dash='solid')
                )
                
                fig_wb.update_layout(
                    title=f"Weekend Range - {current['weekend_start'].strftime('%Y-%m-%d')}",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_wb, use_container_width=True)
            

elif page == "Volume Heatmap":
    render_page_header(
        "Volume Heatmap",
        "Average volume by hour and day of week"
    )
    render_exchange_asset_controls("vh")
    min_date = df_15m.index.min().date()
    max_date = df_15m.index.max().date()
    
    if 'vol_min_date' not in st.session_state:
        st.session_state.vol_min_date = min_date
    if 'vol_max_date' not in st.session_state:
        st.session_state.vol_max_date = max_date
    if 'vol_start_date' not in st.session_state:
        st.session_state.vol_start_date = max_date - timedelta(days=365)
    if 'vol_end_date' not in st.session_state:
        st.session_state.vol_end_date = max_date
    if 'vol_total_days' not in st.session_state:
        st.session_state.vol_total_days = max(1, (st.session_state.vol_end_date - st.session_state.vol_start_date).days)
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date_vol = st.date_input(
            "Start Date",
            min_value=min_date,
            max_value=max_date,
            key="vol_start_date",
            on_change=sync_vol_days_from_dates
        )
    
    with col2:
        end_date_vol = st.date_input(
            "End Date",
            min_value=min_date,
            max_value=max_date,
            key="vol_end_date",
            on_change=sync_vol_days_from_dates
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        total_days_vol = st.slider(
            "Total Days",
            min_value=1,
            max_value=1825,
            value=max(1, st.session_state.vol_total_days),
            step=1,
            key="vol_total_days",
            on_change=sync_vol_end_from_days
        )
    
    with col4:
        total_days_input = st.number_input(
            "Total Days",
            min_value=1,
            max_value=1825,
            value=int(st.session_state.vol_total_days),
            step=1,
            key="vol_total_days_input",
            on_change=sync_vol_days_from_input
        )
    
    df_vol = df_15m[(df_15m.index.date >= start_date_vol) & (df_15m.index.date <= end_date_vol)]
    
    if df_vol.empty:
        st.warning("No data available for the selected date range.")
    else:
        df_hourly = df_vol.resample('1H').agg({'Volume': 'sum'}).dropna()
        df_hourly['day_name'] = df_hourly.index.day_name()
        df_hourly['hour'] = df_hourly.index.hour
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap = df_hourly.pivot_table(
            index='day_name',
            columns='hour',
            values='Volume',
            aggfunc='mean'
        ).reindex(day_order)
        
        heatmap = heatmap.reindex(columns=list(range(24)))
        heatmap = heatmap.fillna(0)
        
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap.values,
                x=heatmap.columns,
                y=heatmap.index,
                colorscale='YlGn',
                colorbar=dict(title="Average Volume"),
                hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Avg Volume: %{z:.2f}<extra></extra>'
            )
        )
        
        fig_heatmap.update_layout(
            title="Average Volume by Hour and Day",
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            plot_bgcolor=plot_bg,
            paper_bgcolor=plot_bg,
            font=dict(color=text_color),
            height=520
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)

elif page == "Round Numbers Front Run":
    render_page_header(
        "Round Numbers Front Run",
        "Track front runs into round levels and probability of taps"
    )
    render_exchange_asset_controls("rnr")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rnr_timeframe = st.selectbox(
            "Timeframe",
            options=['15m', '30m', '1h', '4h', '1D'],
            index=1,
            key="rnr_timeframe"
        )
    
    with col2:
        round_interval = st.number_input(
            "Round Number Interval",
            min_value=10,
            max_value=100000,
            value=1000,
            step=10,
            key="rnr_interval"
        )
    
    with col3:
        lookback_hours = st.number_input(
            "Mitigation Lookback (hours)",
            min_value=1,
            max_value=168,
            value=6,
            step=1,
            key="rnr_lookback_hours"
        )
    
    with col4:
        max_lookforward = st.number_input(
            "Max Look Forward Bars",
            min_value=1,
            max_value=500,
            value=48,
            step=1,
            key="rnr_lookforward"
        )
    
    col4b, col5b = st.columns(2)
    
    with col4b:
        front_run_threshold = st.number_input(
            "Front Run Threshold (±)",
            min_value=0.0,
            max_value=10000.0,
            value=1.0,
            step=0.5,
            key="rnr_front_run_threshold"
        )
    
    with col5b:
        day_filter_rnr = st.multiselect(
            "Day of Week Filter",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            key="rnr_day_filter"
        )
    
    col6, col7 = st.columns(2)
    
    with col6:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()
        
        start_date_rnr = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=365),
            min_value=min_date,
            max_value=max_date,
            key="rnr_start"
        )
    
    with col7:
        end_date_rnr = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="rnr_end"
        )
    
    analyze_rnr = st.button(
        "Analyze Round Numbers Front Runs",
        use_container_width=True,
        type="primary",
        key="analyze_rnr"
    )
    if analyze_rnr:
        with st.spinner("Analyzing round number front runs..."):
            events, df_tf_rnr = analyze_round_number_front_runs(
                df_15m,
                start_date_rnr,
                end_date_rnr,
                day_filter_rnr,
                round_interval,
                lookback_hours,
                max_lookforward,
                rnr_timeframe,
                front_run_threshold
            )
            
            st.session_state.round_numbers_analyzed = True
            st.session_state.round_numbers_results = {
                'events': events,
                'df_tf': df_tf_rnr,
                'max_lookforward': max_lookforward
            }
            st.session_state.round_numbers_chart_index = -1
    
    if st.session_state.round_numbers_analyzed and st.session_state.round_numbers_results:
        events = st.session_state.round_numbers_results['events']
        df_tf_rnr = st.session_state.round_numbers_results['df_tf']
        max_lookforward = st.session_state.round_numbers_results['max_lookforward']
        
        total_events = len(events)
        hit_events = [e for e in events if e['hit']]
        hitrate = (len(hit_events) / total_events * 100) if total_events > 0 else 0
        
        mfe_values = [e['mfe_pct'] for e in events] if total_events > 0 else []
        mfe_avg = np.mean(mfe_values) if mfe_values else 0
        
        bars_to_hit = [e['bars_to_hit'] for e in hit_events if e['bars_to_hit'] is not None]
        median_bars = np.median(bars_to_hit) if bars_to_hit else None
        winsorized_bars = winsorized_mean(bars_to_hit) if bars_to_hit else None
        median_bars_display = f"{median_bars:.1f}" if median_bars is not None else "N/A"
        winsorized_bars_display = f"{winsorized_bars:.1f}" if winsorized_bars is not None else "N/A"
        
        st.subheader("Summary Statistics")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Average MFE %</div>
                <div class="metric-value">{mfe_avg:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Hit Rate %</div>
                <div class="metric-value">{hitrate:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Events</div>
                <div class="metric-value">{total_events}</div>
            </div>
            """, unsafe_allow_html=True)
        
        col_s4, col_s5, col_s6 = st.columns(3)
        with col_s4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Median Bars to Tap</div>
                <div class="metric-value">{median_bars_display}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Winsorised Avg Bars</div>
                <div class="metric-value">{winsorized_bars_display}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_s6:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Hits</div>
                <div class="metric-value">{len(hit_events)}</div>
            </div>
            """, unsafe_allow_html=True)

        # Cumulative probability chart
        if total_events > 0:
            probabilities = []
            for k in range(1, max_lookforward + 1):
                hits_k = sum(1 for e in events if e['hit'] and e['bars_to_hit'] <= k)
                probabilities.append((hits_k / total_events) * 100)
            
            fig_prob = go.Figure()
            fig_prob.add_trace(go.Scatter(
                x=list(range(1, max_lookforward + 1)),
                y=probabilities,
                mode='lines',
                line=dict(color=highlight_color, width=3),
                name="Cumulative Hit Probability"
            ))
            
            fig_prob.update_layout(
                title="Cumulative Probability of Tap by Bars Forward",
                xaxis_title="Bars Forward",
                yaxis_title="Probability (%)",
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                height=420
            )
            
            st.plotly_chart(fig_prob, use_container_width=True)
        
        # Candlestick chart with navigation
        if total_events > 0:
            col_b1, col_b2, col_b3, col_b4 = st.columns([1, 1, 1, 1])
            with col_b1:
                bars_before_rnr = st.number_input(
                    "Bars Before (Chart)",
                    min_value=1,
                    max_value=200,
                    value=30,
                    step=1,
                    key="rnr_bars_before"
                )
            with col_b2:
                bars_after_rnr = st.number_input(
                    "Bars After (Chart)",
                    min_value=1,
                    max_value=200,
                    value=30,
                    step=1,
                    key="rnr_bars_after"
                )

            with col_b3:
                st.write("")
                st.write("")
                if st.button("◀ Previous", key="rnr_prev", type="secondary", use_container_width=True):
                    if st.session_state.round_numbers_chart_index > 0:
                        st.session_state.round_numbers_chart_index -= 1
                    else:
                        st.session_state.round_numbers_chart_index = len(events) - 1
                    st.rerun()

            with col_b4:
                st.write("")
                st.write("")
                if st.button("Next ▶", key="rnr_next", type="secondary", use_container_width=True):
                    if st.session_state.round_numbers_chart_index < len(events) - 1:
                        st.session_state.round_numbers_chart_index += 1
                    else:
                        st.session_state.round_numbers_chart_index = 0
                    st.rerun()
            
            current_idx = st.session_state.round_numbers_chart_index
            if current_idx == -1 or current_idx >= len(events):
                current_idx = len(events) - 1
            
            current_event = events[current_idx]
            event_time = current_event['time']
            level = current_event['level']
            
            st.caption(
                f"Event {current_idx + 1} of {len(events)} | {event_time.strftime('%Y-%m-%d %H:%M')} | "
                f"Level: {level:.0f} | Hit: {'Yes' if current_event['hit'] else 'No'}"
            )
            
            if event_time in df_tf_rnr.index:
                event_idx = df_tf_rnr.index.get_loc(event_time)
                start_idx = max(0, event_idx - bars_before_rnr)
                end_idx = min(len(df_tf_rnr) - 1, event_idx + bars_after_rnr)
                chart_data = df_tf_rnr.iloc[start_idx:end_idx+1]
                
                fig_rnr = go.Figure()
                fig_rnr.add_trace(go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name='Price',
                    increasing_line_color=candle_up,
                    decreasing_line_color=candle_down,
                    increasing_fillcolor=candle_up,
                    decreasing_fillcolor=candle_down,
                    hovertemplate=CANDLE_HOVER_TEMPLATE
                ))
                
                fig_rnr.add_hline(
                    y=level,
                    line_color=highlight_color,
                    line_width=2,
                    line_dash="solid",
                    annotation_text=f"Level {level:.0f}",
                    annotation_position="right"
                )
                
                fig_rnr.add_vline(
                    x=event_time,
                    line_dash="dot",
                    line_color=text_color,
                    line_width=1
                )
                
                fig_rnr.update_layout(
                    title="Round Number Front Run",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_rnr, use_container_width=True)
        
        st.markdown("---")
elif page == "Quartile Opens":
    render_page_header(
        "Quartile Opens Analysis",
        "Test whether X-1 high/low quartile opens are swept"
    )
    render_exchange_asset_controls("qo")
    col1, col2 = st.columns(2)

    with col1:
        timeframe_qo = st.selectbox(
            "Timeframe (Analysis)",
            options=['15m', '30m', '1h', 'Session', 'Daily', 'Weekly', 'Monthly'],
            index=4,  # Default Daily
            key="qo_timeframe"
        )

    with col2:
        lower_timeframe_qo = st.selectbox(
            "Lower Timeframe (Chart)",
            options=['15m', '30m', '1h', '4h', '1D', '1W', '1M'],
            index=1,  # Default 30m
            key="qo_lower_timeframe"
        )

    day_filter_qo = st.multiselect(
        "Day of Week Filter",
        options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        key="qo_day_filter"
    )

    # Session filter (only for Session timeframe)
    if timeframe_qo == 'Session':
        col4, col5 = st.columns(2)
        
        with col4:
            session_filter_qo = st.multiselect(
                "Sessions (select one or more)",
                options=['Asia', 'London', 'New York', 'Close'],
                default=['Asia', 'London', 'New York', 'Close'],
                key="qo_session"
            )
        
        with col5:
            pass  # Spacer
    else:
        session_filter_qo = None
    
    col6, col7 = st.columns(2)
    
    with col6:
        min_date = df_15m.index.min().date()
        max_date = df_15m.index.max().date()
        
        start_date_qo = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=365),
            min_value=min_date,
            max_value=max_date,
            key="qo_start"
        )
    
    with col7:
        end_date_qo = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="qo_end"
        )
    
    analyze_qo = st.button(
        " Analyze Quartile Opens",
        use_container_width=True,
        type="primary",
        key="analyze_qo"
    )
    if analyze_qo:
        # Validate session selection for Session timeframe
        if timeframe_qo == 'Session' and (not session_filter_qo or len(session_filter_qo) == 0):
            st.error("Please select at least one session when using Session timeframe.")
        else:
            with st.spinner("Analyzing quartile opens..."):
                results = analyze_quartile_opens(
                    df_15m, start_date_qo, end_date_qo, timeframe_qo,
                    day_filter_qo, session_filter_qo
                )
                
                st.session_state.quartile_opens_analyzed = True
                st.session_state.quartile_opens_results = results
                st.session_state.quartile_opens_chart_index = -1  # Reset chart index
    
    if 'quartile_opens_analyzed' in st.session_state and st.session_state.quartile_opens_analyzed and 'quartile_opens_results' in st.session_state:
        results = st.session_state.quartile_opens_results
        
        if len(results['times']) == 0:
            st.warning("No quartile opens found matching the criteria. Try adjusting the filters.")
        else:
            results_all_days = analyze_quartile_opens(
                df_15m, start_date_qo, end_date_qo, timeframe_qo,
                None, session_filter_qo
            )

            # Calculate statistics
            upper_opens = [i for i, qt in enumerate(results['quartile_types']) if qt == 'Upper']
            lower_opens = [i for i, qt in enumerate(results['quartile_types']) if qt == 'Lower']
            
            upper_swept = sum([results['swept'][i] for i in upper_opens])
            lower_swept = sum([results['swept'][i] for i in lower_opens])
            
            upper_total = len(upper_opens)
            lower_total = len(lower_opens)
            
            upper_hit_rate = (upper_swept / upper_total * 100) if upper_total > 0 else 0
            lower_hit_rate = (lower_swept / lower_total * 100) if lower_total > 0 else 0
            combined_hit_rate = ((upper_swept + lower_swept) / (upper_total + lower_total) * 100) if (upper_total + lower_total) > 0 else 0

            upper_opens_all = [i for i, qt in enumerate(results_all_days['quartile_types']) if qt == 'Upper']
            lower_opens_all = [i for i, qt in enumerate(results_all_days['quartile_types']) if qt == 'Lower']
            upper_swept_all = sum([results_all_days['swept'][i] for i in upper_opens_all])
            lower_swept_all = sum([results_all_days['swept'][i] for i in lower_opens_all])
            upper_total_all = len(upper_opens_all)
            lower_total_all = len(lower_opens_all)
            upper_hit_rate_all = (upper_swept_all / upper_total_all * 100) if upper_total_all > 0 else 0
            lower_hit_rate_all = (lower_swept_all / lower_total_all * 100) if lower_total_all > 0 else 0
            combined_hit_rate_all = ((upper_swept_all + lower_swept_all) / (upper_total_all + lower_total_all) * 100) if (upper_total_all + lower_total_all) > 0 else 0

            bars_to_sweep = [b for b in results['bars_to_sweep'] if b is not None]
            median_bars = np.median(bars_to_sweep) if bars_to_sweep else None
            winsorized_bars = winsorized_mean(bars_to_sweep) if bars_to_sweep else None
            avg_mfe = np.mean([v for v in results['mae_pct'] if v is not None]) if results['mae_pct'] else 0
            median_bars_display = f"{median_bars:.1f}" if median_bars is not None else "N/A"
            winsorized_bars_display = f"{winsorized_bars:.1f}" if winsorized_bars is not None else "N/A"
            
            # Heatmap (only for Daily timeframe)
            if timeframe_qo == 'Daily':
                st.markdown("---")
                st.subheader("Hit Rate Heatmap by Day of Week")
                
                # Calculate hit rates by day and quartile type
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_data = {'Day': day_order, 'Upper': [], 'Lower': []}
                
                for day in day_order:
                    # Upper quartile for this day
                    day_upper_indices = [i for i in upper_opens if results['day_names'][i] == day]
                    day_upper_swept = sum([results['swept'][i] for i in day_upper_indices])
                    day_upper_rate = (day_upper_swept / len(day_upper_indices) * 100) if len(day_upper_indices) > 0 else 0
                    heatmap_data['Upper'].append(day_upper_rate)
                    
                    # Lower quartile for this day
                    day_lower_indices = [i for i in lower_opens if results['day_names'][i] == day]
                    day_lower_swept = sum([results['swept'][i] for i in day_lower_indices])
                    day_lower_rate = (day_lower_swept / len(day_lower_indices) * 100) if len(day_lower_indices) > 0 else 0
                    heatmap_data['Lower'].append(day_lower_rate)
                
                # Create heatmap
                heatmap_df = pd.DataFrame(heatmap_data).set_index('Day')
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=heatmap_df.T.values,
                    x=heatmap_df.index,
                    y=heatmap_df.columns,
                    colorscale='RdYlGn',
                    text=[[f"{val:.1f}%" for val in row] for row in heatmap_df.T.values],
                    texttemplate='%{text}',
                    textfont={"size": 14},
                    colorbar=dict(title="Hit Rate %")
                ))
                
                fig_heatmap.update_layout(
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(title="Day of Week", color=text_color),
                    yaxis=dict(title="Quartile Type", color=text_color),
                    height=300
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Summary Statistics")

            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Week Upper Hit Rate</div>
                    <div class="metric-value">{upper_hit_rate_all:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Week Lower Hit Rate</div>
                    <div class="metric-value">{lower_hit_rate_all:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Week Combined Hit Rate</div>
                    <div class="metric-value">{combined_hit_rate_all:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            col_s4, col_s5, col_s6 = st.columns(3)
            with col_s4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Filtered Upper Hit Rate</div>
                    <div class="metric-value">{upper_hit_rate:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Filtered Lower Hit Rate</div>
                    <div class="metric-value">{lower_hit_rate:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Filtered Combined Hit Rate</div>
                    <div class="metric-value">{combined_hit_rate:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            col_s7, col_s8, col_s9 = st.columns(3)
            with col_s7:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Median Bars to Fill</div>
                    <div class="metric-value">{median_bars_display}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s8:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Winsorised Avg Bars</div>
                    <div class="metric-value">{winsorized_bars_display}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s9:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Avg MFE %</div>
                    <div class="metric-value">{avg_mfe:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Price Chart - Individual Quartile Opens")
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                bars_before_qo = st.number_input(
                    "Bars Before",
                    min_value=1,
                    max_value=200,
                    value=20,
                    step=5,
                    key="qo_bars_before"
                )
            
            with col2:
                bars_after_qo = st.number_input(
                    "Bars After",
                    min_value=1,
                    max_value=200,
                    value=30,
                    step=5,
                    key="qo_bars_after"
                )
            
            with col3:
                if st.button("◀ Previous Pattern", use_container_width=True, key="prev_qo", type="secondary"):
                    if st.session_state.quartile_opens_chart_index > 0:
                        st.session_state.quartile_opens_chart_index -= 1
                    else:
                        st.session_state.quartile_opens_chart_index = len(results['times']) - 1
                    st.rerun()
            
            with col4:
                if st.button("Next Pattern ▶", use_container_width=True, key="next_qo", type="secondary"):
                    if st.session_state.quartile_opens_chart_index < len(results['times']) - 1:
                        st.session_state.quartile_opens_chart_index += 1
                    else:
                        st.session_state.quartile_opens_chart_index = 0
                    st.rerun()
            
            # Get current quartile open
            current_qo_idx = st.session_state.quartile_opens_chart_index
            if current_qo_idx == -1 or current_qo_idx >= len(results['times']):
                current_qo_idx = len(results['times']) - 1
            
            # Ensure index is within bounds (in case of stale session state)
            current_qo_idx = max(0, min(current_qo_idx, len(results['times']) - 1))
            
            qo_time = results['times'][current_qo_idx]
            qo_type = results['quartile_types'][current_qo_idx]
            swept = results['swept'][current_qo_idx]
            bars_to_sweep = results['bars_to_sweep'][current_qo_idx]
            x_minus_1_high = results['x_minus_1_high'][current_qo_idx]
            x_minus_1_low = results['x_minus_1_low'][current_qo_idx]
            
            # Build caption
            sweep_status = "Yes" if swept else "No"
            bars_text = f"{bars_to_sweep} bars" if bars_to_sweep is not None else "N/A"
            
            st.caption(f"Quartile Open {current_qo_idx + 1} of {len(results['times'])} | {qo_time.strftime('%Y-%m-%d %H:%M')} | {qo_type} Quartile | Swept: {sweep_status} | Bars: {bars_text}")
            
            # Build chart data based on lower_timeframe_qo
            if lower_timeframe_qo == '15m':
                chart_df = df_15m.copy()
            elif lower_timeframe_qo == '30m':
                chart_df = df_15m.resample('30min').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif lower_timeframe_qo == '1h':
                chart_df = df_15m.resample('1H').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif lower_timeframe_qo == '4h':
                chart_df = df_15m.resample('4H').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif lower_timeframe_qo == '1D':
                chart_df = df_15m.resample('1D').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif lower_timeframe_qo == '1W':
                chart_df = df_15m.resample('W-SUN').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            elif lower_timeframe_qo == '1M':
                chart_df = df_15m.resample('M').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
                }).dropna()
            
            # Find qo_time in chart_df
            if qo_time in chart_df.index:
                qo_idx = chart_df.index.get_loc(qo_time)
                
                start_idx = max(0, qo_idx - bars_before_qo)
                end_idx = min(len(chart_df), qo_idx + bars_after_qo + 1)
                
                chart_data = chart_df.iloc[start_idx:end_idx]
                
                fig_chart_qo = go.Figure()
                
                # Add candles
                for idx, row in chart_data.iterrows():
                    is_qo_candle = (idx == qo_time)
                    is_up = row['Close'] > row['Open']
                    
                    if is_qo_candle:
                        color = highlight_color  # Gold for X candle
                    else:
                        color = candle_up if is_up else candle_down
                    
                    # Body
                    fig_chart_qo.add_trace(go.Bar(
                        x=[idx],
                        y=[abs(row['Close'] - row['Open'])],
                        base=[min(row['Open'], row['Close'])],
                        marker_color=color,
                        marker_line_width=0,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # Wicks
                    fig_chart_qo.add_trace(go.Scatter(
                        x=[idx, idx],
                        y=[row['Low'], row['High']],
                        mode='lines',
                        line=dict(color=color, width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                add_ohlc_hover_trace(fig_chart_qo, chart_data)
                
                # Add horizontal ray from X-1 high or low
                target_level = x_minus_1_high if qo_type == 'Upper' else x_minus_1_low
                
                fig_chart_qo.add_trace(go.Scatter(
                    x=[chart_data.index[0], chart_data.index[-1]],
                    y=[target_level, target_level],
                    mode='lines',
                    name=f'X-1 {qo_type} Target',
                    line=dict(color=orange_line, width=2, dash='solid'),
                    hoverinfo='skip'
                ))
                
                # Add sweep marker if swept
                if swept and results['sweep_times'][current_qo_idx] in chart_data.index:
                    sweep_time = results['sweep_times'][current_qo_idx]
                    fig_chart_qo.add_trace(go.Scatter(
                        x=[sweep_time],
                        y=[target_level],
                        mode='markers+text',
                        name='Sweep',
                        marker=dict(size=12, color=blue_color, symbol='star'),
                        text=['S'],
                        textposition='top center',
                        textfont=dict(size=10, color=text_color)
                    ))
                
                fig_chart_qo.update_layout(
                    title=f"{qo_type} Quartile Open - {qo_time.strftime('%Y-%m-%d %H:%M')} ({lower_timeframe_qo})",
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_chart_qo, use_container_width=True)
            else:
                st.warning("Selected quartile open time not found in chart data")
            
            st.markdown("---")
            st.subheader("Hit Rate Evolution")
            rolling_window_qo = st.number_input(
                    "Rolling Window",
                    min_value=1,
                    max_value=100,
                    value=20,
                    step=5,
                    key="qo_rolling_window"
                )
            
            # Composite hit rate (both upper and lower)
            st.write("**Composite Hit Rate (Upper + Lower)**")
            
            all_swept = [1 if swept else 0 for swept in results['swept']]
            df_evolution_composite = pd.DataFrame({
                'date': results['times'],
                'swept': all_swept
            })
            
            df_evolution_composite['cumulative'] = (df_evolution_composite['swept'].expanding().sum() / df_evolution_composite['swept'].expanding().count()) * 100
            df_evolution_composite['rolling'] = df_evolution_composite['swept'].rolling(window=min(rolling_window_qo, len(df_evolution_composite)), min_periods=1).mean() * 100
            
            fig_composite = go.Figure()
            
            fig_composite.add_trace(go.Scatter(
                x=df_evolution_composite['date'],
                y=df_evolution_composite['cumulative'],
                mode='lines',
                name='Cumulative',
                line=dict(color=blue_color, width=2)
            ))
            
            fig_composite.add_trace(go.Scatter(
                x=df_evolution_composite['date'],
                y=df_evolution_composite['rolling'],
                mode='lines',
                name=f'Rolling ({rolling_window_qo})',
                line=dict(color=yellow_color, width=2, dash='dash')
            ))
            
            fig_composite.update_layout(
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(showgrid=False, color=text_color),
                yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=350
            )
            
            st.plotly_chart(fig_composite, use_container_width=True)
            
            # Lower quartile hit rate
            st.write("**Lower Quartile Hit Rate**")
            
            lower_swept_list = [1 if results['swept'][i] else 0 for i in lower_opens]
            df_evolution_lower = pd.DataFrame({
                'date': [results['times'][i] for i in lower_opens],
                'swept': lower_swept_list
            })
            
            if len(df_evolution_lower) > 0:
                df_evolution_lower['cumulative'] = (df_evolution_lower['swept'].expanding().sum() / df_evolution_lower['swept'].expanding().count()) * 100
                df_evolution_lower['rolling'] = df_evolution_lower['swept'].rolling(window=min(rolling_window_qo, len(df_evolution_lower)), min_periods=1).mean() * 100
                
                fig_lower = go.Figure()
                
                fig_lower.add_trace(go.Scatter(
                    x=df_evolution_lower['date'],
                    y=df_evolution_lower['cumulative'],
                    mode='lines',
                    name='Cumulative Lower',
                    line=dict(color=dark_blue, width=2)
                ))
                
                fig_lower.add_trace(go.Scatter(
                    x=df_evolution_lower['date'],
                    y=df_evolution_lower['rolling'],
                    mode='lines',
                    name=f'Rolling Lower ({rolling_window_qo})',
                    line=dict(color=orange_line, width=2, dash='dash')
                ))
                
                fig_lower.update_layout(
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=350
                )
                
                st.plotly_chart(fig_lower, use_container_width=True)
            else:
                st.info("No lower quartile opens found in selected data.")
            
            # Upper quartile hit rate
            st.write("**Upper Quartile Hit Rate**")
            
            upper_swept_list = [1 if results['swept'][i] else 0 for i in upper_opens]
            df_evolution_upper = pd.DataFrame({
                'date': [results['times'][i] for i in upper_opens],
                'swept': upper_swept_list
            })
            
            if len(df_evolution_upper) > 0:
                df_evolution_upper['cumulative'] = (df_evolution_upper['swept'].expanding().sum() / df_evolution_upper['swept'].expanding().count()) * 100
                df_evolution_upper['rolling'] = df_evolution_upper['swept'].rolling(window=min(rolling_window_qo, len(df_evolution_upper)), min_periods=1).mean() * 100
                
                fig_upper = go.Figure()
                
                fig_upper.add_trace(go.Scatter(
                    x=df_evolution_upper['date'],
                    y=df_evolution_upper['cumulative'],
                    mode='lines',
                    name='Cumulative Upper',
                    line=dict(color=light_blue, width=2)
                ))
                
                fig_upper.add_trace(go.Scatter(
                    x=df_evolution_upper['date'],
                    y=df_evolution_upper['rolling'],
                    mode='lines',
                    name=f'Rolling Upper ({rolling_window_qo})',
                    line=dict(color='#FFA500', width=2, dash='dash')
                ))
                
                fig_upper.update_layout(
                    plot_bgcolor=plot_bg,
                    paper_bgcolor=plot_bg,
                    font=dict(color=text_color),
                    xaxis=dict(showgrid=False, color=text_color),
                    yaxis=dict(title="Hit Rate (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=350
                )
                
                st.plotly_chart(fig_upper, use_container_width=True)
            else:
                st.info("No upper quartile opens found in selected data.")
            
            st.markdown("---")
            st.subheader("MAE Distribution")
            st.caption("Maximum Adverse Excursion - how far price moved against the position before sweeping")
            
            # Get MAE percentage values for upper and lower
            upper_mae = [results['mae_pct'][i] for i in upper_opens]
            lower_mae = [results['mae_pct'][i] for i in lower_opens]
            
            fig_mae = go.Figure()
            
            if len(upper_mae) > 0:
                fig_mae.add_trace(go.Histogram(
                    x=upper_mae,
                    name='Upper Quartile MAE',
                    marker_color=light_blue,
                    opacity=0.7,
                    nbinsx=30
                ))
            
            if len(lower_mae) > 0:
                fig_mae.add_trace(go.Histogram(
                    x=lower_mae,
                    name='Lower Quartile MAE',
                    marker_color=orange_line,
                    opacity=0.7,
                    nbinsx=30
                ))
            
            fig_mae.update_layout(
                barmode='overlay',
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(title="MAE (%)", showgrid=True, gridcolor=grid_color, color=text_color),
                yaxis=dict(title="Frequency", showgrid=True, gridcolor=grid_color, color=text_color),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            
            st.plotly_chart(fig_mae, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Hourly Sweep Distribution")
            st.caption("Which hour of day sweeps typically complete")
            
            # Get sweep hours (only for swept trades)
            upper_sweep_hours = [results['sweep_times'][i].hour for i in upper_opens if results['swept'][i] and results['sweep_times'][i] is not None]
            lower_sweep_hours = [results['sweep_times'][i].hour for i in lower_opens if results['swept'][i] and results['sweep_times'][i] is not None]
            
            # Count by hour
            hours = list(range(24))
            upper_counts = [upper_sweep_hours.count(h) for h in hours]
            lower_counts = [lower_sweep_hours.count(h) for h in hours]
            
            fig_hourly = go.Figure()
            
            fig_hourly.add_trace(go.Bar(
                x=hours,
                y=upper_counts,
                name='Upper Quartile',
                marker_color=light_blue
            ))
            
            fig_hourly.add_trace(go.Bar(
                x=hours,
                y=lower_counts,
                name='Lower Quartile',
                marker_color=orange_line
            ))
            
            fig_hourly.update_layout(
                barmode='group',
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                font=dict(color=text_color),
                xaxis=dict(title="Hour of Day (UTC)", showgrid=False, color=text_color, tickmode='linear'),
                yaxis=dict(title="Number of Sweeps", showgrid=True, gridcolor=grid_color, color=text_color),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=400
            )
            
            st.plotly_chart(fig_hourly, use_container_width=True)
