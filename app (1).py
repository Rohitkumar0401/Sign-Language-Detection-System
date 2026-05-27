"""
app.py
======
ASL Sign Language Detection - Streamlit Frontend
- Seedha OpenCV se camera capture
- No WebRTC, No localhost issues
- Base64 encoding se live feed
- predictor.py se fully integrated
"""

import cv2
import time
import base64
import threading
import numpy as np
import streamlit as st
from predictor import ASLPredictor

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASL Sign Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

html, body, [class*="css"] {
    background-color: #050d1a !important;
    color: #c9d8e8 !important;
    font-family: 'Rajdhani', sans-serif;
}

/* ── Header ── */
.header-wrap {
    text-align: center;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid #0d2440;
    margin-bottom: 1.5rem;
}
.header-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00ffaa, #00b4d8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 3px;
}
.header-sub {
    color: #3a6080;
    font-size: 0.95rem;
    letter-spacing: 2px;
    margin-top: 0.3rem;
    font-family: 'Rajdhani', sans-serif;
}

/* ── Camera Box ── */
.cam-wrapper {
    background: #0a1628;
    border: 1px solid #0d2440;
    border-radius: 16px;
    padding: 12px;
    position: relative;
}
.cam-wrapper img {
    border-radius: 10px;
    width: 100%;
}
.cam-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: #00ffaa;
    margin-bottom: 8px;
}
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #00ffaa;
    margin-right: 6px;
    animation: blink 1.2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ── Prediction Card ── */
.pred-card {
    background: #0a1628;
    border: 1px solid #0d2440;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.pred-letter {
    font-family: 'Orbitron', monospace;
    font-size: 6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00ffaa, #00b4d8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    filter: drop-shadow(0 0 20px rgba(0,255,170,0.4));
}
.pred-empty {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    color: #1a3050;
    line-height: 1;
}
.pred-label {
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: #3a6080;
    margin-top: 0.5rem;
    font-family: 'Orbitron', monospace;
}

/* ── Confidence ── */
.conf-card {
    background: #0a1628;
    border: 1px solid #0d2440;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.conf-title {
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: #3a6080;
    font-family: 'Orbitron', monospace;
    margin-bottom: 0.5rem;
}
.conf-num {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    color: #00ffaa;
    margin-bottom: 6px;
}
.conf-track {
    height: 8px;
    background: #0d2440;
    border-radius: 99px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #00ffaa, #00b4d8);
    transition: width 0.3s ease;
}

/* ── History ── */
.hist-card {
    background: #0a1628;
    border: 1px solid #0d2440;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.hist-title {
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: #3a6080;
    font-family: 'Orbitron', monospace;
    margin-bottom: 0.6rem;
}
.hist-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.hist-chip {
    background: #0d2440;
    color: #00b4d8;
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    padding: 4px 12px;
    border-radius: 6px;
    border: 1px solid #1a3a5c;
}

/* ── Stats ── */
.stats-card {
    background: #0a1628;
    border: 1px solid #0d2440;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.stats-title {
    font-size: 0.65rem;
    letter-spacing: 2px;
    color: #3a6080;
    font-family: 'Orbitron', monospace;
    margin-bottom: 0.5rem;
}
.stats-row {
    display: flex;
    gap: 1.5rem;
}
.stat-item { text-align: center; }
.stat-val {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00ffaa;
}
.stat-lbl {
    font-size: 0.7rem;
    color: #3a6080;
    letter-spacing: 1px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #080f1e !important;
    border-right: 1px solid #0d2440;
}
section[data-testid="stSidebar"] * { color: #c9d8e8 !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #00ffaa, #00b4d8) !important;
    color: #000 !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    letter-spacing: 1px !important;
    width: 100% !important;
}

/* Hide streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if "running"      not in st.session_state: st.session_state.running      = False
if "history"      not in st.session_state: st.session_state.history      = []
if "frame_count"  not in st.session_state: st.session_state.frame_count  = 0
if "predictor"    not in st.session_state: st.session_state.predictor    = None
if "cur_letter"   not in st.session_state: st.session_state.cur_letter   = ""
if "cur_conf"     not in st.session_state: st.session_state.cur_conf     = 0.0
if "tts_enabled"  not in st.session_state: st.session_state.tts_enabled  = True

# ── Load Predictor ────────────────────────────────────────────────────────────
@st.cache_resource
def get_predictor():
    return ASLPredictor()

# ── Helper: frame to base64 ───────────────────────────────────────────────────
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <div class="header-title">🤟 ASL DETECTOR</div>
    <div class="header-sub">REAL-TIME AMERICAN SIGN LANGUAGE RECOGNITION</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    start_btn = st.button("▶ START CAMERA")
    stop_btn  = st.button("⏹ STOP CAMERA")

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    st.divider()
    st.markdown("### 🔊 Voice Output")
    st.session_state.tts_enabled = st.toggle("Speak Letters", value=True)

    st.divider()
    st.markdown("### 📖 ASL Reference")
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Asl_alphabet_gallaudet.svg/800px-Asl_alphabet_gallaudet.svg.png",
        caption="ASL Alphabet Chart",
        use_column_width=True
    )

    st.divider()
    if st.button("🗑 Clear History"):
        st.session_state.history    = []
        st.session_state.frame_count = 0

# ── Main Layout ───────────────────────────────────────────────────────────────
col_cam, col_panel = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown("""
    <div class="cam-label">
        <span class="live-dot"></span>LIVE FEED
    </div>
    """, unsafe_allow_html=True)
    cam_placeholder = st.empty()

with col_panel:
    pred_ph  = st.empty()
    conf_ph  = st.empty()
    hist_ph  = st.empty()
    stats_ph = st.empty()

# ── Default UI (Camera Off) ───────────────────────────────────────────────────
def render_idle():
    cam_placeholder.markdown("""
    <div class="cam-wrapper" style="text-align:center;padding:4rem 2rem;">
        <div style="font-size:4rem">📷</div>
        <div style="font-family:'Orbitron',monospace;color:#1a3050;margin-top:1rem;font-size:0.8rem;letter-spacing:2px">
            CLICK START TO BEGIN
        </div>
    </div>
    """, unsafe_allow_html=True)

    pred_ph.markdown("""
    <div class="pred-card">
        <div class="pred-empty">?</div>
        <div class="pred-label">WAITING...</div>
    </div>""", unsafe_allow_html=True)

    conf_ph.markdown("""
    <div class="conf-card">
        <div class="conf-title">CONFIDENCE</div>
        <div class="conf-num">0%</div>
        <div class="conf-track"><div class="conf-fill" style="width:0%"></div></div>
    </div>""", unsafe_allow_html=True)

    hist_ph.markdown("""
    <div class="hist-card">
        <div class="hist-title">DETECTION HISTORY</div>
        <span style="color:#1a3050;font-size:0.9rem">— Waiting for camera —</span>
    </div>""", unsafe_allow_html=True)

    stats_ph.markdown("""
    <div class="stats-card">
        <div class="stats-title">SESSION STATS</div>
        <div class="stats-row">
            <div class="stat-item"><div class="stat-val">0</div><div class="stat-lbl">SIGNS</div></div>
            <div class="stat-item"><div class="stat-val">0</div><div class="stat-lbl">FRAMES</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── Live Camera Loop ──────────────────────────────────────────────────────────
def run_camera():
    predictor = get_predictor()

    # TTS toggle
    if not st.session_state.tts_enabled:
        predictor.speaker._engine = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        st.error("Camera nahi khul raha! Check karo camera connected hai ya nahi.")
        return

    prev_letter = ""

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror

        # ── Run Prediction ────────────────────────────────────────────────────
        annotated, letter, conf = predictor.process(frame)
        st.session_state.frame_count += 1
        st.session_state.cur_letter   = letter
        st.session_state.cur_conf     = conf

        # Update history
        if letter and letter != prev_letter:
            st.session_state.history.append(letter)
            if len(st.session_state.history) > 20:
                st.session_state.history = st.session_state.history[-20:]
        prev_letter = letter

        # ── Show Camera Frame ─────────────────────────────────────────────────
        img_b64 = frame_to_base64(annotated)
        cam_placeholder.markdown(f"""
        <div class="cam-wrapper">
            <img src="data:image/jpeg;base64,{img_b64}" />
        </div>
        """, unsafe_allow_html=True)

        # ── Prediction Badge ──────────────────────────────────────────────────
        if letter:
            pred_ph.markdown(f"""
            <div class="pred-card">
                <div class="pred-letter">{letter}</div>
                <div class="pred-label">DETECTED SIGN</div>
            </div>""", unsafe_allow_html=True)
        else:
            pred_ph.markdown("""
            <div class="pred-card">
                <div class="pred-empty">—</div>
                <div class="pred-label">NO HAND DETECTED</div>
            </div>""", unsafe_allow_html=True)

        # ── Confidence Bar ────────────────────────────────────────────────────
        pct = int(conf * 100)
        conf_ph.markdown(f"""
        <div class="conf-card">
            <div class="conf-title">CONFIDENCE</div>
            <div class="conf-num">{pct}%</div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{pct}%"></div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── History Strip ─────────────────────────────────────────────────────
        chips = "".join(
            f'<span class="hist-chip">{l}</span>'
            for l in st.session_state.history[-12:][::-1]
        ) or '<span style="color:#1a3050">— No detections yet —</span>'

        hist_ph.markdown(f"""
        <div class="hist-card">
            <div class="hist-title">DETECTION HISTORY</div>
            <div class="hist-chips">{chips}</div>
        </div>""", unsafe_allow_html=True)

        # ── Stats ─────────────────────────────────────────────────────────────
        stats_ph.markdown(f"""
        <div class="stats-card">
            <div class="stats-title">SESSION STATS</div>
            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-val">{len(st.session_state.history)}</div>
                    <div class="stat-lbl">SIGNS</div>
                </div>
                <div class="stat-item">
                    <div class="stat-val">{st.session_state.frame_count}</div>
                    <div class="stat-lbl">FRAMES</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        time.sleep(0.03)  # ~30 FPS

    cap.release()

# ── Run ───────────────────────────────────────────────────────────────────────
if st.session_state.running:
    run_camera()
else:
    render_idle()
