import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown

# ─────────────────────────────────────────
# Download model dari Google Drive
# ─────────────────────────────────────────
MODEL_PATH = 'best_model_phase2.keras'
GDRIVE_ID  = '12hVCIZ1Bi5-2vn8JsTRjMTtQAkmcJvJl'

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model... (first time only, ~25MB)'):
            url = f'https://drive.google.com/uc?id={GDRIVE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)

download_model()
import gdown

# ─────────────────────────────────────────
# Google Drive model config
# ─────────────────────────────────────────
MODEL_PATH = 'best_model_phase2.keras'
FILE_ID    = '12hVCIZ1Bi5-2vn8JsTRjMTtQAkmcJvJl'

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1f0f; color: #e8f0e9; font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #1a3d1c 0%, #0d1f0f 60%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
.hero { text-align: center; padding: 3rem 1rem 2rem; }
.hero-icon { font-size: 3.5rem; display: block; margin-bottom: 1rem; filter: drop-shadow(0 0 20px rgba(134,239,172,0.4)); }
.hero h1 { font-family: 'DM Serif Display', serif; font-size: clamp(2rem,5vw,3.2rem); color: #86efac; line-height: 1.1; margin-bottom: 0.75rem; letter-spacing: -0.02em; }
.hero p { color: #9db89e; font-size: 1.05rem; font-weight: 300; max-width: 480px; margin: 0 auto; line-height: 1.6; }
.upload-area { background: rgba(255,255,255,0.03); border: 1.5px dashed rgba(134,239,172,0.25); border-radius: 16px; padding: 2rem; margin: 2rem 0; }
.result-card { background: rgba(134,239,172,0.06); border: 1px solid rgba(134,239,172,0.2); border-radius: 16px; padding: 1.5rem 2rem; margin: 1.5rem 0; }
.result-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #86efac; margin-bottom: 0.4rem; }
.result-disease { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #e8f0e9; line-height: 1.2; margin-bottom: 0.25rem; }
.result-confidence { font-size: 0.95rem; color: #9db89e; font-weight: 300; }
.confidence-bar-bg { background: rgba(255,255,255,0.08); border-radius: 99px; height: 6px; margin-top: 1rem; overflow: hidden; }
.confidence-bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg,#4ade80,#86efac); }
.top3-title { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #9db89e; margin: 1.5rem 0 0.75rem; }
.top3-item { display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.9rem; color: #c8dcc9; }
.top3-item:last-child { border-bottom: none; }
.top3-pct { font-weight: 600; color: #86efac; font-size: 0.85rem; }
.healthy-badge { display: inline-block; background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); border-radius: 99px; padding: 0.2rem 0.8rem; font-size: 0.8rem; font-weight: 600; margin-top: 0.5rem; }
.disease-badge { display: inline-block; background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); border-radius: 99px; padding: 0.2rem 0.8rem; font-size: 0.8rem; font-weight: 600; margin-top: 0.5rem; }
.info-box { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 1rem 1.25rem; margin-top: 2rem; font-size: 0.85rem; color: #7a9b7c; line-height: 1.6; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stFileUploader"] label { color: #9db89e !important; }
.stButton > button { background: linear-gradient(135deg,#166534,#15803d) !important; color: #dcfce7 !important; border: none !important; border-radius: 10px !important; font-family: 'DM Sans',sans-serif !important; font-weight: 500 !important; padding: 0.6rem 1.5rem !important; width: 100% !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load model (download from Drive if needed)
# ─────────────────────────────────────────
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model... (first time only, ~25MB)'):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

def predict(img, model, class_names):
    img_resized = img.convert('RGB').resize((224, 224))
    arr = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    return preds, top3_idx

def format_name(name):
    return name.replace('_', ' ').replace('  ', ' ')

def is_healthy(name):
    return 'healthy' in name.lower()

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon">🌿</span>
    <h1>Plant Disease Classifier</h1>
    <p>Upload a leaf image to instantly detect plant diseases using deep learning.</p>
</div>
""", unsafe_allow_html=True)

with st.spinner('Loading model...'):
    try:
        model, class_names = load_resources()
    except Exception as e:
        st.error(f'Error loading model: {e}')
        st.stop()

st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a leaf image here or click to browse",
    type=['jpg', 'jpeg', 'png'],
    label_visibility='visible'
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded:
    img = Image.open(uploaded)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img, use_column_width=True, caption='Input image')

    with col2:
        with st.spinner('Analyzing...'):
            preds, top3_idx = predict(img, model, class_names)

        top1_name = class_names[top3_idx[0]]
        top1_conf = preds[top3_idx[0]] * 100
        healthy   = is_healthy(top1_name)
        badge     = '<span class="healthy-badge">✓ Healthy</span>' if healthy else '<span class="disease-badge">⚠ Disease Detected</span>'

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Diagnosis</div>
            <div class="result-disease">{format_name(top1_name)}</div>
            <div class="result-confidence">Confidence: {top1_conf:.1f}%</div>
            {badge}
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:{top1_conf:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        top3_html = '<div class="top3-title">Top 3 Predictions</div>'
        for idx in top3_idx:
            name = format_name(class_names[idx])
            conf = preds[idx] * 100
            top3_html += f"""
            <div class="top3-item">
                <span>{name}</span>
                <span class="top3-pct">{conf:.1f}%</span>
            </div>"""
        st.markdown(top3_html, unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    🌱 <strong style="color:#86efac">15 Plant Classes</strong> —
    Pepper (Bell), Potato, and Tomato diseases including bacterial spot, early/late blight,
    leaf mold, septoria, spider mites, target spot, yellow leaf curl virus, mosaic virus, and healthy plants.<br><br>
    📊 <strong style="color:#86efac">Model:</strong> MobileNetV2 fine-tuned on PlantVillage dataset ·
    <strong style="color:#86efac">Accuracy:</strong> 96%
</div>
""", unsafe_allow_html=True)
