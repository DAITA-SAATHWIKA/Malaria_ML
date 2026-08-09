"""
Malaria Cell Detection - Streamlit App
Run with: streamlit run app.py
"""

import datetime
import time

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="Malaria Cell Detection",
    page_icon="M",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root { --navy:#14324a; --blue:#1677a8; --teal:#168b88; --muted:#64748b; --line:#dbe7ee; }
        .stApp { background:#f7fafc; color:var(--navy); }
        html, body, [class*="css"] { font-family:'Inter', sans-serif; }
        #MainMenu, footer, header { visibility:hidden; }
        .block-container { max-width:1040px; padding:2.25rem 1.25rem 2rem; }
        .app-header { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:0 0 1.55rem; border-bottom:1px solid var(--line); }
        .brand { display:flex; align-items:center; gap:.8rem; }
        .brand-icon { width:42px; height:42px; display:grid; place-items:center; border-radius:12px; background:#e2f2f5; color:#087a81; font-size:1.2rem; font-weight:800; }
        .brand-title { color:var(--navy); font-size:1.35rem; line-height:1.2; font-weight:800; }
        .brand-subtitle { color:var(--muted); font-size:.78rem; margin-top:.22rem; }
        .status-badge { display:inline-flex; align-items:center; gap:.38rem; background:#edf8f4; color:#187356; border:1px solid #c9e9dc; border-radius:999px; padding:.42rem .72rem; font-size:.73rem; font-weight:700; white-space:nowrap; }
        .status-dot { width:7px; height:7px; background:#2c9a70; border-radius:50%; }
        .intro { text-align:center; padding:2rem .5rem 1.7rem; }
        .intro h1 { margin:0; color:var(--navy); font-size:1.75rem; letter-spacing:-.035em; }
        .intro p { max-width:640px; margin:.75rem auto 0; color:#50657a; font-size:.96rem; line-height:1.65; }
        .intro .powered { color:#6f8295; font-size:.78rem; margin-top:.45rem; }
        .upload-card, .preview-card, .result-card, .history-card { background:#fff; border:1px solid var(--line); border-radius:16px; box-shadow:0 8px 24px rgba(21,50,74,.055); }
        [data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-color:var(--line); border-radius:16px; box-shadow:0 8px 24px rgba(21,50,74,.055); }
        .upload-card { padding:1.45rem; text-align:center; }
        .section-title { color:var(--navy); font-size:1.08rem; font-weight:750; margin:0; }
        .section-copy { color:var(--muted); font-size:.83rem; margin:.42rem 0 1rem; }
        [data-testid="stFileUploader"] { text-align:left; }
        [data-testid="stFileUploaderDropzone"] { background:#f9fcfd; border:1.5px dashed #9fcbd5; border-radius:12px; padding:.45rem; }
        [data-testid="stFileUploaderDropzone"] > div { color:#597083; }
        [data-testid="stFileUploaderDropzone"] button { border-radius:8px; color:#126f91; border-color:#9bc9d6; background:#fff; }
        .formats { color:#7a8c9c; font-size:.72rem; margin-top:.55rem; }
        .preview-card { padding:1.2rem; margin-top:1.25rem; }
        .eyebrow { color:#638196; font-size:.68rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; margin-bottom:.65rem; }
        .image-meta { color:var(--muted); font-size:.78rem; line-height:1.7; padding-top:.6rem; }
        [data-testid="stImage"] img { border-radius:10px; border:1px solid #e4edf1; max-height:350px; object-fit:contain; }
        .stButton > button { width:100%; background:#147a9d !important; color:#fff !important; border:1px solid #147a9d !important; border-radius:10px !important; padding:.68rem 1.2rem !important; font-weight:700 !important; font-size:.95rem !important; transition:background .15s ease, border-color .15s ease; }
        .stButton > button:hover { background:#0f6684 !important; border-color:#0f6684 !important; }
        .stButton > button:disabled { background:#a8c4ce !important; border-color:#a8c4ce !important; color:#f8fbfc !important; }
        .result-card { padding:1.45rem; margin-top:1.35rem; }
        .result-card.infected { border-left:4px solid #bf5e5e; }
        .result-card.healthy { border-left:4px solid #38836a; }
        .result-grid { display:grid; grid-template-columns:1.35fr .8fr .8fr; gap:1rem; align-items:end; padding-top:.35rem; }
        .result-label { color:#708396; font-size:.7rem; font-weight:750; letter-spacing:.07em; text-transform:uppercase; }
        .prediction { margin-top:.35rem; font-size:1.65rem; font-weight:800; letter-spacing:-.03em; }
        .infected .prediction { color:#9d4141; } .healthy .prediction { color:#276e58; }
        .metric-value { color:var(--navy); margin-top:.35rem; font-size:1.12rem; font-weight:750; }
        .confidence-wrap { margin-top:1.25rem; }
        .confidence-head { display:flex; justify-content:space-between; color:#567086; font-size:.79rem; font-weight:650; margin-bottom:.48rem; }
        .confidence-track { height:8px; overflow:hidden; border-radius:999px; background:#e9f0f3; }
        .confidence-fill { height:100%; border-radius:999px; background:#2183a3; }
        .infected .confidence-fill { background:#b55e5e; } .healthy .confidence-fill { background:#38836a; }
        .model-section { margin-top:1.65rem; }
        .model-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.65rem; margin-top:.7rem; }
        .info-card { background:#fff; border:1px solid var(--line); border-radius:11px; padding:.85rem; }
        .info-label { color:#7a8c9c; font-size:.66rem; text-transform:uppercase; font-weight:750; letter-spacing:.06em; }
        .info-value { color:var(--navy); font-size:.83rem; font-weight:700; margin-top:.32rem; }
        .history-card { overflow:hidden; margin-top:.7rem; }
        .history-row { display:grid; grid-template-columns:1.5fr 1fr .8fr .7fr; gap:.8rem; padding:.8rem 1rem; border-top:1px solid #edf2f4; align-items:center; color:#50657a; font-size:.78rem; }
        .history-row.header { background:#f8fbfc; border-top:0; color:#718395; font-size:.66rem; font-weight:800; letter-spacing:.05em; text-transform:uppercase; }
        .prediction-pill { display:inline-block; border-radius:999px; padding:.25rem .48rem; font-weight:700; font-size:.72rem; }
        .prediction-pill.infected { background:#f9eded; color:#9d4141; } .prediction-pill.healthy { background:#edf7f2; color:#276e58; }
        .empty-history { color:#718395; font-size:.84rem; padding:1.05rem; }
        .app-footer { text-align:center; color:#7b8e9e; border-top:1px solid var(--line); font-size:.72rem; margin-top:2rem; padding-top:1.25rem; }
        @media (max-width:680px) { .block-container { padding:1.25rem 1rem; } .app-header { align-items:flex-start; } .status-badge { margin-top:.2rem; } .result-grid { grid-template-columns:1fr 1fr; } .result-grid > :first-child { grid-column:1/-1; } .model-grid { grid-template-columns:1fr 1fr; } .history-row { grid-template-columns:1.25fr 1fr .75fr; } .history-row > :last-child { display:none; } }
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "total_latency" not in st.session_state:
    st.session_state.total_latency = 0
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


@st.cache_resource
def load_model():
    from keras.layers import Dense

    class PatchedDense(Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    model = tf.keras.models.load_model(
        "malaria_model_final.h5",
        custom_objects={"Dense": PatchedDense},
        compile=False,
    )
    return model


IMG_SIZE = (128, 128)

st.markdown("""
<div class="app-header">
  <div class="brand"><div class="brand-icon">+</div><div><div class="brand-title">Malaria Cell Detection</div><div class="brand-subtitle">AI-Powered Microscopic Blood Cell Analysis</div></div></div>
  <div class="status-badge"><span class="status-dot"></span>AI Model Ready</div>
</div>
<div class="intro"><h1>Clear, focused blood-cell analysis</h1><p>Upload a microscopic blood-cell image and let the deep learning model analyze it for malaria infection.</p><div class="powered">Powered by MobileNetV2 transfer learning.</div></div>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="section-title" style="text-align:center">Upload Blood Cell Image</div><div class="section-copy" style="text-align:center">Upload a clear microscopic image in PNG, JPG, or JPEG format.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload blood cell image", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )
    st.markdown('<div class="formats" style="text-align:center">Accepted formats: PNG, JPG, JPEG</div>', unsafe_allow_html=True)

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Uploaded Sample</div>', unsafe_allow_html=True)
            image_col, details_col = st.columns([1.2, 1])
            with image_col:
                st.image(img, use_container_width=True)
            with details_col:
                st.markdown(
                    f'<div class="image-meta"><strong>File</strong><br>{uploaded_file.name}<br><br><strong>Dimensions</strong><br>{img.size[0]} × {img.size[1]} px<br><br><strong>Format</strong><br>{uploaded_file.type or "Image file"}</div>',
                    unsafe_allow_html=True,
                )

        if st.button("Analyze Sample", use_container_width=True):
            try:
                model = load_model()
                with st.spinner("Analyzing microscopic sample..."):
                    start = time.time()
                    # Existing preprocessing and prediction logic intentionally retained.
                    img_resized = img.resize(IMG_SIZE)
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    prob = float(model.predict(img_array, verbose=0)[0][0])
                    latency_ms = round((time.time() - start) * 1000)

                prediction = "Parasitized" if prob > 0.5 else "Uninfected"
                confidence = prob if prob > 0.5 else 1 - prob
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                result = {"prediction": prediction, "confidence": confidence, "latency_ms": latency_ms, "timestamp": timestamp, "name": uploaded_file.name}
                st.session_state.history.insert(0, result)
                st.session_state.total_latency += latency_ms
                st.session_state.latest_result = result
                st.rerun()
            except Exception:
                st.error("We couldn't analyze this image. Please try another valid blood-cell image.")
    except Exception:
        st.error("We couldn't read that file as an image. Please upload a PNG, JPG, or JPEG image.")

result = st.session_state.latest_result
if result:
    infected = result["prediction"] == "Parasitized"
    state_class = "infected" if infected else "healthy"
    st.markdown(f'''<div class="result-card {state_class}">
        <div class="eyebrow">Analysis Result</div>
        <div class="result-grid">
          <div><div class="result-label">Prediction</div><div class="prediction">{result["prediction"]}</div></div>
          <div><div class="result-label">Confidence</div><div class="metric-value">{result["confidence"]:.1%}</div></div>
          <div><div class="result-label">Inference Time</div><div class="metric-value">{result["latency_ms"]} ms</div></div>
        </div>
        <div class="confidence-wrap"><div class="confidence-head"><span>Confidence</span><span>{result["confidence"]:.1%}</span></div><div class="confidence-track"><div class="confidence-fill" style="width:{result["confidence"] * 100:.1f}%"></div></div></div>
    </div>''', unsafe_allow_html=True)

st.markdown('''<div class="model-section"><div class="section-title">Model Information</div><div class="model-grid">
  <div class="info-card"><div class="info-label">Model</div><div class="info-value">MobileNetV2</div></div>
  <div class="info-card"><div class="info-label">Task</div><div class="info-value">Binary Classification</div></div>
  <div class="info-card"><div class="info-label">Classes</div><div class="info-value">Parasitized / Uninfected</div></div>
  <div class="info-card"><div class="info-label">Framework</div><div class="info-value">TensorFlow / Keras</div></div>
</div></div>''', unsafe_allow_html=True)

st.markdown('<div class="model-section"><div class="section-title">Prediction History</div><div class="history-card">', unsafe_allow_html=True)
if st.session_state.history:
    st.markdown('<div class="history-row header"><span>Sample</span><span>Prediction</span><span>Confidence</span><span>Inference</span></div>', unsafe_allow_html=True)
    for entry in st.session_state.history:
        state_class = "infected" if entry["prediction"] == "Parasitized" else "healthy"
        st.markdown(f'<div class="history-row"><span>{entry.get("name", "Uploaded sample")}</span><span><span class="prediction-pill {state_class}">{entry["prediction"]}</span></span><span>{entry["confidence"]:.1%}</span><span>{entry["latency_ms"]} ms</span></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="empty-history">No analyses yet. Upload a microscopic image to begin.</div>', unsafe_allow_html=True)
st.markdown('</div></div><div class="app-footer">Malaria Cell Detection &bull; Deep Learning for Medical Image Analysis</div>', unsafe_allow_html=True)
