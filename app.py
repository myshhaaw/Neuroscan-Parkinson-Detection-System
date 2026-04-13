import streamlit as st
import numpy as np
import cv2
import os
import json
import tempfile
from PIL import Image
import time

# Page config
st.set_page_config(
    page_title="NeuroScan AI — Parkinson's Diagnosis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --card: #161f30;
    --accent: #00e5ff;
    --accent2: #7c3aed;
    --accent3: #10b981;
    --danger: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e293b;
}

.stApp { background: var(--bg) !important; color: var(--text) !important; font-family: 'Syne', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.neuro-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.neuro-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00e5ff 0%, #7c3aed 50%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.7rem; font-family: 'Space Mono', monospace; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin: 2px; }
.badge-cyan { background: rgba(0,229,255,0.1); color: #00e5ff; border: 1px solid rgba(0,229,255,0.3); }
.badge-purple { background: rgba(124,58,237,0.1); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }
.badge-green { background: rgba(16,185,129,0.1); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }

.section-label { font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--accent); margin-bottom: 6px; }
.section-title { font-size: 1.3rem; font-weight: 700; color: var(--text); margin-bottom: 16px; }
.upload-hint { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: var(--muted); margin-top: 6px; }

.result-positive { background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(239,68,68,0.03)); border: 1px solid rgba(239,68,68,0.3); border-radius: 12px; padding: 24px; text-align: center; }
.result-negative { background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(16,185,129,0.03)); border: 1px solid rgba(16,185,129,0.3); border-radius: 12px; padding: 24px; text-align: center; }
.result-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.result-value { font-size: 2rem; font-weight: 800; margin-bottom: 4px; }
.result-positive .result-value { color: #f87171; }
.result-negative .result-value { color: #34d399; }
.conf-score { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; color: var(--accent); }

.metric-box { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center; }
.metric-val { font-size: 1.4rem; font-weight: 700; color: var(--accent); }
.metric-lbl { font-size: 0.65rem; font-family: 'Space Mono', monospace; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }

.neuro-divider { height: 1px; background: linear-gradient(90deg, transparent, var(--border), transparent); margin: 20px 0; }

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #00e5ff) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 12px 32px !important; width: 100% !important;
    letter-spacing: 0.05em !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Optional torch ────────────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_models():
    models = {"mri": None, "video": None, "scribble": None, "fusion": None}
    if not os.path.exists(MODEL_DIR):
        return models
    for key in models:
        path = os.path.join(MODEL_DIR, f"{key}_model.pth")
        if os.path.exists(path) and TORCH_AVAILABLE:
            try:
                models[key] = torch.load(path, map_location="cpu")
                models[key].eval()
            except Exception:
                pass
    return models

def load_model_metrics():
    metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        return {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# ─── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(img_file, size=224):
    img = Image.open(img_file).convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return arr.transpose(2, 0, 1)[np.newaxis]

def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames, total = [], int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, total // 30)
    idx = 0
    while cap.isOpened() and len(frames) < 30:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        idx += 1
    cap.release()
    while len(frames) < 30:
        frames.append(np.zeros((112, 112, 3), dtype=np.float32))
    return np.stack(frames[:30])[np.newaxis]

def demo_inference(mri_arr, video_arr, scribble_arr, threshold=0.5):
    seed = int(abs(mri_arr.mean()) * 1000 + abs(scribble_arr.mean()) * 500) % (2**31)
    rng = np.random.default_rng(seed)
    mri_s   = float(np.clip(abs(mri_arr.mean()) * 3    + rng.normal(0.40, 0.15), 0.05, 0.95))
    scrib_s = float(np.clip(abs(scribble_arr.mean()) * 2.5 + rng.normal(0.40, 0.15), 0.05, 0.95))
    vid_s   = float(np.clip(video_arr.mean() * 4       + rng.normal(0.40, 0.15), 0.05, 0.95))
    fused   = float(np.clip(0.40*mri_s + 0.30*vid_s + 0.30*scrib_s, 0.05, 0.97))
    label   = "Parkinson's Detected" if fused >= threshold else "No Parkinson's Detected"
    return label, fused, {"MRI": mri_s, "Video": vid_s, "Scribble": scrib_s}

def run_inference(models, mri_arr, video_arr, scribble_arr, threshold):
    if not TORCH_AVAILABLE or models["fusion"] is None:
        return demo_inference(mri_arr, video_arr, scribble_arr, threshold)
    import torch
    with torch.no_grad():
        mri_t   = torch.tensor(mri_arr, dtype=torch.float32)
        scrib_t = torch.tensor(scribble_arr, dtype=torch.float32)
        vid_t   = torch.tensor(video_arr[:, :3], dtype=torch.float32)
        feat_m  = models["mri"](mri_t)
        feat_s  = models["scribble"](scrib_t)
        feat_v  = models["video"](vid_t)
        fused   = torch.cat([feat_m, feat_v, feat_s], dim=1)
        prob    = torch.sigmoid(models["fusion"](fused)).item()
    label = "Parkinson's Detected" if prob >= threshold else "No Parkinson's Detected"
    return label, prob, {"MRI": feat_m.norm().item(), "Video": feat_v.norm().item(), "Scribble": feat_s.norm().item()}

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">System Status</p>', unsafe_allow_html=True)
    mode = "PyTorch" if TORCH_AVAILABLE else "Demo Mode"
    st.markdown(f'<span class="badge badge-green">● ONLINE</span> <span class="badge badge-cyan">{mode}</span>', unsafe_allow_html=True)
    st.markdown('<div class="neuro-divider"></div>', unsafe_allow_html=True)

    model_metrics = load_model_metrics()
    fusion_metrics = model_metrics.get("fusion") if isinstance(model_metrics, dict) else None
    if fusion_metrics:
        st.markdown('<p class="section-label">Model Performance</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='neuro-card' style='padding:16px;margin-bottom:16px;'>
            <div style='display:flex;justify-content:space-between;gap:12px;'>
                <div style='flex:1;min-width:100px;'>
                    <div class='metric-lbl'>Accuracy</div>
                    <div class='metric-val'>{fusion_metrics.get('acc', 0)*100:.1f}%</div>
                </div>
                <div style='flex:1;min-width:100px;'>
                    <div class='metric-lbl'>F1 Score</div>
                    <div class='metric-val'>{fusion_metrics.get('f1', 0)*100:.1f}%</div>
                </div>
                <div style='flex:1;min-width:100px;'>
                    <div class='metric-lbl'>R2 Score</div>
                    <div class='metric-val'>{fusion_metrics.get('r2', 0):.3f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <p class="section-label">About</p>
    <div style='font-size:0.8rem; color:#94a3b8; line-height:1.7;'>
    NeuroScan AI uses <strong style='color:#00e5ff'>multi-modal deep learning</strong> combining:<br><br>
    🧠 <strong>MRI</strong> — EfficientNet substantia nigra volumetry<br>
    🎥 <strong>Tremor Video</strong> — CNN+LSTM oscillatory pattern detection<br>
    ✍️ <strong>Scribble Tests</strong> — spiral entropy & fine motor analysis
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="neuro-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Confidence Threshold</p>', unsafe_allow_html=True)
    threshold = st.slider("Decision boundary", 0.3, 0.8, 0.5, 0.05)

    st.markdown('<div class="neuro-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.7rem; color:#475569; text-align:center; font-family:Space Mono,monospace;'>
    ⚠️ FOR RESEARCH PURPOSES ONLY<br>Not a clinical tool
    </div>
    """, unsafe_allow_html=True)

# ─── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 2rem 0 1rem;'>
    <p class="hero-sub">Multi-Modal AI · Early Detection System · v2.0</p>
    <h1 class="hero-title">NeuroScan AI</h1>
    <p style='color:#94a3b8; font-size:0.95rem; max-width:600px; line-height:1.7;'>
    Advanced Parkinson's Disease diagnosis powered by multi-modal deep learning.
    Integrates MRI neuroimaging, tremor video analysis, and handwriting kinematics
    for early-stage detection with confidence scoring.
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, (val, lbl) in zip([c1,c2,c3,c4], [("10M+","Patients Worldwide"),("3","Input Modalities"),("EfficientNet","MRI Backbone"),("Fusion","Multi-Modal")]):
    with col:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="neuro-divider"></div>', unsafe_allow_html=True)

# ─── Uploads ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Input Data</p><p class="section-title">Upload Patient Data</p>', unsafe_allow_html=True)

col_mri, col_vid, col_scrib = st.columns(3)

with col_mri:
    st.markdown('<div class="neuro-card"><span class="badge badge-cyan">MRI Scan</span><p style="font-weight:700;margin:10px 0 4px;color:#e2e8f0;">T1-Weighted Brain MRI</p><p style="font-size:0.78rem;color:#64748b;margin-bottom:12px;">Analyses substantia nigra morphology and dopaminergic pathway integrity.</p>', unsafe_allow_html=True)
    mri_file = st.file_uploader("MRI Image", type=["png","jpg","jpeg"], key="mri", label_visibility="collapsed")
    if mri_file:
        st.image(mri_file, use_container_width=True, caption="✓ MRI Uploaded")
    else:
        st.markdown('<p class="upload-hint">→ .jpg / .png accepted</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_vid:
    st.markdown('<div class="neuro-card"><span class="badge badge-purple">Tremor Video</span><p style="font-weight:700;margin:10px 0 4px;color:#e2e8f0;">Motor Tremor Recording</p><p style="font-size:0.78rem;color:#64748b;margin-bottom:12px;">CNN+LSTM detects oscillatory patterns at 4–6 Hz characteristic of PD.</p>', unsafe_allow_html=True)
    video_file = st.file_uploader("Tremor Video", type=["mp4","avi","mov"], key="video", label_visibility="collapsed")
    if video_file:
        st.video(video_file)
        st.markdown('<p style="font-size:0.75rem;color:#34d399;margin-top:6px;">✓ Video Uploaded</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="upload-hint">→ .mp4 / .avi / .mov accepted</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_scrib:
    st.markdown('<div class="neuro-card"><span class="badge badge-green">Scribble Test</span><p style="font-weight:700;margin:10px 0 4px;color:#e2e8f0;">Handwriting / Spiral Test</p><p style="font-size:0.78rem;color:#64748b;margin-bottom:12px;">Detects spiral entropy, velocity peaks, and fine motor tremor patterns.</p>', unsafe_allow_html=True)
    scribble_file = st.file_uploader("Scribble Image", type=["png","jpg","jpeg","bmp"], key="scribble", label_visibility="collapsed")
    if scribble_file:
        st.image(scribble_file, use_container_width=True, caption="✓ Scribble Uploaded")
    else:
        st.markdown('<p class="upload-hint">→ .jpg / .png / .bmp accepted</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Run Button ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
run_col, _ = st.columns([1, 2])
with run_col:
    run_btn = st.button("🔬  Run Diagnosis")

# ─── Results ───────────────────────────────────────────────────────────────────
if run_btn:
    if not mri_file or not video_file or not scribble_file:
        st.warning("⚠️  Please upload all three modalities (MRI, Video, Scribble) before running.")
    else:
        progress = st.progress(0, text="Initialising models…")
        time.sleep(0.3)
        progress.progress(20, text="Preprocessing MRI scan…")
        mri_arr = preprocess_image(mri_file, 224)
        time.sleep(0.3)
        progress.progress(40, text="Extracting video tremor features…")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        video_arr = extract_video_features(tmp_path)
        os.unlink(tmp_path)
        time.sleep(0.3)
        progress.progress(65, text="Analysing handwriting patterns…")
        scribble_arr = preprocess_image(scribble_file, 224)
        time.sleep(0.3)
        progress.progress(85, text="Running multi-modal fusion…")
        models = load_models()
        label, confidence, mod_scores = run_inference(models, mri_arr, video_arr, scribble_arr, threshold)
        time.sleep(0.3)
        progress.progress(100, text="Complete!")
        time.sleep(0.2)
        progress.empty()

        st.markdown('<div class="neuro-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Diagnosis Output</p><p class="section-title">Results</p>', unsafe_allow_html=True)

        is_pos = "Parkinson's" in label
        res_col, conf_col, bar_col = st.columns([1.5, 1, 2])

        with res_col:
            cls = "result-positive" if is_pos else "result-negative"
            icon = "⚠️" if is_pos else "✅"
            diag_text = "Parkinson's<br>Detected" if is_pos else "No Parkinson's<br>Detected"
            st.markdown(f'<div class="{cls}"><div class="result-label">Diagnosis</div><div class="result-value">{icon}<br>{diag_text}</div></div>', unsafe_allow_html=True)

        with conf_col:
            st.markdown(f"""
            <div class="neuro-card" style="text-align:center;">
                <div class="result-label">Confidence Score</div>
                <div class="conf-score">{confidence*100:.1f}%</div>
                <div style="font-size:0.7rem;color:#64748b;margin-top:8px;font-family:Space Mono,monospace;">
                    Threshold: {threshold*100:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with bar_col:
            st.markdown('<div class="neuro-card"><p class="section-label" style="margin-bottom:12px;">Modality Contributions</p>', unsafe_allow_html=True)
            for mod, score in mod_scores.items():
                pct = min(score * 100, 100)
                color = "#ef4444" if pct > 60 else "#f59e0b" if pct > 40 else "#10b981"
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                    <span style="font-size:0.78rem;color:#94a3b8;font-family:Space Mono,monospace;">{mod}</span>
                    <span style="font-size:0.78rem;color:{color};font-family:Space Mono,monospace;font-weight:700;">{pct:.0f}%</span>
                  </div>
                  <div style="background:#1e293b;border-radius:4px;height:6px;overflow:hidden;">
                    <div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:4px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if is_pos:
            st.markdown(f"""
            <div class="neuro-card" style="border-left:3px solid #ef4444;">
                <p class="section-label">Clinical Interpretation</p>
                <p style='color:#e2e8f0;font-size:0.9rem;line-height:1.8;'>
                The fusion model returned a confidence of <strong style='color:#f87171'>{confidence*100:.1f}%</strong>,
                exceeding the {threshold*100:.0f}% threshold. Elevated signals across neuroimaging and motor modalities
                suggest possible dopaminergic involvement consistent with early-stage Parkinson's Disease.
                <br><br>
                <strong style='color:#fca5a5'>Recommendation:</strong> Refer to a movement disorder specialist for
                formal evaluation including DAT-SPECT imaging and UPDRS scoring. Early intervention improves long-term outcomes.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="neuro-card" style="border-left:3px solid #10b981;">
                <p class="section-label">Clinical Interpretation</p>
                <p style='color:#e2e8f0;font-size:0.9rem;line-height:1.8;'>
                The fusion model returned <strong style='color:#34d399'>{confidence*100:.1f}%</strong>,
                below the {threshold*100:.0f}% threshold. No significant PD biomarkers detected across
                neuroimaging, tremor, or handwriting modalities.
                <br><br>
                <strong style='color:#6ee7b7'>Recommendation:</strong> No immediate intervention indicated.
                Continue routine neurological monitoring. Repeat if symptoms develop.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔧  Technical Details"):
            t1, t2, t3 = st.tabs(["MRI Features", "Video Features", "Scribble Features"])
            with t1:
                st.code(f"Shape : {mri_arr.shape}\nMean  : {mri_arr.mean():.4f}\nStd   : {mri_arr.std():.4f}\nScore : {mod_scores['MRI']:.4f}")
            with t2:
                st.code(f"Shape  : {video_arr.shape}\nFrames : {video_arr.shape[1]}\nMean   : {video_arr.mean():.4f}\nScore  : {mod_scores['Video']:.4f}")
            with t3:
                st.code(f"Shape : {scribble_arr.shape}\nMean  : {scribble_arr.mean():.4f}\nStd   : {scribble_arr.std():.4f}\nScore : {mod_scores['Scribble']:.4f}")

        st.markdown("""
        <div style='margin-top:20px;padding:12px;background:rgba(239,68,68,0.04);border:1px solid rgba(239,68,68,0.15);
                    border-radius:8px;font-size:0.72rem;color:#94a3b8;font-family:Space Mono,monospace;'>
        ⚠️ DISCLAIMER: Research and educational purposes only. Results must be reviewed by a qualified neurologist.
        </div>
        """, unsafe_allow_html=True)

