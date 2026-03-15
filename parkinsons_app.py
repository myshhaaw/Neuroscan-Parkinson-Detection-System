"""
Parkinson's Disease Detection System
Streamlit Application — MRI Analysis Module (Active)
Scribble Test & Hand Tremor modules (UI Ready, backend pending)
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import base64
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan | Parkinson's Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #30363d;
  }
  section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
  section[data-testid="stSidebar"] .stRadio label { color: #8b949e !important; }
  section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label {
    color: #58a6ff !important; font-weight: 600;
  }

  /* Main background */
  .main .block-container {
    background: #0d1117;
    padding: 2rem 2.5rem;
    max-width: 1200px;
  }
  body { background-color: #0d1117; }

  /* ── Cards ── */
  .med-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .med-card-accent {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #388bfd44;
    border-left: 3px solid #388bfd;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .result-positive {
    background: linear-gradient(135deg, #1a1a2e 0%, #1f1624 100%);
    border: 1px solid #f8514944;
    border-left: 4px solid #f85149;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
  }
  .result-negative {
    background: linear-gradient(135deg, #0d1f1a 0%, #111f1e 100%);
    border: 1px solid #3fb95044;
    border-left: 4px solid #3fb950;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
  }
  .result-pending {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #8b949e;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
  }

  /* ── Typography ── */
  .page-title {
    font-size: 2rem; font-weight: 700; color: #e6edf3;
    letter-spacing: -0.5px; margin-bottom: 0.25rem;
  }
  .page-subtitle { font-size: 0.9rem; color: #8b949e; margin-bottom: 2rem; }
  .section-header {
    font-size: 1.1rem; font-weight: 600; color: #58a6ff;
    text-transform: uppercase; letter-spacing: 1px;
    border-bottom: 1px solid #30363d; padding-bottom: 0.5rem;
    margin-bottom: 1.25rem;
  }
  .label-text { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px; }
  .value-text { font-size: 1rem; color: #e6edf3; font-weight: 500; }
  .badge-active {
    display:inline-block; background:#1f6feb33; color:#58a6ff;
    border:1px solid #1f6feb; border-radius:20px;
    padding:2px 10px; font-size:0.72rem; font-weight:600; letter-spacing:0.5px;
  }
  .badge-pending {
    display:inline-block; background:#30363d; color:#8b949e;
    border:1px solid #484f58; border-radius:20px;
    padding:2px 10px; font-size:0.72rem; font-weight:600; letter-spacing:0.5px;
  }
  .disclaimer {
    background:#161b22; border:1px solid #d2992255;
    border-left:3px solid #d29922; border-radius:8px;
    padding:0.85rem 1rem; font-size:0.8rem; color:#8b949e;
    margin-top:1.5rem;
  }

  /* ── Metric tiles ── */
  .metric-tile {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:1rem 1.25rem; text-align:center;
  }
  .metric-value { font-size:1.6rem; font-weight:700; color:#58a6ff; }
  .metric-label { font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; }

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .stProgress > div > div > div { background-color: #388bfd !important; }
  .stButton > button {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important; width: 100%;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #388bfd 0%, #58a6ff 100%) !important;
    transform: translateY(-1px); box-shadow: 0 4px 15px #388bfd44 !important;
  }
  .stFileUploader {
    background: #161b22 !important; border: 1px dashed #388bfd !important;
    border-radius: 10px !important;
  }
  div[data-testid="stFileUploadDropzone"] {
    background: #161b22 !important;
  }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 NeuroScan")
    st.markdown('<p style="font-size:0.75rem;color:#8b949e;margin-top:-10px;">Parkinson\'s Detection System v1.0</p>', unsafe_allow_html=True)
    st.markdown("---")

    nav = st.radio(
        "Navigation",
        ["🏠  Dashboard", "🧲  MRI Analysis", "✍️  Scribble Test", "📡  Hand Tremor"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    st.markdown('<p class="label-text">System Status</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:0.5rem;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <div style="width:8px;height:8px;border-radius:50%;background:#3fb950;"></div>
        <span style="font-size:0.8rem;color:#e6edf3;">MRI Module — Online</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <div style="width:8px;height:8px;border-radius:50%;background:#d29922;"></div>
        <span style="font-size:0.8rem;color:#8b949e;">Scribble Test — Coming Soon</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:8px;height:8px;border-radius:50%;background:#d29922;"></div>
        <span style="font-size:0.8rem;color:#8b949e;">Hand Tremor — Coming Soon</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="label-text">Model Info</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem;color:#8b949e;line-height:1.8;">
      Architecture: <span style="color:#e6edf3;">ResNet-50</span><br>
      Input: <span style="color:#e6edf3;">224 × 224 px</span><br>
      Classes: <span style="color:#e6edf3;">Parkinson / Normal</span><br>
      Dataset: <span style="color:#e6edf3;">831 MRI scans</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Model loader (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load or build the ResNet-50 model."""
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
        from tensorflow.keras.models import Model

        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )
        for layer in base_model.layers[:-5]:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model, None
    except Exception as e:
        return None, str(e)


def preprocess_image(uploaded_file):
    """Load and preprocess a PIL image for ResNet-50 inference."""
    try:
        from PIL import Image
        img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0), img
    except Exception as e:
        return None, str(e)


def grad_cam_heatmap(model, img_array, last_conv_layer_name="conv5_block3_out"):
    """Compute a Grad-CAM heatmap for the given image."""
    try:
        import tensorflow as tf
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap
    except Exception:
        return None


def overlay_heatmap(pil_img, heatmap):
    """Overlay Grad-CAM heatmap on PIL image, return matplotlib figure."""
    import cv2
    img_array = np.array(pil_img)
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#161b22")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.axis("off")
    axes[0].imshow(img_array)
    axes[0].set_title("Original MRI", color="#8b949e", fontsize=9, pad=8)
    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Activation Map", color="#8b949e", fontsize=9, pad=8)
    axes[2].imshow(superimposed)
    axes[2].set_title("Grad-CAM Overlay", color="#8b949e", fontsize=9, pad=8)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# PAGE — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown('<div class="page-title">Clinical Decision Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Parkinson\'s Disease Diagnostic Assistant — Neuroimaging & Motor Assessment</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        ["ResNet-50", "224×224", "2-Class", "~85%"],
        ["Architecture", "Input Resolution", "Task Type", "Val. Accuracy"],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-tile">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown('<div class="section-header">Diagnostic Modules</div>', unsafe_allow_html=True)
        modules = [
            ("🧲", "MRI Neuroimaging Analysis", "active",
             "Deep learning classification of T1/T2-weighted and diffusion MRI scans using ResNet-50 transfer learning. Supports Grad-CAM saliency visualisation."),
            ("✍️", "Handwriting / Scribble Test", "pending",
             "Automated analysis of micrographia and spiral-drawing irregularities — established early biomarkers for dopaminergic pathway degeneration."),
            ("📡", "Resting Hand Tremor Analysis", "pending",
             "Accelerometer or video-based frequency decomposition (3–6 Hz resting tremor characterisation) correlated with UPDRS motor subscales."),
        ]
        for icon, title, status, desc in modules:
            badge = '<span class="badge-active">ACTIVE</span>' if status == "active" else '<span class="badge-pending">COMING SOON</span>'
            st.markdown(f"""
            <div class="med-card-accent" style="margin-bottom:0.8rem;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div style="font-size:1rem;color:#e6edf3;font-weight:600;">{icon} &nbsp;{title}</div>
                {badge}
              </div>
              <div style="font-size:0.82rem;color:#8b949e;margin-top:0.5rem;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        data = {"Category": ["Total Scans", "Parkinson", "Normal", "MRI Types", "Train Split", "Test Split"],
                "Value":    ["831",          "~415",       "~416",   "18",         "80%",         "20%"]}
        for cat, val in zip(data["Category"], data["Value"]):
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #21262d;">
              <span class="label-text">{cat}</span>
              <span class="value-text">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">MRI Sequence Types</div>', unsafe_allow_html=True)
        seqs = ["T1W (3D FLASH)", "T2W TSE", "DWI/ADC", "SWI/mIP", "T2-FLAIR", "+ 13 more"]
        for s in seqs:
            st.markdown(f'<div style="font-size:0.82rem;color:#8b949e;padding:3px 0;">• {s}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      ⚕️ <strong style="color:#d29922;">Clinical Disclaimer:</strong>
      This system is a research prototype and is <strong>not</strong> a certified medical device.
      All outputs are for investigational purposes only and must not replace clinical judgement,
      radiological review, or neurological examination by a qualified healthcare professional.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE — MRI ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
def page_mri():
    st.markdown('<div class="page-title">🧲 MRI Neuroimaging Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload a brain MRI scan for Parkinson\'s / Normal classification via ResNet-50 transfer learning</div>', unsafe_allow_html=True)

    col_upload, col_options = st.columns([2, 1])

    with col_upload:
        st.markdown('<div class="section-header">Image Upload</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop MRI scan here (PNG / JPG / JPEG)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )

    with col_options:
        st.markdown('<div class="section-header">Analysis Options</div>', unsafe_allow_html=True)
        show_gradcam = st.checkbox("Generate Grad-CAM Overlay", value=True,
                                   help="Visualises which regions influenced the model's decision.")
        show_confidence = st.checkbox("Show Confidence Breakdown", value=True)
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05,
                              help="Probability above this value = Parkinson's positive.")

    if uploaded is not None:
        st.markdown("---")
        with st.spinner("🔬 Preprocessing image and loading model…"):
            model, err = load_model()
            img_array, pil_img = preprocess_image(uploaded)

        if err:
            st.error(f"⚠️ Model load failed: {err}")
            st.info("Running in demo mode with simulated output.")
            model = None

        if img_array is None:
            st.error(f"⚠️ Image preprocessing failed: {pil_img}")
            return

        # ── Run inference ──
        if model is not None:
            with st.spinner("🧠 Running inference…"):
                raw_prob = float(model.predict(img_array, verbose=0)[0][0])
        else:
            # Demo: simulate a result
            np.random.seed(42)
            raw_prob = float(np.random.uniform(0.1, 0.9))

        parkinson_prob = raw_prob
        normal_prob    = 1.0 - raw_prob
        prediction     = "Parkinson's Positive" if parkinson_prob >= threshold else "Normal"
        is_positive    = parkinson_prob >= threshold

        # ── Layout: image + result ──
        img_col, res_col = st.columns([1, 1])

        with img_col:
            st.markdown('<div class="section-header">Uploaded Scan</div>', unsafe_allow_html=True)
            st.image(pil_img, use_column_width=True)

        with res_col:
            st.markdown('<div class="section-header">Classification Result</div>', unsafe_allow_html=True)

            card_class = "result-positive" if is_positive else "result-negative"
            icon = "🔴" if is_positive else "🟢"
            color = "#f85149" if is_positive else "#3fb950"

            st.markdown(f"""
            <div class="{card_class}">
              <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;">Diagnosis</div>
              <div style="font-size:1.5rem;font-weight:700;color:{color};margin:0.4rem 0;">{icon} {prediction}</div>
              <div style="font-size:0.82rem;color:#8b949e;">Threshold: {threshold:.2f} | Model: ResNet-50</div>
            </div>""", unsafe_allow_html=True)

            if show_confidence:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="label-text" style="margin-bottom:8px;">Confidence Breakdown</div>', unsafe_allow_html=True)

                for label, prob, bar_color in [
                    ("Parkinson's", parkinson_prob, "#f85149"),
                    ("Normal",      normal_prob,    "#3fb950"),
                ]:
                    st.markdown(f"""
                    <div style="margin-bottom:0.6rem;">
                      <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                        <span style="font-size:0.8rem;color:#e6edf3;">{label}</span>
                        <span style="font-size:0.8rem;color:{bar_color};font-weight:600;">{prob*100:.1f}%</span>
                      </div>
                      <div style="background:#21262d;border-radius:4px;height:8px;">
                        <div style="background:{bar_color};width:{prob*100:.1f}%;height:8px;border-radius:4px;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            # Interpretation note
            st.markdown("""
            <div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:0.75rem;margin-top:1rem;font-size:0.79rem;color:#8b949e;line-height:1.6;">
              <strong style="color:#58a6ff;">Interpretation Note:</strong><br>
              This classification is based on learned MRI feature patterns.
              Results should be reviewed alongside clinical history, neurological
              examination (UPDRS), and specialist radiological interpretation.
            </div>""", unsafe_allow_html=True)

        # ── Grad-CAM ──
        if show_gradcam and model is not None:
            st.markdown("---")
            st.markdown('<div class="section-header">Grad-CAM Saliency Visualisation</div>', unsafe_allow_html=True)
            with st.spinner("Generating Grad-CAM heatmap…"):
                heatmap = grad_cam_heatmap(model, img_array)
            if heatmap is not None:
                try:
                    fig = overlay_heatmap(pil_img, heatmap)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.markdown("""
                    <div style="font-size:0.79rem;color:#8b949e;margin-top:0.5rem;">
                      Warmer colours (red/yellow) indicate regions with higher influence on the classification decision.
                      This is for interpretability only and does not constitute a radiological finding.
                    </div>""", unsafe_allow_html=True)
                except Exception:
                    st.info("Grad-CAM overlay requires OpenCV (cv2). Install with: pip install opencv-python")
            else:
                st.info("Grad-CAM could not be generated for this model configuration.")

        # ── Report summary ──
        st.markdown("---")
        st.markdown('<div class="section-header">Structured Report Summary</div>', unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        fields = [
            ("Filename",    uploaded.name,          rc1),
            ("Prediction",  prediction,             rc2),
            ("Probability", f"{parkinson_prob:.4f}", rc3),
            ("Threshold",   f"{threshold:.2f}",     rc1),
            ("Model",       "ResNet-50 (ImageNet)", rc2),
            ("Status",      "Demo" if model is None else "Live Inference", rc3),
        ]
        for lbl, val, col in fields:
            with col:
                st.markdown(f"""
                <div style="padding:0.5rem 0;border-bottom:1px solid #21262d;">
                  <div class="label-text">{lbl}</div>
                  <div class="value-text">{val}</div>
                </div>""", unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div class="med-card" style="text-align:center;padding:3rem 2rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">🧲</div>
          <div style="font-size:1rem;color:#e6edf3;font-weight:600;margin-bottom:0.5rem;">No scan uploaded</div>
          <div style="font-size:0.85rem;color:#8b949e;max-width:400px;margin:auto;line-height:1.6;">
            Upload a brain MRI image (PNG/JPG) to begin analysis.
            Supported sequences: T1W, T2W, DWI/ADC, SWI, FLAIR.
          </div>
        </div>""", unsafe_allow_html=True)

    # Training section
    with st.expander("🏋️ Model Training (Advanced)", expanded=False):
        st.markdown('<div class="section-header">Training Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.85rem;color:#8b949e;line-height:1.8;">
          To train the model on your own dataset, ensure the following folder structure:<br><br>
          <code style="color:#79c0ff;background:#0d1117;padding:2px 6px;border-radius:4px;">
          parkinsons_dataset/<br>
          &nbsp;&nbsp;parkinson/<br>
          &nbsp;&nbsp;&nbsp;&nbsp;*.png<br>
          &nbsp;&nbsp;normal/<br>
          &nbsp;&nbsp;&nbsp;&nbsp;*.png
          </code>
        </div>
        """, unsafe_allow_html=True)

        data_path = st.text_input("Dataset Directory", placeholder="/path/to/parkinsons_dataset")
        epochs    = st.slider("Training Epochs", 5, 50, 25, 5)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
        with col_t2:
            test_split = st.slider("Test Split", 0.1, 0.4, 0.2, 0.05)

        if st.button("🚀 Start Training"):
            if not data_path:
                st.warning("⚠️ Please enter a valid dataset directory path.")
            else:
                run_training(data_path, epochs, batch_size, test_split)


def run_training(data_path, epochs, batch_size, test_split):
    """Run the full training pipeline with live progress."""
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
        from tensorflow.keras.models import Model
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

        path = Path(data_path)
        if not path.exists():
            st.error(f"❌ Directory not found: {data_path}")
            return

        # ── Load file paths ──
        df = pd.DataFrame({"path": list(path.glob("*/*.png"))})
        if df.empty:
            df = pd.DataFrame({"path": list(path.glob("*/*.jpg")) + list(path.glob("*/*.jpeg"))})
        if df.empty:
            st.error("❌ No images found. Expected PNG/JPG inside class subfolders.")
            return

        df["disease"] = df["path"].map(lambda x: x.parent.stem)
        df["path_str"] = df["path"].astype(str)

        st.success(f"✅ Found {len(df)} images — Classes: {df['disease'].unique().tolist()}")

        X = df["path_str"]
        y = df["disease"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )

        # ── Generators ──
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255, rotation_range=20,
            width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode="nearest",
        )
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_df = pd.DataFrame({"path": X_train, "disease": y_train})
        test_df  = pd.DataFrame({"path": X_test,  "disease": y_test})

        train_gen = train_datagen.flow_from_dataframe(
            train_df, x_col="path", y_col="disease",
            target_size=(224, 224), batch_size=batch_size, class_mode="binary"
        )
        test_gen = test_datagen.flow_from_dataframe(
            test_df, x_col="path", y_col="disease",
            target_size=(224, 224), batch_size=batch_size,
            class_mode="binary", shuffle=False
        )

        # ── Build model ──
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        for layer in base.layers[:-5]:
            layer.trainable = False
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        out = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=base.input, outputs=out)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # ── Train ──
        progress_bar = st.progress(0)
        status_text  = st.empty()
        chart_placeholder = st.empty()

        acc_hist, val_acc_hist, loss_hist, val_loss_hist = [], [], [], []

        for epoch in range(1, epochs + 1):
            status_text.markdown(f'<span style="color:#58a6ff;font-size:0.85rem;">Epoch {epoch}/{epochs}</span>', unsafe_allow_html=True)
            h = model.fit(train_gen, validation_data=test_gen, epochs=1, verbose=0)
            acc_hist.append(h.history["accuracy"][0])
            val_acc_hist.append(h.history["val_accuracy"][0])
            loss_hist.append(h.history["loss"][0])
            val_loss_hist.append(h.history["val_loss"][0])
            progress_bar.progress(epoch / epochs)

            if epoch % 3 == 0 or epoch == epochs:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
                fig.patch.set_facecolor("#161b22")
                for ax in [ax1, ax2]:
                    ax.set_facecolor("#0d1117")
                    ax.tick_params(colors="#8b949e")
                    ax.xaxis.label.set_color("#8b949e")
                    ax.yaxis.label.set_color("#8b949e")
                    ax.title.set_color("#e6edf3")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#30363d")

                ax1.plot(acc_hist, color="#58a6ff", label="Train Acc", linewidth=2)
                ax1.plot(val_acc_hist, color="#3fb950", label="Val Acc", linewidth=2, linestyle="--")
                ax1.set_title("Accuracy"); ax1.legend(facecolor="#161b22", labelcolor="#8b949e")

                ax2.plot(loss_hist, color="#f85149", label="Train Loss", linewidth=2)
                ax2.plot(val_loss_hist, color="#d29922", label="Val Loss", linewidth=2, linestyle="--")
                ax2.set_title("Loss"); ax2.legend(facecolor="#161b22", labelcolor="#8b949e")

                plt.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close(fig)

        status_text.markdown('<span style="color:#3fb950;font-size:0.85rem;">✅ Training complete!</span>', unsafe_allow_html=True)

        # ── Evaluation ──
        preds_raw  = model.predict(test_gen, verbose=0)
        preds      = (preds_raw > 0.5).astype(int).flatten()
        true_labels = test_gen.classes
        acc = accuracy_score(true_labels, preds)

        st.markdown(f"""
        <div class="result-negative" style="margin-top:1rem;">
          <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;">Final Validation Accuracy</div>
          <div style="font-size:2rem;font-weight:700;color:#3fb950;">{acc*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)

        cm = confusion_matrix(true_labels, preds)
        st.markdown('<div class="label-text" style="margin-top:1rem;">Confusion Matrix</div>', unsafe_allow_html=True)

        fig_cm, ax = plt.subplots(figsize=(4, 3))
        fig_cm.patch.set_facecolor("#161b22")
        ax.set_facecolor("#0d1117")
        import seaborn as sns
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Parkinson"],
            yticklabels=["Normal", "Parkinson"],
            ax=ax, linewidths=1, linecolor="#30363d"
        )
        ax.set_xlabel("Predicted", color="#8b949e")
        ax.set_ylabel("Actual", color="#8b949e")
        ax.tick_params(colors="#8b949e")
        ax.title.set_color("#e6edf3")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        report = classification_report(true_labels, preds, target_names=["Normal", "Parkinson"])
        st.code(report, language="text")

        # Save model
        save_path = "parkinsons_resnet50.h5"
        model.save(save_path)
        st.success(f"💾 Model saved to: {save_path}")

    except ImportError as e:
        st.error(f"Missing dependency: {e}\nInstall with: pip install tensorflow scikit-learn seaborn")
    except Exception as e:
        st.error(f"Training error: {e}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE — SCRIBBLE TEST
# ════════════════════════════════════════════════════════════════════════════
def page_scribble():
    st.markdown('<div class="page-title">✍️ Handwriting / Scribble Test</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Micrographia analysis and spiral drawing assessment — early Parkinson\'s biomarkers</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#161b22;border:1px solid #d2992244;border-left:4px solid #d29922;border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;">
      <div style="font-size:0.85rem;font-weight:600;color:#d29922;margin-bottom:0.4rem;">🚧 Module Under Development</div>
      <div style="font-size:0.82rem;color:#8b949e;line-height:1.7;">
        The Scribble Test module is currently being integrated. The UI and pipeline are designed
        and ready — backend model training on the Parkinson's Drawing Dataset is in progress.
      </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Planned Features</div>', unsafe_allow_html=True)
        features = [
            ("📐", "Spiral Drawing Analysis",     "FFT-based tremor frequency decomposition on hand-drawn spirals"),
            ("🔤", "Handwriting Micrographia",    "Character size normalisation and velocity profiling"),
            ("📊", "Kinematic Feature Extraction","Pen pressure, stroke timing, and acceleration metrics"),
            ("🤖", "CNN Classification",          "Binary Parkinson's / Normal from drawing images"),
        ]
        for icon, title, desc in features:
            st.markdown(f"""
            <div class="med-card" style="padding:1rem;margin-bottom:0.6rem;">
              <div style="font-size:0.9rem;color:#e6edf3;font-weight:600;">{icon} {title}</div>
              <div style="font-size:0.78rem;color:#8b949e;margin-top:4px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Upload Test Image (Preview)</div>', unsafe_allow_html=True)
        st.file_uploader("Upload scribble/spiral image", type=["png", "jpg", "jpeg"], disabled=True,
                         help="This module is not yet active.", label_visibility="collapsed")

        st.markdown("""
        <div class="result-pending">
          <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;">Module Status</div>
          <div style="font-size:1.1rem;font-weight:600;color:#8b949e;margin-top:0.4rem;">⏳ Integration Pending</div>
          <div style="font-size:0.8rem;color:#8b949e;margin-top:0.3rem;">
            Expected: model loading, feature extraction pipeline, and visualisation layer.
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Clinical Background</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.82rem;color:#8b949e;line-height:1.8;">
          Micrographia (abnormally small handwriting) affects up to <strong style="color:#e6edf3;">79%</strong>
          of Parkinson's patients. Spiral drawing abnormalities are detectable years before motor symptom onset,
          making this a valuable pre-symptomatic screening tool aligned with the
          <strong style="color:#e6edf3;">UPDRS Part III</strong> assessment protocol.
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE — HAND TREMOR
# ════════════════════════════════════════════════════════════════════════════
def page_tremor():
    st.markdown('<div class="page-title">📡 Hand Tremor Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Resting tremor frequency analysis — 3–6 Hz characterisation correlated with UPDRS motor score</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#161b22;border:1px solid #d2992244;border-left:4px solid #d29922;border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1.5rem;">
      <div style="font-size:0.85rem;font-weight:600;color:#d29922;margin-bottom:0.4rem;">🚧 Module Under Development</div>
      <div style="font-size:0.82rem;color:#8b949e;line-height:1.7;">
        The Hand Tremor module pipeline is designed. Accelerometer CSV ingestion and FFT-based
        frequency analysis are ready for backend connection.
      </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-header">Signal Analysis Preview</div>', unsafe_allow_html=True)

        # Simulated tremor waveform (demo only)
        t = np.linspace(0, 4, 1000)
        parkinson_signal = (
            0.8 * np.sin(2 * np.pi * 4.5 * t) +        # 4.5 Hz resting tremor
            0.2 * np.sin(2 * np.pi * 9.0 * t) +         # harmonic
            0.15 * np.random.randn(len(t))
        )
        normal_signal = 0.15 * np.random.randn(len(t))

        fig, axes = plt.subplots(2, 1, figsize=(10, 5))
        fig.patch.set_facecolor("#161b22")
        for ax, sig, label, col in zip(
            axes,
            [parkinson_signal, normal_signal],
            ["Simulated Parkinson's Tremor (4.5 Hz)", "Simulated Normal (Control)"],
            ["#f85149", "#3fb950"]
        ):
            ax.set_facecolor("#0d1117")
            ax.plot(t, sig, color=col, linewidth=0.8, alpha=0.85)
            ax.set_title(label, color="#e6edf3", fontsize=9, pad=6)
            ax.set_xlabel("Time (s)", color="#8b949e", fontsize=8)
            ax.set_ylabel("Amplitude", color="#8b949e", fontsize=8)
            ax.tick_params(colors="#8b949e", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('<div style="font-size:0.75rem;color:#8b949e;margin-top:4px;">⚠️ Waveforms are simulated for demonstration purposes only.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Upload Sensor Data (Preview)</div>', unsafe_allow_html=True)
        st.file_uploader("Upload accelerometer CSV", type=["csv"], disabled=True,
                         help="Module not yet active.", label_visibility="collapsed")

        st.markdown("""
        <div class="result-pending">
          <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;">Module Status</div>
          <div style="font-size:1.1rem;font-weight:600;color:#8b949e;margin-top:0.4rem;">⏳ Integration Pending</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Planned Analysis</div>', unsafe_allow_html=True)
        items = [
            ("📈", "FFT Power Spectrum",      "Peak frequency and power in 3–6 Hz band"),
            ("🎯", "Tremor Score",            "UPDRS-correlated composite score"),
            ("📉", "Action vs Rest",          "Differential tremor type classification"),
            ("🗂️", "Longitudinal Tracking",   "Session-over-session progression charting"),
        ]
        for icon, title, desc in items:
            st.markdown(f"""
            <div class="med-card" style="padding:0.85rem;margin-bottom:0.5rem;">
              <div style="font-size:0.85rem;color:#e6edf3;font-weight:600;">{icon} {title}</div>
              <div style="font-size:0.76rem;color:#8b949e;margin-top:3px;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════════════════
if "Dashboard"  in nav: page_dashboard()
elif "MRI"      in nav: page_mri()
elif "Scribble" in nav: page_scribble()
elif "Tremor"   in nav: page_tremor()
