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
        ["🏠  Dashboard", "🧲  MRI Analysis", "✍️  Scribble Test"],
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

    </div>
    """, unsafe_allow_html=True)



# ─── Model loader (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load saved .h5 if available, otherwise return untrained base."""
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam

        # Try to load previously trained model first
        search_paths = [
            Path("parkinsons_resnet50.h5"),
            Path(r"C:\Users\LENOVO\Downloads\archive (5)\parkinsons_resnet50.h5"),
            Path(r"C:\Users\LENOVO\Downloads\parkinsons_resnet50.h5"),
        ]
        for p in search_paths:
            if p.exists():
                model = tf.keras.models.load_model(str(p))
                return model, None, f"trained:{p}"

        # No saved model — build base (needs training before inference)
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers[:140]:
            layer.trainable = False
        for layer in base_model.layers[140:]:
            layer.trainable = True
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
        return model, None, "untrained"
    except Exception as e:
        return None, str(e), "error"


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


    if True:
        st.markdown('<div class="section-header">Diagnostic Modules</div>', unsafe_allow_html=True)
        modules = [
            ("🧲", "MRI Neuroimaging Analysis", "active",
             "Deep learning classification of T1/T2-weighted and diffusion MRI scans using ResNet-50 transfer learning. Supports Grad-CAM saliency visualisation."),
            ("✍️", "Handwriting / Scribble Test", "pending",
             "Automated analysis of micrographia and spiral-drawing irregularities — established early biomarkers for dopaminergic pathway degeneration."),
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
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.35, 0.05,
                              help="Probability above this value = Parkinson's positive.")

    if uploaded is not None:
        st.markdown("---")
        with st.spinner("🔬 Preprocessing image and loading model…"):
            model, err, model_status = load_model()
            img_array, pil_img = preprocess_image(uploaded)

        if err:
            st.error(f"⚠️ Model load failed: {err}")
            model = None

        # Show model status banner
        if model_status == "untrained":
            st.warning("⚠️ **No trained model found.** Predictions will be unreliable. Please train the model first using the panel below, then restart the app.")
        elif model_status.startswith("trained:"):
            st.success(f"✅ Loaded trained model: `{model_status.replace('trained:','')}`")

        if img_array is None:
            st.error(f"⚠️ Image preprocessing failed: {pil_img}")
            return

        # ── Run inference ──
        if model is not None:
            with st.spinner("🧠 Running inference…"):
                raw_prob = float(model.predict(img_array, verbose=0)[0][0])
        else:
            st.error("Cannot run inference — model failed to load.")
            return

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




def run_training(data_path, epochs, batch_size, test_split):
    """Run the full fixed training pipeline with class weights and proper MRI augmentation."""
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
        from sklearn.utils.class_weight import compute_class_weight
        import seaborn as sns

        path = Path(data_path)
        if not path.exists():
            st.error(f"❌ Directory not found: {data_path}")
            st.info("Make sure the path is exactly: C:\\Users\\LENOVO\\Downloads\\archive (5)\\parkinsons_dataset")
            return

        # ── Load file paths ──
        imgs = list(path.glob("*/*.png")) + list(path.glob("*/*.jpg")) + list(path.glob("*/*.jpeg"))
        if not imgs:
            st.error("❌ No images found. Make sure subfolders are named 'parkinson' and 'normal'.")
            return

        df = pd.DataFrame({"path": imgs})
        df["disease"] = df["path"].map(lambda x: x.parent.stem.lower().strip())
        df["path_str"] = df["path"].astype(str)

        class_counts = df["disease"].value_counts()
        st.markdown(f"""
        <div class="med-card-accent">
          <div style="font-size:0.85rem;color:#3fb950;font-weight:600;">✅ Dataset Loaded</div>
          <div style="font-size:0.82rem;color:#8b949e;margin-top:6px;">
            Total: <b style="color:#e6edf3;">{len(df)}</b> images &nbsp;|&nbsp;
            Classes: <b style="color:#e6edf3;">{df['disease'].unique().tolist()}</b><br>
            {' &nbsp;|&nbsp; '.join([f"<b style='color:#e6edf3;'>{k}</b>: {v}" for k,v in class_counts.items()])}
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Stratified split ──
        X_train, X_test, y_train, y_test = train_test_split(
            df["path_str"], df["disease"],
            test_size=test_split, random_state=42, stratify=df["disease"]
        )

        # ── Class weights to fix imbalance bias ──
        classes = np.array(sorted(df["disease"].unique()))
        weights = compute_class_weight("balanced", classes=classes, y=df["disease"])
        class_weight_dict = dict(enumerate(weights))
        st.markdown(f'<div style="font-size:0.8rem;color:#8b949e;">⚖️ Class weights applied: {dict(zip(classes, weights.round(3)))}</div>', unsafe_allow_html=True)

        # ── MRI-specific augmentation (NO horizontal flip — brain asymmetry matters) ──
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=10,          # subtle rotation only
            width_shift_range=0.05,     # minimal shift
            height_shift_range=0.05,
            zoom_range=0.1,             # slight zoom
            brightness_range=[0.9, 1.1],
            fill_mode="nearest",
            # horizontal_flip=False  ← intentionally omitted for MRI
        )
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_df = pd.DataFrame({"path": X_train, "disease": y_train})
        test_df  = pd.DataFrame({"path": X_test,  "disease": y_test})

        train_gen = train_datagen.flow_from_dataframe(
            train_df, x_col="path", y_col="disease",
            target_size=(224, 224), batch_size=batch_size,
            class_mode="binary", shuffle=True
        )
        test_gen = test_datagen.flow_from_dataframe(
            test_df, x_col="path", y_col="disease",
            target_size=(224, 224), batch_size=batch_size,
            class_mode="binary", shuffle=False   # must be False for confusion matrix
        )

        # Show which label maps to 0/1 so user knows orientation
        st.markdown(f'<div style="font-size:0.8rem;color:#8b949e;margin-bottom:8px;">🗂️ Class index mapping: {train_gen.class_indices}</div>', unsafe_allow_html=True)

        # ── Build model — unfreeze more layers for MRI fine-tuning ──
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

        # Freeze first 140 layers, unfreeze last ~10 ResNet blocks
        for layer in base.layers[:140]:
            layer.trainable = False
        for layer in base.layers[140:]:
            layer.trainable = True

        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        out = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=base.input, outputs=out)
        model.compile(
            optimizer=Adam(learning_rate=1e-4),  # lower LR for fine-tuning
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # ── Callbacks ──
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-7, verbose=0
        )
        early_stop = EarlyStopping(
            monitor="val_loss", patience=7,
            restore_best_weights=True, verbose=0
        )

        # ── Train with live charts ──
        progress_bar     = st.progress(0)
        status_text      = st.empty()
        chart_placeholder = st.empty()
        acc_hist, val_acc_hist, loss_hist, val_loss_hist = [], [], [], []

        for epoch in range(1, epochs + 1):
            status_text.markdown(
                f'<span style="color:#58a6ff;font-size:0.85rem;">🔄 Epoch {epoch}/{epochs} — training…</span>',
                unsafe_allow_html=True
            )
            h = model.fit(
                train_gen,
                validation_data=test_gen,
                epochs=1,
                class_weight=class_weight_dict,   # ← KEY FIX
                callbacks=[reduce_lr],
                verbose=0
            )
            acc_hist.append(h.history["accuracy"][0])
            val_acc_hist.append(h.history["val_accuracy"][0])
            loss_hist.append(h.history["loss"][0])
            val_loss_hist.append(h.history["val_loss"][0])
            progress_bar.progress(epoch / epochs)

            if epoch % 2 == 0 or epoch == epochs:
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
                ax1.plot(acc_hist,     color="#58a6ff", label="Train",      linewidth=2)
                ax1.plot(val_acc_hist, color="#3fb950", label="Validation", linewidth=2, linestyle="--")
                ax1.set_title("Accuracy"); ax1.set_ylim(0, 1)
                ax1.legend(facecolor="#161b22", labelcolor="#8b949e")
                ax2.plot(loss_hist,     color="#f85149", label="Train",      linewidth=2)
                ax2.plot(val_loss_hist, color="#d29922", label="Validation", linewidth=2, linestyle="--")
                ax2.set_title("Loss")
                ax2.legend(facecolor="#161b22", labelcolor="#8b949e")
                plt.tight_layout()
                chart_placeholder.pyplot(fig)
                plt.close(fig)

        status_text.markdown('<span style="color:#3fb950;font-size:0.9rem;font-weight:600;">✅ Training complete!</span>', unsafe_allow_html=True)

        # ── Evaluation ──
        preds_raw   = model.predict(test_gen, verbose=0).flatten()
        # Use 0.5 threshold (class weights already balanced bias during training)
        preds       = (preds_raw >= 0.5).astype(int)
        true_labels = test_gen.classes
        acc = accuracy_score(true_labels, preds)

        # Per-class accuracy
        class_idx   = train_gen.class_indices          # e.g. {'normal':0, 'parkinson':1}
        idx_to_class = {v: k for k, v in class_idx.items()}

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="result-negative" style="margin-top:1rem;text-align:center;">
              <div class="label-text">Overall Validation Accuracy</div>
              <div style="font-size:2.2rem;font-weight:700;color:#3fb950;">{acc*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        with col_b:
            # Show how many of each class were correctly predicted
            for cls_name, cls_idx in class_idx.items():
                mask     = true_labels == cls_idx
                cls_acc  = accuracy_score(true_labels[mask], preds[mask]) * 100
                color    = "#f85149" if "park" in cls_name else "#3fb950"
                st.markdown(f"""
                <div style="padding:0.5rem 0;border-bottom:1px solid #21262d;">
                  <span class="label-text">{cls_name.capitalize()} accuracy</span>
                  <span style="float:right;color:{color};font-weight:700;">{cls_acc:.1f}%</span>
                </div>""", unsafe_allow_html=True)

        # ── Confusion matrix ──
        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Confusion Matrix</div>', unsafe_allow_html=True)
        labels_sorted = [idx_to_class[i] for i in sorted(idx_to_class)]
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor("#161b22")
        ax.set_facecolor("#0d1117")
        sns.heatmap(
            confusion_matrix(true_labels, preds),
            annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_sorted, yticklabels=labels_sorted,
            ax=ax, linewidths=1, linecolor="#30363d"
        )
        ax.set_xlabel("Predicted", color="#8b949e")
        ax.set_ylabel("Actual",    color="#8b949e")
        ax.tick_params(colors="#8b949e")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        report = classification_report(true_labels, preds, target_names=labels_sorted)
        st.code(report, language="text")

        # ── Save model ──
        save_path = str(Path(data_path).parent / "parkinsons_resnet50.h5")
        model.save(save_path)
        st.success(f"💾 Model saved → {save_path}")
        st.info("Next time you open the app, place this .h5 file in the same folder as parkinsons_app.py and it will load automatically.")

    except ImportError as e:
        st.error(f"Missing dependency: {e}")
        st.code("pip install tensorflow scikit-learn seaborn opencv-python", language="bash")
    except Exception as e:
        import traceback
        st.error(f"Training error: {e}")
        st.code(traceback.format_exc(), language="text")


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
# ROUTER
# ════════════════════════════════════════════════════════════════════════════
if "Dashboard"  in nav: page_dashboard()
elif "MRI"      in nav: page_mri()
elif "Scribble" in nav: page_scribble()

