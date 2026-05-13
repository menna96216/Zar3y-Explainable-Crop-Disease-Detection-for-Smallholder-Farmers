import io
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="Zar3y AI Studio",
    page_icon="🌱",
    layout="wide"
)

# ─────────────────────────────
# LOAD MODEL
# ─────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ─────────────────────────────
# PREPROCESS
# ─────────────────────────────
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img)

    if img.shape[-1] == 4:
        img = img[..., :3]

    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ─────────────────────────────
# PREDICT (REAL AI)
# ─────────────────────────────
def predict(image):
    img = preprocess(image)
    preds = model.predict(img, verbose=0)[0]

    idx = np.argmax(preds)
    confidence = float(preds[idx])
    label = class_names[idx]

    return label, confidence

# ─────────────────────────────
# UI HEADER
# ─────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#22c55e;'>🌱 Zar3y AI Studio</h1>
<p style='text-align:center; color:#94a3b8;'>Real AI Crop Disease Detection</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────
# INPUT
# ─────────────────────────────
col1, col2 = st.columns(2)

uploaded_file = col1.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])
camera_file = col2.camera_input("Or Capture Image")

file = uploaded_file or camera_file

# ─────────────────────────────
# MAIN
# ─────────────────────────────
if file:

    image = Image.open(file)
    st.image(image, caption="Input Image", use_container_width=True)

    st.markdown("---")

    with st.spinner("🔍 Running AI Model..."):
        label, conf = predict(image)

    # ───────── RESULT ─────────
    st.markdown("## 🧠 Prediction Result")

    st.success(f"**{label}**")
    st.write(f"Confidence: {conf*100:.2f}%")
    st.progress(int(conf * 100))

    # ───────── STATUS ─────────
    if "healthy" in label.lower():
        st.success("🌿 Plant is HEALTHY")
    else:
        st.error("⚠️ Disease Detected")

    # ───────── RAW DEBUG ─────────
    st.markdown("### 🔬 Debug Info")
    st.write("Raw Prediction:", label)

else:
    st.info("Upload or capture image to start analysis 🌱")
