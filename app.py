import io
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
# CLASS NAMES
# ─────────────────────────────
class_names = [
    "Corn_(maize)___Common_rust_",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy"
]

# ─────────────────────────────
# LOAD MODEL
# ─────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

model = load_model()

# ─────────────────────────────
# PREPROCESS IMAGE
# ─────────────────────────────
def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    img = np.array(img)

    # remove alpha channel if exists
    if img.shape[-1] == 4:
        img = img[..., :3]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ─────────────────────────────
# PREDICTION
# ─────────────────────────────
def predict(image: Image.Image):
    img = preprocess(image)
    preds = model.predict(img, verbose=0)[0]

    idx = np.argmax(preds)
    confidence = float(preds[idx])
    label = class_names[idx]

    return label, confidence

# ─────────────────────────────
# UI HEADER
# ─────────────────────────────
st.markdown(
    """
    <h1 style='text-align:center; color:#22c55e;'>🌱 Zar3y AI Studio</h1>
    <p style='text-align:center; color:#94a3b8;'>
    Crop Disease Detection using Deep Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ─────────────────────────────
# INPUT SECTION
# ─────────────────────────────
col1, col2 = st.columns(2)

uploaded_file = col1.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

camera_file = col2.camera_input("Or Capture Image")

file = uploaded_file or camera_file

# ─────────────────────────────
# MAIN LOGIC
# ─────────────────────────────
if file:

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

    st.markdown("---")

    with st.spinner("🔍 Running AI Model..."):
        label, conf = predict(image)

    # ───────── RESULT ─────────
    st.markdown("## 🧠 Prediction Result")

    st.success(f"Prediction: **{label}**")
    st.write(f"Confidence: **{conf * 100:.2f}%**")

    st.progress(int(conf * 100))

    # ───────── STATUS ─────────
    if "healthy" in label.lower():
        st.success("🌿 Plant is HEALTHY")
    else:
        st.error("⚠️ Disease Detected")

    # ───────── DEBUG ─────────
    with st.expander("🔬 Debug Info"):
        st.write("Class Index:", np.argmax(model.predict(preprocess(image), verbose=0)))
        st.write("Raw Label:", label)

else:
    st.info("Upload or capture a leaf image to start analysis 🌱")
