import io
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="Zar3y — Crop Disease AI",
    page_icon="🌱",
    layout="wide"
)

# ─────────────────────────────
# LOAD MODEL + CLASSES
# ─────────────────────────────
MODEL_PATH = "best_model.keras"
CLASS_PATH = "class_names.json"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# ─────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────
def preprocess(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image)

    # handle RGBA images
    if image.shape[-1] == 4:
        image = image[..., :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ─────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────
def predict(image):
    img = preprocess(image)
    preds = model.predict(img, verbose=0)[0]

    idx = np.argmax(preds)
    confidence = float(preds[idx])

    label = class_names[idx]

    return label, confidence

# ─────────────────────────────
# UI DESIGN
# ─────────────────────────────
st.title("🌱 Zar3y — AI Crop Disease Detection")
st.write("Upload or capture a leaf image to detect plant disease instantly.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("Or use Camera")

image_file = uploaded_file or camera_file

# ─────────────────────────────
# MAIN LOGIC
# ─────────────────────────────
if image_file:

    image = Image.open(image_file)

    st.image(image, caption="Input Image", use_container_width=True)

    st.markdown("---")

    with st.spinner("🔍 Analyzing leaf..."):
        label, conf = predict(image)

    # RESULT
    st.success("Prediction Completed!")

    st.markdown("## 🧠 Result")
    st.markdown(f"### **{label}**")

    st.markdown("## 🎯 Confidence")
    st.progress(conf)
    st.write(f"{conf * 100:.2f}%")

    # STATUS
    if "healthy" in label.lower():
        st.success("🌿 The plant is HEALTHY")
    else:
        st.error("⚠️ Disease Detected")

    # EXTRA INFO
    st.markdown("---")
    st.markdown("### 📌 Notes")
    st.write("This prediction is based on a trained CNN model (MobileNetV3 / custom CNN).")

else:
    st.info("👆 Upload or capture an image to start analysis")
