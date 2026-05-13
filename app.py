import io
import json
import numpy as np
import streamlit as st
from PIL import Image
import random

# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="Zar3y — Crop Disease AI",
    page_icon="🌱",
    layout="wide"
)

# ─────────────────────────────
# LOAD CLASSES
# ─────────────────────────────
CLASS_PATH = "class_names.json"

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# ─────────────────────────────
# IMAGE PREPROCESSING (optional - kept for future upgrade)
# ─────────────────────────────
def preprocess(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image)

    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    image = image / 255.0
    return image

# ─────────────────────────────
# PREDICTION (MOCK AI for Streamlit Cloud)
# ─────────────────────────────
def predict(image):
    label = random.choice(class_names)
    confidence = round(random.uniform(0.75, 0.99), 2)
    return label, confidence

# ─────────────────────────────
# UI
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
    st.progress(int(conf * 100))
    st.write(f"{conf * 100:.2f}%")

    # STATUS
    if "healthy" in label.lower():
        st.success("🌿 The plant is HEALTHY")
    else:
        st.error("⚠️ Disease Detected")

    # EXTRA INFO
    st.markdown("---")
    st.markdown("### 📌 Notes")
    st.write("This is a demo version running on Streamlit Cloud (no backend, no TensorFlow).")

else:
    st.info("👆 Upload or capture an image to start analysis")
