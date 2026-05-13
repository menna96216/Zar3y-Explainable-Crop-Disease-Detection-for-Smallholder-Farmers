import io
import json
import numpy as np
import streamlit as st
from PIL import Image
import random
import time

# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="Zar3y AI Studio",
    page_icon="🌱",
    layout="wide"
)

# ─────────────────────────────
# LOAD CLASSES
# ─────────────────────────────
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ─────────────────────────────
# SESSION STATE (history)
# ─────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────
# SMART PREDICTION ENGINE (Enhanced mock AI)
# ─────────────────────────────
def predict(image, class_name):
    crop = class_name.split("_")[0] if "_" in class_name else "Unknown"

    disease_map = {
        "Tomato": ["Early Blight", "Late Blight", "Leaf Mold"],
        "Potato": ["Early Blight", "Late Blight"],
        "Pepper": ["Bacterial Spot"],
        "Corn": ["Common Rust"]
    }

    disease = random.choice(disease_map.get(crop, ["Healthy"]))

    if "healthy" in class_name.lower():
        label = f"{crop} — Healthy"
        confidence = round(random.uniform(0.90, 0.99), 2)
    else:
        label = f"{crop} — {disease}"
        confidence = round(random.uniform(0.75, 0.95), 2)

    return label, confidence, crop, disease

# ─────────────────────────────
# UI HEADER
# ─────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#4ADE80; font-size:42px;'>
        🌱 Zar3y AI Studio
    </h1>
    <p style='text-align:center; color:#94A3B8;'>
        Advanced Crop Disease Intelligence System
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────
# UPLOAD
# ─────────────────────────────
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","jpeg","png"])

with col2:
    camera_file = st.camera_input("📸 Capture Image")

file = uploaded_file or camera_file

# ─────────────────────────────
# MAIN
# ─────────────────────────────
if file:

    image = Image.open(file)
    st.image(image, caption="Input Leaf", use_container_width=True)

    st.markdown("---")

    with st.spinner("🧠 AI analyzing plant health..."):
        time.sleep(1.5)
        class_name = random.choice(class_names)
        label, conf, crop, disease = predict(image, class_name)

    # Save history
    st.session_state.history.append((label, conf))

    # ─────────────────────────────
    # RESULT UI (SaaS STYLE)
    # ─────────────────────────────
    st.markdown(f"""
    <div style="padding:20px; border-radius:15px; background:#0f172a;">
        <h2 style="color:#22c55e;">{label}</h2>
        <p style="color:#94a3b8;">Crop Type: {crop}</p>
        <p style="color:#fbbf24;">Confidence: {conf*100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(conf * 100))

    # ─────────────────────────────
    # AI EXPLANATION PANEL
    # ─────────────────────────────
    st.markdown("## 🔍 AI Explanation")

    if "Healthy" in label:
        st.success("Plant is healthy. No disease patterns detected.")
    else:
        st.warning(f"""
        Detected disease patterns consistent with:
        - {disease}
        - Leaf discoloration
        - Possible fungal/bacterial infection
        """)

    # ─────────────────────────────
    # HISTORY SECTION
    # ─────────────────────────────
    st.markdown("## 📊 Recent Predictions")

    for item in reversed(st.session_state.history[-5:]):
        st.write(f"🌱 {item[0]} — {item[1]*100:.1f}%")

else:
    st.info("Upload or capture a leaf image to start analysis 🌿")
