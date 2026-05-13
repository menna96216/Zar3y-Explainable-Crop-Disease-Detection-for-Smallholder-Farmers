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
# LOAD TFLITE MODEL
# ─────────────────────────────
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# class names
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

    # مهم جدًا في quantized models
    input_dtype = input_details[0]["dtype"]

    if input_dtype == np.uint8:
        img = img.astype(np.uint8)
    else:
        img = img / 255.0
        img = img.astype(np.float32)

    return np.expand_dims(img, axis=0)

# ─────────────────────────────
# PREDICT
# ─────────────────────────────
def predict(image):
    img = preprocess(image)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    idx = np.argmax(preds)
    confidence = float(preds[idx])
    label = class_names[idx]

    return label, confidence

# ─────────────────────────────
# UI
# ─────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#22c55e;'>🌱 Zar3y AI Studio</h1>
<p style='text-align:center;'>TFLite Quantized Crop Disease Detection</p>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

uploaded_file = col1.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])
camera_file = col2.camera_input("Or Capture Image")

file = uploaded_file or camera_file

if file:

    image = Image.open(file)
    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("🔍 Running AI Model..."):
        label, conf = predict(image)

    st.markdown("## 🧠 Prediction Result")
    st.success(f"**{label}**")
    st.write(f"Confidence: {conf*100:.2f}%")
    st.progress(int(conf * 100))

    if "healthy" in label.lower():
        st.success("🌿 Plant is HEALTHY")
    else:
        st.error("⚠️ Disease Detected")

else:
    st.info("Upload or capture image to start analysis 🌱")
