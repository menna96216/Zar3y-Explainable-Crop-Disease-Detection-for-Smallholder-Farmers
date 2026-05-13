import io
import base64
import requests
import streamlit as st

from PIL import Image

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Zar3y",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Zar3y")
st.subheader("AI Crop Disease Detection")

uploaded_file = st.file_uploader(
    "Upload leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):

        uploaded_file.seek(0)

        files = {
            "file": (
                "leaf.jpg",
                uploaded_file.read(),
                "image/jpeg"
            )
        }

        response = requests.post(
            f"{BACKEND_URL}/predict",
            files=files
        )

    if response.status_code == 200:

        result = response.json()

        with col2:

            st.success(result["display_name"])

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.write("### Description")
            st.write(result["description"])

            st.write("### Recommended Action")
            st.write(result["next_step"])

            st.write("### Inference Time")
            st.write(f"{result['inference_ms']} ms")

        gradcam_bytes = base64.b64decode(
            result["grad_cam_base64"]
        )

        gradcam_image = Image.open(
            io.BytesIO(gradcam_bytes)
        )

        st.write("## Grad-CAM")

        g1, g2 = st.columns(2)

        with g1:
            st.image(image, caption="Original")

        with g2:
            st.image(gradcam_image, caption="Heatmap")

    else:
        st.error(response.text)
