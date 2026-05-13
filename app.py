"""
Zar3y — Streamlit Demo (Requirement 5) - Premium SaaS UI Redesign

Features:
1. Upload a photo or use device camera
2. Send image to FastAPI backend (POST /predict)
3. Display prediction results:
   - Predicted disease class + confidence
   - Grad-CAM overlay (what the model looked at)
   - Plain-language description of the disease
   - Next-step suggestion
"""

import io
import base64
import time
import requests
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{BACKEND_URL}/predict"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Zar3y — AI Crop Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS for Dark SaaS Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Base dark theme and font settings */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    /* Typography improvements */
    h1, h2, h3, h4, h5, h6 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    p {
        color: #94A3B8;
        line-height: 1.6;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1.5rem;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: rgba(34, 197, 94, 0.2);
    }

    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(135deg, #22C55E 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: inline-block;
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        letter-spacing: 0.025em;
    }
    .status-badge.online {
        background-color: rgba(34, 197, 94, 0.1);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    .status-badge.offline {
        background-color: rgba(239, 68, 68, 0.1);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    /* Confidence Bars */
    .confidence-container {
        width: 100%;
        background-color: #1E293B;
        border-radius: 9999px;
        height: 0.5rem;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 1s ease-in-out;
    }
    .fill-high { background: linear-gradient(90deg, #22C55E, #10B981); }
    .fill-medium { background: linear-gradient(90deg, #F59E0B, #D97706); }
    .fill-low { background: linear-gradient(90deg, #EF4444, #DC2626); }

    /* Result severity colors */
    .text-healthy { color: #4ADE80 !important; }
    .text-warning { color: #F87171 !important; }

    /* Custom Streamlit File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(30, 41, 59, 0.4);
        border: 2px dashed #334155;
        border-radius: 16px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #22C55E;
        background-color: rgba(34, 197, 94, 0.05);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    /* Dividers */
    hr {
        border-color: #1E293B;
        margin: 2.5rem 0;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: #F8FAFC;
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8;
        font-weight: 500;
    }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
@st.cache_data(ttl=60)
def check_backend_status():
    """Check if the FastAPI backend is online."""
    try:
        start_time = time.time()
        response = requests.get(HEALTH_ENDPOINT, timeout=3)
        latency = (time.time() - start_time) * 1000
        if response.status_code == 200:
            return True, latency
    except requests.ConnectionError:
        pass
    return False, 0.0

def generate_confidence_bar(confidence):
    """Generate HTML for a styled confidence progress bar."""
    if confidence >= 0.8:
        fill_class = "fill-high"
    elif confidence >= 0.5:
        fill_class = "fill-medium"
    else:
        fill_class = "fill-low"
        
    percentage = int(confidence * 100)
    
    return f"""
    <div class="confidence-container">
        <div class="confidence-fill {fill_class}" style="width: {percentage}%;"></div>
    </div>
    """

# ─────────────────────────────────────────────
# Sidebar Navigation & Info
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<h2 style="margin-bottom: 0;">🌱 Zar3y <span style="font-weight:300; font-size:1rem; color:#94A3B8;">v1.0</span></h2>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.9rem; margin-top: 0;">AI Crop Disease Detection</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # System Status
    st.markdown("### System Status")
    is_online, latency = check_backend_status()
    
    if is_online:
        st.markdown(
            f'<div class="status-badge online" style="margin-bottom: 1rem;">'
            f'<span style="margin-right: 6px;">●</span> API Online ({int(latency)}ms)</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-badge offline" style="margin-bottom: 1rem;">'
            '<span style="margin-right: 6px;">●</span> API Offline</div>', 
            unsafe_allow_html=True
        )
        st.error("Backend not reachable. Start with `uvicorn backend.main:app --port 8000`")

    # Supported Crops Accordion
    with st.expander("🌾 Supported Crops", expanded=True):
        st.markdown("""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
            <div style="background: rgba(30,41,59,0.5); padding: 8px; border-radius: 8px; text-align: center;">🍅 Tomato</div>
            <div style="background: rgba(30,41,59,0.5); padding: 8px; border-radius: 8px; text-align: center;">🥔 Potato</div>
            <div style="background: rgba(30,41,59,0.5); padding: 8px; border-radius: 8px; text-align: center;">🫑 Pepper</div>
            <div style="background: rgba(30,41,59,0.5); padding: 8px; border-radius: 8px; text-align: center;">🌽 Corn</div>
        </div>
        """, unsafe_allow_html=True)

    # Model Info Accordion
    with st.expander("🔬 AI Model Details", expanded=False):
        st.markdown("""
        <ul style="color: #94A3B8; font-size: 0.9rem; padding-left: 1.2rem;">
            <li><b>Architecture:</b> MobileNetV3-Small</li>
            <li><b>Optimization:</b> INT8 Quantized TFLite</li>
            <li><b>Classes:</b> 10 (Healthy + Diseases)</li>
            <li><b>Explainability:</b> Grad-CAM Heatmaps</li>
            <li><b>Inference:</b> CPU Optimized (< 50ms)</li>
        </ul>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; color:#64748B; font-size:0.8rem;'>"
        "Designed for Egyptian smallholder farmers.<br>Running on localhost environment."
        "</div>", 
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────
# Main Header / Hero Section
# ─────────────────────────────────────────────
st.markdown(
    """
    <div style="padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem; line-height: 1.2;">
            Instant AI Diagnosis for <span class="gradient-text">Healthier Crops</span>
        </h1>
        <p style="font-size: 1.2rem; max-width: 800px; margin-top: 1rem;">
            Empowering smallholder farmers with production-grade computer vision. 
            Upload a photo of a leaf to detect diseases, view confidence metrics, 
            and understand AI reasoning through Grad-CAM visualisations.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ─── Dashboard Stats (Mockup metrics for SaaS feel) ───
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(label="Supported Classes", value="10", delta="4 Crops")
with m2:
    st.metric(label="Model Size", value="1.1 MB", delta="INT8 Optimized", delta_color="normal")
with m3:
    st.metric(label="Avg Inference", value="~45 ms", delta="CPU Fast", delta_color="inverse")
with m4:
    st.metric(label="Explainability", value="Grad-CAM", delta="Visual Trust")

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Upload Section
# ─────────────────────────────────────────────
st.markdown('<h3>1. Capture or Upload Leaf Image</h3>', unsafe_allow_html=True)
st.markdown('<p style="margin-bottom: 1.5rem;">Ensure the leaf is well-lit and centered for the most accurate diagnosis.</p>', unsafe_allow_html=True)

col_upload, col_camera = st.columns(2, gap="large")

with col_upload:
    st.markdown("""
    <div class="glass-card" style="height: 100%;">
        <h4 style="margin-top:0; display:flex; align-items:center; gap:8px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
            File Upload
        </h4>
        <p style="font-size:0.9rem; margin-bottom:1rem;">Upload a clear photo from your gallery (.jpg, .png)</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

with col_camera:
    st.markdown("""
    <div class="glass-card" style="height: 100%;">
        <h4 style="margin-top:0; display:flex; align-items:center; gap:8px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>
            Live Camera
        </h4>
        <p style="font-size:0.9rem; margin-bottom:1rem;">Capture a live photo of the leaf directly</p>
    </div>
    """, unsafe_allow_html=True)
    camera_input = st.camera_input("", label_visibility="collapsed")

image_source = uploaded_file or camera_input


# ─────────────────────────────────────────────
# Analysis & Results Section
# ─────────────────────────────────────────────
if image_source is not None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<h3>2. AI Diagnosis Results</h3>', unsafe_allow_html=True)
    
    # Process image
    image = Image.open(image_source)
    
    with st.spinner("Analyzing leaf patterns using MobileNetV3..."):
        try:
            image_source.seek(0)
            file_bytes = image_source.read()
            files = {"file": ("leaf.jpg", file_bytes, "image/jpeg")}
            
            # Simulated slight delay for visual UX if inference is too fast
            start_req = time.time()
            resp = requests.post(PREDICT_ENDPOINT, files=files, timeout=30)
            if (time.time() - start_req) < 0.5:
                time.sleep(0.5)

            if resp.status_code == 200:
                result = resp.json()

                # Extract data
                display_name = result.get("display_name", result["class_name"])
                confidence = result["confidence"]
                description = result["description"]
                next_step = result["next_step"]
                inference_ms = result.get("inference_ms", "—")
                grad_cam_b64 = result.get("grad_cam_base64", "")
                
                is_healthy = "healthy" in result["class_name"].lower()
                status_icon = "🟢" if is_healthy else "🔴"
                text_color_class = "text-healthy" if is_healthy else "text-warning"
                
                # Split crop and disease from display name
                if "—" in display_name:
                    crop_type, disease_name = display_name.split("—", 1)
                else:
                    crop_type, disease_name = "Unknown", display_name

                # Main Results Grid
                r_col1, r_col2 = st.columns([1, 1.5], gap="large")
                
                with r_col1:
                    st.markdown("""
                    <div class="glass-card" style="padding: 1rem; text-align: center;">
                        <span style="color: #94A3B8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">Input Image</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(image, use_container_width=True, output_format="JPEG")
                    st.caption(f"⏱️ Inference Latency: **{inference_ms} ms**")

                with r_col2:
                    # Primary Diagnosis Card
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <span style="background: rgba(255,255,255,0.1); padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; color: #CBD5E1;">{crop_type.strip()}</span>
                                <h2 style="margin-top: 0.5rem; margin-bottom: 0.2rem;" class="{text_color_class}">
                                    {status_icon} {disease_name.strip()}
                                </h2>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.8rem; font-weight: 700; color: #F8FAFC;">{confidence * 100:.1f}%</div>
                                <div style="font-size: 0.8rem; color: #94A3B8;">Confidence Score</div>
                            </div>
                        </div>
                        {generate_confidence_bar(confidence)}
                    </div>
                    """, unsafe_allow_html=True)

                    # Insights & Action Card
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4 style="margin-top: 0; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                            Clinical Description
                        </h4>
                        <p style="margin-bottom: 1.5rem;">{description}</p>
                        
                        <h4 style="margin-top: 0; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                            Recommended Action
                        </h4>
                        <p style="margin-bottom: 0; color: #E2E8F0;">{next_step}</p>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.error(f"Backend Error [{resp.status_code}]: {resp.text}")

        except requests.ConnectionError:
            st.error("❌ Connection to API failed. Ensure backend is running.")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {e}")

    # ─────────────────────────────────────────────
    # Explainability (Grad-CAM)
    # ─────────────────────────────────────────────
    if "result" in locals() and result.get("grad_cam_base64"):
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 1.5rem;">
                <div>
                    <h3 style="margin-bottom: 0;">3. Visual Explainability (Grad-CAM)</h3>
                    <p style="margin-bottom: 0;">Understanding what the AI model focused on to make its decision.</p>
                </div>
                <span style="background: rgba(59, 130, 246, 0.1); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.2); padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">
                    AI Transparency
                </span>
            </div>
            """, 
            unsafe_allow_html=True
        )

        gradcam_bytes = base64.b64decode(result["grad_cam_base64"])
        gradcam_image = Image.open(io.BytesIO(gradcam_bytes))

        # Grad-CAM Side-by-side
        g_col1, g_col2 = st.columns(2, gap="large")
        
        with g_col1:
            st.markdown("""
            <div class="glass-card" style="padding: 1rem; text-align: center; margin-bottom: 0.5rem;">
                <span style="color: #F8FAFC; font-weight: 600;">Original Image</span>
            </div>
            """, unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
        with g_col2:
            st.markdown("""
            <div class="glass-card" style="padding: 1rem; text-align: center; margin-bottom: 0.5rem; border-color: rgba(239, 68, 68, 0.3);">
                <span style="color: #F8FAFC; font-weight: 600;">Attention Heatmap</span>
            </div>
            """, unsafe_allow_html=True)
            st.image(gradcam_image, use_container_width=True)

        st.markdown("""
        <div class="glass-card" style="margin-top: 1rem; padding: 1rem 1.5rem;">
            <p style="margin: 0; font-size: 0.95rem;">
                <strong>How to read this:</strong> The heatmap highlights the exact regions of the leaf that most strongly influenced the model's prediction. 
                <span style="color: #EF4444; font-weight: 600;">Red areas</span> indicate high attention (usually lesions or spots), while <span style="color: #3B82F6; font-weight: 600;">blue areas</span> indicate low attention (usually healthy tissue or background).
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # ─── Empty State / Hassan Scenario ───
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 4rem 2rem; margin-top: 2rem; border-style: dashed;">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#22C55E" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 1rem; opacity: 0.8;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
            <h3 style="margin-bottom: 0.5rem;">Waiting for Image Input</h3>
            <p style="max-width: 500px; margin: 0 auto; color: #94A3B8;">
                Please upload a photo of a leaf or capture one using your device's camera to begin the AI analysis process.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    with st.expander("📖 Read Hassan's Story (Project Context)", expanded=False):
        st.markdown(
            """
            > **Hassan, a smallholder farmer in Beheira**, sees yellow spots on his tomato leaves. 
            > He doesn't know if it's nutrient deficiency, fungal infection, or pest damage. 
            > He has 30 minutes before he needs to decide whether to spray fungicide. 
            > He takes a photo with his phone...
            
            This dashboard represents the frontend of the **Zar3y** application, designed to provide 
            Hassan with an immediate, accurate, and explainable diagnosis.
            """
        )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; padding-top: 2rem; border-top: 1px solid #1E293B; margin-top: 2rem;">
        <p style="color: #64748B; font-size: 0.85rem; margin-bottom: 0.2rem;">
            Zar3y — End-to-End AI Engineering Pipeline
        </p>
        <p style="color: #475569; font-size: 0.8rem;">
            Computer Vision • MobileNetV3 • TFLite Quantization • FastAPI • Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
