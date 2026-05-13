import base64
import time
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import TFLitePredictor
from settings import DISEASE_INFO, DISPLAY_NAMES

app = FastAPI(
    title="Zar3y API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = TFLitePredictor()


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image")

    start = time.perf_counter()

    image_bytes = await file.read()

    class_name, confidence, probs = predictor.predict(image_bytes)

    overlay = predictor.generate_gradcam(image_bytes)

    _, buffer = cv2.imencode(".png", overlay)

    gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

    info = DISEASE_INFO.get(class_name, {})

    elapsed_ms = (time.perf_counter() - start) * 1000

    return JSONResponse(
        content={
            "class_name": class_name,
            "display_name": DISPLAY_NAMES.get(class_name, class_name),
            "confidence": round(confidence, 4),
            "description": info.get("description", ""),
            "next_step": info.get("next_step", ""),
            "grad_cam_base64": gradcam_base64,
            "inference_ms": round(elapsed_ms, 1),
        }
    )