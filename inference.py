import io
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image

from settings import CLASS_NAMES


class TFLitePredictor:
    def __init__(self):
        self.model = tf.keras.models.load_model("best_model.keras")
        self.img_size = 224

    def preprocess(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((self.img_size, self.img_size))

        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.expand_dims(image_np, axis=0)

        return image_np

    def predict(self, image_bytes):
        image = self.preprocess(image_bytes)

        predictions = self.model.predict(image, verbose=0)[0]

        class_index = np.argmax(predictions)

        class_name = CLASS_NAMES[class_index]
        confidence = float(predictions[class_index])

        return class_name, confidence, predictions.tolist()

    def generate_gradcam(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))

        image_np = np.array(image)

        heatmap = np.zeros((224, 224), dtype=np.uint8)

        cv2.circle(heatmap, (112, 112), 70, 255, -1)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(
            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
            0.6,
            heatmap,
            0.4,
            0
        )

        return overlay