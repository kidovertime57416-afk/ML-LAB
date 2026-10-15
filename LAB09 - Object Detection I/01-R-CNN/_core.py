from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "model.keras"
CLASSES = ["person", "rider", "car", "bus", "truck", "bike", "motor",
           "traffic light", "traffic sign", "train"]
IMG_SIZE = 64


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def detect_demo(model, frame, threshold=0.85, max_regions=25):
    # สร้างกรอบแบบ sliding window สำหรับ demo การจำแนก crop
    height, width = frame.shape[:2]
    boxes = []
    for scale in (0.2, 0.35, 0.5):
        bw, bh = max(16, int(width * scale)), max(16, int(height * scale))
        for cy in np.linspace(bh // 2, max(bh // 2, height - bh // 2), 4):
            for cx in np.linspace(bw // 2, max(bw // 2, width - bw // 2), 5):
                x1, y1 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
                x2, y2 = min(width, x1 + bw), min(height, y1 + bh)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        if len(boxes) >= max(1, max_regions):
            break
    boxes = boxes[:max(1, max_regions)]
    if not boxes:
        return []
    # เตรียมภาพเหมือนตอน train: RGB ขนาด 64x64 และ pixel ช่วง 0-1
    crops = np.asarray([
        cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (IMG_SIZE, IMG_SIZE)),
                     cv2.COLOR_BGR2RGB)
        for x1, y1, x2, y2 in boxes
    ], dtype=np.float32) / 255.0
    scores = model.predict(crops, verbose=0)
    predictions = []
    for box, score in zip(boxes, scores):
        class_id = int(np.argmax(score))
        confidence = float(score[class_id])
        if confidence >= threshold:
            predictions.append((*box, class_id, confidence))
    return predictions


def draw(frame, predictions):
    # วาดกรอบ ชื่อคลาส และ confidence ของ crop ที่ผ่าน threshold
    result = frame.copy()
    for x1, y1, x2, y2, class_id, confidence in predictions:
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, f"{CLASSES[class_id]} {confidence:.2f}",
                    (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
    return result
