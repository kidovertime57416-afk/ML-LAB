import json
import random
from pathlib import Path

import cv2
import numpy as np

from data_loader import CLASSES, preprocess, read_boxes, split_paths
from experiments import load_model


def test_image(output_dir, split="test", image_path=None):
    output_dir = Path(output_dir)
    image_dir, labels_dir = split_paths(split)
    if image_path:
        path = Path(image_path)
    else:
        files = sorted(p for p in image_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                       and (labels_dir / (p.stem + ".txt")).is_file())
        # โน้ต: สุ่มด้วย seed เดียวกัน ให้ R-CNN กับ Fast R-CNN ดูภาพเดียวกัน
        path = random.Random(42).choice(files)
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(str(path))
    boxes = read_boxes(path, labels_dir, frame)
    if not boxes:
        raise ValueError(f"No ground-truth boxes in {path}")
    model = load_model(output_dir)
    crops = np.asarray([preprocess(frame[y1:y2, x1:x2])
                        for x1, y1, x2, y2, _ in boxes], dtype=np.float32)
    probabilities = model.predict(crops, verbose=0)
    predictions = []
    for (x1, y1, x2, y2, actual), row in zip(boxes, probabilities):
        predicted = int(np.argmax(row))
        confidence = float(row[predicted])
        # โน้ต: กรอบนี้มาจาก label ไม่ใช่กรอบที่โมเดลหาเอง สีแดงคือทายคลาสผิด
        color = (0, 160, 0) if predicted == actual else (0, 0, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{CLASSES[predicted]} {confidence:.2f}",
                    (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        predictions.append({"box": [x1, y1, x2, y2], "actual": CLASSES[actual],
                            "predicted": CLASSES[predicted], "confidence": confidence})
    cv2.imwrite(str(output_dir / "image_result.jpg"), frame)
    (output_dir / "image_predictions.json").write_text(
        json.dumps({"source_image": path.name, "boxes_from": "ground_truth", "predictions": predictions}, indent=2),
        encoding="utf-8")
    print(f"Saved {len(boxes)} crop predictions for {path.name}")


def detect_demo(model, frame, threshold=0.85, max_regions=25):
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
    crops = np.asarray([preprocess(frame[y1:y2, x1:x2]) for x1,y1,x2,y2 in boxes])
    scores = model.predict(crops, verbose=0)
    return [(*box, int(row.argmax()), float(row.max())) for box, row in zip(boxes, scores)
            if row.max() >= threshold]


def draw(frame, predictions):
    result = frame.copy()
    for x1, y1, x2, y2, class_id, confidence in predictions:
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result, f"{CLASSES[class_id]} {confidence:.2f}",
                    (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return result


def webcam_demo(output_dir, camera=0, threshold=0.85, regions=25):
    model = load_model(output_dir)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera {camera}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # โน้ต: คง sliding-window demo เดิม ไม่มี background class จึงอาจทายเกิน
            predictions = detect_demo(model, frame, threshold, regions)
            cv2.imshow("Aquarium crop-classification demo", draw(frame, predictions))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
