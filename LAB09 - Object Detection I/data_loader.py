from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
# โน้ต: ใช้รูปจาก LAB10 ตรง ๆ ไม่ต้อง copy dataset มาซ้ำ
DATASET = ROOT.parent / "LAB10-Object Detection II" / "dataset"
CONFIG = yaml.safe_load((DATASET / "data.yaml").read_text(encoding="utf-8"))
CLASSES = CONFIG["names"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 64


def split_paths(split):
    image_dir = DATASET / CONFIG[split]
    return image_dir, image_dir.parent / "labels"


def preprocess(crop):
    # โน้ต: คง 64x64 เหมือนเดิม แล้วแปลง BGR -> RGB และหาร 255
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def read_boxes(image_path, labels_dir, image):
    label = labels_dir / (image_path.stem + ".txt")
    if not label.is_file():
        return []
    height, width = image.shape[:2]
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, xc, yc, bw, bh = map(float, parts)
        if not class_id.is_integer() or not 0 <= class_id < NUM_CLASSES:
            continue
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in [xc, yc, bw, bh]):
            continue
        x1, y1 = max(0, int((xc - bw / 2) * width)), max(0, int((yc - bh / 2) * height))
        x2, y2 = min(width, int((xc + bw / 2) * width)), min(height, int((yc + bh / 2) * height))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2, int(class_id)))
    return boxes


def load_crops(split="train", limit=0):
    image_dir, labels_dir = split_paths(split)
    if not image_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset: {image_dir}, {labels_dir}")
    files = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if limit > 0:
        files = files[:limit]
    crops, labels = [], []
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        for x1, y1, x2, y2, class_id in read_boxes(path, labels_dir, image):
            crops.append(preprocess(image[y1:y2, x1:x2]))
            labels.append(class_id)
    if not crops:
        raise ValueError(f"No labeled crops in {image_dir}")
    print(f"{split}: {len(files)} images -> {len(crops)} crops")
    return np.asarray(crops, dtype=np.float32), np.asarray(labels, dtype=np.int32)
