from pathlib import Path
import json

import yaml

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
RUNS_DIR = ROOT / "runs" / "aquarium"
TRAIN_PROJECT = RUNS_DIR / "training"
SELECTED_MODEL = RUNS_DIR / "selected_model.json"


def check_dataset(splits=("train", "val")):
    # เช็กโฟลเดอร์ก่อนเริ่มเทรน จะได้รู้ทันทีว่าข้อมูลขาดตรงไหน
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    dataset_root = Path(data.get("path") or DATA_YAML.parent)
    if not dataset_root.is_absolute():
        dataset_root = DATA_YAML.parent / dataset_root
    for split in splits:
        images = dataset_root / data[split]
        labels = images.parent / "labels"
        if not images.is_dir() or not labels.is_dir():
            raise FileNotFoundError(f"Missing {split} images/labels: {images}, {labels}")
        image_files = [p for p in images.iterdir()
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not image_files:
            raise ValueError(f"No images in {images}")
        # ภาพ background อาจไม่มี label แต่ทั้ง split ต้องมี annotation ให้ประเมิน
        label_files = [labels / f"{p.stem}.txt" for p in image_files]
        if not any(p.is_file() and p.read_text(encoding="utf-8").strip() for p in label_files):
            raise ValueError(f"No object annotations in {labels}")
    return data


def load_model():
    info = model_info()
    weights = ROOT / info["weights"]
    if not weights.is_file():
        raise FileNotFoundError(f"Model not found: {weights}")
    from ultralytics import YOLO
    model = YOLO(str(weights))
    expected = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))["names"]
    if list(model.names.values()) != expected:
        raise ValueError("Model classes do not match Aquarium dataset")
    return model


def model_info():
    # ใช้โมเดลที่เลือกจาก validation ของ Aquarium เท่านั้น
    if SELECTED_MODEL.is_file():
        return json.loads(SELECTED_MODEL.read_text(encoding="utf-8"))
    return {"weights": "runs/aquarium/training/aquarium/weights/best.pt", "imgsz": 480, "batch": 8}


def get_device():
    import torch
    # ใช้ GPU ถ้ามี ไม่มีก็ให้รัน CPU ได้
    return 0 if torch.cuda.is_available() else "cpu"
