import json
from pathlib import Path

import numpy as np

from compare_experiments import run_experiments
from data_loader import load_data
from preprocessing import to_features
from split_data import split_dataset
from test_vgg import test_vgg

BASE_DIR = Path(__file__).resolve().parent
# ใช้รูปชุดเดิมจาก LAB07 ไม่ต้อง copy dataset มาซ้ำ
DATA_PATH = BASE_DIR.parent.parent / "LAB07 - CNN" / "Animal"
OUTPUT_DIR = BASE_DIR / "outputs"

# ตั้งค่าที่ใช้ทดลองไว้ตรงนี้เหมือนสคริปต์เดิม
IMG_SIZE = 100
CONFIG_EPOCHS = 20
EPOCH_LIST = [10, 20, 30, 50]
BATCH_SIZE = 32


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    images, y, classes = load_data(DATA_PATH, IMG_SIZE)
    X = to_features(images)
    # ใช้สัดส่วนและ seed เหมือน LAB07: train 70%, val 10%, test 20%
    splits = split_dataset(X, y, test_size=0.2, val_size=0.1)
    for name, values in zip(
            ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"], splits):
        np.save(output_dir / f"{name}.npy", values)
    (output_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    metadata = {
        "data_path": str(DATA_PATH.resolve()), "classes": classes,
        "class_counts": {name: int((y == i).sum()) for i, name in enumerate(classes)},
        "image_size": IMG_SIZE, "seed": 42, "normalization": "Rescaling(1/255)",
        "split_counts": dict(zip(["train", "validation", "test"], [len(s) for s in splits[:3]])),
        "config_epochs": CONFIG_EPOCHS, "epoch_checkpoints": EPOCH_LIST,
        "batch_size": BATCH_SIZE,
    }
    (output_dir / "run_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    run_experiments(*splits, classes, output_dir, CONFIG_EPOCHS,
                    EPOCH_LIST, BATCH_SIZE)
    # โหลดโมเดลที่เลือกด้วย validation แล้วสุ่ม test 4 รูปมาแสดง
    test_vgg(n_samples=4, output_dir=output_dir)


if __name__ == "__main__":
    main()
