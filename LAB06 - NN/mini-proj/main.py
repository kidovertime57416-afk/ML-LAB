"""Main End-to-End Pipeline for ML-LAB-06 Neural Network Image Classification.

สคริปต์หลักสำหรับรันกระบวนการ Machine Learning Pipeline แบบครบวงจร 7 ขั้นตอน:
1. โหลดข้อมูลภาพจากโฟลเดอร์ Animal/ (data_loader.py)
2. ปรับขนาดและแปลงระบบสีของภาพ (preprocessing.py)
3. แบ่งชุดข้อมูล Train 70%, Validation 10%, Test 20% (split_data.py)
4. สร้างและฝึกสอนโมเดลหลัก Pure Neural Network (nn_model.py)
5. ประเมินผลโมเดล: คำนวณ Metrics, วาด Confusion Matrix และ Training History (evaluate.py)
6. ทำการทดลองเปรียบเทียบตามใบงาน: เปรียบเทียบ Configurations และ Epochs (compare_experiments.py)
7. สุ่มทดสอบการพยากรณ์ภาพตัวอย่าง 4 ภาพ (test_nn.py)
"""

import json
import os
import matplotlib

# กำหนด backend ของ matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_experiments import (
    run_configuration_comparison,
    run_epoch_comparison
)
from data_loader import load_data
from evaluate import (
    evaluate_model,
    plot_config_comparison,
    plot_epoch_comparison,
    plot_history,
    plot_prediction_sample
)
from nn_model import predict_model, train_model
from preprocessing import to_features
from split_data import split_dataset
from test_nn import test_nn

# กำหนดเส้นทางโฟลเดอร์สำหรับอ่านข้อมูลและบันทึกผลลัพธ์
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Animal")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ไฮเปอร์พารามิเตอร์หลัก
IMG_SIZE = 100         # ขนาดภาพ 100x100 พิกเซล
TEST_SIZE = 0.2        # สัดส่วน Test Set 20%
VAL_SIZE = 0.1         # สัดส่วน Validation Set 10%
MAX_PER_CLASS = 3000   # ขีดจำกัดสูงสุดต่อคลาส (None = ทั้งหมด)
EPOCHS = 30            # จำนวนรอบการฝึกสอนโมเดลหลัก
BATCH_SIZE = 32        # ขนาด Batch Size


def main():
    """ฟังก์ชันหลักสำหรับรัน Pipeline ทั้ง 7 ขั้นตอน"""
    print("=" * 60)
    print("ML-LAB-06: Neural Network Image Recognition (Tiger vs Lion)")
    print("=" * 60)

    # สร้างโฟลเดอร์ outputs หากยังไม่มี
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------
    # ขั้นตอนที่ 1: โหลดรูปภาพจากโฟลเดอร์ชุดข้อมูล
    # -------------------------------------------------------------
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    # บันทึก Label และรายชื่อคลาส
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
    with open(os.path.join(OUTPUT_DIR, "classes.json"), "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # -------------------------------------------------------------
    # ขั้นตอนที่ 2: เตรียมข้อมูลรูปภาพ (Preprocessing)
    # -------------------------------------------------------------
    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels

    np.save(os.path.join(OUTPUT_DIR, "features.npy"), X)
    print(f"Feature shape: {X.shape}")

    # -------------------------------------------------------------
    # ขั้นตอนที่ 3: แบ่งชุดข้อมูล Train 70% / Val 10% / Test 20%
    # -------------------------------------------------------------
    print("\n[Step 3] Splitting dataset (Train 70% / Val 10% / Test 20%)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    # บันทึกชุดข้อมูลทั้งหมดเป็นไฟล์ .npy
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    # -------------------------------------------------------------
    # ขั้นตอนที่ 4: สร้างและฝึกสอนโมเดลหลัก Pure Neural Network
    # -------------------------------------------------------------
    print("\n[Step 4] Training primary Neural Network model...")
    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE
    )

    print("Training completed.")

    # -------------------------------------------------------------
    # ขั้นตอนที่ 5: ประเมินผลโมเดลบนชุดข้อมูลทดสอบ
    # -------------------------------------------------------------
    print("\n[Step 5] Evaluating primary model on test set...")
    predictions = predict_model(model, X_test)
    
    # คำนวณ Accuracy, Classification Report และบันทึก confusion_matrix.png
    evaluate_model(
        y_test, predictions, classes,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )
    # บันทึกกราฟแสดงประวัติการฝึกสอน training_history.png
    plot_history(
        history,
        save_path=os.path.join(OUTPUT_DIR, "training_history.png")
    )
    # บันทึกภาพตัวอย่างการพยากรณ์ prediction_sample.png
    plot_prediction_sample(
        X_test, y_test, model, classes,
        save_path=os.path.join(OUTPUT_DIR, "prediction_sample.png")
    )

    # -------------------------------------------------------------
    # ขั้นตอนที่ 6: การทดลองเปรียบเทียบตามข้อกำหนดของใบงาน Lab 06
    # -------------------------------------------------------------
    print("\n[Step 6] Running Lab 06 Comparison Experiments...")
    
    # 6.1 เปรียบเทียบ Configurations (Hidden Layers & Neurons) พร้อมบันทึก comparison_configs.png
    config_results = run_configuration_comparison(
        X_train, y_train, X_val, y_val, X_test, y_test, epochs=20, batch_size=BATCH_SIZE
    )
    plot_config_comparison(
        config_results,
        save_path=os.path.join(OUTPUT_DIR, "comparison_configs.png")
    )

    # 6.2 เปรียบเทียบจำนวน Epochs (10, 20, 30, 50 Epochs) พร้อมบันทึก comparison_epochs.png
    epoch_results = run_epoch_comparison(
        X_train, y_train, X_val, y_val, X_test, y_test, epoch_list=(10, 20, 30, 50), batch_size=BATCH_SIZE
    )
    plot_epoch_comparison(
        epoch_results,
        save_path=os.path.join(OUTPUT_DIR, "comparison_epochs.png")
    )

    # -------------------------------------------------------------
    # ขั้นตอนที่ 7: สุ่มทดสอบการพยากรณ์ภาพตัวอย่าง 4 ภาพ
    # -------------------------------------------------------------
    print("\n[Step 7] Testing model inference on random test samples...")
    test_nn(n_samples=4)

    print("\n" + "=" * 60)
    print("All ML-LAB-06 steps and experiments completed successfully!")
    print(f"All generated outputs saved in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
