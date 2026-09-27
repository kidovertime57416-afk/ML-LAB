"""สคริปต์หลักสำหรับรันกระบวนการทั้งหมดของโมเดล CNN:
โหลดข้อมูล -> Preprocessing -> แบ่งชุดข้อมูล -> ฝึกสอนโมเดล -> ประเมินผล -> เปรียบเทียบผลการทดลอง -> ทดสอบการทำนาย
"""

import json
import os
import matplotlib

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
from cnn_model import predict_model, train_model
from preprocessing import to_features
from split_data import split_dataset
from test_cnn import test_cnn

# กำหนดโฟลเดอร์สำหรับอ่านข้อมูลและบันทึกผลลัพธ์
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Animal")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ตั้งค่าพารามิเตอร์เริ่มต้น
IMG_SIZE = 100         # ปรับขนาดภาพเป็น 100x100 พิกเซล
TEST_SIZE = 0.2        # แบ่งชุดทดสอบ 20%
VAL_SIZE = 0.1         # แบ่งชุดตรวจสอบ 10%
MAX_PER_CLASS = 3000   # จำนวนภาพสูงสุดต่อคลาส
EPOCHS = 30            # จำนวนรอบในการฝึกสอน
BATCH_SIZE = 32        # ขนาด batch size


def main():
    print("=" * 60)
    print("ML-LAB-07: Convolutional Neural Network Image Recognition (Tiger vs Lion)")
    print("=" * 60)

    # สร้างโฟลเดอร์เก็บผลลัพธ์
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. โหลดข้อมูลภาพจากโฟลเดอร์
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    # บันทึก labels และรายชื่อคลาส
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
    with open(os.path.join(OUTPUT_DIR, "classes.json"), "w") as f:
        json.dump(classes, f)

    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # 2. ปรับแต่งและแปลงข้อมูลภาพเป็น feature array
    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels
    np.save(os.path.join(OUTPUT_DIR, "features.npy"), X)
    print(f"Feature shape: {X.shape}")

    # 3. แบ่งชุดข้อมูล Train 70%, Validation 10%, Test 20%
    print("\n[Step 3] Splitting dataset (Train 70% / Val 10% / Test 20%)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    # บันทึกชุดข้อมูลลงไฟล์ .npy
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    # 4. สร้างและฝึกสอนโมเดล CNN
    print("\n[Step 4] Training primary CNN model...")
    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        output_dir=OUTPUT_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE
    )

    # 5. ประเมินประสิทธิภาพของโมเดลบนชุดทดสอบ
    print("\n[Step 5] Evaluating primary model on test set...")
    predictions = predict_model(model, X_test)
    
    # คำนวณ metrics และบันทึกกราฟ confusion matrix
    evaluate_model(
        y_test, predictions, classes,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )
    # บันทึกกราฟ accuracy, loss และ learning rate
    plot_history(
        history,
        save_path=os.path.join(OUTPUT_DIR, "training_history.png")
    )
    # บันทึกภาพตัวอย่างผลการทำนาย
    plot_prediction_sample(
        X_test, y_test, model, classes,
        save_path=os.path.join(OUTPUT_DIR, "prediction_sample.png")
    )

    # 6. รันการทดลองเปรียบเทียบสถาปัตยกรรมโมเดลและจำนวน epochs
    print("\n[Step 6] Running Comparison Experiments...")
    
    # เปรียบเทียบ configurations (Conv layers, filters, neurons)
    config_results = run_configuration_comparison(
        X_train, y_train, X_val, y_val, X_test, y_test, epochs=20, batch_size=BATCH_SIZE
    )
    plot_config_comparison(
        config_results,
        save_path=os.path.join(OUTPUT_DIR, "comparison_configs.png")
    )

    # เปรียบเทียบจำนวน epochs (10, 20, 30, 50)
    epoch_results = run_epoch_comparison(
        X_train, y_train, X_val, y_val, X_test, y_test, epoch_list=(10, 20, 30, 50), batch_size=BATCH_SIZE
    )
    plot_epoch_comparison(
        epoch_results,
        save_path=os.path.join(OUTPUT_DIR, "comparison_epochs.png")
    )

    # 7. สุ่มภาพจากชุดทดสอบมาทำนายผล
    print("\n[Step 7] Testing CNN model inference on random test samples...")
    test_cnn(n_samples=4)

    print("\n" + "=" * 60)
    print("Process finished successfully!")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
