"""ML-LAB-06 Experimentation & Comparison Script.

สคริปต์สำหรับการทดลองเปรียบเทียบตามข้อกำหนดของใบงาน ML-LAB-06:
1. การเปรียบเทียบสถาปัตยกรรมโมเดล Neural Network (จำนวน Hidden Layers และจำนวน Neurons)
2. การเปรียบเทียบจำนวน Epochs ในการฝึกสอน (10, 20, 30, 50 Epochs)
"""

import json
import os
import time

import matplotlib

# กำหนด backend ของ matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from evaluate import plot_config_comparison, plot_epoch_comparison

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def set_seed(seed=42):
    """กำหนดค่า Seed สำหรับ NumPy และ TensorFlow เพื่อให้ผลการทดลองสามารถทำซ้ำได้ (Reproducibility)"""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_config_1_mlp_1layer(input_shape):
    """
    Config 1: โมเดล Pure Neural Network ขนาด 1 Hidden Layer (128 Neurons)
    - โครงสร้าง: Input -> Rescaling -> Flatten -> Dense(128) + Dropout(0.2) -> Dense(1, Sigmoid)
    """
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ], name="Config_1_NN_1Layer_128N")


def build_config_2_mlp_2layer_small(input_shape):
    """
    Config 2: โมเดล Pure Neural Network ขนาด 2 Hidden Layers (256 -> 64 Neurons)
    - โครงสร้าง: Input -> Rescaling -> Flatten -> Dense(256) -> Dense(64) -> Dense(1, Sigmoid)
    """
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Flatten(),
        layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ], name="Config_2_NN_2Layers_256_64N")


def build_config_3_mlp_2layer_wide(input_shape):
    """
    Config 3: โมเดล Pure Neural Network ขนาด 2 Hidden Layers แบบขยายกว้าง (512 -> 128 Neurons)
    - โครงสร้าง: Input -> Rescaling -> Flatten -> Dense(512) -> Dense(128) -> Dense(1, Sigmoid)
    """
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Flatten(),
        layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ], name="Config_3_NN_2Layers_512_128N")


def build_config_4_mlp_3layer(input_shape):
    """
    Config 4: โมเดล Pure Neural Network ขนาด 3 Hidden Layers (256 -> 128 -> 64 Neurons)
    - โครงสร้าง: Input -> Rescaling -> Flatten -> Dense(256) -> Dense(128) -> Dense(64) -> Dense(1, Sigmoid)
    """
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Flatten(),
        layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ], name="Config_4_NN_3Layers_256_128_64N")


def run_configuration_comparison(X_train, y_train, X_val, y_val, X_test, y_test, epochs=25, batch_size=32):
    """
    ทำการทดลองเปรียบเทียบสถาปัตยกรรมโมเดล Neural Network ทั้ง 4 แบบ
    
    บันทึกตัวชี้วัด:
    - Training Accuracy
    - Validation Accuracy
    - Test Accuracy & Test Loss
    - เวลาที่ใช้ในการฝึกสอน (วินาที)
    
    Returns:
        list: รายการผลการทดลองของแต่ละ Configuration
    """
    print("\n" + "=" * 60)
    print("Part 1: Comparing Neural Network Configurations (Hidden Layers & Neurons)")
    print("=" * 60)

    builders = [
        ("Config 1 (1 Hidden Layer: 128N)", "1 Layer [128]", build_config_1_mlp_1layer),
        ("Config 2 (2 Hidden Layers: 256 -> 64N)", "2 Layers [256, 64]", build_config_2_mlp_2layer_small),
        ("Config 3 (2 Hidden Layers: 512 -> 128N)", "2 Layers [512, 128]", build_config_3_mlp_2layer_wide),
        ("Config 4 (3 Hidden Layers: 256 -> 128 -> 64N)", "3 Layers [256, 128, 64]", build_config_4_mlp_3layer),
    ]

    results = []

    for name, desc, builder in builders:
        set_seed(42)
        model = builder(X_train.shape[1:])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        elapsed_time = time.time() - start_time

        train_acc = history.history["accuracy"][-1] * 100
        val_acc = history.history["val_accuracy"][-1] * 100
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        test_acc *= 100

        print(f"[{name}]")
        print(f"  Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {elapsed_time:.1f}s")

        results.append({
            "name": name,
            "architecture": desc,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "test_loss": float(test_loss),
            "time_sec": elapsed_time
        })

    return results


def run_epoch_comparison(X_train, y_train, X_val, y_val, X_test, y_test, epoch_list=(10, 20, 30, 50), batch_size=32):
    """
    ทำการทดลองเปรียบเทียบผลการฝึกสอนตามจำนวน Epochs ต่างๆ (10, 20, 30, 50 รอบ)
    
    Returns:
        list: รายการผลการทดลองของแต่ละจำนวน Epochs
    """
    print("\n" + "=" * 60)
    print("Part 2: Comparing Different Numbers of Epochs (Neural Network)")
    print("=" * 60)

    results = []

    for ep in epoch_list:
        set_seed(42)
        model = build_config_4_mlp_3layer(X_train.shape[1:])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=ep,
            batch_size=batch_size,
            verbose=0
        )
        elapsed_time = time.time() - start_time

        train_acc = history.history["accuracy"][-1] * 100
        val_acc = history.history["val_accuracy"][-1] * 100
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        test_acc *= 100

        print(f"[Epochs: {ep:2d}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {elapsed_time:.1f}s")

        results.append({
            "epochs": ep,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "test_loss": float(test_loss),
            "time_sec": elapsed_time
        })

    return results


def main():
    """ฟังก์ชันหลักสำหรับรันการทดลองเปรียบเทียบแบบ Standalone"""
    print("=" * 60)
    print("ML-LAB-06: Running Comparison Experiments")
    print("=" * 60)

    # โหลดชุดข้อมูลที่ผ่าน Preprocessing แล้ว
    X_train = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(OUTPUT_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(OUTPUT_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

    print(f"Loaded Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 1. เปรียบเทียบ Configurations และสร้างกราฟ
    config_results = run_configuration_comparison(X_train, y_train, X_val, y_val, X_test, y_test, epochs=25)
    plot_config_comparison(config_results, os.path.join(OUTPUT_DIR, "comparison_configs.png"))

    # 2. เปรียบเทียบจำนวน Epochs และสร้างกราฟ
    epoch_results = run_epoch_comparison(X_train, y_train, X_val, y_val, X_test, y_test, epoch_list=(10, 20, 30, 50))
    plot_epoch_comparison(epoch_results, os.path.join(OUTPUT_DIR, "comparison_epochs.png"))


if __name__ == "__main__":
    main()

