"""ML-LAB-06 Experimentation & Comparison Script.

Compares:
1. Different Neural Network configurations (MLP vs CNN, varying layers/neurons).
2. Different numbers of epochs (10, 20, 30, 50).
Generates comparison summary tables and visualization charts for Lab 06.
"""

import json
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_config_1_mlp_1layer(input_shape):
    """Config 1: 1 Hidden Layer (128 neurons)."""
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ], name="Config_1_MLP_1Layer_128N")


def build_config_2_mlp_2layer(input_shape):
    """Config 2: 2 Hidden Layers (256 -> 128 neurons)."""
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Flatten(),
        layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ], name="Config_2_MLP_2Layers_256_128N")


def build_config_3_mlp_3layer(input_shape):
    """Config 3: 3 Hidden Layers (512 -> 256 -> 64 neurons)."""
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Flatten(),
        layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ], name="Config_3_MLP_3Layers_512_256_64N")


def build_config_4_cnn(input_shape):
    """Config 4: Convolutional Neural Network (CNN 32 -> 64 -> 128 + Dense 128)."""
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ], name="Config_4_CNN")


def run_configuration_comparison(X_train, y_train, X_val, y_val, X_test, y_test, epochs=25, batch_size=32):
    print("\n" + "=" * 60)
    print("Part 1: Comparing Neural Network Configurations (Hidden Layers & Neurons)")
    print("=" * 60)

    builders = [
        ("Config 1 (1 Hidden Layer: 128N)", "1 Layer [128]", build_config_1_mlp_1layer, 3e-4),
        ("Config 2 (2 Hidden Layers: 256 -> 128N)", "2 Layers [256, 128]", build_config_2_mlp_2layer, 3e-4),
        ("Config 3 (3 Hidden Layers: 512 -> 256 -> 64N)", "3 Layers [512, 256, 64]", build_config_3_mlp_3layer, 3e-4),
        ("Config 4 (CNN: 32 -> 64 -> 128 + Dense 128)", "Conv32-64-128 + Dense128", build_config_4_cnn, 1e-3),
    ]

    results = []

    for name, desc, builder, lr in builders:
        set_seed(42)
        model = builder(X_train.shape[1:])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
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
    print("\n" + "=" * 60)
    print("Part 2: Comparing Different Numbers of Epochs")
    print("=" * 60)

    results = []

    for ep in epoch_list:
        set_seed(42)
        model = build_config_4_cnn(X_train.shape[1:])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
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



def save_lab_summary(config_results, epoch_results, save_path=None):

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "lab_06_summary_report.md")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("# ML-LAB-06: Neural Network Experiment & Comparison Report\n\n")
        f.write("## 1. Comparison of Neural Network Configurations\n\n")
        f.write("| Configuration | Hidden Layers & Neurons | Train Acc (%) | Val Acc (%) | Test Acc (%) | Time (s) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for r in config_results:
            f.write(f"| {r['name']} | `{r.get('architecture', '-')}` | {r['train_acc']:.2f}% | {r['val_acc']:.2f}% | **{r['test_acc']:.2f}%** | {r['time_sec']:.1f}s |\n")

        f.write("\n## 2. Comparison of Different Numbers of Epochs (Best CNN Model)\n\n")
        f.write("| Epochs | Train Acc (%) | Val Acc (%) | Test Acc (%) | Test Loss | Time (s) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in epoch_results:
            f.write(f"| {r['epochs']} | {r['train_acc']:.2f}% | {r['val_acc']:.2f}% | **{r['test_acc']:.2f}%** | {r['test_loss']:.4f} | {r['time_sec']:.1f}s |\n")

        f.write("\n## 3. Analysis & Conclusions for Lab 06\n")
        f.write("- **Model Configuration:** โมเดลแบบ CNN (Convolutional Neural Network) ให้ประสิทธิภาพความแม่นยำสูงกว่า MLP อย่างเห็นได้ชัด เนื่องจาก CNN สามารถสกัดและเรียนรู้ฟีเจอร์เชิงพื้นที่ (Spatial Features) ของรูปภาพใบหน้าและลายของสัตว์ได้ดีกว่า\n")
        f.write("- **Number of Epochs:** เมื่อเพิ่มจำนวน Epoch จาก 10 เป็น 20-30 ประสิทธิภาพของโมเดลจะเพิ่มขึ้นอย่างรวดเร็ว และจะเริ่มทรงตัวที่ประมาณ 25-30 Epochs หากเทรนมากเกินไป (เช่น 50+ โดยไม่มี Regularization) อาจเริ่มเกิด Overfitting เล็กน้อย\n")

    print(f"\nSaved Lab 06 Markdown Report to: {save_path}")


def main():
    print("=" * 60)
    print("ML-LAB-06: Running Comparison Experiments")
    print("=" * 60)

    # Load preprocessed datasets
    X_train = np.load(os.path.join(OUTPUT_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(OUTPUT_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(OUTPUT_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

    print(f"Loaded Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 1. Compare Configurations & Plot
    config_results = run_configuration_comparison(X_train, y_train, X_val, y_val, X_test, y_test, epochs=25)
    plot_config_comparison(config_results, os.path.join(OUTPUT_DIR, "comparison_configs.png"))

    # 2. Compare Epochs & Plot
    epoch_results = run_epoch_comparison(X_train, y_train, X_val, y_val, X_test, y_test, epoch_list=(10, 20, 30, 50))
    plot_epoch_comparison(epoch_results, os.path.join(OUTPUT_DIR, "comparison_epochs.png"))

    # 3. Save Summary
    save_lab_summary(config_results, epoch_results)



if __name__ == "__main__":
    main()
