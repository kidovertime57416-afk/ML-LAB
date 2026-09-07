import json
import os
import matplotlib

# Set backend before pyplot so it works in any environment
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from compare_experiments import (
    run_configuration_comparison,
    run_epoch_comparison,
    save_lab_summary
)
from data_loader import load_data
from evaluate import (
    evaluate_model,
    plot_history,
    plot_prediction_sample
)
from nn_model import predict_model, train_model
from preprocessing import to_features
from split_data import split_dataset
from test_nn import test_nn

# Paths are relative to this file, so the script runs from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Animal")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

IMG_SIZE = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.1
MAX_PER_CLASS = 3000   # None = use all images
EPOCHS = 30
BATCH_SIZE = 32


def plot_config_comparison(results, save_path):
    """Plot bar chart comparing different NN configurations (MLP layers/neurons vs CNN)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels_full = [r["name"] for r in results]
    x = np.arange(len(labels_full))
    width = 0.25

    train_bars = ax.bar(x - width, [r["train_acc"] for r in results], width, label="Train Accuracy", color="#2b5c8f")
    val_bars = ax.bar(x, [r["val_acc"] for r in results], width, label="Val Accuracy", color="#e27c38")
    test_bars = ax.bar(x + width, [r["test_acc"] for r in results], width, label="Test Accuracy", color="#3fa34d")

    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Neural Network Configuration Performance Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_full, rotation=12, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    for bar in test_bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_epoch_comparison(results, save_path):
    """Plot line chart comparing performance across different numbers of epochs."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    epochs_arr = [r["epochs"] for r in results]
    train_accs = [r["train_acc"] for r in results]
    val_accs = [r["val_acc"] for r in results]
    test_accs = [r["test_acc"] for r in results]

    ax.plot(epochs_arr, train_accs, marker="o", linewidth=2.2, color="#2b5c8f", label="Train Accuracy")
    ax.plot(epochs_arr, val_accs, marker="s", linewidth=2.2, color="#e27c38", label="Validation Accuracy")
    ax.plot(epochs_arr, test_accs, marker="^", linewidth=2.5, color="#3fa34d", label="Test Accuracy")

    for x_val, y_val_pt in zip(epochs_arr, test_accs):
        ax.annotate(f"{y_val_pt:.1f}%", xy=(x_val, y_val_pt), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=10, fontweight="bold", color="#246930")

    ax.set_xlabel("Number of Epochs", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Comparison Across Different Epochs (CNN Model)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(epochs_arr)
    ax.set_ylim(40, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")



def main():

    print("=" * 60)
    print("ML-LAB-06: Neural Network Image Recognition (Tiger vs Lion)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load Dataset
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
    with open(os.path.join(OUTPUT_DIR, "classes.json"), "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")

    # Step 2: Preprocessing
    print("\n[Step 2] Preprocessing images...")
    X = to_features(images)
    y = labels

    np.save(os.path.join(OUTPUT_DIR, "features.npy"), X)
    print(f"Feature shape: {X.shape}")

    # Step 3: Split Dataset
    print("\n[Step 3] Splitting dataset (Train 70% / Val 10% / Test 20%)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X, y, TEST_SIZE, VAL_SIZE
    )

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"Training samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples   : {len(X_test)}")

    # Step 4: Train Primary Model
    print("\n[Step 4] Training primary CNN model...")
    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes),
        OUTPUT_DIR, EPOCHS, BATCH_SIZE
    )
    print("Training completed.")

    # Step 5: Prediction & Evaluation
    print("\n[Step 5] Evaluating primary model on test set...")
    predictions = predict_model(model, X_test)
    evaluate_model(
        y_test, predictions, classes,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )
    plot_history(
        history,
        save_path=os.path.join(OUTPUT_DIR, "training_history.png")
    )
    plot_prediction_sample(
        X_test, y_test, model, classes,
        save_path=os.path.join(OUTPUT_DIR, "prediction_sample.png")
    )

    # Step 6: Lab 06 Comparison Experiments (Configurations & Epochs)
    print("\n[Step 6] Running Lab 06 Comparison Experiments...")
    
    # 6.1 เปรียบเทียบ Configurations (Hidden Layers & Neurons) พร้อมสร้างรูปภาพ
    config_results = run_configuration_comparison(
        X_train, y_train, X_val, y_val, X_test, y_test, epochs=20, batch_size=BATCH_SIZE
    )
    plot_config_comparison(
        config_results,
        save_path=os.path.join(OUTPUT_DIR, "comparison_configs.png")
    )

    # 6.2 เปรียบเทียบจำนวน Epochs (10, 20, 30, 50 Epochs) พร้อมสร้างรูปภาพ
    epoch_results = run_epoch_comparison(
        X_train, y_train, X_val, y_val, X_test, y_test, epoch_list=(10, 20, 30, 50), batch_size=BATCH_SIZE
    )
    plot_epoch_comparison(
        epoch_results,
        save_path=os.path.join(OUTPUT_DIR, "comparison_epochs.png")
    )

    # 6.3 บันทึกตารางสรุปผลการทดลอง
    save_lab_summary(
        config_results, epoch_results,
        save_path=os.path.join(OUTPUT_DIR, "lab_06_summary_report.md")
    )

    # Step 7: Sample Testing Preview
    print("\n[Step 7] Testing model inference on random test samples...")
    test_nn(n_samples=4)

    print("\n" + "=" * 60)
    print("All ML-LAB-06 steps and experiments completed successfully!")
    print(f"All generated outputs and reports saved in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
