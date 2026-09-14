import matplotlib

# กำหนด backend ของ matplotlib เป็น Agg เพื่อให้สามารถบันทึกภาพลงไฟล์ได้โดยไม่ต้องเปิดหน้าต่าง GUI
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path=None):
    """
    ประเมินประสิทธิภาพของโมเดลบนชุดข้อมูลทดสอบ (Test Set)
    
    ขั้นตอนการประเมิน:
    1. แปลงค่าการทำนาย (Predictions/Probabilities) ให้อยู่ในรูป Label ตัวเลข (0 หรือ 1)
    2. คำนวณค่า Accuracy Score (ร้อยละความถูกต้องโดยรวม)
    3. สร้าง Classification Report (Precision, Recall, F1-Score สำหรับแต่ละคลาส)
    4. คำนวณ Confusion Matrix (ตารางแสดงความถูกต้องและการสับสนระหว่างคลาส)
    5. พล็อตและบันทึกรูปภาพ Confusion Matrix หากมีการระบุ save_path
    
    Args:
        y_test: Label จริงของชุดข้อมูลทดสอบ
        predictions: ค่าผลลัพธ์ที่โมเดลทำนาย (อาจเป็น Probability หรือ Class Index)
        classes: รายชื่อคลาส (เช่น ['Lion', 'tiger'])
        save_path: เส้นทางบันทึกไฟล์ภาพ Confusion Matrix
        
    Returns:
        float: ค่า Accuracy ของโมเดล
    """
    preds = np.asarray(predictions)
    # กรณีเป็นค่าความน่าจะเป็นแบบ Binary (Sigmoid)
    if preds.ndim == 2 and preds.shape[1] == 1:
        pred_labels = (preds.ravel() >= 0.5).astype(int)
    # กรณีเป็นค่าความน่าจะเป็นแบบ Multi-class (Softmax)
    elif preds.ndim == 2 and preds.shape[1] > 1:
        pred_labels = preds.argmax(axis=1)
    else:
        pred_labels = preds.astype(int)

    labels = list(range(len(classes)))

    # คำนวณค่าความแม่นยำรวม (Accuracy)
    accuracy = accuracy_score(y_test, pred_labels)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # แสดงตารางรายงาน Precision, Recall, F1-score
    print("\nClassification Report:")
    report = classification_report(
        y_test,
        pred_labels,
        labels=labels,
        target_names=classes,
        zero_division=0
    )
    print(report)

    # คำนวณและแสดงตาราง Confusion Matrix
    print("Confusion Matrix:")
    matrix = confusion_matrix(y_test, pred_labels, labels=labels)
    print(matrix)

    # บันทึกรูปภาพ Confusion Matrix
    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):
    """
    วาดและบันทึกภาพ Confusion Matrix แบบ Heatmap พร้อมตัวเลขกำกับ
    
    Args:
        matrix: ข้อมูล Confusion Matrix จาก sklearn
        classes: รายชื่อคลาส
        save_path: เส้นทางไฟล์สำหรับบันทึกภาพ
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    # กำหนดแกน X (ค่าที่ทำนาย) และแกน Y (ค่าจริง)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)

    # เพิ่มแถบสเกลสี (Colorbar)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ใส่ตัวเลขจำนวนภาพในแต่ละช่อง พร้อมปรับสีตัวอักษรให้อ่านง่าย
    threshold = matrix.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = matrix[i, j]
            ax.text(j, i, f"{val}", ha="center", va="center",
                    color="white" if val > threshold else "black",
                    fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history(history, save_path):
    """
    วาดกราฟเปรียบเทียบ Loss และ Accuracy ตลอดการฝึกสอน (Training Curves)
    
    กราฟประกอบด้วย 2 ส่วนย่อย (Subplots):
    1. กราฟด้านซ้าย: แสดงเส้น Accuracy ของชุดฝึกสอน (Train) และชุดตรวจสอบ (Validation)
    2. กราฟด้านขวา: แสดงเส้น Loss ของชุดฝึกสอน (Train) และชุดตรวจสอบ (Validation)
    
    Args:
        history: ออบเจกต์ History จากโมเดล Keras
        save_path: เส้นทางไฟล์สำหรับบันทึกภาพ
    """
    hist = history.history if hasattr(history, "history") else history
    epochs = range(1, len(hist["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # กราฟย่อยที่ 1: Accuracy Curves
    axes[0].plot(epochs, hist["accuracy"], label="Train Accuracy", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    if "val_accuracy" in hist:
        axes[0].plot(epochs, hist["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4)
    axes[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    axes[0].set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right", frameon=True)

    # กราฟย่อยที่ 2: Loss Curves
    axes[1].plot(epochs, hist["loss"], label="Train Loss", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    if "val_loss" in hist:
        axes[1].plot(epochs, hist["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4)
    axes[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Loss", fontsize=11, fontweight="bold")
    axes[1].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_prediction_sample(
    X_test,
    y_test,
    predictions_or_model,
    classes,
    save_path="outputs/prediction_sample.png",
    n_samples=4
):
    """
    สุ่มตัวอย่างรูปภาพจาก Test Set มาพยากรณ์และแสดงผลในรูปแบบตาราง Grid 2x2
    
    รายละเอียด:
    - สุ่มเลือกภาพจำนวน n_samples (4 รูป) โดยใช้ Seed คงที่เพื่อให้ผลลัพธ์ทำซ้ำได้
    - คำนวณค่าความมั่นใจจริง (Confidence Probability %)
    - แสดงหัวข้อระบุชื่อคลาสที่ทำนาย (Pred) เทียบกับคลาสจริง (True)
    - ใช้สีเขียวเมื่อทำนายถูกต้อง และสีแดงเมื่อทำนายผิดพลาด
    
    Args:
        X_test: ข้อมูลภาพทดสอบ
        y_test: Label จริง
        predictions_or_model: โมเดลหรือผลการพยากรณ์
        classes: รายชื่อคลาส
        save_path: เส้นทางบันทึกภาพ
        n_samples: จำนวนภาพตัวอย่าง (default: 4)
    """
    np.random.seed(42)
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    X_sample = X_test[indices]
    y_sample = y_test[indices]

    if hasattr(predictions_or_model, "predict"):
        probs = predictions_or_model.predict(X_sample, verbose=0)
    else:
        probs = np.asarray(predictions_or_model)[indices]

    # คำนวณ Confidence Percentage จากความน่าจะเป็น
    if probs.ndim == 2 and probs.shape[1] == 1:
        probs_1d = probs.ravel()
        pred_indices = (probs_1d >= 0.5).astype(int)
        confidences = np.where(pred_indices == 1, probs_1d, 1.0 - probs_1d)
    elif probs.ndim == 1:
        if probs.max() <= 1.0 and np.any((probs > 0) & (probs < 1)):
            pred_indices = (probs >= 0.5).astype(int)
            confidences = np.where(pred_indices == 1, probs, 1.0 - probs)
        else:
            pred_indices = probs.astype(int)
            confidences = np.ones(len(probs))
    else:
        pred_indices = probs.argmax(axis=1)
        confidences = probs.max(axis=1)

    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    correct_count = 0

    for i in range(n_samples):
        img = X_sample[i]

        # ปรับมิติรูปภาพหากถูก Flatten
        if img.ndim == 1:
            if img.shape[0] == 100 * 100 * 3:
                img = img.reshape(100, 100, 3)
            elif img.shape[0] == 100 * 100:
                img = img.reshape(100, 100)

        # แปลงค่าภาพให้อยู่ในสเกล uint8 (0-255) สำหรับการแสดงผลผ่าน imshow
        if img.dtype != np.uint8 and img.max() <= 1.0:
            img_disp = (img * 255).astype(np.uint8)
        else:
            img_disp = img.astype(np.uint8)

        pred_idx = int(pred_indices[i])
        true_idx = int(y_sample[i])
        is_correct = (pred_idx == true_idx)
        if is_correct:
            correct_count += 1

        color = "#1b8a2e" if is_correct else "#d32f2f"

        if img_disp.ndim == 2:
            axes[i].imshow(img_disp, cmap="gray")
        else:
            axes[i].imshow(img_disp)

        conf_pct = confidences[i] * 100
        axes[i].set_title(
            f"Pred: {classes[pred_idx]} ({conf_pct:.1f}%)\nTrue: {classes[true_idx]}",
            color=color,
            fontsize=12,
            fontweight="bold"
        )
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for j in range(n_samples, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Prediction: {correct_count}/{n_samples} correct",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_config_comparison(results, save_path):
    """
    วาดกราฟแท่งเปรียบเทียบประสิทธิภาพของแต่ละ Configuration (Hidden Layers & Neurons)
    
    Args:
        results (list): ผลลัพธ์จากการทดลองเปรียบเทียบในรูปแบบ Dict
        save_path (str): เส้นทางบันทึกไฟล์ภาพ
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    labels_full = [r["name"] for r in results]
    x = np.arange(len(labels_full))
    width = 0.25

    # วาดแท่งเปรียบเทียบ 3 ค่า: Train, Validation และ Test Accuracy
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

    # ใส่ตัวเลขร้อยละกำกับบนยอดแท่ง Test Accuracy
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
    """
    วาดกราฟเส้นเปรียบเทียบประสิทธิภาพตามจำนวน Epochs (10, 20, 30, 50 Epochs)
    
    Args:
        results (list): ผลลัพธ์จากการทดลองเปรียบเทียบ Epochs
        save_path (str): เส้นทางบันทึกไฟล์ภาพ
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    epochs_arr = [r["epochs"] for r in results]
    train_accs = [r["train_acc"] for r in results]
    val_accs = [r["val_acc"] for r in results]
    test_accs = [r["test_acc"] for r in results]

    # วาดเส้นกราฟความแม่นยำ Train, Val และ Test
    ax.plot(epochs_arr, train_accs, marker="o", linewidth=2.2, color="#2b5c8f", label="Train Accuracy")
    ax.plot(epochs_arr, val_accs, marker="s", linewidth=2.2, color="#e27c38", label="Validation Accuracy")
    ax.plot(epochs_arr, test_accs, marker="^", linewidth=2.5, color="#3fa34d", label="Test Accuracy")

    # แสดงตัวเลขความแม่นยำบนจุดของ Test Accuracy
    for x_val, y_val_pt in zip(epochs_arr, test_accs):
        ax.annotate(f"{y_val_pt:.1f}%", xy=(x_val, y_val_pt), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=10, fontweight="bold", color="#246930")

    ax.set_xlabel("Number of Epochs", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Comparison Across Different Epochs (Neural Network)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(epochs_arr)
    ax.set_ylim(40, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

