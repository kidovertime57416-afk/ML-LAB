import matplotlib

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
    คำนวณ Accuracy, Classification Report, Confusion Matrix และบันทึกกราฟ
    
    Args:
        y_test: Label จริงของชุดทดสอบ
        predictions: ค่าผลลัพธ์ที่โมเดลทำนาย
        classes: รายชื่อคลาส
        save_path: ตำแหน่งบันทึกไฟล์ภาพ Confusion Matrix
        
    Returns:
        float: ค่า Accuracy ของโมเดล
    """
    preds = np.asarray(predictions)
    # แปลงผลการทำนายเป็น label ตัวเลข
    if preds.ndim == 2 and preds.shape[1] == 1:
        pred_labels = (preds.ravel() >= 0.5).astype(int)
    elif preds.ndim == 2 and preds.shape[1] > 1:
        pred_labels = preds.argmax(axis=1)
    else:
        pred_labels = preds.astype(int)

    labels = list(range(len(classes)))

    # คำนวณความแม่นยำรวม
    accuracy = accuracy_score(y_test, pred_labels)

    print("\n------------ CNN Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # แสดงผล Precision, Recall, F1-score
    print("\nClassification Report:")
    report = classification_report(
        y_test,
        pred_labels,
        labels=labels,
        target_names=classes,
        zero_division=0
    )
    print(report)

    # แสดง Confusion Matrix
    print("Confusion Matrix:")
    matrix = confusion_matrix(y_test, pred_labels, labels=labels)
    print(matrix)

    # วาดและบันทึกรูปภาพ Confusion Matrix
    if save_path:
        plot_confusion_matrix(matrix, classes, save_path, accuracy=accuracy)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path, accuracy=None):
    """
    วาด Confusion Matrix ในรูปแบบ Heatmap สเกลเปอร์เซ็นต์ (0-100%) พร้อมตัวเลขกำกับ
    
    Args:
        matrix: ข้อมูล Confusion Matrix จาก sklearn (จำนวนตัวอย่างจริง)
        classes: รายชื่อคลาส
        save_path: ตำแหน่งบันทึกไฟล์ภาพ
        accuracy: ค่าความแม่นยำรวมสำหรับแสดงที่หัวข้อ
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    # คำนวณเป็นร้อยละต่อคลาสจริง (0 - 100%)
    matrix_pct = np.divide(
        matrix.astype(float) * 100.0,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=200)
    
    # วาด heatmap สเกล 0-100 ด้วย colormap Blues
    im = ax.imshow(matrix_pct, cmap="Blues", interpolation="nearest", vmin=0, vmax=100)

    # ตั้งชื่อแกนและ labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=12, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold", labelpad=10)

    title = "CNN Confusion Matrix (%)"
    if accuracy is not None:
        title += f"\nOverall Test Accuracy: {accuracy * 100:.2f}%"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    # แถบ Colorbar ระบุเปอร์เซ็นต์ 0 - 100%
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentage (%)", fontsize=11, fontweight="bold", labelpad=10)
    cbar.set_ticks(np.linspace(0, 100, 6))
    cbar.ax.set_yticklabels([f"{int(t)}%" for t in np.linspace(0, 100, 6)], fontsize=10)

    # แสดงตัวเลขเปอร์เซ็นต์และจำนวนนับในแต่ละช่อง
    for i in range(len(classes)):
        for j in range(len(classes)):
            pct = matrix_pct[i, j]
            count = matrix[i, j]
            total = row_sums[i, 0]
            
            # ปรับสีตัวอักษรตามความเข้มของพื้นหลัง
            text_color = "white" if pct > 50 else "#1a2530"
            cell_text = f"{pct:.1f}%\n({count}/{total})"
            
            ax.text(
                j, i, cell_text,
                ha="center", va="center",
                color=text_color,
                fontsize=13,
                fontweight="bold",
                linespacing=1.3
            )

    # ปรับเส้นขอบตาราง
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(1.2)

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_history(history, save_path):
    """
    วาดกราฟ Accuracy, Loss และ Learning Rate ตลอดการฝึกสอน
    
    Args:
        history: ออบเจกต์ History จาก Keras หรือ Dict ข้อมูลประวัติ
        save_path: ตำแหน่งบันทึกไฟล์ภาพ
    """
    hist = history.history if hasattr(history, "history") else history
    epochs = list(range(1, len(hist["accuracy"]) + 1))

    # ตรวจสอบว่ามีข้อมูล Learning Rate หรือไม่
    lr_key = "learning_rate" if "learning_rate" in hist else ("lr" if "lr" in hist else None)
    has_lr = lr_key is not None and len(hist[lr_key]) > 0

    num_plots = 3 if has_lr else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(5.2 * num_plots, 4.6), dpi=200)

    # 1. กราฟ Train/Val Accuracy
    axes[0].plot(epochs, hist["accuracy"], label="Train Accuracy", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    if "val_accuracy" in hist:
        axes[0].plot(epochs, hist["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4)
    axes[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    axes[0].set_title("CNN Training & Validation Accuracy", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right", frameon=True)
    axes[0].set_xticks(epochs if len(epochs) <= 16 else np.linspace(1, len(epochs), 10, dtype=int))

    # 2. กราฟ Train/Val Loss
    axes[1].plot(epochs, hist["loss"], label="Train Loss", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    if "val_loss" in hist:
        axes[1].plot(epochs, hist["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4)
    axes[1].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Loss", fontsize=11, fontweight="bold")
    axes[1].set_title("CNN Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].set_xticks(epochs if len(epochs) <= 16 else np.linspace(1, len(epochs), 10, dtype=int))

    # 3. กราฟ Learning Rate Schedule
    if has_lr:
        lr_vals = hist[lr_key]
        axes[2].plot(epochs, lr_vals, label="Learning Rate", color="#2ca02c", linewidth=2.2, marker="d", markersize=5)
        axes[2].set_xlabel("Epoch", fontsize=11, fontweight="bold")
        axes[2].set_ylabel("Learning Rate", fontsize=11, fontweight="bold")
        axes[2].set_title("Learning Rate Schedule (ReduceLROnPlateau)", fontsize=12, fontweight="bold")
        axes[2].grid(True, linestyle="--", alpha=0.6)
        axes[2].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        axes[2].set_xticks(epochs if len(epochs) <= 16 else np.linspace(1, len(epochs), 10, dtype=int))
        axes[2].legend(loc="upper right", frameon=True)

        # ใส่ตัวเลขกำกับจุดที่มีการปรับลด Learning Rate
        for idx in range(1, len(lr_vals)):
            if lr_vals[idx] != lr_vals[idx - 1]:
                axes[2].annotate(
                    f"{lr_vals[idx]:.1e}",
                    xy=(epochs[idx], lr_vals[idx]),
                    xytext=(0, 14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#196924",
                    arrowprops=dict(arrowstyle="->", color="#196924", lw=1.2)
                )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
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
    สุ่มตัวอย่างภาพจากชุดทดสอบมาทำนายและแสดงผลในตาราง Grid
    
    Args:
        X_test: ข้อมูลภาพทดสอบ
        y_test: label จริง
        predictions_or_model: โมเดลหรือผลการทำนาย
        classes: รายชื่อคลาส
        save_path: ตำแหน่งบันทึกไฟล์ภาพ
        n_samples: จำนวนตัวอย่างภาพ
    """
    np.random.seed(42)
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    X_sample = X_test[indices]
    y_sample = y_test[indices]

    if hasattr(predictions_or_model, "predict"):
        probs = predictions_or_model.predict(X_sample, verbose=0)
    else:
        probs = np.asarray(predictions_or_model)[indices]

    # คำนวณความมั่นใจในการทำนาย (Confidence %)
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

        # แปลงสเกลค่าพิกเซลเป็น uint8 0-255 สำหรับ imshow
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
        f"CNN Prediction: {correct_count}/{n_samples} correct",
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
    วาดกราฟแท่งเปรียบเทียบค่าความแม่นยำของแต่ละโมเดล configuration
    """
    fig, ax = plt.subplots(figsize=(10, 5.2))
    labels_full = [r["name"] for r in results]
    x = np.arange(len(labels_full))
    width = 0.25

    # วาดกราฟแท่ง Train, Validation และ Test Accuracy
    train_bars = ax.bar(x - width, [r["train_acc"] for r in results], width, label="Train Accuracy", color="#2b5c8f")
    val_bars = ax.bar(x, [r["val_acc"] for r in results], width, label="Val Accuracy", color="#e27c38")
    test_bars = ax.bar(x + width, [r["test_acc"] for r in results], width, label="Test Accuracy", color="#3fa34d")

    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("CNN Configuration Performance Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_full, rotation=12, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    # ใส่ตัวเลขร้อยละบนยอดแท่ง Test Accuracy
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
    วาดกราฟเส้นเปรียบเทียบค่าความแม่นยำตามจำนวน epochs ต่างๆ
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    epochs_arr = [r["epochs"] for r in results]
    train_accs = [r["train_acc"] for r in results]
    val_accs = [r["val_acc"] for r in results]
    test_accs = [r["test_acc"] for r in results]

    # วาดเส้น Train, Validation และ Test Accuracy
    ax.plot(epochs_arr, train_accs, marker="o", linewidth=2.2, color="#2b5c8f", label="Train Accuracy")
    ax.plot(epochs_arr, val_accs, marker="s", linewidth=2.2, color="#e27c38", label="Validation Accuracy")
    ax.plot(epochs_arr, test_accs, marker="^", linewidth=2.5, color="#3fa34d", label="Test Accuracy")

    # แสดงตัวเลขค่าความแม่นยำบนจุดของ Test Accuracy
    for x_val, y_val_pt in zip(epochs_arr, test_accs):
        ax.annotate(f"{y_val_pt:.1f}%", xy=(x_val, y_val_pt), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=10, fontweight="bold", color="#246930")

    ax.set_xlabel("Number of Epochs", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Performance Comparison Across Different Epochs (CNN)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(epochs_arr)
    ax.set_ylim(40, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")
