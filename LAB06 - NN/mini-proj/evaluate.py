import matplotlib

# Set backend before pyplot, so it works without a display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path=None):
    # If predictions are probabilities, convert to class indices
    preds = np.asarray(predictions)
    if preds.ndim == 2 and preds.shape[1] == 1:
        pred_labels = (preds.ravel() >= 0.5).astype(int)
    elif preds.ndim == 2 and preds.shape[1] > 1:
        pred_labels = preds.argmax(axis=1)
    else:
        pred_labels = preds.astype(int)

    # Pin label order so target_names always matches the columns
    labels = list(range(len(classes)))

    # Calculate accuracy
    accuracy = accuracy_score(y_test, pred_labels)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    report = classification_report(
        y_test,
        pred_labels,
        labels=labels,
        target_names=classes,
        zero_division=0
    )
    print(report)

    print("Confusion Matrix:")
    matrix = confusion_matrix(y_test, pred_labels, labels=labels)
    print(matrix)

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)

    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

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
    """Accuracy and loss curves — formatted with 1-based epochs and gridlines."""
    hist = history.history if hasattr(history, "history") else history

    epochs = range(1, len(hist["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Accuracy subplot
    axes[0].plot(epochs, hist["accuracy"], label="Train Accuracy", color="#1f77b4", linewidth=2.2, marker="o", markersize=4)
    if "val_accuracy" in hist:
        axes[0].plot(epochs, hist["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2.2, marker="s", markersize=4)
    axes[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    axes[0].set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right", frameon=True)

    # Loss subplot
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
    """Plot sample predictions with actual probability confidences and clean layout."""
    np.random.seed(42)  # For reproducible sample preview
    indices = np.random.choice(len(X_test), n_samples, replace=False)

    X_sample = X_test[indices]
    y_sample = y_test[indices]

    if hasattr(predictions_or_model, "predict"):
        probs = predictions_or_model.predict(X_sample, verbose=0)
    else:
        probs = np.asarray(predictions_or_model)[indices]

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

        # Reshape if flattened
        if img.ndim == 1:
            if img.shape[0] == 100 * 100 * 3:
                img = img.reshape(100, 100, 3)
            elif img.shape[0] == 100 * 100:
                img = img.reshape(100, 100)

        # Scale uint8 0-255 or float 0-1 properly
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
