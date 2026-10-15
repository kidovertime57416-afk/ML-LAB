import csv
import json
import shutil
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from data_loader import CLASSES, load_crops

# โน้ต: เปลี่ยนแค่ dropout จะได้เห็นผลของมันชัด ๆ
CONFIGS = [("baseline", 0.0), ("dropout_03", 0.3)]
SEED = 42


def write_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2), encoding="utf-8")


def dataset(X, y, batch, training=False):
    data = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        data = data.shuffle(len(y), seed=SEED, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return data.batch(batch).with_options(options).prefetch(1)


def summarize(y, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, average="macro", labels=list(range(len(CLASSES))), zero_division=0)
    return {"accuracy": float(accuracy_score(y, predictions)),
            "precision_macro": float(precision), "recall_macro": float(recall),
            "f1_macro": float(f1)}


def plot_history(history, path):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, ax in [("accuracy", axes[0]), ("loss", axes[1])]:
        ax.plot(epochs, history[key], label="train")
        ax.plot(epochs, history["val_" + key], label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.title())
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiments(make_model, output_dir, epochs=10, batch=32, limit=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = load_crops("train", limit)
    # โน้ต: ใช้ valid ที่แยกภาพไว้แล้ว ไม่สุ่ม crop จากภาพเดียวกันมาปน train/val
    X_val, y_val = load_crops("val")
    rows = []
    for name, dropout in CONFIGS:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        model = make_model(dropout=dropout)
        folder = output_dir / "experiments" / name
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Experiment: {name}, dropout={dropout}, epochs={epochs}", flush=True)
        start = perf_counter()
        history = model.fit(dataset(X_train, y_train, batch, training=True),
                            validation_data=dataset(X_val, y_val, batch),
                            epochs=epochs, verbose=2)
        seconds = perf_counter() - start
        val_loss, _ = model.evaluate(dataset(X_val, y_val, batch), verbose=0)
        predictions = model.predict(X_val, batch_size=batch, verbose=0).argmax(axis=1)
        row = {"configuration": name, "dropout": dropout, "epochs": epochs,
               "seed": SEED, "batch_size": batch, "train_crops": len(y_train),
               "validation_crops": len(y_val), "val_loss": float(val_loss),
               **summarize(y_val, predictions), "seconds": seconds}
        rows.append(row)
        model.save(folder / "model.keras")
        write_json(folder / "history.json", history.history)
        plot_history(history.history, folder / "training_history.png")
        write_json(output_dir / "experiments.json", rows)
    # โน้ต: fish มีมากกว่าคลาสอื่น เลยดู macro F1 ที่ให้น้ำหนักทุกคลาสเท่ากัน
    chosen = max(rows, key=lambda row: (row["f1_macro"], -row["val_loss"]))
    selected_folder = output_dir / "experiments" / chosen["configuration"]
    for filename in ["model.keras", "history.json", "training_history.png"]:
        shutil.copy2(selected_folder / filename, output_dir / filename)
    write_json(output_dir / "selection.json", {"rule": "highest validation macro F1, then lowest loss", **chosen})
    write_json(output_dir / "classes.json", CLASSES)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(rows))
    for offset, metric, label in [(-0.18, "accuracy", "Validation accuracy"),
                                 (0.18, "f1_macro", "Validation macro F1")]:
        bars = ax.bar(x + offset, [100 * row[metric] for row in rows], 0.36, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=2)
    ax.set_xticks(x, [row["configuration"] for row in rows])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=150)
    plt.close(fig)
    print(f"Selected by validation: {chosen['configuration']}", flush=True)


def load_model(output_dir):
    model = tf.keras.models.load_model(Path(output_dir) / "model.keras")
    if model.output_shape[-1] != len(CLASSES):
        raise ValueError("Model must be trained for the seven Aquarium classes")
    return model


def evaluate_selected(output_dir, split="test", limit=0, batch=32):
    output_dir = Path(output_dir)
    X, y = load_crops(split, limit)
    model = load_model(output_dir)
    predictions = model.predict(X, batch_size=batch, verbose=0).argmax(axis=1)
    values = {"task": "ground_truth_crop_classification", "split": split,
              "crops": len(y), "classes": CLASSES, **summarize(y, predictions)}
    values["report"] = classification_report(
        y, predictions, labels=list(range(len(CLASSES))), target_names=CLASSES,
        output_dict=True, zero_division=0)
    matrix = confusion_matrix(y, predictions, labels=list(range(len(CLASSES))))
    values["confusion_matrix"] = matrix.tolist()
    write_json(output_dir / "evaluation.json", values)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Aquarium crop classification")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max()/2 else "black")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k:v for k,v in values.items() if k not in {"report","confusion_matrix"}}, indent=2))
    return values
