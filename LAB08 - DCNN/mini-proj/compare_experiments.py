import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from evaluate import evaluate_model, plot_history
from vgg_model import CONFIGS, make_dataset, predict_model, train_model


def save_json(path, values):
    Path(path).write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")


def save_table(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(rows, labels, path, title):
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, key, label in [(-0.25, "train_accuracy", "Train"),
                               (0, "val_accuracy", "Validation"),
                               (0.25, "test_accuracy", "Test")]:
        bars = ax.bar(x + offset, [100 * row[key] for row in rows], width=0.25, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def measure(model, datasets):
    values = {}
    # วัดแบบ inference เหมือนกันทุก split ไม่เปิด dropout/augmentation
    for name, dataset in zip(["train", "val", "test"], datasets):
        loss, accuracy = model.evaluate(dataset, verbose=0)
        values[f"{name}_accuracy"] = float(accuracy)
        values[f"{name}_loss"] = float(loss)
    return values


def validation_rank(row):
    # เลือกด้วย val accuracy ถ้าเท่ากันค่อยดู val loss ห้ามใช้ test เลือก
    return row["val_accuracy"], -row["val_loss"]


def run_experiments(X_train, X_val, X_test, y_train, y_val, y_test,
                    classes, output_dir, config_epochs=20,
                    epoch_list=(10, 20, 30, 50), batch_size=32):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [make_dataset(X, y, batch_size) for X, y in
                [(X_train, y_train), (X_val, y_val), (X_test, y_test)]]
    config_rows = []
    config_dir = output_dir / "configurations"
    config_dir.mkdir(exist_ok=True)

    for name, config in CONFIGS.items():
        print(f"\nConfiguration: {name}, {config_epochs} epochs", flush=True)
        start = perf_counter()
        model, history = train_model(
            X_train, y_train, X_val, y_val, len(classes),
            epochs=config_epochs, batch_size=batch_size, **config)
        seconds = perf_counter() - start
        row = {"configuration": name,
               "conv_layers": sum(n for _, n in config["blocks"]),
               "dense_units": config["dense_units"], "parameters": model.count_params(),
               "epochs": config_epochs, **measure(model, datasets), "seconds": seconds}
        config_rows.append(row)
        save_json(config_dir / f"{name}_history.json", history.history)
        plot_history(history, config_dir / f"{name}_history.png")
        # บันทึกทีละรอบ เผื่อรันไม่จบยังมีผลส่วนที่ทำไปแล้ว
        save_table(output_dir / "comparison_configs.csv", config_rows)

    chosen = max(config_rows, key=validation_rank)["configuration"]
    print(f"\nSelected configuration by validation: {chosen}", flush=True)
    epoch_list = sorted(set(epoch_list))
    epoch_rows = []

    class EpochComparison(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.started = perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            number = epoch + 1
            if number not in epoch_list:
                return
            # อ่านค่าที่ epoch 10/20/30/50 จาก training run เดียวกัน
            # น้ำหนักและ optimizer ต่อเนื่องกัน ไม่ restart และไม่หยุดก่อนครบ
            elapsed = perf_counter() - self.started
            row = {"configuration": chosen, "epochs": number,
                   **measure(self.model, datasets), "seconds": elapsed}
            epoch_rows.append(row)
            if max(epoch_rows, key=validation_rank) is row:
                self.model.save(output_dir / "vgg_model.keras")
            save_table(output_dir / "comparison_epochs.csv", epoch_rows)
            print(f"Checkpoint {number}: {row}", flush=True)

    # เริ่มจาก random seed เดิมอีกรอบเพื่อศึกษา epochs ของ config ที่เลือก
    model, history = train_model(
        X_train, y_train, X_val, y_val, len(classes), epochs=max(epoch_list),
        batch_size=batch_size, callbacks=[EpochComparison()], **CONFIGS[chosen])
    save_json(output_dir / "history.json", history.history)
    plot_history(history, output_dir / "training_history.png")
    plot_comparison(config_rows, [r["configuration"] for r in config_rows],
                    output_dir / "comparison_configs.png", "DCNN configurations (same epochs)")
    plot_comparison(epoch_rows, [str(r["epochs"]) for r in epoch_rows],
                    output_dir / "comparison_epochs.png", f"Epoch checkpoints: {chosen}")

    selected = max(epoch_rows, key=validation_rank)
    # โหลด checkpoint ที่เลือกจาก val กลับมา ไม่ใช้โมเดลรอบสุดท้ายโดยอัตโนมัติ
    model = keras.models.load_model(output_dir / "vgg_model.keras")
    predictions = predict_model(model, X_test)
    accuracy = evaluate_model(y_test, predictions, classes,
                              save_path=output_dir / "confusion_matrix.png")
    probabilities = model.predict(X_test, verbose=0)
    confidence = (np.maximum(probabilities.ravel(), 1 - probabilities.ravel())
                  if probabilities.shape[-1] == 1 else probabilities.max(axis=1))
    save_table(output_dir / "predictions.csv", [
        {"test_index": i, "actual": classes[int(y_test[i])],
         "predicted": classes[int(predictions[i])], "confidence": float(confidence[i])}
        for i in range(len(y_test))])
    report = classification_report(y_test, predictions, labels=list(range(len(classes))),
                                   target_names=classes, zero_division=0, output_dict=True)
    save_json(output_dir / "metrics.json", {
        "selection_rule": "highest validation accuracy, then lowest validation loss",
        "selected_configuration": chosen, "selected_epoch": selected["epochs"],
        "selected_checkpoint": selected, "test_accuracy": float(accuracy),
        "confusion_matrix": confusion_matrix(y_test, predictions,
                                             labels=list(range(len(classes)))).tolist(),
        "classification_report": report,
    })

    # ทำตารางจากผลจริงไว้ใช้เขียนรายงาน ไม่พิมพ์คะแนนขึ้นมาเอง
    lines = ["# LAB08 results", "", "## Configuration comparison", "",
             "| Config | Conv | Dense | Train % | Val % | Test % | Seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in config_rows:
        lines.append(f"| {row['configuration']} | {row['conv_layers']} | {row['dense_units']} | "
                     f"{100*row['train_accuracy']:.2f} | {100*row['val_accuracy']:.2f} | "
                     f"{100*row['test_accuracy']:.2f} | {row['seconds']:.1f} |")
    lines += ["", f"## Epoch comparison ({chosen})", "",
              "| Epoch | Train % | Val % | Test % | Test loss |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for row in epoch_rows:
        lines.append(f"| {row['epochs']} | {100*row['train_accuracy']:.2f} | "
                     f"{100*row['val_accuracy']:.2f} | {100*row['test_accuracy']:.2f} | "
                     f"{row['test_loss']:.4f} |")
    lines += ["", f"Selected by validation: {chosen}, epoch {selected['epochs']}.",
              f"Selected model test accuracy: {100*accuracy:.2f}%."]
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
