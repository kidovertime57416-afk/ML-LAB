import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from settings import DATA_YAML, ROOT, RUNS_DIR, SELECTED_MODEL, check_dataset, get_device
from train import train_model

# เปลี่ยนทีละอย่าง จะได้ดูผลของ image size และ batch size แยกกัน
CONFIGS = [
    {"name": "img480_batch8", "imgsz": 480, "batch": 8},
    {"name": "img320_batch8", "imgsz": 320, "batch": 8},
    {"name": "img480_batch16", "imgsz": 480, "batch": 16},
]
EPOCHS = 10
EPOCH_CHECKPOINTS = (1, 5, 10)
METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
}


def read_history(csv_path):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as file:
        rows = [{key.strip(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(file)]
    if not rows:
        raise ValueError(f"Empty training history: {csv_path}")
    for row in rows:
        # ตรวจว่ามี metric ครบก่อนนำไปทำตารางหรือกราฟ
        for column in ["epoch", *METRICS.values()]:
            float(row[column])
    return rows


def scores(row):
    return {name: float(row[column]) for name, column in METRICS.items()}


def save_table(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_epochs(csv_path, output_dir, checkpoints=EPOCH_CHECKPOINTS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = read_history(csv_path)
    rows = [{"epoch": int(float(row["epoch"])), **scores(row)}
            for row in history if int(float(row["epoch"])) in checkpoints]
    if len(rows) != len(set(checkpoints)):
        raise ValueError("Training history does not contain all requested checkpoints")
    save_table(output_dir / "comparison_epochs.csv", rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in METRICS:
        ax.plot([row["epoch"] for row in rows],
                [100 * row[metric] for row in rows], marker="o", label=metric)
    ax.set_xticks([row["epoch"] for row in rows])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation metric (%)")
    ax.set_title("YOLOv8n: epoch checkpoints from one training run")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_epochs.png", dpi=150)
    plt.close(fig)
    return rows


def plot_configs(rows, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(rows))
    width = 0.2
    for index, metric in enumerate(METRICS):
        bars = ax.bar(positions + (index - 1.5) * width,
                      [100 * row[metric] for row in rows], width, label=metric)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(positions, [row["configuration"] for row in rows])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Validation metric (%)")
    ax.set_title("YOLOv8n: training configurations at 10 epochs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    check_dataset()
    project = RUNS_DIR / "comparisons"
    rows = []
    for config in CONFIGS:
        # เริ่มจาก pretrained เดียวกัน ใช้ dataset และ seed เดียวกันทุกรอบ
        save_dir = train_model(epochs=EPOCHS, project=project, **config)
        history = read_history(save_dir / "results.csv")
        # วัด best checkpoint ของแต่ละ config บน val เหมือนกันทุกตัว
        from ultralytics import YOLO
        model = YOLO(str(save_dir / "weights" / "best.pt"))
        metrics = model.val(data=str(DATA_YAML), split="val", imgsz=config["imgsz"],
                            batch=config["batch"], device=get_device(), workers=0,
                            plots=False, project=str(save_dir), name="validation", exist_ok=True)
        row = {"configuration": config["name"], "imgsz": config["imgsz"],
               "batch": config["batch"], "epochs": int(float(history[-1]["epoch"])),
               "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
               "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
               "run_dir": save_dir.relative_to(ROOT).as_posix()}
        rows.append(row)
        summarize_epochs(save_dir / "results.csv", save_dir)
        # ใช้ validation เทียบ config เก็บ test ไว้ประเมินโมเดลสุดท้าย
        save_table(project / "comparison_configs.csv", rows)
    plot_configs(rows, project / "comparison_configs.png")
    chosen = max(rows, key=lambda row: row["mAP50_95"])
    selected = {**chosen, "weights": chosen["run_dir"] + "/weights/best.pt",
                "selection_rule": "highest validation mAP50-95"}
    SELECTED_MODEL.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    summarize_epochs(ROOT / chosen["run_dir"] / "results.csv", RUNS_DIR / "summary")
    print(f"Selected by validation: {chosen['configuration']}", flush=True)


if __name__ == "__main__":
    main()
