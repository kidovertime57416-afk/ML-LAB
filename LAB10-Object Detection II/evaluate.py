import json
import numpy as np

from settings import DATA_YAML, RUNS_DIR, check_dataset, get_device, load_model, model_info


def main():
    # test ใช้วัดผลสุดท้าย แยกจาก val ที่ใช้ดูผลระหว่างเทรน
    check_dataset(("test",))
    model = load_model()
    info = model_info()
    metrics = model.val(
        data=str(DATA_YAML), split="test", imgsz=info["imgsz"], batch=info["batch"],
        device=get_device(), workers=0, plots=True,
        project=str(RUNS_DIR), name="evaluation", exist_ok=True,
    )
    scores = {
        "split": "test", "configuration": info.get("configuration", "aquarium"),
        "weights": info["weights"], "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr), "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        # แปลงตัวเลข NumPy ให้เป็นชนิดปกติก่อนเขียน JSON
        "per_class": [
            {key: value.item() if isinstance(value, np.generic) else value
             for key, value in row.items()}
            for row in metrics.summary()
        ],
    }
    # เก็บตัวเลขคู่กับรูป evaluation ไว้ดูทีหลัง
    (metrics.save_dir / "metrics.json").write_text(
        json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
