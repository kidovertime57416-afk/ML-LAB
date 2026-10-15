from pathlib import Path
import random

from settings import DATA_YAML, RUNS_DIR, get_device, load_model, model_info
import yaml


def main():
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    dataset_root = Path(data.get("path") or DATA_YAML.parent)
    if not dataset_root.is_absolute():
        dataset_root = DATA_YAML.parent / dataset_root
    image_dir = dataset_root / data["test"]
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found: {image_dir}")
    images = sorted(p for p in image_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise ValueError(f"No test images in {image_dir}")

    model = load_model()
    info = model_info()
    # fix seed ให้สุ่มได้ภาพเดิมตอนกลับมาดูผล
    selected = random.Random(42).sample(images, min(5, len(images)))
    for image in selected:
        # save=True เก็บภาพพร้อม box/class/confidence ไว้ใน runs/aquarium/image_test
        results = model.predict(
            source=str(image), conf=0.25, imgsz=info["imgsz"], device=get_device(),
            save=True, project=str(RUNS_DIR), name="image_test", exist_ok=True,
        )
        for result in results:
            print(f"Image: {image.name}")
            for box in result.boxes:
                print(f"{model.names[int(box.cls[0])]}: {float(box.conf[0]):.2f}")


if __name__ == "__main__":
    main()
