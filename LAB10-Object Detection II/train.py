from pathlib import Path

from settings import DATA_YAML, ROOT, TRAIN_PROJECT, check_dataset, get_device


def train_model(epochs=10, imgsz=480, batch=8, name="aquarium", project=TRAIN_PROJECT):
    check_dataset()
    from ultralytics import YOLO

    # โหลด pretrained ใหม่ทุกรอบ เพื่อให้แต่ละ config เริ่มจากน้ำหนักเดียวกัน
    model = YOLO(str(ROOT / "yolov8n.pt"))
    model.train(
        data=str(DATA_YAML), epochs=epochs, imgsz=imgsz, batch=batch,
        device=get_device(), workers=0, amp=True, cache=False,
        project=str(project), name=name,
        exist_ok=(name == "aquarium" and project == TRAIN_PROJECT),
        seed=0, deterministic=True, patience=0, save=True, plots=True,
        optimizer="AdamW", lr0=0.001, warmup_epochs=1,
    )
    # อ่าน save_dir จริง เพราะถ้ามีชื่อซ้ำ Ultralytics จะเพิ่มเลขให้
    save_dir = Path(model.trainer.save_dir)
    print(f"Best model: {save_dir / 'weights' / 'best.pt'}")
    print(f"Training CSV: {save_dir / 'results.csv'}")
    return save_dir


def main():
    train_model()


if __name__ == "__main__":
    main()
