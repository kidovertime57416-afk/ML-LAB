from settings import ROOT, RUNS_DIR, get_device, load_model, model_info

VIDEO_PATH = ROOT / "dataset" / "test" / "video.mp4"


def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    model = load_model()
    info = model_info()
    # stream=True อ่านผลทีละ frame ไม่เก็บทั้งวิดีโอไว้ใน RAM
    results = model.predict(
        source=str(VIDEO_PATH), stream=True, conf=0.25, imgsz=info["imgsz"],
        device=get_device(), save=True, project=str(RUNS_DIR),
        name="video_test", exist_ok=False,
    )
    for result in results:
        print(f"Frame detected objects: {len(result.boxes)}")


if __name__ == "__main__":
    main()
