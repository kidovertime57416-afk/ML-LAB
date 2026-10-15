import cv2

from settings import get_device, load_model, model_info


def main():
    model = load_model()
    info = model_info()
    device = get_device()
    # เปิด webcam ตัวแรกของเครื่อง
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open webcam (camera index 0)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # ทายทีละ frame แล้ววาด bounding box กลับลงบนภาพ
            results = model.predict(source=frame, conf=0.25, imgsz=info["imgsz"],
                                    device=device, verbose=False)
            cv2.imshow("Aquarium YOLO Real-time Detection", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # คืนกล้องให้ระบบเมื่อปิดหน้าต่างหรืออ่านภาพไม่สำเร็จ
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
