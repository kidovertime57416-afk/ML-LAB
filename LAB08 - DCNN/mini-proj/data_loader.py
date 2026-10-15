import os
import cv2
import numpy as np

from preprocessing import preprocess_image

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_data(data_path, img_size=100, max_per_class=None):

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {os.path.abspath(data_path)}\n"
        )

    images = []
    labels = []

    # ใช้ชื่อโฟลเดอร์เป็นคลาส เรียงชื่อให้ label ตรงกันทุกครั้ง
    classes = sorted([
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ])
    print("Detected classes:", classes)
    if len(classes) < 2:
        raise ValueError("Dataset must contain at least two class folders")

    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0
        skipped = 0
        for filename in filenames:
            if max_per_class and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            # ย่อภาพตอนอ่านเลย จะได้ไม่กิน RAM กับภาพต้นฉบับ
            image = preprocess_image(image, img_size)

            # ถ้ารูปเสียให้นับไว้แล้วข้าม
            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")
        if loaded < 10:
            raise ValueError(f"Class {class_name} needs at least 10 readable images for stratified splitting")

    return np.stack(images), np.array(labels), classes
