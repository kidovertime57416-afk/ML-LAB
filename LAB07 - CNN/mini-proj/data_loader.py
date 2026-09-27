import os
import cv2
import numpy as np

from preprocessing import preprocess_image

# กำหนดนามสกุลไฟล์ภาพที่รองรับ
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_data(data_path, img_size=100, max_per_class=None):
    """
    โหลดไฟล์ภาพจากโฟลเดอร์แยกตามคลาส ย่อขนาดภาพ และแปลงเป็น NumPy array
    
    Args:
        data_path (str): โฟลเดอร์เก็บข้อมูลภาพหลัก
        img_size (int): ขนาดภาพที่ต้องการย่อ (กว้าง x สูง)
        max_per_class (int, optional): จำนวนภาพสูงสุดต่อคลาส
        
    Returns:
        tuple: (images_array, labels_array, class_names)
    """
    # ตรวจสอบว่ามีโฟลเดอร์ข้อมูลอยู่จริงหรือไม่
    if not os.path.exists(data_path):
        alt_path = os.path.join(os.path.dirname(data_path), "..", "LAB06 - NN", "Animal")
        if os.path.exists(alt_path):
            data_path = alt_path

    images = []
    labels = []

    # ดึงรายชื่อโฟลเดอร์ย่อยมาใช้เป็นชื่อคลาส
    classes = sorted([
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ])
    print("Detected classes:", classes)

    # วนลูปอ่านรูปภาพทีละคลาส
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

            # ย่อขนาดภาพและแปลงระบบสีเป็น RGB
            image = preprocess_image(image, img_size)

            # ข้ามไฟล์ที่อ่านไม่ได้หรือไฟล์เสียหาย
            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")

    # รวมเป็น array มิติ (N, Height, Width, Channels)
    return np.stack(images), np.array(labels), classes
