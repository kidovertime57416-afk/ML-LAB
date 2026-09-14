import os
import cv2
import numpy as np

from preprocessing import preprocess_image

# กำหนดนามสกุลไฟล์ภาพที่ระบบรองรับ
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_data(data_path, img_size=100, max_per_class=None):
    """
    โหลดและรวบรวมรูปภาพจากโฟลเดอร์ของแต่ละคลาส พร้อมแปลงเป็น NumPy Array
    
    ขั้นตอนการทำงาน:
    1. ตรวจหาโฟลเดอร์ย่อยใน data_path เพื่อกำหนดเป็นรายชื่อคลาสโดยอัตโนมัติ (เช่น Lion, tiger)
    2. วนลูปอ่านไฟล์ภาพในแต่ละโฟลเดอร์คลาสตามนามสกุลที่ถูกต้อง
    3. เรียกฟังก์ชัน preprocess_image เพื่อแปลงระบบสี (BGR -> RGB) และย่อขนาดภาพ (Resize)
    4. คัดกรองและข้ามไฟล์ภาพที่เสียหายหรือไม่สามารถอ่านได้
    5. รวบรวมรูปภาพและ Label ที่แปลงเป็นตัวเลข (0, 1) ส่งกลับในรูป NumPy Array
    
    Args:
        data_path (str): เส้นทางโฟลเดอร์เก็บข้อมูลภาพหลัก (เช่น ../Animal)
        img_size (int): ขนาดความกว้างและความสูงของภาพหลังการ Resize (default: 100)
        max_per_class (int, optional): จำนวนภาพสูงสุดที่จะโหลดต่อคลาส (None = โหลดทั้งหมด)
        
    Returns:
        tuple: (images_array, labels_array, class_names)
    """
    images = []
    labels = []

    # ตรวจหาโฟลเดอร์ย่อยใน data_path โดยอัตโนมัติเพื่อใช้เป็นชื่อคลาส
    classes = sorted([
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ])
    print("Detected classes:", classes)

    # วนลูปอ่านรูปภาพจากโฟลเดอร์ของแต่ละคลาส
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0
        skipped = 0
        for filename in filenames:
            # ตรวจสอบขีดจำกัดจำนวนภาพสูงสุดต่อคลาส
            if max_per_class and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            # ย่อขนาดและปรับระบบสีทันทีเพื่อประหยัดการใช้หน่วยความจำ (RAM)
            image = preprocess_image(image, img_size)

            # ตรวจสอบและข้ามภาพที่ไม่สามารถอ่านได้หรือไฟล์เสียหาย
            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")

    # รวมภาพทั้งหมดเป็น NumPy Array มิติ (N, Height, Width, Channels)
    return np.stack(images), np.array(labels), classes
