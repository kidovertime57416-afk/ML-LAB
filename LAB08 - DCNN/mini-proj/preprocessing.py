import cv2
import numpy as np


def preprocess_image(image, img_size=100):

    if image is None or image.size == 0:
        return None

    # OpenCV อ่านเป็น BGR เลยสลับเป็น RGB ให้สีตรงกับภาพที่แสดง
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # resize ให้ทุกภาพมีขนาดเท่ากันก่อนเข้าโมเดล
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image


def to_features(images):
    # เก็บเป็น uint8 ก่อน ส่วน normalize ทำใน Rescaling ของโมเดล
    return np.ascontiguousarray(images, dtype=np.uint8)
