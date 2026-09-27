import cv2
import numpy as np


def preprocess_image(image, img_size=100):
    """
    แปลงระบบสีของภาพเป็น RGB และปรับขนาดภาพ (Resize)
    
    Args:
        image (numpy.ndarray): ภาพต้นฉบับจาก cv2.imread
        img_size (int): ขนาดเป้าหมาย (กว้างและสูง)
        
    Returns:
        numpy.ndarray or None: ภาพที่แปลงสีและย่อขนาดแล้ว
    """
    # ตรวจสอบความถูกต้องของข้อมูลภาพ
    if image is None or image.size == 0:
        return None

    # แปลงระบบสีเป็น RGB (ภาพ grayscale แปลงเป็น 3 channels, ภาพ BGR แปลงเป็น RGB)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ย่อขนาดภาพเป็น img_size x img_size ด้วย interpolation INTER_AREA
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image


def to_features(images):
    """
    แปลง list หรือ array ของรูปภาพให้อยู่ในรูปแบบ C-contiguous uint8 array
    
    Args:
        images: ข้อมูลรูปภาพ (N, H, W, 3)
        
    Returns:
        numpy.ndarray: array ชนิด uint8 ต่อเนื่องในหน่วยความจำ
    """
    return np.ascontiguousarray(images, dtype=np.uint8)
