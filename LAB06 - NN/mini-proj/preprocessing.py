import cv2
import numpy as np


def preprocess_image(image, img_size=100):
    """
    เตรียมความพร้อมของรูปภาพเดี่ยว: แปลงระบบสีและย่อขนาดภาพ
    
    ขั้นตอนการทำงาน:
    1. ตรวจสอบความถูกต้องของข้อมูลภาพ หากเป็น None หรือว่างเปล่าจะส่งคืน None
    2. แปลงระบบสีของภาพ: OpenCV จะอ่านไฟล์เป็น BGR โดยค่าเริ่มต้น จึงต้องแปลงเป็น RGB
       เพื่อให้ภาพแสดงผลสีได้ถูกต้องตามธรรมชาติและตรงกับโมเดลการเรียนรู้
    3. ปรับขนาดภาพ (Resize) ให้เป็นขนาดมาตรฐาน img_size x img_size พิกเซล
       โดยใช้การประมาณค่าแบบ INTER_AREA ซึ่งเหมาะสมที่สุดสำหรับการย่อขนาดภาพ
       
    Args:
        image (numpy.ndarray): ข้อมูลภาพต้นฉบับจาก OpenCV
        img_size (int): ขนาดเป้าหมาย (กว้างและสูงเท่ากัน)
        
    Returns:
        numpy.ndarray or None: ภาพที่ผ่านการแปลงระบบสีและ Resize แล้ว
    """
    if image is None or image.size == 0:
        return None

    # OpenCV อ่านภาพเข้ามาในรูปแบบ BGR จึงต้องแปลงเป็น RGB เพื่อให้สีถูกต้อง
    if image.ndim == 2:
        # กรณีเป็นภาพขาวดำ (Grayscale) ให้แปลงเป็น RGB 3 ช่องสี
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        # กรณีเป็นภาพสี BGR แปลงเป็น RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ปรับขนาดภาพ (cv2.INTER_AREA ให้คุณภาพการย่อภาพที่ดีที่สุด)
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image


def to_features(images):
    """
    แปลงชุดข้อมูลภาพเป็น Feature Array ที่พร้อมสำหรับป้อนเข้าโมเดล
    
    รายละเอียด:
    - เก็บข้อมูลในรูปแบบ uint8 ต่อเนื่องในหน่วยความจำ (ascontiguousarray) เพื่อประหยัด RAM
    - การปรับสเกลค่าพิกเซล 0-255 เป็น 0-1 จะถูกทำผ่านชั้น Rescaling ภายในโมเดล Keras โดยตรง
      เพื่อป้องกันข้อผิดพลาดในการลืม Normalization ระหว่างขั้นตอน Inference หรือทดสอบ
      
    Args:
        images (list หรือ numpy.ndarray): รายการของรูปภาพขนาด (N, H, W, 3)
        
    Returns:
        numpy.ndarray: Array ต่อเนื่องในรูปแบบ uint8 มิติ (N, H, W, 3)
    """
    return np.ascontiguousarray(images, dtype=np.uint8)
