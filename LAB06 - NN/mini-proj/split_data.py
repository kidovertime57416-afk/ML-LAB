import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, val_size=0.1):
    """
    แบ่งชุดข้อมูลออกเป็น 3 ส่วน: Training Set, Validation Set และ Test Set
    
    หลักการแบ่งข้อมูล:
    1. ใช้การแบ่งแบบ Stratified Sampling (`stratify=y`) เพื่อรักษาสัดส่วนของแต่ละคลาส (Lion และ Tiger)
       ให้มีความสมดุลเท่ากันในทุกชุดข้อมูล ป้องกันปัญหา Class Imbalance
    2. ขั้นแรก: แยก Test Set ออกมาตามสัดส่วน test_size (เช่น 20% หรือ 92 ภาพ)
    3. ขั้นสอง: คำนวณอัตราส่วนที่เหลือ แล้วแยก Validation Set ออกมาตามสัดส่วน val_size (เช่น 10% หรือ 46 ภาพ)
    4. ส่วนที่เหลือทั้งหมดเป็น Training Set (70% หรือ 322 ภาพ)
    
    Args:
        X (numpy.ndarray): ข้อมูล Feature Array รูปภาพ (N, 100, 100, 3)
        y (numpy.ndarray): ข้อมูล Label (N,)
        test_size (float): สัดส่วนของชุดข้อมูลทดสอบ (default: 0.2)
        val_size (float): สัดส่วนของชุดข้อมูลตรวจสอบ (default: 0.1)
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    y = np.asarray(y)

    # ขั้นที่ 1: แยก Test Set ออกมาจากข้อมูลทั้งหมดก่อน (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # ขั้นที่ 2: คำนวณสัดส่วน Validation Set จากข้อมูลที่เหลืออยู่ (10% ของทั้งหมด)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        random_state=42,
        stratify=y_train
    )

    # ส่งคืนชุดข้อมูลทั้ง 3 ส่วน
    return X_train, X_val, X_test, y_train, y_val, y_test
