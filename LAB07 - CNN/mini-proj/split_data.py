import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, val_size=0.1):
    """
    แบ่งชุดข้อมูลเป็น Train, Validation และ Test Set แบบรักษาสัดส่วนคลาส (Stratified)
    
    Args:
        X (numpy.ndarray): ข้อมูลภาพ (N, 100, 100, 3)
        y (numpy.ndarray): ข้อมูล label (N,)
        test_size (float): สัดส่วน test set จากทั้งหมด
        val_size (float): สัดส่วน validation set จากทั้งหมด
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    y = np.asarray(y)

    # 1. แยกชุดทดสอบ (Test Set) ออกมาก่อน
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # 2. แยกชุดตรวจสอบ (Validation Set) จากข้อมูลที่เหลือ
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_ratio,
        random_state=42,
        stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
