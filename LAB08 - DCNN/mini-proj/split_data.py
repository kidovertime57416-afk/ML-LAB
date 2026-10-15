import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, val_size=0.1):
    y = np.asarray(y)

    if not (0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1):
        raise ValueError("test_size and val_size must be positive and sum to less than 1")
    # แยก test เก็บไว้ก่อน ใช้ stratify ให้สัดส่วนแต่ละคลาสใกล้กัน
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=42,
        stratify=y
    )

    # val_size คิดจากข้อมูลทั้งหมด เลยต้องหารด้วยสัดส่วนที่เหลือ
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio,
        random_state=42,
        stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
