import json
import os

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes=2, hidden_layers=(256, 128, 64), dropout_rate=0.3):
    """
    สร้างสถาปัตยกรรมโครงข่ายประสาทเทียม (Pure Feedforward Neural Network / MLP)
    
    โครงสร้างสถาปัตยกรรมโมเดล:
    1. Input Layer: รับข้อมูลรูปภาพมิติ (100, 100, 3)
    2. Rescaling Layer: แปลงค่าพิกเซลจาก [0, 255] ให้อยู่ในช่วง [0, 1] เพื่อความเสถียรในการคำนวณ Gradient
    3. Flatten Layer: คลี่รูปภาพ 2D (100x100x3) ให้กลายเป็น 1D Vector ขนาด 30,000 มิติ
    4. Hidden Dense Layers: ชั้น Fully-Connected พร้อมฟังก์ชันกระตุ้น ReLU 
       และ L2 Regularization เพื่อควบคุมค่าน้ำหนักไม่ให้สูงเกินไป
    5. Dropout Layers: สุ่มปิดการทำงานของโหนดบางส่วน (30%) ในระหว่างฝึกสอน เพื่อป้องกัน Overfitting
    6. Output Layer: 
       - สำหรับ Binary Classification (2 คลาส): ใช้ 1 Node พร้อม Activation 'sigmoid' (ช่วง 0.0 - 1.0)
       - สำหรับ Multi-class: ใช้จำนวน Node เท่ากับคลาส พร้อม Activation 'softmax'
       
    Args:
        input_shape (tuple): มิติข้อมูลนำเข้า เช่น (100, 100, 3)
        num_classes (int): จำนวนคลาสของผลลัพธ์ (default: 2 สำหรับ Lion vs Tiger)
        hidden_layers (tuple): จำนวน Hidden Neurons ในแต่ละชั้น เช่น (256, 128, 64)
        dropout_rate (float): อัตราการสุ่มตัดโหนด Dropout (default: 0.3)
        
    Returns:
        keras.Model: โมเดล Keras Sequential ที่คอมไพล์พร้อมสำหรับการฝึกสอน
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Flatten(),
    ])

    # เพิ่มชั้น Hidden Dense และ Dropout ตามโครงสร้างที่กำหนด
    for units in hidden_layers:
        model.add(layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-4)
        ))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # ชั้น Output Layer สำหรับการจำแนกประเภท
    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    ))

    # กำหนด Optimizer, Loss Function และ Metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=30, batch_size=32, hidden_layers=(256, 128, 64)):
    """
    สร้าง ฝึกสอน และบันทึกโมเดล Neural Network พร้อมประวัติการเรียนรู้
    
    การควบคุมการฝึกสอน (Callbacks):
    - EarlyStopping: หยุดการฝึกสอนอัตโนมัติหากค่า val_loss ไม่ดีขึ้นต่อเนื่อง 7 รอบ เพื่อป้องกัน Overfitting
      พร้อมคืนค่าน้ำหนักที่ดีที่สุดกลับมา (restore_best_weights=True)
    - ReduceLROnPlateau: ลดอัตราการเรียนรู้ (Learning Rate) ลง 50% หากค่า val_loss คงที่ 3 รอบ
    
    Args:
        X_train, y_train: ข้อมูลและ Label สำหรับการฝึกสอน
        X_val, y_val: ข้อมูลและ Label สำหรับการตรวจสอบ
        num_classes: จำนวนคลาส
        output_dir: โฟลเดอร์ปลายทางสำหรับบันทึกโมเดลและประวัติ (history.json)
        epochs: จำนวนรอบสูงสุดในการฝึกสอน
        batch_size: ขนาดของชุดข้อมูลในแต่ละรอบการปรับค่าน้ำหนัก
        hidden_layers: สถาปัตยกรรม Hidden Layers
        
    Returns:
        tuple: (model, history)
    """
    model = build_model(X_train.shape[1:], num_classes, hidden_layers=hidden_layers)
    model.summary()

    # ตั้งค่า Callbacks เพื่อควบคุมประสิทธิภาพและป้องกัน Overfitting
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    print("\nTraining Neural Network...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # บันทึกไฟล์โมเดล .keras และประวัติ history.json
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "nn_model.keras")
        history_path = os.path.join(output_dir, "history.json")

        model.save(model_path)
        with open(history_path, "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f, indent=2)

        print(f"Saved: {model_path}")

    return model, history


def predict_model(model, X_test):
    """
    พยากรณ์คลาสของข้อมูลทดสอบจากโมเดลที่ผ่านการเทรนแล้ว
    
    Args:
        model: โมเดล Keras ที่ฝึกสอนแล้ว
        X_test: ข้อมูลภาพทดสอบ (N, 100, 100, 3)
        
    Returns:
        numpy.ndarray: ค่า Label ผลการทำนาย (0 หรือ 1)
    """
    probabilities = model.predict(X_test, verbose=0)

    # กรณี Binary Head (Sigmoid): หากค่าความน่าจะเป็น >= 0.5 จะทำนายเป็นคลาส 1 มิฉะนั้นคลาส 0
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() >= 0.5).astype(int)

    # กรณี Multi-class Head (Softmax): เลือกคลาสที่มีความน่าจะเป็นสูงสุด
    return probabilities.argmax(axis=1)



