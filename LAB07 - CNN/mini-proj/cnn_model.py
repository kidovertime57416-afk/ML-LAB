import json
import os

from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    input_shape,
    num_classes=2,
    conv_blocks=((32, (3, 3)), (64, (3, 3)), (128, (3, 3))),
    dense_units=128,
    dropout_rate=0.3
):
    """
    สร้างโครงสร้างโมเดล CNN (Conv2D -> MaxPooling2D -> Flatten -> Dense -> Output)
    
    Args:
        input_shape (tuple): ขนาดมิติภาพนำเข้า เช่น (100, 100, 3)
        num_classes (int): จำนวนคลาสผลลัพธ์
        conv_blocks (tuple): รายการ (filters, kernel_size) ของแต่ละชั้น Conv
        dense_units (int): จำนวนโหนดในชั้น Dense
        dropout_rate (float): อัตราการสุ่มตัดโหนด Dropout
        
    Returns:
        keras.Model: โมเดล Keras Sequential ที่คอมไพล์แล้ว
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    
    # ปรับสเกลค่าพิกเซลจาก 0-255 ให้อยู่ในช่วง 0-1
    model.add(layers.Rescaling(1.0 / 255))

    # เพิ่มชั้น Conv2D และ MaxPooling2D ตาม conv_blocks
    for idx, (filters, kernel_size) in enumerate(conv_blocks):
        model.add(layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name=f"conv2d_layer_{idx+1}"
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), name=f"maxpool_layer_{idx+1}"))

    # คลี่ Feature Map เป็นเวกเตอร์ 1D
    model.add(layers.Flatten(name="flatten_layer"))

    # ชั้น Fully-Connected และ Dropout
    if dense_units > 0:
        model.add(layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name="dense_fc"
        ))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name="dropout_fc"))

    # ชั้น Output (sigmoid สำหรับ 2 คลาส, softmax สำหรับ multi-class)
    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax",
        name="output_layer"
    ))

    # คอมไพล์โมเดลด้วย Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    X_train, y_train, X_val, y_val, num_classes,
    output_dir=None, epochs=30, batch_size=32,
    conv_blocks=((32, (3, 3)), (64, (3, 3)), (128, (3, 3))),
    dense_units=128
):
    """
    สร้างและเทรนโมเดล CNN พร้อมบันทึกประวัติและตัวโมเดลลงไฟล์
    """
    model = build_model(
        X_train.shape[1:],
        num_classes=num_classes,
        conv_blocks=conv_blocks,
        dense_units=dense_units
    )
    model.summary()

    # ตั้งค่า Callbacks: หยุดเมื่อ val_loss ไม่ดีขึ้น และลด learning rate เมื่อ loss นิ่ง
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    print("\nTraining CNN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # บันทึกโมเดลและประวัติการฝึกสอนลงโฟลเดอร์ปลายทาง
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "cnn_model.keras")
        history_path = os.path.join(output_dir, "history.json")

        model.save(model_path)
        with open(history_path, "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f, indent=2)

        print(f"Saved: {model_path}")

    return model, history


def predict_model(model, X_test):
    """
    ทำนายผลลัพธ์จากชุดข้อมูลทดสอบ
    
    Args:
        model: โมเดล Keras ที่ฝึกสอนแล้ว
        X_test: ข้อมูลภาพทดสอบ (N, H, W, C)
        
    Returns:
        numpy.ndarray: class label ที่ทำนายได้ (0 หรือ 1)
    """
    probabilities = model.predict(X_test, verbose=0)

    # ถ้าเป็น binary output (sigmoid) ใช้ threshold 0.5
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() >= 0.5).astype(int)

    # ถ้าเป็น multi-class output (softmax) เลือกค่าความน่าจะเป็นสูงสุด
    return probabilities.argmax(axis=1)
