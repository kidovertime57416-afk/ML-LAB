import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# ใช้ Conv 3x3 ซ้อนกันแบบ VGG แต่ลด filters ให้เทรนบน CPU ได้
# แต่ละ tuple คือ (จำนวน filters, จำนวน Conv ใน block)
VGG16 = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]
VGG_SMALL = [(16, 2), (32, 2), (64, 2)]
CONFIGS = {
    "vgg_6conv": {"blocks": VGG_SMALL, "dense_units": 64},
    "vgg_8conv": {"blocks": [(16, 2), (32, 2), (64, 2), (128, 2)], "dense_units": 128},
    "vgg_10conv": {"blocks": [(16, 2), (32, 2), (64, 3), (128, 3)], "dense_units": 128},
}


def vgg_block(model, filters, n_conv):
    # Conv หลายชั้นช่วยหาลาย/ขอบภาพ แล้วใช้ pooling ย่อ feature map
    for _ in range(n_conv):
        model.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(2))


def build_model(input_shape, num_classes, blocks=VGG_SMALL, dense_units=64):
    if num_classes < 2:
        raise ValueError("Need at least two classes")
    if min(input_shape[:2]) < 2 ** len(blocks):
        raise ValueError("Image too small for the number of pooling blocks")
    model = keras.Sequential([keras.Input(shape=input_shape)])
    # รับภาพ 0-255 แล้ว normalize เป็น 0-1 ทั้งตอน train และ predict
    model.add(layers.Rescaling(1.0 / 255))
    model.add(layers.RandomFlip("horizontal"))
    for filters, n_conv in blocks:
        vgg_block(model, filters, n_conv)
    # Flatten เก็บตำแหน่ง feature ไว้ แล้วส่งเข้า Dense เพื่อแยกคลาส
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1 if num_classes == 2 else num_classes,
                           activation="sigmoid" if num_classes == 2 else "softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_dataset(X, y, batch_size=32, training=False):
    import tensorflow as tf

    data = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        data = data.shuffle(len(y), seed=42, reshuffle_each_iteration=True)
    # จำกัด threads ไม่ให้การทดลองใช้ CPU มากเกินไป
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return data.batch(batch_size).with_options(options).prefetch(1)


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=20, batch_size=32, blocks=VGG_SMALL,
                dense_units=64, callbacks=None):
    keras.backend.clear_session()
    keras.utils.set_random_seed(42)
    model = build_model(X_train.shape[1:], num_classes, blocks, dense_units)
    # ไม่ใช้ early stopping ในการเปรียบเทียบ จะได้ครบ epochs ที่กำหนดจริง
    history = model.fit(
        make_dataset(X_train, y_train, batch_size, training=True),
        validation_data=make_dataset(X_val, y_val, batch_size),
        epochs=epochs, callbacks=callbacks or [], verbose=2,
    )
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(output_dir / "vgg_model.keras")
        (output_dir / "history.json").write_text(
            json.dumps(history.history, indent=2), encoding="utf-8")
    return model, history


def predict_model(model, X_test):
    probabilities = model.predict(X_test, verbose=0)
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() >= 0.5).astype(int)
    return np.argmax(probabilities, axis=1)
