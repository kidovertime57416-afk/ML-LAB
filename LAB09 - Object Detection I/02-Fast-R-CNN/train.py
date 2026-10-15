from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "outputs"

import tensorflow as tf
from data_loader import IMG_SIZE, NUM_CLASSES
from experiments import run_experiments


# โน้ตผลรอบนี้: Dropout 0.3 เพิ่ม val accuracy 77.23% -> 79.10%
# แต่ val macro F1 ลด 0.6734 -> 0.6440 เลยเลือก baseline ตามเกณฑ์เดิม
def make_model(dropout=0.0):
    # โน้ต: ตัวนี้ยังเป็น crop classifier แบบเดิม ไม่ได้เพิ่ม RPN หรือ box regression
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    # โน้ต: pool_size=4 ย่อ crop feature แบบเดิม เป็นแค่ approximation ของ ROI pooling
    x = tf.keras.layers.MaxPooling2D(pool_size=4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout, seed=42)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="FastRCNN_crop_classifier")
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.limit < 0:
        parser.error("epochs/batch must be positive; limit must be zero or positive")
    run_experiments(make_model, OUT, args.epochs, args.batch, args.limit)


if __name__ == "__main__":
    main()
