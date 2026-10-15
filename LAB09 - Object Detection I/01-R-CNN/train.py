from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "outputs"

import tensorflow as tf
from data_loader import IMG_SIZE, NUM_CLASSES
from experiments import run_experiments


# โน้ตผลรอบนี้: Aquarium, seed 42, 10 epochs
# Dropout 0.3 ทำให้ val macro F1 จาก 0.4676 เหลือ 0.3840 เลยเลือก baseline
def make_model(dropout=0.0):
    # โน้ต: คง Conv + GlobalAveragePooling แบบเดิม เพื่อเทียบผล dropout อย่างเดียว
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        # โน้ต: 0.0 เป็น baseline, 0.3 ลองลดการจำ crop ตอน train
        tf.keras.layers.Dropout(dropout, seed=42),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="RCNN_crop_classifier")
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
