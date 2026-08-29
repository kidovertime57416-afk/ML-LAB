"""Test the trained model on random test images (2x2 grid).

Random sample every run. Run main.py first.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

N_SAMPLES = 4


def test_nn(n_samples=N_SAMPLES):
    model_path = os.path.join(OUTPUT_DIR, "nn_model.keras")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run main.py first.")
        return

    # Load model and test set
    model = keras.models.load_model(model_path)
    X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
    with open(os.path.join(OUTPUT_DIR, "classes.json")) as f:
        classes = json.load(f)

    # Pick random images
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]

    # Predict
    probabilities = model.predict(X_sample, verbose=0)
    if probabilities.shape[-1] == 1:
        probabilities = probabilities.ravel()
        predictions = (probabilities >= 0.5).astype(int)
        confidence = np.where(predictions == 1, probabilities, 1.0 - probabilities)
    else:
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)

    # Show results in a grid
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]
        true = classes[y_sample[i]]
        correct = (predictions[i] == y_sample[i])
        color = "#1b8a2e" if correct else "#d32f2f"

        ax.imshow(X_sample[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Pred: {pred} ({confidence[i] * 100:.1f}%)\nTrue: {true}",
            color=color,
            fontsize=12,
            fontweight="bold"
        )

        print(f"[{i + 1}] Pred: {pred:<6} True: {true:<6} "
              f"conf {confidence[i] * 100:5.1f}%  "
              f"{'OK' if correct else 'WRONG'}")

    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples} ({correct_total/n_samples*100:.1f}%)")

    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "prediction_sample.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    test_nn()

