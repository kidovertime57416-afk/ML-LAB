import json
import os

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes=2, hidden_layers=(256, 128, 64), dropout_rate=0.3):
    """Fully-connected Neural Network (Multi-Layer Perceptron - MLP) for Image Classification.
    
    Architecture:
    - Input layer (100, 100, 3)
    - Rescaling(1./255) to standardize pixel values to [0, 1]
    - Flatten() to convert 2D image into 1D feature vector
    - Hidden Dense layers with ReLU activation and Dropout regularization
    - Output Dense layer with Sigmoid activation for binary classification
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Flatten(),
    ])

    for units in hidden_layers:
        model.add(layers.Dense(units, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    ))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(X_train, y_train, X_val, y_val, num_classes,
                output_dir=None, epochs=30, batch_size=32, hidden_layers=(256, 128, 64)):
    """Build, train and save the Neural Network model. Returns (model, history)."""

    model = build_model(X_train.shape[1:], num_classes, hidden_layers=hidden_layers)
    model.summary()

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

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        model.save(os.path.join(output_dir, "nn_model.keras"))
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.history.items()}, f, indent=2)

        print(f"Saved: {os.path.join(output_dir, 'nn_model.keras')}")

    return model, history


def predict_model(model, X_test):
    """Predict class labels for given inputs."""
    probabilities = model.predict(X_test, verbose=0)

    # Binary head outputs one probability, multiclass outputs one per class
    if probabilities.shape[-1] == 1:
        return (probabilities.ravel() >= 0.5).astype(int)

    return probabilities.argmax(axis=1)



