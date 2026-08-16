# ML-05: Support Vector Machine (SVM) Image Classification

## Overview

A Python pipeline for binary image classification between alligators and crocodiles using Support Vector Machines (SVM). The pipeline handles image loading, grayscale conversion, normalization, PCA dimensionality reduction, and RBF-kernel SVM classification.

---

## Dataset

* **Source:** [Kaggle Alligator vs Crocodile Dataset](https://www.kaggle.com/datasets/azharn/alligator-vs-crocodile1)
* **Classes:** `Alligator`, `Crocodile`
* **Target Location:** `mini-proj/Alligator-vs-Crocodile/`

---

## What It Does

* **`data_loader.py`:** Automatically scans class subdirectories, reads images via OpenCV, handles image resizing, and filters damaged or unreadable files.
* **`preprocessing.py`:** Converts images to grayscale, resizes them to 100x100 pixels, flattens pixel arrays into 1D vectors (10,000 features), and scales pixel values to the range [0, 1].
* **`split_data.py`:** Splits processed feature arrays into training (80%) and testing (20%) subsets with class stratification.
* **`svm_model.py`:** Builds a Scikit-Learn `Pipeline` combining `StandardScaler`, `PCA` (150 components with whitening), and an RBF-kernel `SVC` (C=10) for high-dimensional classification.
* **`evaluate.py`:** Evaluates test predictions, prints accuracy, precision, recall, and F1-score, and exports the confusion matrix plot (`outputs/confusion_matrix.png`).
* **`test_svm.py`:** Samples random test images to perform visual inference, rendering a labeled prediction grid (`outputs/prediction_sample.png`).
* **`main.py`:** Orchestrates the end-to-end workflow and serializes dataset splits, labels, fitted pipelines, and models to disk.

---

## Tech Stack

* **Language:** Python
* **Computer Vision & Image Processing:** OpenCV (`cv2`)
* **Machine Learning & Preprocessing:** Scikit-Learn (`SVC`, `PCA`, `StandardScaler`, `Pipeline`, `train_test_split`, `metrics`)
* **Numerical Computing & Persistence:** NumPy, Joblib
* **Visualization:** Matplotlib (`Agg` headless backend)

---

## Project Structure

```text
ML-05-SVM/
├── mini-proj/
│   ├── Alligator-vs-Crocodile/
│   │   ├── Alligator/
│   │   └── Crocodile/
│   │
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── svm_model.py
│   ├── evaluate.py
│   ├── test_svm.py
│   ├── main.py
│   └── outputs/
│       ├── images.npy
│       ├── labels.npy
│       ├── classes.json
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── scaler.pkl
│       ├── svm_model.pkl
│       ├── confusion_matrix.png
│       └── prediction_sample.png
│
└── README.md
```

---

## Reference

* [Machine-Learning-Course](https://github.com/aproot-en/Machine-Learning-Course.git) — Course materials, labs, datasets, and source code for Machine Learning.

