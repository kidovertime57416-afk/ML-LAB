# ML-05-Support Vector Machine (SVM)

Build a simple SVM pipeline using Python, including image data loading, preprocessing, feature scaling, dimensionality reduction, model training, evaluation, and visual prediction.

This project is a **rewritten and refactored implementation** of the original code structure. The code was reorganized into separate modular scripts to make the SVM image classification pipeline easier to understand, maintain, and extend.

## Data

Kaggle Alligator vs Crocodile Dataset:  
https://www.kaggle.com/datasets/azharn/alligator-vs-crocodile1

## Rewrite Code

The original implementation was rewritten and reorganized into a modular structure.

The main improvements include:

- Separating image loading, validation, and resizing into a dedicated module (`data_loader.py`).
- Isolating grayscale conversion, flattening, and pixel normalization (`preprocessing.py`).
- Adding stratified train-test splitting logic into an independent script (`split_data.py`).
- Building a streamlined Scikit-Learn pipeline combining `StandardScaler`, `PCA`, and RBF-kernel `SVC` (`svm_model.py`).
- Separating metric reporting, confusion matrix plotting, and random sample visual testing (`evaluate.py`, `test_svm.py`).
- Improving code readability, memory efficiency, and artifact persistence while keeping the original project concept.

## Reference

* [Machine-Learning-Course](https://github.com/aproot-en/Machine-Learning-Course.git) — Course materials, labs, datasets, and source code for Machine Learning.

## Structure

```text
ML-05-SVM/
├── mini-proj/
│   ├── Alligator-vs-Crocodile/
│   │   ├── Alligator/
│   │   │   ├── alligator1.jpg
│   │   │   ├── alligator2.jpg
│   │   │   └── ...
│   │   │
│   │   └── Crocodile/
│   │       ├── crocodile1.jpg
│   │       ├── crocodile2.jpg
│   │       └── ...
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

# Summary

The project uses SVM for alligator and crocodile image recognition. Images are loaded from class directories, resized, converted into feature vectors, scaled, and then used to train an SVM classifier. The trained model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.
