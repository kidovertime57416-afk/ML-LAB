# ML-04: KNN & Clustering Mini-Project

## Overview

A modular machine learning pipeline using the **Kaggle Automobile Dataset**. This project demonstrates custom **multi-class classification** and **clustering** using TensorFlow tensor operations.

Rewritten and refactored from: [Machine-Learning-Course](https://github.com/aproot-en/Machine-Learning-Course.git)

---

## Dataset

- **Source:** [Kaggle Automobile Dataset](https://www.kaggle.com/datasets/tawfikelmetwally/automobile-dataset)
- **Target (Classification):** `origin`
  - `europe` = 0
  - `japan` = 1
  - `usa` = 2
- **Features Used:**
  - `mpg`
  - `cylinders`
  - `displacement`
  - `horsepower`
  - `weight`
  - `acceleration`
  - `model_year`

---

## What It Does

### 1. Classification

Located in `mini-proj/classification/`

- **`data_loader.py`**
  - Cleans missing data.
  - Converts target labels to integers.
  - Splits the dataset into Training (60%), Validation (20%), and Testing (20%).
  - Applies `StandardScaler` to the features.

- **`knn_tf.py`**
  - Implements a custom KNN Classifier using TensorFlow.
  - Calculates Euclidean distance using tensor broadcasting.
  - Performs majority voting using `tf.math.top_k`.

- **`main.py` & `evaluate.py`**
  - Finds the best `k` using validation data.
  - Generates:
    - `01_k_curve.png`
    - `02_confusion_matrix.png`
  - Compares the custom KNN with:
    - Scikit-Learn KNN
    - Majority-class baseline
  - Saves final predictions to `predictions.csv`.

### 2. Clustering

Located in `mini-proj/clustering/`

- **`data_loader.py`**
  - Cleans missing data.
  - Scales numerical features.
  - Keeps raw values for easier interpretation.

- **`kmeans_tf.py`**
  - Implements K-Means clustering from scratch using TensorFlow.
  - Includes safeguards for empty clusters.

- **`knn_tools.py`**
  - Provides `KNNClusterAssigner`.
  - Assigns new data points to existing clusters without retraining K-Means.

- **`main.py` & `visualize.py`**
  - Evaluates different cluster counts using:
    - Elbow Method
    - Silhouette Score
  - Generates:
    - `01_elbow.png`
    - `02_clusters.png`
  - Exports:
    - `cluster_summary.csv`
    - `clustered_cars.csv`

---

## Tech Stack

- **Language:** Python
- **Machine Learning:** TensorFlow, Scikit-Learn
- **Data Processing:** Pandas, NumPy
- **Preprocessing:** Scikit-Learn `StandardScaler`
- **Visualization:** Matplotlib

---

## Project Structure

```text
ML-4-KNN/
├── mini-proj/
│   ├── classification/
│   │   ├── main.py
│   │   ├── data_loader.py
│   │   ├── knn_tf.py
│   │   ├── evaluate.py
│   │   └── outputs/
│   │       ├── 01_k_curve.png
│   │       ├── 02_confusion_matrix.png
│   │       └── predictions.csv
│   │
│   ├── clustering/
│   │   ├── main.py
│   │   ├── data_loader.py
│   │   ├── kmeans_tf.py
│   │   ├── knn_tools.py
│   │   ├── visualize.py
│   │   └── outputs/
│   │       ├── 01_elbow.png
│   │       ├── 02_clusters.png
│   │       ├── cluster_summary.csv
│   │       └── clustered_cars.csv
│   │
│   └── data-car/
│       └── Automobile.csv
│
└── README.md
```

---

## Outputs

### Classification

| File | Description |
|---|---|
| `01_k_curve.png` | Accuracy comparison for different K values |
| `02_confusion_matrix.png` | Confusion matrix of the final classifier |
| `predictions.csv` | Test-set predictions |

### Clustering

| File | Description |
|---|---|
| `01_elbow.png` | Elbow Method and Silhouette Score results |
| `02_clusters.png` | 2D visualization of the generated clusters |
| `cluster_summary.csv` | Cluster centroid profiles |
| `clustered_cars.csv` | Automobile data with assigned cluster labels |

---

## Objectives

This mini-project demonstrates:

1. Multi-class classification using **K-Nearest Neighbors (KNN)**.
2. Implementing KNN operations using **TensorFlow tensors**.
3. Selecting the optimal `k` using validation data.
4. Evaluating classification performance with accuracy and confusion matrix.
5. Implementing **K-Means clustering from scratch**.
6. Selecting an appropriate number of clusters using the Elbow Method and Silhouette Score.
7. Assigning new data to existing clusters using KNN.
8. Comparing custom implementations with Scikit-Learn approaches.