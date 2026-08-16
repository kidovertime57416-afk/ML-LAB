# ML-04-K-Nearest Neighbors (KNN)

Build a simple KNN pipeline using Python, including data loading, preprocessing, feature scaling, model training, evaluation, and prediction.

This project is a **rewritten and refactored implementation** of the original code structure. The code was reorganized into separate modules to make the KNN pipeline easier to understand, maintain, and extend.

## Data

Kaggle Car Information Dataset:  
https://www.kaggle.com/datasets/tawfikelmetwally/automobile-dataset

## Rewrite Code

The original implementation was rewritten and reorganized into a modular structure.

The main improvements include:

- Separating data loading and preprocessing from the model implementation.
- Implementing KNN operations using TensorFlow tensor operations.
- Separating model evaluation and visualization from the main pipeline.
- Organizing classification and clustering into separate modules.
- Improving code readability and maintainability while keeping the original project concept.

## Reference

* [Machine-Learning-Course](https://github.com/aproot-en/Machine-Learning-Course.git) — Course materials, labs, datasets, and source code for Machine Learning.


## Structure

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

## Summary

This project demonstrates modular machine learning workflows for **KNN classification** and **K-Means clustering** on vehicle data using custom TensorFlow tensor operations.
The pipeline is implemented entirely in Python with supporting libraries such as **NumPy, Pandas, Scikit-Learn, TensorFlow, and Matplotlib**.
The code was rewritten and refactored to provide a cleaner and more modular implementation while preserving the main functionality of the original project.