# LAB 03: Regression & Classification on Face Images (age_gender.csv)

This project runs two tasks side by side on the `age_gender.csv` dataset:

- **Regression** — predict age from 48x48 grayscale face images
- **Classification** — predict gender from the same images

Everything is built with scikit-learn, no deep learning involved. The focus is on comparing a bare-bones model against a full pipeline with PCA and regularization.

## Dataset

- File: `age_gender.csv`
- Each row has `age`, `gender`, and `pixels` (pixel values of a 48x48 image, space-separated)
- Pixel values are converted to float and normalized with `/255.0` before training
- Gender: `0 = Male`, `1 = Female`

## Structure

### 1. Regression — Age Prediction

| Model | Feature | Pipeline |
|---|---|---|
| Simple Linear Regression | Mean pixel value of the image (1 feature) | LinearRegression |
| Multiple Linear Regression | All pixels | StandardScaler → PCA(150) → Ridge(alpha=1.0) |

Metrics: MAE, MSE, R²
Predictions are clipped to the [1, 116] range to stay within a realistic age span.

### 2. Classification — Gender Prediction

- Pipeline: StandardScaler → PCA(100) → LogisticRegression(C=1.0)
- Includes a decision boundary plot on a 2D PCA projection before training the actual model on 100 PCA components
- Evaluated with Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC curve, and AUC

### 3. Model Comparison

- Simple vs. Multiple Linear Regression compared on MAE / MSE / R²
- Overfitting check by comparing train vs. test R²
- Summary table comparing Regression vs. Classification (target, algorithm, main metric)
- Full classification report plus detailed regression metrics (MAE, RMSE, R²)

## Dependencies

- numpy
- pandas
- matplotlib
- scikit-learn

## Notes

- `random_state` is fixed at 42 everywhere randomness is involved (train/test split, PCA, sampled images for plotting) so results are reproducible.
- All models are classical ML — no neural networks used.

## Credit
### Nipun Arora
- AGE, GENDER AND ETHNICITY (FACE DATA) CSV
- https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv