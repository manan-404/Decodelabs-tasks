# Week 02 — Data Classification Using AI

A basic machine learning classification model trained on the Iris dataset
using the K-Nearest Neighbors (KNN) algorithm.

## How to Run

Install dependencies:

    pip install scikit-learn pandas

Then run:

    python classifier.py

## Dataset

**Iris Dataset** — a classic ML benchmark dataset.

- 150 samples, 3 classes (Setosa, Versicolor, Virginica)
- 4 features: Sepal Length, Sepal Width, Petal Length, Petal Width
- Perfectly balanced — 50 samples per class

## Pipeline

    Raw Data → Feature Scaling → Train/Test Split → KNN Training → Validation

## Concepts Demonstrated

- Loading and exploring a dataset using `pandas`
- Feature scaling with `StandardScaler` (required for distance-based algorithms)
- Shuffling and splitting data — 80% training, 20% testing
- Training a K-Nearest Neighbors classifier (`k=5`)
- Model validation using Accuracy, F1 Score, Precision, Recall, and Confusion Matrix

## Project Structure

    Week-02-Data-Classification/
    ├── classifier.py
    └── README.md

## Sample Output

    Accuracy  : 100.00%
    F1 Score  : 1.0000
    Precision : 1.0000
    Recall    : 1.0000

    Confusion Matrix:
                     Pred: setosa  Pred: versicolor  Pred: virginica
    Actual: setosa             10                 0                0
    Actual: versicolor          0                 9                0
    Actual: virginica           0                 0               11

> Note: High accuracy is expected on Iris — it is a simple benchmark
> dataset designed to be separable. The focus of this project is the
> ML pipeline, not the score.
