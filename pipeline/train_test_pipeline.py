"""
End-to-End Training & Evaluation Pipeline for Linear Regression
---------------------------------------------------------------
This script demonstrates a clean ML workflow:
- Load data
- Train/Test split
- Train a simple linear regression model
- Evaluate using reusable metrics
"""

import numpy as np
from evaluation.model_metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)


def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def fit_linear_regression(X, y):
    """
    Fits y = b0 + b1*x using least squares
    """
    X = np.c_[np.ones(len(X)), X]
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta


def predict(X, theta):
    X = np.c_[np.ones(len(X)), X]
    return X @ theta


def main():
    # Example synthetic data (replace with real feature/target if needed)
    X = np.array([500, 800, 1200, 1500, 1800, 2200, 2600])
    y = np.array([150000, 200000, 280000, 330000, 360000, 420000, 480000])

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    theta = fit_linear_regression(X_train, y_train)
    y_pred = predict(X_test, theta)

    print("Evaluation Results:")
    print("MSE :", mean_squared_error(y_test, y_pred))
    print("RMSE:", root_mean_squared_error(y_test, y_pred))
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("R2  :", r2_score(y_test, y_pred))


if __name__ == "__main__":
    main()
