"""
Model Evaluation Metrics for Regression
---------------------------------------
This module provides reusable evaluation metrics
commonly used in regression problems.
"""

import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Computes Mean Squared Error (MSE)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Computes Root Mean Squared Error (RMSE)
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """
    Computes Mean Absolute Error (MAE)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Computes R-squared (R²) score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    # Simple sanity check
    y_actual = [100, 150, 200, 250]
    y_predicted = [110, 140, 210, 240]

    print("MSE :", mean_squared_error(y_actual, y_predicted))
    print("RMSE:", root_mean_squared_error(y_actual, y_predicted))
    print("MAE :", mean_absolute_error(y_actual, y_predicted))
    print("R2  :", r2_score(y_actual, y_predicted))
