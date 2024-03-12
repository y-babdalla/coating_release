import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from lightgbm import LGBMRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_squared_error,
)
from scipy.stats import pearsonr

import scienceplots


def test_models(
        X,
        y,
        model_name,
):
    """
        Perform nested cross-validation on a given dataset with specified model and parameters.

        Parameters:
        X: test X data.
        y: test y data.
        model_name: string name of the model used

        Returns:
        pandas DataFrame containing metrics for each fold.
    """
    plt.style.use(["science", "no-latex", 'seaborn-darkgrid'])

    with open(f'models/best_{model_name}_new.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    y_pred = loaded_model.predict(X)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    correlation, _ = pearsonr(y, y_pred)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(y, y_pred, edgecolor="k", s=100, alpha=0.6)
    plt.plot(
        [min(y), max(y)],
        [min(y), max(y)],
        "k--",
        lw=2,
    )

    plt.text(0.9, 0.1, f'Pearson\'s r: {correlation:.2f}',
             fontsize=14,
             ha='right',
             va='bottom',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))

    plt.text(0.9, 0.03, f'R2: {r2:.2f}',
             fontsize=14,
             ha='right',
             va='bottom',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(f"Actual vs Predicted - {model_name}", fontsize=14)
    plt.savefig(f"new/valid/test_predicted_vs_real_{model_name}.png")
    plt.show()

    return {"r2": r2, "mae": mae, "mse": mse}, list(y_pred)
