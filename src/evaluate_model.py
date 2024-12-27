"""Module for evaluating a trained model on test data.

This module loads a pre-trained model from a pickle file and evaluates its
performance using common regression metrics. It also generates a scatter plot
comparing actual and predicted values.
"""

import pickle

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401  (Required for custom plotting styles)
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    x: pd.DataFrame, y: pd.Series, model_name: str
) -> tuple[dict[str, float], list[float]]:
    """Evaluate a saved model on test data.

    This function loads a previously saved model from disk and evaluates
    its performance on the given test features and target values. It computes
    regression metrics and creates a scatter plot that compares the actual
    and predicted values.

    Parameters:
        x: The test features as a pandas DataFrame.
        y: The true target values as a pandas Series.
        model_name: The string name of the pre-trained model to load.

    Returns:
        A tuple containing:
            - A dictionary of evaluation metrics (R², MAE, MSE).
            - A list of predicted values from the model.
    """
    plt.style.use(["science", "no-latex", "seaborn-darkgrid"])

    with open(f"models/best_{model_name}_new.pkl", "rb") as file:
        loaded_model = pickle.load(file)

    y_pred = loaded_model.predict(x)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    correlation, _ = pearsonr(y, y_pred)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred, edgecolor="k", s=100, alpha=0.6)
    plt.plot([min(y), max(y)], [min(y), max(y)], "k--", lw=2)

    plt.text(
        0.9,
        0.1,
        f"Pearson's r: {correlation:.2f}",
        fontsize=14,
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.text(
        0.9,
        0.03,
        f"R²: {r2:.2f}",
        fontsize=14,
        ha="right",
        va="bottom",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(f"Actual vs Predicted - {model_name}", fontsize=14)
    plt.savefig(f"new/valid/test_predicted_vs_real_{model_name}.png")
    plt.show()

    return {"r2": r2, "mae": mae, "mse": mse}, list(y_pred)
