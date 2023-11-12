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
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor



def nested_cross_validation(
    X,
    y,
    model_name,
    num_iter=100,
    num_loops=10,
    n_splits=5,
    scoring="r2",
    plot=True,
    n_jobs=10,
):
    """
        Perform nested cross-validation on a given dataset with specified model and parameters.

        Parameters:
    X: pandas DataFrame or numpy array
            Feature dataset.
        y: pandas Series or numpy array
            Target variable.
        model_name: string
            Name of the model to be used. Choices: "lightGBM", "XGBoost", "KNN", "SVM", "RF".
        num_iter: int, optional
            Number of iterations for RandomizedSearchCV. Default is 100.
        num_loops: int, optional
            Number of loops for outer cross-validation. Default is 10.
        n_splits: int, optional
            Number of splits for KFold cross-validation. Default is 5.
        scoring: string, optional
            Scoring metric for model evaluation. Default is 'r2'.
        plot: bool, optional
            Whether to plot real vs predicted values. Default is True.
        n_jobs: int, optional
            Number of CPU workers used for computation. Default is 1.

        Returns:
        pandas DataFrame containing metrics for each fold.
    """

    if model_name == "lightGBM":
        model = LGBMRegressor(verbosity=-1)
        model_parameters = {
            "n_estimators": [100, 150, 300, 400],
            "max_depth": [5, 10, 15, 20],
            "learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "subsample": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_samples": [10, 20, 30, 40],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "reg_alpha": [0.001, 0.01, 0.1],
            "reg_lambda": [0.001, 0.01, 0.1],
        }
    elif model_name == "XGBoost":
        model = XGBRegressor()
        model_parameters = {
            "booster": ["gbtree", "dart"],
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [3, 6, 9, 12],
            "gamma": [0, 1, 3, 5],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.5, 0.7, 0.9],
            "min_child_weight": [1, 2, 3, 4],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [1, 0.1, 0.01],
        }
    elif model_name == "KNN":
        model = KNeighborsRegressor()
        model_parameters = {
            "n_neighbors": np.arange(1, 31),
            "p": [1, 2, 3],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        }
    elif model_name == "SVM":
        model = svm.SVR()
        model_parameters = {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [1, 0.1, 0.01, 0.001, 0.0001, "scale", "auto"],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
        }
    elif model_name == "RF":
        model = RandomForestRegressor()
        model_parameters = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [10, 20, 30, 40, 50, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False],
        }
    else:
        raise ValueError("Unsupported model name")

    results = []
    scaler = MinMaxScaler()
    all_y_test, all_y_pred = [], []
    best_model_params = None
    best_score = 0

    for i in range(num_loops):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=i
        )
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        cv_inner = KFold(n_splits=n_splits)
        search = RandomizedSearchCV(
            model,
            model_parameters,
            n_iter=num_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv_inner,
            random_state=42,
        )
        search.fit(X_train_scaled, y_train)
        current_best_score = search.best_score_

        if current_best_score > best_score:
            best_score = current_best_score
            best_model_params = search.best_params_

        y_pred = search.predict(X_test_scaled)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        results.append(
            {
                "Iteration": i + 1,
                "R2 Score": r2,
                "MAE": mae,
                "MSE": mse,
                "Best Model Parameters": search.best_params_,
            }
        )


    results_df = pd.DataFrame(results)
    results_df.to_csv(f"scores/{model_name}_cv_scores.csv")

    # Re-fit the best model on the entire dataset
    X_scaled = scaler.fit_transform(X)
    best_model = model.set_params(**best_model_params)
    best_model.fit(X_scaled, y)

    with open(f'models/best_{model_name}.pkl', 'wb') as file:
        pickle.dump(model, file)

    if plot:
        plt.style.use('seaborn-darkgrid')
        plt.figure(figsize=(10, 6))
        sns.scatterplot(all_y_test, all_y_pred, edgecolor="k", s=100, colour='blue', alpha=0.6)
        plt.plot(
            [min(all_y_test), max(all_y_test)],
            [min(all_y_test), max(all_y_test)],
            "k--",
            lw=2,
        )
        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.title(f"Actual vs Predicted - {model_name}", fontsize=14)
        plt.savefig(f"plots/predicted_vs_real_{model_name}.png")
        plt.show()

    return results_df

