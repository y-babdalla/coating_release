"""Module for performing nested cross-validation on a given dataset."""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor


def get_model_and_params(model_name: str) -> tuple[BaseEstimator, dict]:
    """Return an uninitialised model and a parameter grid for the given model name."""
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
        raise ValueError("Unsupported model name.")
    return model, model_parameters


def scale_data(
    x_train: pd.DataFrame | np.ndarray, x_test: pd.DataFrame | np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Scale training and test data using MinMaxScaler and return scaled data with the scaler."""
    scaler = MinMaxScaler()
    scaler.fit(x_train)

    x_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_train.columns)

    return x_train_scaled, x_test_scaled, scaler


def apply_pls(
    x_train_scaled: pd.DataFrame,
    x_test_scaled: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    pls_components: int,
) -> tuple[np.ndarray, np.ndarray, PLSRegression]:
    """Perform PLS transformation on selected columns of the scaled train and test data."""
    plsr = PLSRegression(n_components=pls_components)
    x_train_region = x_train_scaled.loc[:, 2001.063477:158.482422]
    x_test_region = x_test_scaled.loc[:, 2001.063477:158.482422]

    plsr.fit(x_train_region, y_train)
    x_train_pls = pd.DataFrame(plsr.transform(x_train_region))
    x_test_pls = pd.DataFrame(plsr.transform(x_test_region))

    x_train_other = x_train_scaled.loc[:, "medium":"time"]
    x_test_other = x_test_scaled.loc[:, "medium":"time"]

    x_train_final = np.array(pd.concat([x_train_pls, x_train_other], axis=1))
    x_test_final = np.array(pd.concat([x_test_pls, x_test_other], axis=1))

    return x_train_final, x_test_final, plsr


def run_randomised_search(
    model: BaseEstimator,
    params: dict,
    x_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    scoring: str,
    n_jobs: int,
    n_iter: int,
    cv_inner: KFold,
    random_state: int,
) -> RandomizedSearchCV:
    """Perform a RandomizedSearchCV on the provided model and return the fitted search object."""
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv_inner,
        random_state=random_state,
    )
    search.fit(x_train, y_train)
    return search


def nested_cross_validation(
    x: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    model_name: str,
    num_iter: int = 100,
    num_loops: int = 5,
    n_splits: int = 5,
    scoring: str = "r2",
    plot: bool = True,
    n_jobs: int = 10,
    pls: int | None = None,
) -> pd.DataFrame:
    """Perform nested cross-validation on a given dataset with a specified model and parameters.

    Parameters:
        x: Feature dataset, either a pandas DataFrame or numpy array.
        y: Target variable, either a pandas Series or numpy array.
        model_name: Name of the model to be used.
            Choices: 'lightGBM', 'XGBoost', 'KNN', 'SVM', 'RF'.
        num_iter: Number of iterations for RandomizedSearchCV. Defaults to 100.
        num_loops: Number of loops for the outer cross-validation. Defaults to 5.
        n_splits: Number of splits for the inner KFold cross-validation. Defaults to 5.
        scoring: Scoring metric for model evaluation. Defaults to 'r2'.
        plot: Whether to plot real vs. predicted values. Defaults to True.
        n_jobs: Number of CPU workers to use for computation. Defaults to 10.
        pls: Number of PLS components to apply to a specified spectral region, if any.
            Defaults to None.

    Returns:
        A pandas DataFrame containing performance metrics for each fold.
    """
    plt.style.use(["science", "no-latex"])

    model, model_parameters = get_model_and_params(model_name)
    results: list[dict] = []
    all_y_test, all_y_pred = [], []
    best_model_params = None
    best_score = float("-inf")

    for i in range(num_loops):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
        x_train_scaled, x_test_scaled, scaler = scale_data(x_train, x_test)

        if pls is not None:
            x_train_scaled, x_test_scaled, plsr = apply_pls(
                x_train_scaled, x_test_scaled, y_train, pls
            )

        cv_inner = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        search = run_randomised_search(
            model=model,
            params=model_parameters,
            x_train=x_train_scaled,
            y_train=y_train,
            scoring=scoring,
            n_jobs=n_jobs,
            n_iter=num_iter,
            cv_inner=cv_inner,
            random_state=42,
        )

        current_best_score = search.best_score_
        if current_best_score > best_score:
            best_score = current_best_score
            best_model_params = search.best_params_

        y_pred = search.predict(x_test_scaled)
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
    results_df.to_csv(f"new/{model_name}_cv_scores.csv", index=False)

    x_scaled = pd.DataFrame(MinMaxScaler().fit_transform(x), columns=x.columns)

    if pls is not None:
        plsr_full = PLSRegression(n_components=pls)
        x_region_full = x_scaled.loc[:, 2001.063477:158.482422]
        plsr_full.fit(x_region_full, y)
        x_pls_full = pd.DataFrame(plsr_full.transform(x_region_full))

        x_other_full = x_scaled.loc[:, "medium":"time"]
        x_scaled = np.array(pd.concat([x_pls_full, x_other_full], axis=1))

    if best_model_params is None:
        raise ValueError("No best model parameters found. Please check your data or parameters.")

    best_model = model.set_params(**best_model_params)
    best_model.fit(x_scaled, y)

    with open(f"models/best_{model_name}_new.pkl", "wb") as file:
        pickle.dump(best_model, file)

    if plot:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=all_y_test, y=all_y_pred, edgecolor="k", s=100, alpha=0.6)
        min_val = min(*all_y_test, *all_y_pred)
        max_val = max(*all_y_test, *all_y_pred)
        plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.title(f"Actual vs. Predicted - {model_name}", fontsize=14)
        plt.savefig(f"new/predicted_vs_real_{model_name}_plsr.png", dpi=300)
        plt.show()

    return results_df
