"""
Run models using cross-validation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate, cross_val_score
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

sns.set(style="darkgrid")


def make_predictions(
        X,
        y,
        cv=5,
        n_jobs=1,
        scoring=None,
        title="",
        models=None,
        plot=False,
        scale=True,
):
    """
    Function to run ML models on the data sets
    :param X: numpy array of images
    :param y: numpy array of image labels
    :param cv: number of folds in cross validation, default is 5
    :param n_jobs: number of cpus used, default is 1
    :param scoring: scoring to be used for cross validation, default is "f1_macro", "roc_auc", "neg_brier_score", and "neg_log_loss" for binary classifation or "f1_macro", "precision_macro", and "recall_macro" for multiclass classification
    :param title: title of the plots and graph
    :param models: a list of tuples (name, model)
    :param plot: boolean, True if plot results, false if not, default is True
    :param scale: whether the data should be scaled
    :return: None
    """
    if scoring is None:
        scoring = ["neg_mean_absolute_error", "r2"]

    # Instantiate models
    if models is None:
        models = [
            ("RF", RandomForestRegressor()),
            ("SVM", svm.SVR()),
            ("XGR", XGBRegressor()),
            ("MLP", MLPRegressor()),
            ("KNN", KNeighborsRegressor()),
        ]

    r2_scores = []
    mae_scores = []
    for model in tqdm(models, desc="Running ML models"):
        if scale:
            pipeline_items = [("norm", MinMaxScaler()),
                              ("model", model[1])
                              ]

            regressor = Pipeline(pipeline_items)

        else:
            regressor = model[1]

        ml_scores = cross_validate(
            regressor, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs
        )

        df_results = pd.DataFrame(ml_scores)

        r2_scores.append(ml_scores["test_r2"])
        mae_scores.append(ml_scores["test_neg_mean_absolute_error"])

    # Call plotting function

    r2_scores = np.stack(r2_scores)
    r2_scores = pd.DataFrame(
        r2_scores.T,
        columns=[model[0] for model in models],
    )
    mae_scores = np.stack(mae_scores)
    mae_scores = pd.DataFrame(
        mae_scores.T,
        columns=[model[0] for model in models],
    )

    scores = [
        ("R2", r2_scores),
        ("MAE", mae_scores),
    ]

    if plot:
        for score in scores:
            plot_data(score[1], title=title, score=score[0])

    print(r2_scores)
    print(mae_scores)
    exit()

    return r2_scores, mae_scores



# Create function for plotting results
def plot_data(df, title, score):
    """
    Function to plot box plots
    :param df: DataFrame to be plotted
    :param title: Title of the graph
    :param score: Machine Learning scoring used
    :return: None
    """
    df = pd.DataFrame(df.stack(), index=None)
    df = df.reset_index().drop(["level_0"], axis=1)
    df.rename(columns={0: f"{score}", "level_1": "Model"}, inplace=True)

    sns.boxplot(x="Model", y=f"{score}", data=df)
    plt.ylim(0, 1)
    plt.title(f"{title}")
    plt.savefig(f"plots/{title}-{score}.png")
    plt.show()
