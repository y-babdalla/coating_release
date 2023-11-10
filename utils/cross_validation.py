import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import shap

from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

min_max_Scaler = MinMaxScaler()


def nested_cross_validation(
        X,
        y,
        model,
        model_parameters=None,
        num_iter=100,
        num_loops=10,
        n_splits=5,
        scoring="neg_mean_absolute_error",
        plot=True,
        shapley=False,
        n_jobs=1,
        title="",
        task="regression",
):
    """
    Function to train ML Regression models using cross-validation
    :param X: X
    :param y: y_data
    :param model: model to be used
    :param model_parameters: dictionary of model parameters for random search
    :param num_iter: number of iterations, default is 100
    :param num_loops: number of loops, default is 10
    :param n_splits: number of splits in cross validation, default is 5
    :param scoring: scoring metric
    :param plot: bool, whether to plot real vs predicted values, default is True
    :param shapley: bool, whether to plot shap values, default is False
    :param n_jobs: number of CPU workers used
    :param title: saved model title
    :param task: the type of task carried out, classification or regression, default is classification
    :param generate: whether to generate new data, defualt is false
    :param generate_params: parameters for generation, if None then default parameters are used
    :return: r2 and mae if regression and accuracy and auroc if classification
    """

    itr_number = []
    outer_results = []
    inner_results = []
    model_params = []
    y_test_list = []

    pred_list = []

    for i in range(num_loops):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i, test_size=0.3)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_test_list.append(y_test)

        min_max_Scaler.fit(X_train)
        X_train = min_max_Scaler.transform(X_train)
        X_test = min_max_Scaler.transform(X_test)

        # configure the cross-validation procedure - inner loop
        cv_inner = KFold(n_splits=n_splits)

        # Define search space
        search = RandomizedSearchCV(
            model,
            model_parameters,
            n_iter=num_iter,
            verbose=0,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv_inner,
        )

        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_
        best_score = abs(result.best_score_)
        inner_results.append(best_score)

        # evaluate model on the hold out dataset
        yhat = np.round(best_model.predict(X_test), 3)

        # store the predictions
        pred_list.append(yhat)

        if task == "regression":
            r = r2_score(y_test, yhat)
            mae = mean_absolute_error(y_test, yhat)
            mse = mean_squared_error(y_test, yhat)
            if scoring == "neg_mean_absolute_error":
                acc = mae
            elif scoring == "r2":
                acc = r
        elif task == "classification":
            accuracy = accuracy_score(y_test, yhat)
            auroc = roc_auc_score(y_test, yhat)
            if scoring == "accuracy":
                acc = accuracy
            elif scoring == "auroc":
                acc = auroc
        else:
            raise ValueError("Incorrect task, task can be 'regression' or 'classification'")

        # store the result
        itr_number.append(i + 1)
        outer_results.append(acc)
        model_params.append(result.best_params_)

        # report progress at end of each inner loop
        print(
            "\n################################################################\n\nSTATUS REPORT:"
        )
        print("Iteration " + str(i + 1) + " of " + str(num_loops) + " completed")
        if task == "regression":
            print(
                "MAE: %.3f, MSE:%.3f, r2: %.3f, Best_Valid_Score: %.3f, \n\nBest_Model_Params: \n%s"
                % (mae, mse, r, best_score, result.best_params_)
            )
        elif task == "classification":
            print(
                "Accuracy: %.3f, AUROC: %.3f, Best_Valid_Score: %.3f, \n\nBest_Model_Params: \n%s"
                % (accuracy, auroc, best_score, result.best_params_)
            )
        print("\n################################################################\n ")

    # Merge all the model outputs into a single list
    list_of_tuples = list(
        zip(
            itr_number,
            inner_results,
            outer_results,
            model_params,
            y_test_list,
            pred_list,
        )
    )
    # Convert into a pandas dataframe
    cv_dataset = pd.DataFrame(
        list_of_tuples,
        columns=[
            "Iter",
            "Valid Score",
            "Test Score",
            "Model Parms",
            "Real_AT",
            "Predicted_AT",
        ],
    )

    # Group by dataframe model iterations that best fit the data
    cv_dataset["Score_difference"] = abs(
        cv_dataset["Valid Score"] - cv_dataset["Test Score"]
    )

    # Sort by difference in scores (valid vs test)
    cv_dataset.sort_values(
        by=["Score_difference", "Test Score"], ascending=True, inplace=True
    )
    cv_dataset = cv_dataset.reset_index(drop=True)  # Reset index

    best_model_params = cv_dataset.iloc[0, 3]  # assign the best model paramaters
    model = model.set_params(**best_model_params)  # set params from the best model
    model = model.fit(X_train, y_train)
    yhat = np.round(model.predict(X_test), 3)

    # Save dataframe as pickle file
    cv_dataset.to_pickle(f"{title}.pkl", protocol=5)
    with open(f"{title}.pkl", "wb") as file:  # Save the Model to pickle file
        pickle.dump(model, file)

    # Plot real vs predicted values
    if plot:
        fig = plt.figure()

        fig.add_subplot(111, aspect="equal")

        g = sns.scatterplot(
            cv_dataset["Real_AT"][3],
            cv_dataset["Predicted_AT"][7],
            s=100,
            edgecolor="k",
            zorder=1,
        )

        g.plot([0, 25], [0, 25], lw=2, color="k", zorder=0, linestyle="--")
        g.set_xlim(-0.5, 27.5)
        g.set_ylim(-0.5, 27.5)
        g.set_xlabel("Real AT values (Hours)", size=12)
        g.set_ylabel("Predicted AT values (Hours)", size=12)
        sns.despine()
        plt.savefig(f"{title}_plot.pdf")
        plt.show()

        # Compute and plot SHAP values
        if shapley:
            pred = model.predict(X_test, output_margin=True)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            shap.summary_plot(shap_values, X_test, plot_type="bar")

            shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig(f"shap_{title}.pdf")
            plt.show()

            for name in X_train.columns:
                shap.dependence_plot(name, shap_values, X_test, display_features=X_test)

    if task == "regression":
        return r2_score(y_test, yhat), mean_absolute_error(y_test, yhat), mean_squared_error(y_test, yhat)
    elif task == "classification":
        return accuracy_score(y_test, yhat), roc_auc_score(y_test, yhat)
