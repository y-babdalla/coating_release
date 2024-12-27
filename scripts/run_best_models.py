"""Script for evaluating multiple regression models on a test dataset."""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from src.evaluate_model import evaluate_model
from src.process_spectra import process_spectrum_dataframe

warnings.filterwarnings("ignore")
plt.style.use(["science", "no-latex"])


def main() -> None:
    """Orchestrate the evaluation of multiple regression models on a test dataset.

    Steps:
        1. Load and process the training and test data from an Excel file.
        2. Downsample the spectra, encode the medium feature, and scale the features.
        3. For each model in 'model_names', load the pre-trained model and evaluate it.
        4. Save the resulting predictions and metrics to disk.
        5. Plot the evaluation metrics as bar charts.
    """
    model_names: list[str] = ["lightGBM", "XGBoost", "KNN", "SVM", "RF"]

    train_df = pd.read_excel("data/coating_release.xlsx", sheet_name="all_data")
    train_df = train_df.drop(["polysaccharide name"], axis=1)

    processed_data = process_spectrum_dataframe(train_df, downsample=20, encode_labels=False)
    x_train = processed_data.drop(["release", "index"], axis=1)

    test_full = pd.read_excel("data/coating_release.xlsx", sheet_name="test")
    test_df = test_full.drop(["polysaccharide name"], axis=1)

    processed_test = process_spectrum_dataframe(test_df, downsample=20, encode_labels=False)
    x_test = processed_test.drop(["release", "index"], axis=1)
    y_test = processed_test["release"].to_numpy()

    scaler = MinMaxScaler()
    label_encoder = LabelEncoder()
    x_train["medium"] = label_encoder.fit_transform(x_train["medium"])
    x_test["medium"] = label_encoder.transform(x_test["medium"])
    scaler.fit(x_train)

    x_test_scaled = scaler.transform(x_test)
    metrics_results: dict[str, dict[str, float]] = {}

    for model_name in tqdm(model_names, desc="Models"):
        results, predictions = evaluate_model(x=x_test_scaled, y=y_test, model_name=model_name)
        metrics_results[model_name] = results
        test_full[f"{model_name}_pred"] = predictions

    test_full.to_csv("data/new_test_predictions.csv", index=False)

    metrics_df: pd.DataFrame = pd.DataFrame(metrics_results).T
    metrics_df.to_csv("new/valid/valid_test_scores.csv")

    for metric in ["r2", "mae", "mse"]:
        plt.figure(figsize=(10, 6))
        metrics_df[metric].plot(kind="bar")
        plt.title(f"Comparison of {metric.upper()}")
        plt.ylabel(metric.upper())
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"new/valid/test_{metric}_performance.png")
        plt.show()


if __name__ == "__main__":
    main()
