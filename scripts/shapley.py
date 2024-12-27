"""Script for generating SHAP explanations of a RandomForestRegressor."""

import pickle
import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401 (for plotting styles)
import shap
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.process_spectra import process_spectrum_dataframe

if TYPE_CHECKING:
    import numpy as np

warnings.filterwarnings("ignore")
plt.style.use(["science", "no-latex"])


def main() -> None:
    """Run the SHAP analysis on a RandomForestRegressor for spectral data.

    Loads the training and test sets, performs the necessary feature
    processing and encoding, and then computes SHAP values for each test
    instance. Generates a bar and summary plot to highlight feature
    importance.
    """
    df = pd.read_excel("data/coating_release.xlsx", sheet_name="all_data")
    df = df.drop(["polysaccharide name"], axis=1)
    train_data: pd.DataFrame = process_spectrum_dataframe(df, downsample=20, encode_labels=False)

    x_train = train_data.drop(["release", "index"], axis=1)

    test_df = pd.read_excel("data/coating_release.xlsx", sheet_name="test")
    test_df = test_df.drop(["polysaccharide name"], axis=1)
    processed_test = process_spectrum_dataframe(test_df, downsample=20, encode_labels=False)
    x_test = processed_test.drop(["release", "index"], axis=1)

    label_encoder = LabelEncoder()
    x_train["medium"] = label_encoder.fit_transform(x_train["medium"])
    x_test["medium"] = label_encoder.transform(x_test["medium"])

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_test_scaled: np.ndarray = scaler.transform(x_test)

    x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_train.columns)

    with open("models/best_RF_new.pkl", "rb") as model_file:
        rf_model = pickle.load(model_file)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(x_test_scaled_df)

    plt.rcParams.update({"font.size": 18})

    shap.summary_plot(shap_values, x_test_scaled_df, plot_type="bar")
    plt.savefig("new/shap_RF_summ.png", dpi=600)

    shap.summary_plot(shap_values, x_test_scaled_df, show=True)
    plt.savefig("new/shap_RF.png", dpi=600)


if __name__ == "__main__":
    main()
