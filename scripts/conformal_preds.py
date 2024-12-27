"""Module for training and evaluating a RandomForestRegressor using spectral data.

This script:
1. Loads training and test data from an Excel file.
2. Processes spectral data by downsampling.
3. Encodes and scales features.
4. Trains a RandomForestRegressor with predefined hyperparameters.
5. Calibrates prediction intervals using the absolute calibration error distribution.
6. Makes predictions on the test set and saves the resulting DataFrame with
   prediction intervals to an Excel file.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.process_spectra import process_spectrum_dataframe


def main() -> None:
    """Execute the training and evaluation for a RandomForestRegressor on spectral data.

    This function:
    - Loads and processes the data.
    - Encodes and scales features.
    - Splits data into training and calibration sets.
    - Trains a RandomForestRegressor with specified parameters.
    - Calibrates a prediction interval based on calibration errors.
    - Predicts on the test set, calculates lower and upper bounds for each prediction,
      and saves results to 'data/conformal_preds_full.xlsx'.
    """
    df = pd.read_excel("data/coating_release.xlsx", sheet_name="all_data")
    df = df.drop(["polysaccharide name"], axis=1)
    processed_data: pd.DataFrame = process_spectrum_dataframe(
        df, downsample=20, encode_labels=False
    )

    x = processed_data.drop(["release", "index"], axis=1)
    y = processed_data["release"]

    test_full = pd.read_excel("data/coating_release.xlsx", sheet_name="test_full")
    test_full = test_full.drop(["polysaccharide name"], axis=1)
    processed_test = process_spectrum_dataframe(test_full, downsample=20, encode_labels=False)

    x_test = processed_test.drop(["release", "index"], axis=1)

    # Note: 'medium' is categorical, so we label-encode it before scaling
    scaler = MinMaxScaler()
    label_encoder = LabelEncoder()
    x["medium"] = label_encoder.fit_transform(x["medium"])
    x_test["medium"] = label_encoder.transform(x_test["medium"])

    x_scaled = scaler.fit_transform(x)
    x_test_scaled = scaler.transform(x_test)

    x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)

    x_train, x_calibrate, y_train, y_calibrate = train_test_split(
        x_scaled, y, test_size=0.2, random_state=3
    )

    model_params: dict[str, int | str | bool] = {
        "n_estimators": 200,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "auto",
        "max_depth": 30,
        "bootstrap": True,
    }

    rf_model = RandomForestRegressor(**model_params)
    rf_model.fit(x_train, y_train)

    calibration_predictions = rf_model.predict(x_calibrate)
    calibration_errors = np.abs(calibration_predictions - y_calibrate)
    error_threshold = np.quantile(calibration_errors, 0.9)

    test_predictions = rf_model.predict(x_test_scaled_df)
    lower_bounds = test_predictions - error_threshold
    upper_bounds = test_predictions + error_threshold

    test_full["pred"] = test_predictions
    test_full["upper_bound"] = upper_bounds
    test_full["lower_bound"] = lower_bounds

    test_full.to_excel("data/conformal_preds_full.xlsx", index=False)


if __name__ == "__main__":
    main()
