import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def process_spectrum_dataframe(df, downsample=None, label=True):
    columns_to_keep = ["medium", "time", "release", "index"]
    columns_to_process = df.drop(columns=columns_to_keep).columns

    if label:
        le = LabelEncoder()
        df["medium"] = le.fit_transform(df["medium"])

    processed_df = pd.DataFrame()

    for i, row in df.iterrows():
        row_to_keep = row[columns_to_keep]
        row_to_process = row[columns_to_process].astype("float64")

        row_smoothed_downsampled = process_individual_spectrum(
            row_to_process, downsample=downsample,
        )

        # Apply the same downsampling to the column labels
        if downsample is not None:
            downsampled_columns = columns_to_process[::downsample]
        else:
            downsampled_columns = columns_to_process

        if len(row_smoothed_downsampled) != len(downsampled_columns):
            raise ValueError("Processed data length does not match the downsampled column count.")

        new_row_data = {col: row_smoothed_downsampled[idx] for idx, col in enumerate(downsampled_columns)}
        new_row = pd.Series({**new_row_data, **row_to_keep})

        processed_df = pd.concat([processed_df, pd.DataFrame([new_row])], ignore_index=True)

    return processed_df


def process_individual_spectrum(df, plot=False, downsample=15):
    x = pd.read_excel("data/coating_release.xlsx", sheet_name="x")["x"]

    # Correct the baseline
    coefficients = np.polyfit(x, df, 3)
    baseline = np.polyval(coefficients, x)
    y_corrected = df - baseline

    # Apply Savitzky-Golay smoothing
    y_smoothed = savgol_filter(y_corrected, window_length=51, polyorder=3)

    # Downsample to reduce dimensions
    x_downsampled = x[::downsample]
    data_downsampled = df[::downsample]

    data_smoothed_downsampled = y_smoothed[::downsample]

    if plot:
        plt.figure()
        plt.plot(x, df, label="Original Spectrum")
        plt.plot(
            x_downsampled,
            data_smoothed_downsampled,
            label="Smoothed downsampled spectrum",
        )
        plt.plot(x, y_corrected, label="Corrected Spectrum")
        plt.plot(x, y_smoothed, label="Smoothed Spectrum")
        plt.plot(x_downsampled, data_downsampled, label="Downsampled Spectrum")

        plt.legend()
        plt.show()

    return data_smoothed_downsampled
