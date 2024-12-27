"""Process spectral data by baseline correction, smoothing, and downsampling."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelEncoder


def process_spectrum_dataframe(
    spectrum_df: pd.DataFrame, downsample: int | None = None, encode_labels: bool = True
) -> pd.DataFrame:
    """Process a DataFrame containing spectral data.

    Process spectral data in a DataFrame by optionally downsampling and smoothing
    the spectra. Optionally encode the 'medium' column using a LabelEncoder.
    The resulting DataFrame contains the processed spectra alongside the original
    metadata columns.

    Parameters:
        spectrum_df: The input DataFrame containing spectral data.
        downsample: The downsampling factor. If None, no downsampling is performed.
        encode_labels: Whether to label-encode the 'medium' column.

    Returns:
        A new DataFrame with processed (downsampled and smoothed) spectral data
        and original metadata.

    Raises:
        ValueError: If the length of the processed data does not match the
            downsampled column count.
    """
    columns_to_keep = ["medium", "time", "release", "index"]
    columns_to_process = spectrum_df.drop(columns=columns_to_keep).columns

    if encode_labels:
        label_encoder = LabelEncoder()
        spectrum_df["medium"] = label_encoder.fit_transform(spectrum_df["medium"])

    processed_df = pd.DataFrame()

    for _row_index, row_data in spectrum_df.iterrows():
        row_metadata = row_data[columns_to_keep]
        row_spectrum = row_data[columns_to_process].astype("float64")

        row_smoothed_downsampled = process_individual_spectrum(
            y=row_spectrum, downsample=downsample, show_plot=False
        )

        if downsample is not None:
            downsampled_columns = columns_to_process[::downsample]
        else:
            downsampled_columns = columns_to_process

        if len(row_smoothed_downsampled) != len(downsampled_columns):
            msg = "Processed data length does not match the downsampled " "column count."
            raise ValueError(msg)

        new_row_data = {
            col: row_smoothed_downsampled[i] for i, col in enumerate(downsampled_columns)
        }
        new_row = pd.Series({**new_row_data, **row_metadata})

        processed_df = pd.concat([processed_df, pd.DataFrame([new_row])], ignore_index=True)

    return processed_df


def process_individual_spectrum(
    y: pd.Series | np.ndarray, show_plot: bool = False, downsample: int | None = 15
) -> np.ndarray:
    """Process an individual spectrum by baseline correction, smoothing, and downsampling.

    Perform polynomial baseline correction (degree 3), Savitzky-Golay smoothing,
    and optional downsampling on the given spectrum. Optionally plot the
    intermediate steps for visual inspection.

    Parameters:
        y: The original spectrum data as a pandas Series or NumPy array.
        show_plot: Whether to plot the processing steps.
        downsample: The downsampling factor. If None, downsampling is skipped.

    Returns:
        A NumPy array containing the processed (smoothed and downsampled) spectrum.
    """
    x_values = pd.read_excel("data/coating_release.xlsx", sheet_name="x")["x"].to_numpy()

    coefficients = np.polyfit(x_values, y, deg=3)
    baseline = np.polyval(coefficients, x_values)
    y_corrected = y - baseline

    y_smoothed = savgol_filter(y_corrected, window_length=51, polyorder=3)

    if downsample is not None:
        x_values_downsampled = x_values[::downsample]
        y_downsampled = y[::downsample]
        y_smoothed_downsampled = y_smoothed[::downsample]
    else:
        x_values_downsampled = x_values
        y_downsampled = y
        y_smoothed_downsampled = y_smoothed

    if show_plot:
        plt.figure()
        plt.plot(x_values, y, label="Original Spectrum")
        plt.plot(x_values, y_corrected, label="Baseline-Corrected Spectrum")
        plt.plot(x_values, y_smoothed, label="Smoothed Spectrum")
        plt.plot(x_values_downsampled, y_downsampled, label="Downsampled Spectrum")
        plt.plot(
            x_values_downsampled, y_smoothed_downsampled, label="Smoothed & Downsampled Spectrum"
        )
        plt.legend()
        plt.show()

    return y_smoothed_downsampled
