"""Script for detecting peaks in Raman spectra and visualising the results.

Reads spectral data from an Excel file, processes the data (downsampling
where desired), detects peaks using scipy.signal.find_peaks, and then
creates and saves bar plots and density (KDE) plots of the detected peaks.
"""

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401  (Required for custom plotting styles)
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from scipy.signal import find_peaks

from src.process_spectra import process_spectrum_dataframe

warnings.filterwarnings("ignore")


def main() -> None:
    """Detect and plot peaks from Raman spectra in an Excel file.

    Steps:
        1. Load and process spectral data from the specified Excel sheet.
        2. Detect peaks in each individual spectrum.
        3. Plot a bar chart showing the number of peaks for each sample.
        4. Combine all peak locations across samples and visualise them using a KDE plot.
        5. Save both plots to disk.
    """
    plt.style.use(["science", "no-latex"])

    df = pd.read_excel("data/coating_release.xlsx", sheet_name="test_media")
    df_clean = df.drop(["polysaccharide name"], axis=1)
    processed_data = process_spectrum_dataframe(df_clean, downsample=1)
    x_data = processed_data.drop(["index"], axis=1).to_numpy()

    peak_counts: list[int] = []
    for spectrum in x_data:
        peaks, _ = find_peaks(spectrum, height=0.1)
        peak_counts.append(len(peaks))

    table = pd.DataFrame()
    table["name"] = df["polysaccharide name"]
    table["peaks"] = peak_counts

    plt.figure(figsize=(12, 12))
    plt.rcParams.update({"font.size": 20})
    sns.barplot(x="name", y="peaks", data=table, ci=90, capsize=0.1, palette="Set2")
    plt.xticks(rotation=90, fontsize=20)
    plt.ylim(0, 22)
    plt.tight_layout()

    plt.gca().yaxis.set_major_locator(MultipleLocator(2))
    plt.ylabel("Number of Peaks", fontsize=20)
    plt.savefig("new/peaks_test.png")
    plt.show()

    peak_locations: list[int] = []
    for spectrum in x_data:
        peaks, _ = find_peaks(spectrum, height=0.1)
        peak_locations.extend(peaks)

    plt.rcParams.update({"font.size": 18})
    plt.figure(figsize=(12, 6))
    sns.kdeplot(peak_locations, bw_adjust=0.25, fill=True)
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Density")
    plt.ylim(0, 0.002)

    plt.savefig("new/peak_locations_test.png")
    plt.show()


if __name__ == "__main__":
    main()
