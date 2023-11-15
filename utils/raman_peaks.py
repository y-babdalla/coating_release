from sklearn.preprocessing import MinMaxScaler


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt

from utils.process_spectra import process_spectrum_dataframe
import scienceplots
import warnings
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')


scaler = MinMaxScaler()

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="test_media")
data = df.drop(["polysaccharide name"], axis=1)

data = process_spectrum_dataframe(data, downsample=1)
X = np.array(data.drop(["release", "index", "medium", "time"], axis=1))

# Detecting peaks in each spectrum
peak_counts = []
for spectrum in X:
    peaks, _ = find_peaks(spectrum, height=0.1)
    # Adjust the height as needed
    peak_counts.append(len(peaks))
    print(peak_counts)

# Creating a bar plot
plt.figure(figsize=(14, 8))
plt.bar(list(df["polysaccharide name"]), peak_counts)
plt.xlabel('Coating')
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.ylabel('Number of Peaks')
plt.title('Number of Peaks in Each Raman Spectrum')
plt.savefig("plots/peaks_train.png")
plt.show()


# Detecting peaks and collecting their locations
peak_locations = []
for spectrum in X:
    peaks, _ = find_peaks(spectrum, height=0.1)
    peak_locations.extend(peaks)

# Number of bins
num_bins = 100

# Create a larger figure to accommodate the rotated labels
plt.figure(figsize=(12, 6))

# Creating a histogram of peak locations
n, bins, patches = plt.hist(peak_locations, bins=num_bins, alpha=0.75)

# Setting labels and title
plt.xlabel('Peak Location (Wavenumber)')
plt.ylabel('Frequency of Peaks')
plt.title('Distribution of Peak Locations in Raman Spectra')

# Adjust layout to prevent clipping of labels
plt.tight_layout()

plt.savefig("plots/peak_locations_test.png")

# Show plot
plt.show()