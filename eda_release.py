from matplotlib.ticker import MaxNLocator, FuncFormatter, MultipleLocator
from sklearn.preprocessing import MinMaxScaler


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from utils.process_spectra import process_spectrum_dataframe
import scienceplots
import warnings
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')

scaler = MinMaxScaler()
plt.style.use(["science", "no-latex"])
random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="train_media")
data = df.drop(["polysaccharide name"], axis=1)

data = process_spectrum_dataframe(data, downsample=1)
X = np.array(data.drop(["index"], axis=1))

# Detecting peaks in each spectrum
peak_counts = []
for spectrum in X:
    peaks, _ = find_peaks(spectrum, height=0.1)
    # Adjust the height as needed
    peak_counts.append(len(peaks))

table = pd.DataFrame([])
table["name"] = list(df["polysaccharide name"])
table["peaks"]  = peak_counts
# Creating a bar plot
plt.figure(figsize=(12, 12))
plt.rcParams.update({'font.size': 20})
sns.barplot(x='name', y='peaks', data=table, ci=90, capsize=.1, palette="Set2")
# plt.bar(list(df["polysaccharide name"]), peak_counts, palette="Set2")
plt.xticks(rotation=90, fontsize=20)
plt.tight_layout()
plt.gca().yaxis.set_major_locator(MultipleLocator(2))
plt.ylabel('Number of Peaks', fontsize=20)
plt.savefig("new/peaks_train.png")
plt.show()

# Detecting peaks and collecting their locations
peak_locations = []
for spectrum in X:
    peaks, _ = find_peaks(spectrum, height=0.1)
    peak_locations.extend(peaks)

# Create a larger figure to accommodate the rotated labels
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(12, 6))

# Creating a KDE plot of peak locations
sns.kdeplot(peak_locations, bw_adjust=0.25, fill=True)


# Setting labels and title
plt.xlabel('Peak Location (cm-1)')
plt.ylabel('Density')
plt.rcParams.update({'font.size': 18})
plt.ylim(0, 0.002)


plt.title('Distribution of Peak Locations in Raman Spectra')

plt.savefig("new/peak_locations_train.png")

# Show plot
plt.show()