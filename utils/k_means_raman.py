import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from utils.process_spectra import process_spectrum_dataframe
import scienceplots
import warnings
warnings.filterwarnings('ignore')


scaler = MinMaxScaler()

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="media")
df = df.drop(["polysaccharide name"], axis=1)

data = process_spectrum_dataframe(df, downsample=1)
X = data.drop(["release", "index", "medium", "time"], axis=1)

X = scaler.fit_transform(X)

for j in range(1, 10):
    n_clusters = j # replace with your chosen number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed).fit(X)

    # Assigning cluster labels to each spectrum
    labels = kmeans.labels_

    rows = 2  # for example
    cols = (n_clusters + 1) // rows  # to ensure all clusters are included

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))

    # Flatten the axes array for easy indexing
    axes = axes.flatten()

    # Plot each cluster in its own subplot
    for i in range(n_clusters):
        axes[i].plot(np.mean(X[labels == i, :], axis=0))
        axes[i].set_title(f'Cluster {i + 1}')
        axes[i].set_xlabel('Wavenumber')
        axes[i].set_ylabel('Intensity')

    # If there are more subplots than clusters, turn off the extra subplots
    for k in range(n_clusters, len(axes)):
        axes[k].axis('off')

    # Adjust layout for a clean look
    plt.tight_layout()
    plt.savefig(f"plots/kmeans/k_means_{j}.png")
    plt.show()
