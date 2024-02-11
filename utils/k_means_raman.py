import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import umap

from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from utils.process_spectra import process_spectrum_dataframe
import scienceplots
import warnings
warnings.filterwarnings('ignore')


scaler = MinMaxScaler()

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="train_media")
df = df.drop(["polysaccharide name"], axis=1)

data = process_spectrum_dataframe(df, downsample=1)
X = data.drop(["release", "index", "medium", "time"], axis=1)

X = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=random_seed).fit(X)

# Assigning cluster labels to each spectrum
labels = kmeans.labels_
print(labels)

new_labels = ["Oligosaccharides/Sugars", "Starches"]
reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric='euclidean')
embedding = reducer.fit_transform(X)
plt.rcParams.update({'font.size': 16})
# Visualizing the Results
plt.figure(figsize=(10, 6))
plt.style.use(["science", "no-latex"])
color_map = {0: 'blue', 1: 'green', 2: 'orange'}
colors = [color_map[label] for label in labels]

plt.scatter(embedding[:, 0], embedding[:, 1], c=colors)
plt.tick_params(left=False, bottom=False)
plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Creating a custom legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=new_labels[i],
                          markersize=10, markerfacecolor=color_map[i]) for i in range(len(new_labels))]
plt.legend(handles=legend_elements)

plt.savefig("plots/eda/umap_k_means.png", dpi=600)
plt.show()

# Create a figure and a grid of subplots
fig, axes = plt.subplots(1, 1, figsize=(15, 10))

# # Flatten the axes array for easy indexing
# axes = axes.flatten()

# Plot each cluster in its own subplot

axes.plot(np.mean(X[labels == 0, :], axis=0))
# axes.set_title(f'Cluster {i + 1}')
axes.set_xlabel('Wavenumber (cm-1)')
axes.set_ylabel('Intensity')
axes.set_ylim(0.1, 0.9)


# Adjust layout for a clean look
plt.tight_layout()
# plt.savefig(f"new/k_means_test.png")
plt.show()
