
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from utils.cross_validation import nested_cross_validation
from utils.process_spectra import process_spectrum_dataframe
import scienceplots
import warnings
warnings.filterwarnings('ignore')
plt.style.use(["science", "no-latex"])

def plot_scores(plot_data, score_types):
    sns.set(style="whitegrid")
    font = 26
    plt.rcParams.update({'font.size': font})

    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    for i, score_type in enumerate(score_types):
        ax = axes[i]

        sns.stripplot(x="Model", y=score_type, data=plot_data[score_type], dodge=True, marker='o', alpha=0.7,
                      color='black', ax=ax)

        sns.pointplot(x="Model", y=score_type, data=plot_data[score_type], dodge=0.5, join=False, capsize=0.2, ci='sd',
                      scale=2, marker="o", errwidth=2, ax=ax, palette="Set2")

        ax.set_title(f'{score_type}', fontsize=font)
        ax.set_xlabel('Model', fontsize=font)
        ax.set_ylabel(score_type, fontsize=font)
        ax.tick_params(axis='x', labelsize=font)
        ax.tick_params(axis='y', labelsize=font)

        ax.grid(False)

    # Adjust layout
    plt.tight_layout()

    # Save and show plot
    plt.savefig(f'new/model_summary_grid.png', dpi=300)
    plt.show()

model_names = ["LightGBM", "XGBoost", "KNN", "SVM", "RF"]

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="all_data")
df = df.drop(["polysaccharide name"], axis=1)

data = process_spectrum_dataframe(df, downsample=20)
X = data.drop(["release", "index"], axis=1)
y = data["release"]


scores_data = {model: {score: [] for score in ["R2 Score", "MAE", "MSE"]} for model in model_names}

for model_name in tqdm(model_names, desc="Models"):
    results_df = nested_cross_validation(X=X, y=y, model_name=model_name)
    scores_data[model_name]["R2 Score"].extend(results_df["R2 Score"].tolist())
    scores_data[model_name]["MAE"].extend(results_df["MAE"].tolist())
    scores_data[model_name]["MSE"].extend(results_df["MSE"].tolist())

# Aggregating results for plotting
plot_data = {}
for score_type in ["R2 Score", "MAE", "MSE"]:
    scores = []
    models = []
    for model, data in scores_data.items():
        scores.extend(data[score_type])
        models.extend([model] * len(data[score_type]))
    plot_data[score_type] = pd.DataFrame({score_type: scores, "Model": models})

plot_scores(plot_data, ["R2 Score", "MAE", "MSE"])