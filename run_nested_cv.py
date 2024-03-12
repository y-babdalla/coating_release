
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
    font = 22
    plt.rcParams.update({'font.size': font})
    for score_type in score_types:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x="Model", y=score_type, data=plot_data[score_type], palette="Set2")
        # plt.title(f'Box Plot of {score_type} for Different Models')
        plt.ylabel(score_type, fontsize=font)
        plt.xlabel('Model', fontsize=font)
        plt.xticks(fontsize=font)
        plt.yticks(fontsize=font)
        plt.savefig(f'new/model_summary_{score_type}.png')
        plt.show()

model_names = ["lightGBM", "XGBoost", "KNN", "SVM", "RF"]

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="all_data")
df = df.drop(["polysaccharide name"], axis=1)

data = process_spectrum_dataframe(df, downsample=20)
X = data.drop(["release", "index"], axis=1)
y = data["release"]


scores_data = {model: {score: [] for score in ["R2 Score", "MAE", "MSE"]} for model in model_names}

for model_name in tqdm(model_names, desc="Models"):
    results_df = nested_cross_validation(X, y, model_name=model_name)
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