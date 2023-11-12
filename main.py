import csv

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from utils.preliminary_cross_val import make_predictions


from utils.process_spectra import process_spectrum_dataframe


downsample_values = [15, 16, 17, 18, 19, 20]

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="full")
df = df.drop(["polysaccharide name"], axis=1)

all_r2_scores = pd.DataFrame()
all_mae_scores = pd.DataFrame()

# Iterate through downsample values
for downsample_value in tqdm(downsample_values, desc="Downsample"):
    data = process_spectrum_dataframe(df, downsample=downsample_value)
    X = data.drop(["release", "index"], axis=1)
    y = data["release"]

    r2_scores, mae_scores = make_predictions(X, y)

    # Add the downsample value column
    r2_scores["Downsample"] = downsample_value
    mae_scores["Downsample"] = downsample_value

    # Concatenate results
    all_r2_scores = pd.concat([all_r2_scores, r2_scores], ignore_index=True)
    all_mae_scores = pd.concat([all_mae_scores, mae_scores], ignore_index=True)


# Function to plot box plots and save to CSV for each model
def plot_and_save(scores_df, metric_name):
    models = scores_df.columns[:-1]
    for model in models:
        model_df = scores_df[["Downsample", model]].rename(columns={model: metric_name})
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Downsample", y=metric_name, data=model_df)
        plt.title(f"{metric_name} Scores for {model}")
        plt.xlabel("Downsample Value")
        plt.ylabel(f"{metric_name} Score")
        plt.savefig(f"plots/{model}_{metric_name}_boxplot.png")
        plt.show()

        # Save to CSV
        model_df.to_csv(f"scores/{model}_{metric_name}_scores.csv", index=False)

    plt.figure(figsize=(12, 8))
    for model in models:
        sns.lineplot(data=scores_df, x="Downsample", y=model, label=model)
    plt.title(f"Comparison of {metric_name} Scores Across Models")
    plt.xlabel("Downsample Value")
    plt.ylabel(f"{metric_name} Score")
    plt.legend(title="Model")
    plt.savefig(f"plots/Comparison_{metric_name}_scores_lineplot.png")
    plt.show()


# Plot and save R2 and MAE scores for each model
plot_and_save(all_r2_scores, "R2")
plot_and_save(all_mae_scores, "MAE")
