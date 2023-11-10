import csv

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from utils.preliminary_cross_val import make_predictions



from utils.process_spectra import process_spectrum_dataframe


downsample_values = [5, 10, 15, 20, 25, 30]

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="full")
df = df.drop(["polysaccharide name"], axis=1)

# Dictionaries to store DataFrames for each model
r2_scores = {}
mae_scores = {}

# Iterate through downsample values
for downsample_value in tqdm(downsample_values, desc="Downsample"):
    data = process_spectrum_dataframe(df, downsample=downsample_value)
    X = data.drop(["release", "index"], axis=1)
    y = data["release"]

    r2_score_df, mae_score_df = make_predictions(X, y)

    # Iterate through each model and append scores
    for model in r2_score_df.columns:
        r2_scores.setdefault(model, []).append(r2_score_df.loc[0, model])
        mae_scores.setdefault(model, []).append(mae_score_df.loc[0, model])


# Function to save scores to CSV and plot for each model
def save_and_plot(model_scores, metric_name):
    for model, scores in model_scores.items():
        # Create a DataFrame for the model
        model_df = pd.DataFrame({'Downsample': downsample_values, f'{metric_name}_Score': scores})

        # Save to CSV
        model_df.to_csv(f'{model}_{metric_name}_scores.csv', index=False)

        # Plotting
        plt.figure()
        plt.plot(downsample_values, scores, marker='o')
        plt.title(f'{metric_name} Scores for {model}')
        plt.xlabel('Downsample Value')
        plt.ylabel(f'{metric_name} Score')
        plt.grid(True)
        plt.savefig(f'{model}_{metric_name}_scores_plot.png')
        plt.show()


# Save and plot R2 and MAE scores for each model
save_and_plot(r2_scores, 'R2')
save_and_plot(mae_scores, 'MAE')



