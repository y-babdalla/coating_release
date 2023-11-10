import csv

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import pandas as pd
import numpy as np



from utils.process_spectra import process_spectrum_dataframe


downsample_values = [5, 10, 15, 20, 25, 30]

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="full")
df = df.drop(["polysaccharide name"], axis=1)

for downsample_value in downsample_values:
    data = process_spectrum_dataframe(df, downsample=downsample_value)
    X = data.drop(["release", "index"], axis=1)
    y = data["release"]

