
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from utils.cross_validation import nested_cross_validation


from utils.process_spectra import process_spectrum_dataframe

import warnings
warnings.filterwarnings('ignore')


model_names = ["lightGBM", "XGBoost", "KNN", "SVM", "RF"]

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="full")
df = df.drop(["polysaccharide name"], axis=1)

all_r2_scores = pd.DataFrame()
all_mae_scores = pd.DataFrame()

# Iterate through downsample values
for model_name in tqdm(model_names, desc="Models"):
    data = process_spectrum_dataframe(df, downsample=15)
    X = data.drop(["release", "index"], axis=1)
    y = data["release"]

    scores = nested_cross_validation(X=X, y=y, model_name=model_name)
