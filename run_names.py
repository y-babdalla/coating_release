from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from utils.preliminary_cross_val import make_predictions

import warnings
warnings.filterwarnings('ignore')

model_names = ["lightGBM", "XGBoost", "KNN", "SVM", "RF"]

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="name")

le = LabelEncoder()
X = df.drop(["release"], axis=1)
y = df["release"]

X["medium"] = le.fit_transform(X["medium"])
X["polysaccharide name"] = le.fit_transform(X["polysaccharide name"])

r2_scores, mae_scores = make_predictions(X, y, title="Name", plot=True)

