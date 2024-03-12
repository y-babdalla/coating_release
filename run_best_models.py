from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import pickle

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils.test_models import test_models


from utils.process_spectra import process_spectrum_dataframe
import scienceplots
import warnings
warnings.filterwarnings('ignore')
plt.style.use(["science", "no-latex"])

model_names = ["lightGBM", "XGBoost", "KNN", "SVM", "RF"]

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="all_data")
df = df.drop(["polysaccharide name"], axis=1)

data = process_spectrum_dataframe(df, downsample=20, label=False)
X_train = data.drop(["release", "index"], axis=1)
y_train = data["release"]

test_full = pd.read_excel("data/coating_release.xlsx", sheet_name="test")
test = test_full.drop(["polysaccharide name"], axis=1)

test = process_spectrum_dataframe(test, downsample=20, label=False)
X_test = test.drop(["release", "index"], axis=1)
y_test = np.array(test["release"])

scaler = MinMaxScaler()
le = LabelEncoder()
X_train["medium"] = le.fit_transform(X_train["medium"])
X_test["medium"] = le.transform(X_test["medium"])

scaler.fit(X_train)
X_test = scaler.transform(X_test)

metrics_results = {}

for model_name in tqdm(model_names, desc="Models"):
    results, pred = test_models(X=X_test, y=y_test, model_name=model_name)
    metrics_results[model_name] = results

    test_full[f"{model_name}_pred"] = pred

test_full.to_csv("data/new_test_predictions.csv")


metrics_df = pd.DataFrame(metrics_results).T
metrics_df.to_csv("new/valid/valid_test_scores.csv")

for metric in ['r2', 'mae', 'mse']:
    plt.figure(figsize=(10, 6))
    metrics_df[metric].plot(kind='bar')
    plt.title(f'Comparison of {metric}')
    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.savefig(f"new/valid/test_{metric}_performance.png")
    plt.show()
