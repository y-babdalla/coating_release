from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import shap

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils.test_models import test_models


from utils.process_spectra import process_spectrum_dataframe
import scienceplots
import warnings
warnings.filterwarnings('ignore')
plt.style.use(["science", "no-latex"])

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="full")
df = df.drop(["polysaccharide name"], axis=1)

data = process_spectrum_dataframe(df, downsample=15, label=False)
X_train = data.drop(["release", "index"], axis=1)
y_train = data["release"]

test = pd.read_excel("data/coating_release.xlsx", sheet_name="test")
test = test.drop(["polysaccharide name"], axis=1)

test = process_spectrum_dataframe(test, downsample=15, label=False)
X_test = test.drop(["release", "index"], axis=1)
y_test = np.array(test["release"])

scaler = MinMaxScaler()
le = LabelEncoder()
X_train["medium"] = le.fit_transform(X_train["medium"])
X_test["medium"] = le.transform(X_test["medium"])

scaler.fit(X_train)
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=X_train.columns)
print(X_test)

plt.style.use(["science", "no-latex", 'seaborn-darkgrid'])

with open(f'models/best_lightGBM.pkl', 'rb') as file:
    model = pickle.load(file)

pred = model.predict(X_test, output_margin=True)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(f"shap/shap_lightGBM.png")
plt.show()


for name in X_train.columns:
    shap.dependence_plot(name, shap_values, X_test, display_features=X_test)
    plt.savefig(f"shap/shap_dependence_{name}.png")
    plt.show()

