from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import shap

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


from utils.process_spectra import process_spectrum_dataframe
import scienceplots
import warnings
warnings.filterwarnings('ignore')
plt.style.use(["science", "no-latex"])

random_seed = 42
df = pd.read_excel("data/coating_release.xlsx", sheet_name="all_data")
df = df.drop(["polysaccharide name"], axis=1)

data = process_spectrum_dataframe(df, downsample=20, label=False)
X_train = data.drop(["release", "index"], axis=1)
y_train = data["release"]

test = pd.read_excel("data/coating_release.xlsx", sheet_name="test")
test = test.drop(["polysaccharide name"], axis=1)

test = process_spectrum_dataframe(test, downsample=20, label=False)
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

plt.style.use(["science", "no-latex"])

with open(f'models/best_RF_new.pkl', 'rb') as file:
    model = pickle.load(file)

pred = model.predict(X_test)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
plt.rcParams.update({'font.size': 18})
shap.summary_plot(shap_values, X_test, plot_type="bar")
# plt.savefig(f"new/shap_RF_summ.png", dpi=600)

shap.summary_plot(shap_values, X_test, show=True)
# plt.savefig(f"new/shap_RF.png", dpi=600)
plt.show()

