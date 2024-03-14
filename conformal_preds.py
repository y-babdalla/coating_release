from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from utils.process_spectra import process_spectrum_dataframe

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

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Assuming X, y are your features and target variable
X_train, X_calibrate, y_train, y_calibrate = train_test_split(X_train, y_train, test_size=0.2, random_state=3)


model_params = {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 30, 'bootstrap': True}
# Train your regression model
model = RandomForestRegressor(**model_params)
model.fit(X_train, y_train)

# Calibration
calibration_predictions = model.predict(X_calibrate)
calibration_errors = np.abs(calibration_predictions - y_calibrate)
alpha = 0.9  # For a 90% prediction interval
error_threshold = np.quantile(calibration_errors, alpha)

# Making predictions with intervals
test_predictions = model.predict(X_test)
lower_bounds = test_predictions - error_threshold
upper_bounds = test_predictions + error_threshold
# The prediction interval for each instance in your test set
prediction_intervals = list(zip(lower_bounds, upper_bounds))

test_full["pred"] = test_predictions
test_full["upper_bound"] = upper_bounds
test_full["lower_bound"] = lower_bounds

test_full.to_excel("data/conformal_preds.xlsx")
