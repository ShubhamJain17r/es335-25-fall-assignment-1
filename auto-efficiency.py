import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
cols = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin","car_name"]
data = pd.read_csv(url, delim_whitespace=True, header=None, names=cols, na_values="?")

# Clean the above data by removing redundant columns and rows with junk values
data = data.dropna(subset=["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin"])

data["cylinders"] = data["cylinders"].astype(int)
data["model_year"] = data["model_year"].astype(int)
data["origin"] = data["origin"].astype(int)

X = data.drop(columns=["mpg","car_name"]).reset_index(drop=True)
y = data["mpg"].reset_index(drop=True)

# 70/30 split
N = len(X)
idx = np.random.permutation(N)
cut = int(0.7 * N)
tr_idx, te_idx = idx[:cut], idx[cut:]
X_train, X_test = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
y_train, y_test = y.iloc[tr_idx].reset_index(drop=True), y.iloc[te_idx].reset_index(drop=True)

dt = DecisionTree(criterion="information_gain", max_depth=6)
dt.fit(X_train, y_train)
yhat = dt.predict(X_test)

print("== Our DecisionTree (Regression) ==")
print("RMSE:", round(rmse(yhat, y_test), 3))
print("MAE :", round(mae(yhat, y_test), 3))
print("\nTree:")
dt.plot()

# Compare the performance of your model with the decision tree module from scikit learn
sk = DecisionTreeRegressor(max_depth=6, random_state=42)
sk.fit(X_train, y_train)
yhat_sk = pd.Series(sk.predict(X_test))

print("\n== sklearn.tree.DecisionTreeRegressor ==")
print("RMSE:", round(rmse(yhat_sk, y_test), 3))
print("MAE :", round(mae(yhat_sk, y_test), 3))

# plot predictions vs ground truth
plt.figure()
plt.scatter(y_test, yhat, label="Our DT", alpha=0.7)
plt.scatter(y_test, yhat_sk, label="sklearn DT", alpha=0.7, marker="x")
miny, maxy = float(y_test.min()), float(y_test.max())
plt.plot([miny, maxy], [miny, maxy])
plt.xlabel("True MPG"); plt.ylabel("Predicted MPG")
plt.legend(); plt.title("Auto MPG: Predictions vs True")
plt.show()