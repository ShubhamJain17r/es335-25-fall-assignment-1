"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Test case 1
# Real Input and Real Output

print("\n\n\n"+"="*25 + "\t\tReal Input and Real Output\t\t\t" + "="*25 + "\n\n\n")

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria, max_depth=4)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("RMSE: ", round(rmse(y_hat, y), 3))
    print("MAE: ", round(mae(y_hat, y), 3))

# Test case 2
# Real Input and Discrete Output
print("\n\n\n"+"="*25 + "\t\tReal Input and Discrete Output\t\t\t" + "="*25 + "\n\n\n")

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria, max_depth=4)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    for cls in y.unique():
        print("Precision: ", round(precision(y_hat, y, cls), 3))
        print("Recall: ", round(recall(y_hat, y, cls), 3))


# # Test case 3
# # Discrete Input and Discrete Output
print("\n\n\n"+"="*25 + "\t\tDiscrete Input and Discrete Output\t\t\t" + "="*25 + "\n\n\n")

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria, max_depth=2)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    for cls in y.unique():
        print("Precision: ", round(precision(y_hat, y, cls), 3))
        print("Recall: ", round(recall(y_hat, y, cls), 3))

# # Test case 4
# # Discrete Input and Real Output
print("\n\n\n"+"="*25 + "\t\tDiscrete Input and Real Output\t\t\t" + "="*25 + "\n\n\n")

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria, max_depth=2)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("RMSE: ", round(rmse(y_hat, y), 3))
    print("MAE: ", round(mae(y_hat, y), 3))
