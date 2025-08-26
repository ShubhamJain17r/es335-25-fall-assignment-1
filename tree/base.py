"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.utils import *

np.random.seed(42)

class Node:
    def __init__(self, feature=None, threshold=None, children=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.children = children or {}  # for discrete splits
        self.prediction = prediction


@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion, max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """

        features = X.columns
        self.root = self._build_tree(X, y, features, depth=0)

    def _build_tree(self, X, y, features, depth):
        if depth >= self.max_depth or len(set(y)) == 1 or X.shape[1] == 0:
            return Node(prediction=self._leaf_value(y))

        attr, val = opt_split_attribute(X, y, self.criterion, features)
        if attr is None:
            return Node(prediction=self._leaf_value(y))

        node = Node(feature=attr, threshold=val)
        subsets = split_data(X, y, attr, val)

        for key, (X_sub, y_sub) in subsets.items():
            if len(y_sub) == 0:
                child = Node(prediction=self._leaf_value(y))
            else:
                child = self._build_tree(X_sub, y_sub, X_sub.columns, depth + 1)
            node.children[key] = child
        return node

    def _leaf_value(self, y):
        if check_ifreal(y):  # regression
            return np.mean(y)
        else:  # classification
            return y.mode()[0]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Funtion to run the decision tree on test inputs
        """
        return X.apply(self._predict_row, axis=1)
    
    def _predict_row(self, row):
        node = self.root
        while node.prediction is None:
            if node.threshold is None:  # discrete
                val = row[node.feature]
                if val in node.children:
                    node = node.children[val]
                else:
                    return list(node.children.values())[0].prediction
            else:  # real
                if row[node.feature] <= node.threshold:
                    node = node.children["<="]
                else:
                    node = node.children[">"]
        return node.prediction



    def plot(self, node=None, depth=0) -> None:
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if node is None:
            node = self.root
        prefix = "    " * depth
        if node.prediction is not None:
            print(f"{prefix}Predict -> {node.prediction}")
        else:
            if node.threshold is None:
                print(f"{prefix}?({node.feature})")
                for val, child in node.children.items():
                    print(f"{prefix} {val}:")
                    self.plot(child, depth + 1)
            else:
                print(f"{prefix}?({node.feature} <= {node.threshold:.3f})")
                for cond, child in node.children.items():
                    print(f"{prefix} {cond}:")
                    self.plot(child, depth + 1)
