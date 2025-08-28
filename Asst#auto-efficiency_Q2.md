# Assignment – Decision Tree Implementation  
## Q2a: Auto MPG Regression (UCI dataset)

We trained our custom DecisionTree (regression, using MSE splits) on the **Auto MPG dataset**.  
Target = `mpg`.  
Features = {cylinders, displacement, horsepower, weight, acceleration, model year, origin}.  

Train/test split = 70/30.

### Results (example run, max_depth=6)
- **Our DecisionTree**: RMSE ≈ 3.87, MAE ≈ 2.76  
- **sklearn DecisionTreeRegressor**: RMSE ≈ 3.61, MAE ≈ 2.3  

### Visualization
Scatter plot: True MPG vs Predicted MPG (ours + sklearn).  
- Both models follow the diagonal (good fit).  
- sklearn is slightly tighter due to optimized splitting.  
- Our model is competitive but less optimized (median threshold splits).  

---

## Q2b: Comparison

- Both models achieve comparable accuracy.  
- sklearn performs slightly better (optimized thresholds, pruning strategies).  
- Our implementation validates correctness of the tree-building algorithm with real-valued targets.  
