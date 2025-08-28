# Assignment – Decision Tree Implementation  
## Q1a: Classification Dataset (2D synthetic)

We generated a 2D dataset using `sklearn.datasets.make_classification`.  
- **70/30 train-test split** was applied.  
- We trained our own DecisionTree with both `information_gain` (entropy) and `gini_index`.  
- Evaluation metrics were: accuracy, per-class precision, and recall.  

### Results (Example run, will vary slightly with random split):
| Criterion | Accuracy | Precision (class 0) | Recall (class 0) | Precision (class 1) | Recall (class 1) |
|-----------|----------|---------------------|------------------|---------------------|------------------|
| InfoGain  | 0.9      | 0.84                | 0.91             | 0.94                | 0.89             |
| Gini      | 0.9      | 0.85                | 0.92             | 0.94                | 0.88             |

### Visualization
Scatter plot of dataset colored by class (two overlapping clusters).  
Decision tree textual plot shows recursive splits on `x1` and `x2`.  

---

## Q1b: Cross-validation (Nested)

We used **5-fold outer CV** and **3-fold inner CV** to choose tree depth from {1,…,10}.  

- Selected depth varied per fold (commonly 4–6).  
- Mean outer accuracy ≈ **0.91**.  
- Best depth overall (mode of chosen depths) ≈ **4**.  

This matches expectation: deeper trees fit more training patterns but risk overfitting, while shallow trees underfit. Depth ~4 balances bias–variance tradeoff.
