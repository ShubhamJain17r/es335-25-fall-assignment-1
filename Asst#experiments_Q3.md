# Assignment – Decision Tree Implementation  
## Q3: Runtime Complexity Experiments

We created synthetic datasets with:
- N samples ∈ {200, 400, 800, 1600}  
- M features ∈ {8, 16, 32}  
- Four cases:  
  1. Discrete X, Discrete y  
  2. Real X, Discrete y  
  3. Real X, Real y  
  4. Discrete X, Real y  

### Measurements
For each case we measured:
- **Fit time** (tree training)  
- **Predict time** (on test set of size N)  

Each measurement averaged over 3–5 runs.

### Observations
- **Fit time** grows nearly linearly with N and M, with slight superlinear factor (log N from depth).  
- **Predict time** grows linearly with N, independent of M (since prediction only follows depth ≈ log N).  
- Regression vs classification shows similar scaling.  
- Discrete inputs are slightly faster since splits are value-based, not threshold-search.  

### Theoretical vs Empirical
- Theoretical complexity:  
  - **Training**: O(N * M * log N)  
  - **Prediction**: O(N * log N)  
- Plots confirmed this trend:  
  - Fit time curves rise steeply with M.  
  - Predict time curves remain nearly parallel across different M.  

### Visualization
Plots:  
- Fit time vs N (separate curves for M=8,16,32)  
- Predict time vs N (same)  
for each of the 4 cases.  

All results align well with the expected computational complexity.
