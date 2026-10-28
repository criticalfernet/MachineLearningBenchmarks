A Performance Benchmark of Numerical and Machine Learning Algorithms.

Evaluates the latency, throughput, scalability, memory efficiency, and numerical accuracy of C++ implementations against established Python numerical and machine learning libraries

## Module A: Numerical computing

| Algorithm | benchmark |
|---|---|
| Matrix multiplication | Throughput, scaling with matrix size |
| Linear regression | Training time, prediction latency |
| PCA | Decomposition time, reconstruction error |
| Covariance matrix | Computation time and memory usage |
| Cholesky decomposition | Runtime, numerical stability |
| Vectorized dot products | Latency and throughput |

## Module B: Machine learning

| # | Algorithm | Python baseline | C++ implementation |
|---|---|---|---|
| 1 | Logistic Regression | scikit-learn | Eigen |
| 2 | Naive Bayes | scikit-learn | Eigen |
| 3 | Ridge Regression | scikit-learn | Eigen |
| 4 | Elastic Net Regression | scikit-learn | Eigen |
| 5 | K-Nearest Neighbors (KNN) | scikit-learn | Eigen |
| 6 | K-means | scikit-learn | Eigen |
| 7 | Hierarchical Agglomerative Clustering | scikit-learn | Eigen |
| 8 | DBSCAN | scikit-learn | Eigen |
| 9 | CART Decision Tree | scikit-learn | Custom C++ |
| 10 | Random Forest | scikit-learn | Custom C++ |
| 11 | Gradient Boosted Trees | XGBoost | Custom C++ |


## Module C: Quantitative finance
1. Monte Carlo option pricing
2. Correlated asset simulation
3. Portfolio risk estimation


## Repository Structure
```
/
│
├── build/  #Compiled files (gitignored)
├── eigen-5.0.0/  #Eigen library (gitignored)
│
├── Algorithms/ # The implementation lies here
│
├── datasets/ # (gitignored)
│
├── results/
│
├── .gitignore
├── Makefile
└── README.md
```