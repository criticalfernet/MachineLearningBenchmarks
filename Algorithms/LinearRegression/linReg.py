import numpy as np
from scipy.linalg import solve

def linear_regression(X, y):
    XtX = X.T @ X
    Xty = X.T @ y
    return solve(XtX, Xty, assume_a="pos")