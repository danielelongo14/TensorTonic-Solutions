import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    N = len(X)
    mean = np.mean(X, axis=0)
    X_c = X - mean

    return (X_c.T @ X_c) / (N - 1)