import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    N = len(X)

    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0)
    
    X_c = X - mean
    #Now calculate covariance_matrix

    cov_m = X_c.T @ X_c / (N - 1)

    #now standard deviation for each feature
    sd = np.std(X, axis=0, ddof=1)

    #Now we need to create the matrix with the standard deviation
    denom = np.outer(sd, sd)

    return cov_m / denom