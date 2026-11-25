import numpy as np

def matrix_multiply(A: list, B: list) -> np.ndarray:
    """
    Returns the matrix product as a float64 array.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    return A @ B