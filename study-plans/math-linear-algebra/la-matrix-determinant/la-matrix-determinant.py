import numpy as np

def matrix_determinant(A: list) -> float:
    """
    Returns the determinant as a Python float.
    """
    return float(np.linalg.det(A))