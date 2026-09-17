import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """
    Returns a sorted NumPy array of real eigenvalues.
    """
    matrix = np.asarray(matrix, dtype=float)

    eigenvalues = np.linalg.eigvals(matrix)

    #we only take real numbers
    eigenvalues = np.real(eigenvalues).astype(float)

    #return them in order
    return np.sort(eigenvalues)