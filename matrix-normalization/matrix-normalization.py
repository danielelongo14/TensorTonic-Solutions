import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    matrix = np.asarray(matrix, dtype=float)

    if norm_type == "l1":
        norm = np.sum(matrix, axis=axis, keepdims=True)

    if norm_type == "l2":
        norm = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))

    if norm_type == "max":
        norm = np.max(np.abs(matrix), axis=axis, keepdims=True)


    safe_norm = np.where(norm == 0.0, 1.0, norm)

    return matrix / safe_norm