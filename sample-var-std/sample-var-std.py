import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    mean = np.mean(x)

    squared_sum = 0
    for i in x:
        squared_sum += (i - mean)**2

    sample_variance = squared_sum / (len(x) - 1)

    return {
        'variance': float(sample_variance),
        'standard_deviation': float(np.sqrt(sample_variance)),        
    }
        