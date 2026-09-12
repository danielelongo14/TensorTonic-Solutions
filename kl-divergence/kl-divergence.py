import numpy as np

def kl_divergence(p: list, q: list, eps: float = 1e-12) -> float:
    """
    Returns the divergence as a float.
    """
    kl_div = 0
    for p, q in zip(p, q):
        if p == 0:
            continue
        q = np.maximum(q, eps)

        kl_div += p * np.log(p / q)
        
    return float(kl_div)