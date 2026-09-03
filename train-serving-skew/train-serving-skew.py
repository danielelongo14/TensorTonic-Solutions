import numpy as np

def detect_skew(train_dist: dict, serving_dist: dict, threshold: float = 0.2, eps: float = 1e-10) -> dict:
    """
    Returns a dictionary of feature PSI scores and skew flags.
    """
    result = {}
    for key in train_dist.keys():
        train_array = np.asarray(train_dist[key], dtype=float)
        serving_array = np.asarray(serving_dist[key], dtype=float)

        skewed = 0
        for training, serving in zip(train_array, serving_array):
            training += eps
            serving += eps
            skewed += (serving - training) * np.log(serving / training)

        current_res = {}
        current_res["psi"] = skewed
        if skewed >= threshold:
            current_res["skewed"] = True
        else:
            current_res["skewed"] = False

        result[key] = current_res

    return result

            
            
        