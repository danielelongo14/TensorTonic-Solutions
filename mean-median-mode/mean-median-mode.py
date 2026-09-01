from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    x = sorted(x)
    result = {}

    list_len = len(x)
    mean = sum(x) / list_len

    result["mean"] = float(mean)

    median = 0
    if len(x) % 2 == 0:
        median = (x[list_len // 2 - 1] + x[list_len // 2]) / 2
    else:
        median = x[list_len // 2]

    result["median"] = float(median)
    count = Counter(x)

    mode = count.most_common(1)[0][0]

    result["mode"] = float(mode)

    return result