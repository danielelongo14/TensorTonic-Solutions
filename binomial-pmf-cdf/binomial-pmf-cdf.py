import math

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # PMF for i: comb(n, i) * (p**i) * ((1 - p)**(n - i))
    def get_pmf(i: int) -> float:
        return math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))

    # P(X = k)
    pmf = get_pmf(k)

    # P(X <= k) = sum of all pmf to K
    cdf = sum(get_pmf(i) for i in range(k + 1))

    return {
        "pmf": float(pmf),
        "cdf": float(cdf)
    }