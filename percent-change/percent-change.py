def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    p_change = []
    for i in range(1, len(series)):
        if series[i - 1] == 0:
            p_change.append(0.0)
        else:
            p_change.append((series[i] - series[i - 1]) / series[i - 1])

    return p_change