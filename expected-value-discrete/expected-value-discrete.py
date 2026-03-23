import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    e = 0
    if np.sum(p) == 1:
        for a, b in zip(x, p):
            e += a*b
        return e
    else:
        raise ValueError
