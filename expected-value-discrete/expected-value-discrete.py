import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError
    return np.dot(x, p)
