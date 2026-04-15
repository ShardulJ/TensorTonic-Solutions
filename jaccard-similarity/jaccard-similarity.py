def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    a = set(set_a)
    b = set(set_b)
    intersection = a & b
    union = a | b
    if union:
        return len(intersection)/ len(union)
    else:
        return 0