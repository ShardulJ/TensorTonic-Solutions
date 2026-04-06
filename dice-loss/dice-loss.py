import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    p = np.array(p, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return 1 - ((2 * np.sum(p * y)) + eps)/ (np.sum(p) + np.sum(y) + eps)