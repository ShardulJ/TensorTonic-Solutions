import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    out =np.full((len(seqs), max_len), pad_value, dtype=np.asarray(seqs[0]).dtype)
    for i, s in enumerate(seqs):
        n = min(len(s), max_len)
        out[i, :n] = s[:n]
    return out