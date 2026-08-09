"""
Landmark sequence normalization and padding.

Owner: Mariam Ashraf Tobar

Each frame is a 126-length vector: 21 landmarks * 3 coords (x,y,z) per hand,
left hand first (indices 0:63), right hand second (indices 63:126).
"""
import numpy as np

MAX_FRAMES = 30
FEATURES = 126


def normalize_landmarks(sequence: np.ndarray) -> np.ndarray:
    """Center each hand on its wrist and scale to unit range, per frame.

    Args:
        sequence: array of shape (T, 126)
    Returns:
        normalized array of shape (T, 126)
    """
    normalized = np.zeros_like(sequence)
    for i, frame in enumerate(sequence):
        lh = frame[:63].reshape(21, 3)
        rh = frame[63:].reshape(21, 3)

        if np.any(lh != 0):
            wrist = lh[0]
            lh = lh - wrist
            scale = np.max(np.abs(lh)) + 1e-6
            lh = lh / scale

        if np.any(rh != 0):
            wrist = rh[0]
            rh = rh - wrist
            scale = np.max(np.abs(rh)) + 1e-6
            rh = rh / scale

        normalized[i] = np.concatenate([lh.flatten(), rh.flatten()])
    return normalized


def pad_sequence(seq: np.ndarray, max_len: int = MAX_FRAMES) -> np.ndarray:
    """Truncate/pad a raw landmark sequence to `max_len` frames, then normalize."""
    if len(seq) >= max_len:
        seq = seq[:max_len]
    else:
        pad = np.zeros((max_len - len(seq), FEATURES))
        seq = np.vstack([seq, pad])
    return normalize_landmarks(seq)
