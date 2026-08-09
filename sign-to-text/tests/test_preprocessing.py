import numpy as np

from src.preprocessing.normalize import pad_sequence, normalize_landmarks, MAX_FRAMES, FEATURES


def test_pad_sequence_pads_short_sequences():
    seq = np.random.rand(10, FEATURES)
    padded = pad_sequence(seq)
    assert padded.shape == (MAX_FRAMES, FEATURES)


def test_pad_sequence_truncates_long_sequences():
    seq = np.random.rand(50, FEATURES)
    padded = pad_sequence(seq)
    assert padded.shape == (MAX_FRAMES, FEATURES)


def test_normalize_landmarks_zero_frame_stays_zero():
    seq = np.zeros((5, FEATURES))
    normalized = normalize_landmarks(seq)
    assert np.allclose(normalized, 0)
