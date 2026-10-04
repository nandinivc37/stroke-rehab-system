import numpy as np
from scipy.signal import savgol_filter


def smooth_sequence(sequence, window_length=7, polyorder=2):
    """
    Smooth a sequence of 3D joint positions using
    a Savitzky-Golay filter.

    Args:
        sequence: shape (frames, 25, 3)
        window_length: smoothing window
        polyorder: polynomial order

    Returns:
        Smoothed sequence with the same shape.
    """

    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 3 or sequence.shape[1:] != (25, 3):
        raise ValueError(
            f"Expected shape (frames, 25, 3), "
            f"got {sequence.shape}"
        )

    frame_count = sequence.shape[0]

    # Window must be odd and smaller than the sequence.
    if window_length >= frame_count:
        window_length = frame_count if frame_count % 2 == 1 else frame_count - 1

    if window_length <= polyorder:
        raise ValueError(
            "window_length must be greater than polyorder."
        )

    return savgol_filter(
        sequence,
        window_length=window_length,
        polyorder=polyorder,
        axis=0
    ).astype(np.float32)