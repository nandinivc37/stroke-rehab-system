import numpy as np


def normalize_frame(frame):
    """
    Convert one Toronto skeleton frame into a
    participant-centric coordinate system.

    Args:
        frame: numpy array of shape (25, 3)

    Returns:
        normalized frame of shape (25, 3)
    """

    frame = np.asarray(frame, dtype=np.float32)

    if frame.shape != (25, 3):
        raise ValueError(
            f"Expected frame shape (25, 3), got {frame.shape}"
        )

    # Kinect joint indices
    SPINE_BASE = 0
    SPINE_SHOULDER = 20
    SHOULDER_LEFT = 4
    SHOULDER_RIGHT = 8

    # -------------------------------------------------
    # 1. Use spine-shoulder as the origin
    # -------------------------------------------------

    origin = frame[SPINE_SHOULDER]

    centered = frame - origin

    # -------------------------------------------------
    # 2. Normalize by shoulder width
    # -------------------------------------------------

    shoulder_width = np.linalg.norm(
        frame[SHOULDER_LEFT] -
        frame[SHOULDER_RIGHT]
    )

    if shoulder_width == 0:
        return centered

    normalized = centered / shoulder_width

    return normalized


def normalize_sequence(sequence):
    """
    Normalize every frame in a sequence.

    Args:
        sequence: shape (frames, 25, 3)

    Returns:
        normalized sequence: shape (frames, 25, 3)
    """

    sequence = np.asarray(sequence, dtype=np.float32)

    if sequence.ndim != 3 or sequence.shape[1:] != (25, 3):
        raise ValueError(
            f"Expected shape (frames, 25, 3), "
            f"got {sequence.shape}"
        )

    return np.stack(
        [normalize_frame(frame) for frame in sequence]
    )