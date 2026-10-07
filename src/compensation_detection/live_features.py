import numpy as np


# MediaPipe Pose landmark indices
NOSE = 0

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24


TEMPORAL_FEATURE_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "trunk_vertical_angle",
    "shoulder_x_angle",
    "shoulder_height_difference",
    "trunk_forward_displacement",
    "trunk_length",
]


FEATURE_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "trunk_vertical_angle",
    "shoulder_x_angle",
    "shoulder_height_difference",
    "trunk_forward_displacement",
    "trunk_length",
    "left_elbow_angle_delta",
    "right_elbow_angle_delta",
    "trunk_vertical_angle_delta",
    "shoulder_x_angle_delta",
    "shoulder_height_difference_delta",
    "trunk_forward_displacement_delta",
    "trunk_length_delta",
    "left_elbow_angle_rolling_std",
    "right_elbow_angle_rolling_std",
    "trunk_vertical_angle_rolling_std",
    "shoulder_x_angle_rolling_std",
    "shoulder_height_difference_rolling_std",
    "trunk_forward_displacement_rolling_std",
    "trunk_length_rolling_std",
]


def vector(a, b):
    """Vector from point a to point b."""
    return b - a


def vector_angle(v1, v2):
    """Angle between two vectors in degrees."""

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def joint_angle(a, b, c):
    """Calculate angle ABC."""

    ba = a - b
    bc = c - b

    return vector_angle(ba, bc)


def distance(a, b):
    """Euclidean distance."""

    return float(np.linalg.norm(a - b))


def landmarks_to_points(results):
    """
    Convert MediaPipe Pose landmarks into a
    dictionary of normalized 3D points.

    MediaPipe provides x, y, z values.
    """

    landmarks = results.pose_landmarks.landmark

    points = {}

    for index, landmark in enumerate(landmarks):
        points[index] = np.array(
            [
                landmark.x,
                landmark.y,
                landmark.z,
            ],
            dtype=np.float32,
        )

    return points


def extract_live_features(results):
    """
    Extract the seven base biomechanical features
    used by the Toronto temporal model.

    Returns:
        dict containing the seven base features.
    """

    points = landmarks_to_points(results)

    left_shoulder = points[LEFT_SHOULDER]
    right_shoulder = points[RIGHT_SHOULDER]

    left_elbow = points[LEFT_ELBOW]
    right_elbow = points[RIGHT_ELBOW]

    left_wrist = points[LEFT_WRIST]
    right_wrist = points[RIGHT_WRIST]

    left_hip = points[LEFT_HIP]
    right_hip = points[RIGHT_HIP]

    # -------------------------------------------------
    # Elbow angles
    # -------------------------------------------------

    left_elbow_angle = joint_angle(
        left_shoulder,
        left_elbow,
        left_wrist,
    )

    right_elbow_angle = joint_angle(
        right_shoulder,
        right_elbow,
        right_wrist,
    )

    # -------------------------------------------------
    # Shoulder width
    # -------------------------------------------------

    shoulder_width = distance(
        left_shoulder,
        right_shoulder,
    )

    # Prevent division-related instability.
    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    # -------------------------------------------------
    # Trunk
    # -------------------------------------------------

    hip_center = (
        left_hip + right_hip
    ) / 2.0

    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2.0

    trunk_vector = vector(
        hip_center,
        shoulder_center,
    )

    trunk_vertical_angle = vector_angle(
        trunk_vector,
        np.array([0.0, 1.0, 0.0]),
    )

    # -------------------------------------------------
    # Shoulder orientation
    # -------------------------------------------------

    shoulder_vector = vector(
        left_shoulder,
        right_shoulder,
    )

    shoulder_x_angle = vector_angle(
        shoulder_vector,
        np.array([1.0, 0.0, 0.0]),
    )

    # -------------------------------------------------
    # Shoulder height difference
    # -------------------------------------------------

    shoulder_height_difference = (
        left_shoulder[1]
        - right_shoulder[1]
    )

    # Normalize by shoulder width.
    shoulder_height_difference /= shoulder_width

    # -------------------------------------------------
    # Trunk forward displacement
    # -------------------------------------------------

    trunk_forward_displacement = (
        shoulder_center[2]
        - hip_center[2]
    )

    trunk_forward_displacement /= shoulder_width

    # -------------------------------------------------
    # Trunk length
    # -------------------------------------------------

    trunk_length = distance(
        hip_center,
        shoulder_center,
    )

    trunk_length /= shoulder_width

    return {
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "trunk_vertical_angle": trunk_vertical_angle,
        "shoulder_x_angle": shoulder_x_angle,
        "shoulder_height_difference": shoulder_height_difference,
        "trunk_forward_displacement": trunk_forward_displacement,
        "trunk_length": trunk_length,
    }


class LiveTemporalFeatureBuffer:
    """
    Maintains the temporal features required by the
    trained BiLSTM.

    Each frame produces:
        7 static features
        7 frame-to-frame deltas
        7 rolling standard deviations

    Total = 21 features/frame.

    A small history beyond the 20-frame model window
    is retained so that the first frame of the LSTM
    window can use its preceding frame and proper
    rolling history.
    """

    def __init__(
        self,
        window_size=20,
        rolling_window=5,
    ):
        self.window_size = window_size
        self.rolling_window = rolling_window

        # Keep enough history for:
        # 20 model frames + previous/rolling context.
        self.max_history = (
            self.window_size
            + self.rolling_window
            - 1
        )

        self.history = []

    def reset(self):
        self.history.clear()

    def update(self, base_features):
        """
        Add one frame and return its 21-dimensional
        temporal feature vector.

        Returns:
            np.ndarray shape (21,)
        """

        current = np.array(
            [
                base_features[name]
                for name in TEMPORAL_FEATURE_NAMES
            ],
            dtype=np.float32,
        )

        self.history.append(current)

        if len(self.history) > self.max_history:
            self.history.pop(0)

        return self._feature_from_history_index(
            len(self.history) - 1
        )

    @property
    def ready(self):
        return len(self.history) >= self.window_size

    def get_window(self):
        """
        Return the most recent 20-frame sequence.

        Shape:
            (20, 21)
        """

        if not self.ready:
            return None

        start = len(self.history) - self.window_size
        end = len(self.history)

        return np.asarray(
            [
                self._feature_from_history_index(i)
                for i in range(start, end)
            ],
            dtype=np.float32,
        )

    def _feature_from_history_index(self, index):
        """
        Construct the 21 temporal features for
        one historical frame.
        """

        current = self.history[index]

        # ---------------------------------------------
        # Frame-to-frame delta
        # ---------------------------------------------

        if index == 0:
            delta = np.zeros(
                len(current),
                dtype=np.float32,
            )
        else:
            delta = (
                self.history[index]
                - self.history[index - 1]
            )

        # ---------------------------------------------
        # Rolling standard deviation
        # ---------------------------------------------

        start = max(
            0,
            index - self.rolling_window + 1,
        )

        rolling_history = np.asarray(
            self.history[start:index + 1],
            dtype=np.float32,
        )

        rolling_std = np.std(
            rolling_history,
            axis=0,
        ).astype(np.float32)

        # ---------------------------------------------
        # Combine
        # ---------------------------------------------

        return np.concatenate(
            [
                current,
                delta,
                rolling_std,
            ]
        ).astype(np.float32)