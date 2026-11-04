"""
Joint-angle measurements from MediaPipe Pose results.

Two families of measurements exist:

* image_plane_*: the ORIGINAL 2D measurement in normalized image
  coordinates. Kept unchanged for regression tests and for interpreting
  historical sessions recorded with it.

* World-landmark (3D) measurements using results.pose_world_landmarks
  (metres, hip-centred). Angles are expressed in a trunk-fixed frame
  built from the shoulders and hips, so they do not depend on where the
  person sits in the image or on the image aspect ratio.

Assumed MediaPipe world axes: x image-right, y down, z away from the
camera. The forward axis of the trunk frame depends on this assumption.
It has NOT been verified with a live webcam.
"""

import math
import os

import numpy as np

from src.exercises.config import AngleReading

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

ARM_LANDMARKS = {
    "LEFT": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "RIGHT": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
}

TRUNK_LANDMARKS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

LANDMARK_NAMES = {
    LEFT_SHOULDER: "left shoulder",
    RIGHT_SHOULDER: "right shoulder",
    LEFT_ELBOW: "left elbow",
    RIGHT_ELBOW: "right elbow",
    LEFT_WRIST: "left wrist",
    RIGHT_WRIST: "right wrist",
    LEFT_HIP: "left hip",
    RIGHT_HIP: "right hip",
}


def arm_landmarks(side):
    return ARM_LANDMARKS[side.upper()]


def arm_and_trunk_landmarks(side):
    shoulder, elbow, wrist = arm_landmarks(side)
    return tuple(dict.fromkeys(
        (shoulder, elbow, wrist) + TRUNK_LANDMARKS
    ))


# ---------------------------------------------------------
# Original 2D measurement
# ---------------------------------------------------------

def image_plane_arm_elevation(results, side):
    """
    ORIGINAL measurement: angle between the shoulder->wrist vector and
    straight down, in normalized image coordinates.

    Image x and y are normalized by different pixel dimensions, so the
    value depends on the camera aspect ratio, and forward raises are
    foreshortened. Retained for regression and historical comparison.
    """

    shoulder, _, wrist = arm_landmarks(side)
    landmarks = results.pose_landmarks.landmark

    dx = landmarks[wrist].x - landmarks[shoulder].x
    dy = landmarks[wrist].y - landmarks[shoulder].y

    length = (dx ** 2 + dy ** 2) ** 0.5

    if length <= 0:
        return AngleReading(0.0)

    cosine = max(-1.0, min(1.0, dy / length))

    return AngleReading(math.degrees(math.acos(cosine)))


def image_plane_curve(frames_2d, side):
    """Original 2D angle for each frame of a (n, 33, 2) sequence."""

    shoulder, _, wrist = arm_landmarks(side)
    curve = []

    for frame in frames_2d:
        dx = frame[wrist][0] - frame[shoulder][0]
        dy = frame[wrist][1] - frame[shoulder][1]
        length = (dx ** 2 + dy ** 2) ** 0.5

        if length > 0:
            cosine = max(-1.0, min(1.0, dy / length))
            curve.append(math.degrees(math.acos(cosine)))

    return curve


# ---------------------------------------------------------
# World-landmark helpers
# ---------------------------------------------------------

def world_array(results):
    """(33, 3) float array of world landmarks, or None if unavailable."""

    world = getattr(results, "pose_world_landmarks", None)

    if world is None:
        return None

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in world.landmark],
        dtype=np.float64,
    )

    if points.shape != (33, 3) or not np.all(np.isfinite(points)):
        return None

    return points


CAMERA_UP = np.array([0.0, -1.0, 0.0])


def hips_visible(results, threshold=0.5):
    landmarks = results.pose_landmarks.landmark
    return all(
        landmarks[i].visibility >= threshold for i in (LEFT_HIP, RIGHT_HIP)
    )


def trunk_frame(points, use_hips=True):
    """
    Orthonormal trunk frame from world points.

    Returns (lateral, up, forward) unit vectors or None if degenerate.
        lateral: towards the subject's LEFT shoulder
        up:      hip midpoint -> shoulder midpoint (or the camera
                 vertical when use_hips is False), orthogonal to lateral
        forward: the direction the chest faces
    """

    lateral = points[LEFT_SHOULDER] - points[RIGHT_SHOULDER]

    if use_hips:
        shoulder_mid = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2.0
        hip_mid = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2.0
        up = shoulder_mid - hip_mid
    else:
        up = CAMERA_UP.copy()

    lateral_norm = np.linalg.norm(lateral)

    if lateral_norm < 1e-6:
        return None

    lateral = lateral / lateral_norm
    up = up - (up @ lateral) * lateral
    up_norm = np.linalg.norm(up)

    if up_norm < 1e-6:
        return None

    up = up / up_norm
    forward = np.cross(lateral, up)

    return lateral, up, forward


def humeral_elevation(points, side, use_hips=True):
    """
    Elevation of the upper arm (shoulder -> elbow) relative to the
    trunk, and its plane of elevation.

    elevation: 0 = arm hanging along the trunk, 90 = horizontal,
               180 = overhead.
    plane:     0 = arm moving forward (flexion), 90 = sideways
               (abduction), negative = behind the body. None when the
               arm is too close to the trunk for the plane to be stable.
    """

    frame = trunk_frame(points, use_hips)

    if frame is None:
        return None

    lateral, up, forward = frame
    shoulder, elbow, _ = arm_landmarks(side)

    arm = points[elbow] - points[shoulder]
    arm_norm = np.linalg.norm(arm)

    if arm_norm < 1e-6:
        return None

    arm = arm / arm_norm

    down_component = -(arm @ up)
    elevation = math.degrees(math.acos(max(-1.0, min(1.0, down_component))))

    outward = lateral if side.upper() == "LEFT" else -lateral
    forward_component = arm @ forward
    outward_component = arm @ outward

    horizontal = math.hypot(forward_component, outward_component)

    plane = None
    if horizontal > math.sin(math.radians(15.0)):
        plane = math.degrees(math.atan2(outward_component, forward_component))

    return elevation, plane


# ---------------------------------------------------------
# Depth-robust humeral elevation (front-facing camera)
# ---------------------------------------------------------

# AngleReading.plane_status values.
PLANE_OK = "ok"
PLANE_NEAR_VERTICAL = "arm_near_vertical"
PLANE_CALIBRATING = "calibrating"
PLANE_DEPTH_AMBIGUOUS = "depth_ambiguous"
# Plane unknown for a reason other than the arm being near vertical:
# the movement direction could not be verified.
PLANE_UNVERIFIED = (PLANE_CALIBRATING, PLANE_DEPTH_AMBIGUOUS)

PLANE_MIN_HORIZONTAL = math.sin(math.radians(15.0))


# Assumed horizontal field of view of the webcam (degrees) for the
# perspective model of the DIAGNOSTIC image-based plane. Typical webcams are
# 60-70°; not measured for any particular camera. Set REHAB_CAMERA_HFOV
# to the camera's actual value (see docs/webcam_validation_checklist.md).
CAMERA_HFOV_ENV = "REHAB_CAMERA_HFOV"
DEFAULT_CAMERA_HORIZONTAL_FOV = 65.0


def _camera_horizontal_fov():
    try:
        value = float(os.environ.get(CAMERA_HFOV_ENV, ""))
    except ValueError:
        return DEFAULT_CAMERA_HORIZONTAL_FOV
    return value if 20.0 <= value <= 150.0 else DEFAULT_CAMERA_HORIZONTAL_FOV


CAMERA_HORIZONTAL_FOV = _camera_horizontal_fov()


def calibrated_humeral_elevation(points, side, rest_length, use_hips=True,
                                 length_tolerance=0.05,
                                 min_depth_agreement=0.25,
                                 image=None, image_aspect=1.0,
                                 horizontal_fov=CAMERA_HORIZONTAL_FOV):
    """
    Humeral elevation and plane for a front-facing camera that do not
    trust MediaPipe's depth MAGNITUDE.

    With the patient facing the camera, a forward raise moves the elbow
    almost only along the camera's depth axis. If the estimated depth
    is compressed, the measured shoulder->elbow vector is shorter than
    the real upper arm, elevation is under-estimated below 90° (and
    over-estimated above) and the plane drifts towards abduction.

    The upper arm is rigid, so its length is taken from the patient's
    own resting arm (rest_length; hanging arm, nearly parallel to the
    image). Elevation is then acos(down / length) and the forward
    excursion is the out-of-image component implied by foreshortening,
    sqrt(length^2 - down^2 - outward^2). MediaPipe depth is used only for
    the SIGN of that excursion (in front of / behind the body), and only
    when it agrees well enough to be trusted.

    Plane: from the same world vector, exactly as in commit 7ee51f7
    (outward component, forward excursion signed by MediaPipe depth).

    Diagnostic only: with image landmarks (image: (33, 2) coordinates in
    units of the image height, x scaled by image_aspect = width / height)
    a second plane is computed from the 2D image with a pinhole camera
    model (_image_plane_components) and reported in details. It does NOT
    decide the plane: on a 26-minute recording (2026-11-01) it rejected
    12 repetitions the world plane accepted, when the trunk was turned
    away from the camera or the image and world landmarks disagreed
    about the elevation itself. Close-range front-facing flexion remains
    unresolved (README, "Shoulder flexion: open issue").

    When the measured length is not shorter than
    rest_length * (1 - length_tolerance), the result is identical to
    humeral_elevation(); the tolerance keeps
    landmark jitter on a hanging arm from appearing as elevation.

    Returns (elevation, plane, plane_status, details) or None if
    degenerate. details: world_plane (None when near vertical),
    plane_source ("world": the source of the returned plane),
    depth_agreement (world); diagnostic only: image_plane,
    image_plane_status, image_depth_agreement (see _choose_root),
    camera_distance (shoulder to camera in MediaPipe world units under
    the assumed field of view), None if unavailable.
    """

    frame = trunk_frame(points, use_hips)

    if frame is None:
        return None

    lateral, up, forward = frame
    shoulder, elbow, _ = arm_landmarks(side)

    arm = points[elbow] - points[shoulder]
    measured = float(np.linalg.norm(arm))
    # An arm pointing straight at the camera with no depth estimate has
    # the elbow on top of the shoulder: still 90 deg by foreshortening.
    length = max(measured, rest_length * (1.0 - length_tolerance))

    if length < 1e-6:
        return None

    outward_axis = lateral if side.upper() == "LEFT" else -lateral
    down = -float(arm @ up)
    outward = float(arm @ outward_axis)
    depth_reported = float(arm @ forward)

    elevation = math.degrees(math.acos(max(-1.0, min(1.0, down / length))))

    depth = math.sqrt(max(0.0, length ** 2 - down ** 2 - outward ** 2))

    details = {
        "world_plane": None,
        "image_plane": None,
        "image_plane_status": None,
        "plane_source": "world",
        "depth_agreement": (
            abs(depth_reported) / depth if depth > 1e-9 else None
        ),
        "image_depth_agreement": None,
        "camera_distance": None,
    }

    # Diagnostic only: the plane from the image landmarks (pinhole
    # model). It never decides the plane returned below.
    if image is not None:
        image_reading = _image_plane_components(
            image, points, side, rest_length, use_hips, image_aspect,
            horizontal_fov)
        if image_reading is not None:
            candidates, scale, details["camera_distance"] = image_reading
            (details["image_plane"], details["image_plane_status"],
             details["image_depth_agreement"]) = _choose_root(
                candidates, scale, depth_reported, min_depth_agreement)

    # Decision: world landmarks, exactly as in commit 7ee51f7.
    if math.hypot(outward, depth) <= PLANE_MIN_HORIZONTAL * length:
        return elevation, None, PLANE_NEAR_VERTICAL, details

    details["world_plane"] = math.degrees(math.atan2(
        outward, -depth if depth_reported < 0 else depth))

    if (
        depth > PLANE_MIN_HORIZONTAL * length
        and abs(depth_reported) < min_depth_agreement * depth
    ):
        # Foreshortening says the arm points clearly out of the image
        # plane, but MediaPipe's depth is too small to tell whether it
        # is in front of or behind the body.
        return elevation, None, PLANE_DEPTH_AMBIGUOUS, details

    return elevation, details["world_plane"], PLANE_OK, details


def _choose_root(candidates, length, depth_reported, min_depth_agreement):
    """
    DIAGNOSTIC ONLY (image plane; never decides the returned plane).

    Plane from the two arm directions consistent with the image,
    ((outward, forward) nearer the camera, (outward, forward) farther;
    forward positive towards the camera), chosen with MediaPipe's
    reported forward component (depth_reported).

    MediaPipe depth is trusted for its sign and to within a compression
    factor of min_depth_agreement, not for its magnitude. The nearer
    direction (forward f_near) is supported when the reported depth is
    at least min(m * f_near, f_near); the farther (f_far) when it is at
    most max(m * f_far, f_far) (m = min_depth_agreement). So a direction
    behind the body needs a reported depth clearly behind it, and one in
    front of the body but nearer the frontal plane than the other is
    supported by anything up to its own depth. Exactly one supported
    direction is used; otherwise, if the two differ clearly (half their
    forward separation above sin 15° of the length), the plane is
    ambiguous. For an orthographic view the two are (outward, ±depth)
    and this is the sign of the reported depth, ambiguous when
    |reported| < m * depth.

    Returns (plane or None if near vertical, plane_status, agreement),
    agreement = reported depth relative to the two directions: 0 at the
    farther, 1 at the nearer (None if they coincide).
    """

    (near_out, near_fwd), (far_out, far_fwd) = candidates
    half = (near_fwd - far_fwd) / 2.0
    slack = 1e-3 * length  # numerical only (float32 landmarks)
    near_ok = depth_reported >= min(min_depth_agreement * near_fwd,
                                    near_fwd) - slack
    far_ok = depth_reported <= max(min_depth_agreement * far_fwd,
                                   far_fwd) + slack
    agreement = None
    if half > 1e-9:
        agreement = (depth_reported - far_fwd) / (2.0 * half)

    if near_ok != far_ok:
        use_near = near_ok
    else:
        use_near = (abs(depth_reported - near_fwd)
                    <= abs(depth_reported - far_fwd))
    outward, forward = (
        (near_out, near_fwd) if use_near else (far_out, far_fwd)
    )

    if math.hypot(outward, forward) <= PLANE_MIN_HORIZONTAL * length:
        return None, PLANE_NEAR_VERTICAL, agreement

    plane = math.degrees(math.atan2(outward, forward))
    if near_ok == far_ok and half > PLANE_MIN_HORIZONTAL * length:
        # The image is consistent with two clearly different arm
        # directions (e.g. in front of / behind the body) and MediaPipe's
        # depth does not support exactly one of them.
        return plane, PLANE_DEPTH_AMBIGUOUS, agreement

    return plane, PLANE_OK, agreement


def _image_plane_components(image, points, side, rest_length, use_hips,
                            aspect=1.0,
                            horizontal_fov=CAMERA_HORIZONTAL_FOV):
    """
    The two upper-arm directions consistent with the image elbow, as
    ((outward, forward) nearer the camera, (outward, forward) farther),
    with the arm length and the shoulder-to-camera distance:
    (candidates, length, distance), or None if degenerate. The trunk
    frame's lateral and up axes lie in the image (shoulder line;
    perpendicular to it, checked against hips->shoulders); forward is
    towards the camera. Lengths are in world units (rest_length =
    calibrated resting upper arm).

    Pinhole camera with the principal point at the image centre and the
    given horizontal field of view. The image scale at the shoulders
    (image units per world unit) is measured EVERY frame from the image
    shoulder width over the in-image (x, y) world shoulder width, so it
    follows the patient leaning towards or away from the camera. The
    elbow lies on its camera ray at rest_length from the shoulder: two
    solutions. If the ray misses that sphere, the image elbow is farther
    out than the arm can reach and the single closest point is used
    (arm across the ray, i.e. close to the frontal plane).

    Why perspective matters: live recordings (2026-11-01) had the
    shoulders about 0.6 m from the camera and the patient leaning in
    during the raise (image shoulder width +31%). A forward-moving elbow
    then slides AWAY from the image centre, i.e. outward for a shoulder
    off the optical axis, and grows. The previous orthographic model
    with a fixed image rest length read that as an arm lying in the
    frontal plane (forward component clamped to 0, plane exactly 90°).
    Close to the camera the second solution of a sideways arm can also
    lie in front of the body, so the choice uses MediaPipe's depth
    (_choose_root) and may be ambiguous.

    horizontal_fov=None: orthographic (same per-frame scale); the two
    solutions are then mirror images in front of / behind the body.
    """

    shoulder, elbow, _ = arm_landmarks(side)

    lateral = image[LEFT_SHOULDER] - image[RIGHT_SHOULDER]
    lateral_norm = np.linalg.norm(lateral)
    world_width = math.hypot(*(points[LEFT_SHOULDER]
                               - points[RIGHT_SHOULDER])[:2])
    if lateral_norm < 1e-9 or world_width < 1e-9 or rest_length < 1e-9:
        return None
    scale = lateral_norm / world_width
    lateral = lateral / lateral_norm

    # Perpendicular to the shoulder line, pointing up the image (y down).
    up = np.array([lateral[1], -lateral[0]])
    if up[1] > 0:
        up = -up
    if use_hips:
        trunk = (
            (image[LEFT_SHOULDER] + image[RIGHT_SHOULDER])
            - (image[LEFT_HIP] + image[RIGHT_HIP])
        ) / 2.0
        if trunk @ up <= 0:
            return None

    outward_axis = lateral if side.upper() == "LEFT" else -lateral

    if horizontal_fov is None:
        arm = (image[elbow] - image[shoulder]) / scale
        length = max(float(np.linalg.norm(arm)), rest_length)
        down = -float(arm @ up)
        outward = float(arm @ outward_axis)
        depth = math.sqrt(max(0.0, length ** 2 - down ** 2 - outward ** 2))
        return ((outward, depth), (outward, -depth)), length, None

    # Camera coordinates: x right, y down (image units / focal), z away.
    focal = (aspect / 2.0) / math.tan(math.radians(horizontal_fov) / 2.0)
    centre = np.array([aspect / 2.0, 0.5])
    distance = focal / scale
    shoulder_3d = np.append((image[shoulder] - centre) / scale, distance)
    ray = np.append((image[elbow] - centre) / focal, 1.0)
    # |t * ray - shoulder_3d| = rest_length
    a = float(ray @ ray)
    b = float(ray @ shoulder_3d)
    root = math.sqrt(max(0.0, b * b - a * (shoulder_3d @ shoulder_3d
                                           - rest_length ** 2)))
    candidates = []
    for t in ((b - root) / a, (b + root) / a):
        arm = t * ray - shoulder_3d
        candidates.append((float(arm[:2] @ outward_axis), -float(arm[2])))
    length = max(rest_length, float(np.linalg.norm(b / a * ray
                                                   - shoulder_3d)))
    return tuple(candidates), length, distance


def image_array(results, aspect=1.0):
    """(33, 2) image landmarks in units of the image height (x scaled by
    width / height), or None if unavailable."""

    landmarks = getattr(results, "pose_landmarks", None)
    if landmarks is None:
        return None
    points = np.array(
        [[lm.x * aspect, lm.y] for lm in landmarks.landmark],
        dtype=np.float64,
    )
    if points.shape != (33, 2) or not np.all(np.isfinite(points)):
        return None
    return points


class CalibratedHumeralElevation:
    """
    Stateful per-session measurement for front-facing shoulder flexion.

    Until a stable resting upper-arm length has been observed (arm
    hanging, rest_frames consecutive frames whose lengths vary by at
    most rest_stability of their median; 5 frames is about
    0.17 s at 30 fps and 0.7 s at 7 fps) readings carry the uncalibrated
    3D elevation with plane None and plane_status "calibrating", so the
    movement plane counts as unverified rather than as valid flexion.

    set_image_size() must be called with the processed frame size for
    the image-based plane to be geometrically correct (x and y of
    MediaPipe image landmarks are normalised by different dimensions);
    the default aspect of 1.0 suits synthetic poses. horizontal_fov is
    the assumed camera field of view (None = orthographic, as in the
    synthetic poses). last_details holds the latest frame's diagnostics
    (see calibrated_humeral_elevation).

    Thresholds are engineering defaults; not clinically validated.
    """

    def __init__(self, rest_frames=5, rest_elevation=30.0,
                 rest_stability=0.10, length_tolerance=0.05,
                 min_depth_agreement=0.25,
                 horizontal_fov=CAMERA_HORIZONTAL_FOV):
        self.rest_frames = rest_frames
        self.rest_elevation = rest_elevation
        self.rest_stability = rest_stability
        self.length_tolerance = length_tolerance
        self.min_depth_agreement = min_depth_agreement
        self.horizontal_fov = horizontal_fov
        self.image_aspect = 1.0
        self.reset()

    def reset(self):
        self.rest_length = None
        self._rest_samples = []
        self.last_details = None

    def set_image_size(self, width, height):
        self.image_aspect = width / height if width and height else 1.0

    def _observe(self, points, side, elevation):
        if elevation > self.rest_elevation:
            self._rest_samples = []
            return

        shoulder, elbow, _ = arm_landmarks(side)
        self._rest_samples.append(
            float(np.linalg.norm(points[elbow] - points[shoulder]))
        )
        self._rest_samples = self._rest_samples[-self.rest_frames:]

        if len(self._rest_samples) < self.rest_frames:
            return

        middle = float(np.median(self._rest_samples))
        spread = max(self._rest_samples) - min(self._rest_samples)
        if middle > 1e-6 and spread <= self.rest_stability * middle:
            self.rest_length = middle

    def __call__(self, results, side):
        points = world_array(results)

        if points is None:
            return None

        image = image_array(results, self.image_aspect)
        use_hips = hips_visible(results)
        reference_frame = "hips" if use_hips else "camera"

        if self.rest_length is None:
            measurement = humeral_elevation(points, side, use_hips)
            self.last_details = None
            if measurement is None:
                return None
            self._observe(points, side, measurement[0])
            if self.rest_length is None:
                return AngleReading(
                    measurement[0], None, reference_frame,
                    PLANE_CALIBRATING,
                )

        measurement = calibrated_humeral_elevation(
            points, side, self.rest_length, use_hips,
            self.length_tolerance, self.min_depth_agreement,
            image, self.image_aspect, self.horizontal_fov,
        )

        if measurement is None:
            self.last_details = None
            return None

        elevation, plane, status, self.last_details = measurement
        return AngleReading(elevation, plane, reference_frame, status)


# ---------------------------------------------------------
# World-landmark measurements used by exercise configs
# ---------------------------------------------------------

def world_humeral_elevation(results, side):
    points = world_array(results)

    if points is None:
        return None

    use_hips = hips_visible(results)
    measurement = humeral_elevation(points, side, use_hips)

    if measurement is None:
        return None

    elevation, plane = measurement
    return AngleReading(
        elevation, plane, "hips" if use_hips else "camera"
    )


def world_elbow_flexion(results, side):
    """
    Elbow flexion in degrees: 0 = straight arm, increasing as the elbow
    bends (180 minus the shoulder-elbow-wrist angle).
    """

    points = world_array(results)

    if points is None:
        return None

    shoulder, elbow, wrist = arm_landmarks(side)

    upper = points[shoulder] - points[elbow]
    fore = points[wrist] - points[elbow]

    norms = np.linalg.norm(upper) * np.linalg.norm(fore)

    if norms < 1e-9:
        return None

    cosine = max(-1.0, min(1.0, float(upper @ fore / norms)))
    interior = math.degrees(math.acos(cosine))

    return AngleReading(180.0 - interior)


def world_forward_reach(results, side):
    """
    Forward reach as a percentage of arm length: the wrist's
    displacement in front of the shoulder along the trunk's forward
    axis, divided by (upper-arm + forearm) length.

    Depends heavily on MediaPipe's depth estimate.
    """

    points = world_array(results)

    if points is None:
        return None

    use_hips = hips_visible(results)
    frame = trunk_frame(points, use_hips)

    if frame is None:
        return None

    _, _, forward = frame
    shoulder, elbow, wrist = arm_landmarks(side)

    arm_length = (
        np.linalg.norm(points[elbow] - points[shoulder])
        + np.linalg.norm(points[wrist] - points[elbow])
    )

    if arm_length < 1e-6:
        return None

    reach = (points[wrist] - points[shoulder]) @ forward

    return AngleReading(
        100.0 * float(reach) / float(arm_length),
        None,
        "hips" if use_hips else "camera",
    )
