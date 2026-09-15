import math

def calculate_angle(a, b, c):
    """
    Calculate angle ABC in degrees.

    a = shoulder
    b = elbow (joint)
    c = wrist

    Each point is (x, y)
    """

    ax, ay = a
    bx, by = b
    cx, cy = c

    # Create vectors BA and BC
    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    # Dot product
    dot = ba[0] * bc[0] + ba[1] * bc[1]

    # Magnitudes
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba == 0 or mag_bc == 0:
        return 0

    cos_theta = dot / (mag_ba * mag_bc)

    # Clamp to avoid floating point errors
    cos_theta = max(-1.0, min(1.0, cos_theta))

    angle = math.degrees(math.acos(cos_theta))

    return round(angle, 1)