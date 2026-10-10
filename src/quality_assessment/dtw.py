import math


def angle_distance(a, b):
    """
    Absolute difference between two shoulder
    elevation angles.
    """
    return abs(float(a) - float(b))


def dtw_distance(reference, patient):
    """
    Dynamic Time Warping distance between
    two one-dimensional angle trajectories.
    """

    n = len(reference)
    m = len(patient)

    if n == 0 or m == 0:
        return float("inf")

    dtw = [
        [float("inf")] * (m + 1)
        for _ in range(n + 1)
    ]

    dtw[0][0] = 0.0

    for i in range(1, n + 1):

        for j in range(1, m + 1):

            cost = angle_distance(
                reference[i - 1],
                patient[j - 1],
            )

            dtw[i][j] = cost + min(
                dtw[i - 1][j],
                dtw[i][j - 1],
                dtw[i - 1][j - 1],
            )

    distance = dtw[n][m] / (n + m)

    return distance


def dtw_similarity(reference, patient):
    """
    Convert average DTW angular error into
    a 0-100 similarity score.

    0° average error  -> 100%
    90° average error -> 0%
    """

    distance = dtw_distance(
        reference,
        patient,
    )

    if math.isinf(distance):
        return 0.0

    similarity = (
        100.0 * (1.0 - distance / 90.0)
    )

    similarity = max(
        0.0,
        min(100.0, similarity),
    )

    return round(similarity, 2)