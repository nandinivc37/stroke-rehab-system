"""
Dynamic Time Warping between two one-dimensional angle trajectories.

dtw_alignment() returns the accumulated cost and the length of the
optimal warping path. Two normalisations turn the cost into a mean
angular error (degrees):

  "n_plus_m"  cost / (n + m): the original rule, kept for historical
              sessions and the characterization tests. Because a warping
              path has between max(n, m) and n + m - 1 steps, this
              understates the mean error by up to about a factor of 2.
  "path"      cost / path length: the mean absolute difference between
              the samples DTW actually matched (metrics v2).

dtw_similarity() maps the mean error linearly to 0-100 (0 deg -> 100,
90 deg or more -> 0). The 90 deg scale is an engineering convention, not
a calibrated or clinical threshold.
"""

import math

NORMALIZATIONS = ("n_plus_m", "path")


def angle_distance(a, b):
    """
    Absolute difference between two shoulder
    elevation angles.
    """
    return abs(float(a) - float(b))


def dtw_alignment(reference, patient):
    """
    (accumulated cost, warping-path length) for two sequences, or
    (inf, 0) when either is empty. Ties between predecessors prefer the
    diagonal, then the shorter path.
    """

    n = len(reference)
    m = len(patient)

    if n == 0 or m == 0:
        return float("inf"), 0

    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    steps = [[0] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min(
                (cost[i - 1][j - 1], steps[i - 1][j - 1]),
                (cost[i - 1][j], steps[i - 1][j]),
                (cost[i][j - 1], steps[i][j - 1]),
            )
            cost[i][j] = (
                angle_distance(reference[i - 1], patient[j - 1]) + best[0]
            )
            steps[i][j] = best[1] + 1

    return cost[n][m], steps[n][m]


def dtw_distance(reference, patient, normalization="n_plus_m"):
    """Mean angular error (degrees) under the chosen normalisation."""

    if normalization not in NORMALIZATIONS:
        raise ValueError(f"Unknown DTW normalization {normalization!r}")

    total, path_length = dtw_alignment(reference, patient)

    if math.isinf(total):
        return float("inf")

    if normalization == "path":
        return total / path_length

    return total / (len(reference) + len(patient))


def similarity_from_error(mean_error):
    """0 deg -> 100, 90 deg or more -> 0 (linear, clipped)."""

    if math.isinf(mean_error):
        return 0.0

    similarity = 100.0 * (1.0 - mean_error / 90.0)
    return round(max(0.0, min(100.0, similarity)), 2)


def dtw_similarity(reference, patient, normalization="n_plus_m"):
    """
    Convert the mean DTW angular error into
    a 0-100 similarity score.

    0° average error  -> 100%
    90° average error -> 0%
    """

    return similarity_from_error(
        dtw_distance(reference, patient, normalization)
    )
