# src/assessment/recommendation.py


EXERCISE_LIBRARY = {
    "Shoulder Flexion": {
        "joint": "Shoulder",
        "body_region": "Upper Limb",
        "supported_impairments": [
            "Reduced ROM",
            "Reduced strength",
            "Poor movement coordination",
        ],
    },

    "Elbow Flexion-Extension": {
        "joint": "Elbow",
        "body_region": "Upper Limb",
        "supported_impairments": [
            "Reduced ROM",
            "Reduced strength",
            "Poor movement coordination",
        ],
    },

    "Wrist Flexion-Extension": {
        "joint": "Wrist",
        "body_region": "Upper Limb",
        "supported_impairments": [
            "Reduced ROM",
            "Reduced strength",
            "Poor movement coordination",
        ],
    },

    "Hip Flexion-Extension": {
        "joint": "Hip",
        "body_region": "Lower Limb",
        "supported_impairments": [
            "Reduced ROM",
            "Reduced strength",
            "Poor movement coordination",
        ],
    },

    "Knee Flexion-Extension": {
        "joint": "Knee",
        "body_region": "Lower Limb",
        "supported_impairments": [
            "Reduced ROM",
            "Reduced strength",
            "Poor movement coordination",
        ],
    },
}


def recommend_exercise(
    diagnosis,
    affected_side,
    body_region,
    joint,
    impairment,
    baseline_rom=None,
    restrictions=None,
):
    """
    Generate a candidate rehabilitation exercise.

    This is a rule-based recommendation system.
    The therapist must review and approve the recommendation
    before it is assigned to the patient.
    """

    # --------------------------------------------------
    # Basic diagnosis check
    # --------------------------------------------------

    supported_diagnoses = [
        "Stroke",
        "Post-stroke hemiparesis",
    ]

    if diagnosis not in supported_diagnoses:
        return {
            "success": False,
            "exercise": None,
            "difficulty": None,
            "reason": "Diagnosis is not currently supported.",
        }

    # --------------------------------------------------
    # Find an exercise matching the affected joint
    # --------------------------------------------------

    matching_exercises = []

    for exercise_name, exercise in EXERCISE_LIBRARY.items():

        if exercise["joint"] != joint:
            continue

        if exercise["body_region"] != body_region:
            continue

        if impairment not in exercise["supported_impairments"]:
            continue

        matching_exercises.append(exercise_name)

    # --------------------------------------------------
    # No matching exercise
    # --------------------------------------------------

    if not matching_exercises:
        return {
            "success": False,
            "exercise": None,
            "difficulty": None,
            "reason": (
                "No suitable exercise is currently available "
                "for the selected clinical profile."
            ),
        }

    # --------------------------------------------------
    # Select first matching exercise
    # --------------------------------------------------

    exercise_name = matching_exercises[0]

    # New exercises start at easy difficulty.
    # Difficulty will later be adapted using performance.
    difficulty = "Easy"

    # --------------------------------------------------
    # Generate recommendation reason
    # --------------------------------------------------

    reason = (
        f"{exercise_name} recommended for {affected_side.lower()} "
        f"{joint.lower()} based on the recorded diagnosis and "
        f"{impairment.lower()}."
    )

    return {
        "success": True,
        "exercise": exercise_name,
        "difficulty": difficulty,
        "reason": reason,
    }