# src/assessment/recommendation.py


from src.exercises.catalogue import implemented_exercises

# Diagnoses that can be recorded in a clinical assessment (Assessment
# page and the Patients page diagnosis editor).
DIAGNOSIS_OPTIONS = [
    "Stroke",
    "Post-stroke hemiparesis",
    "Other",
]

# Diagnoses for which an exercise can be recommended.
SUPPORTED_DIAGNOSES = ["Stroke", "Post-stroke hemiparesis"]

# Only exercises that are implemented in the exercise catalogue can be
# recommended. Previously this module kept its own list (including
# wrist, hip and knee exercises with no analysis implementation), and
# approving one of those crashed the rehabilitation session page.
EXERCISE_LIBRARY = {
    config.display_name: {
        "exercise_id": config.exercise_id,
        "joint": config.joint,
        "body_region": config.body_region,
        "supported_impairments": list(config.supported_impairments),
    }
    for config in implemented_exercises()
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

    if diagnosis not in SUPPORTED_DIAGNOSES:
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
        "exercise_id": EXERCISE_LIBRARY[exercise_name]["exercise_id"],
        "difficulty": difficulty,
        "reason": reason,
    }