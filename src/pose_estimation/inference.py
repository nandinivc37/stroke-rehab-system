import cv2
import mediapipe as mp

from src.utils.angles import calculate_angle
from src.action_recognition.rep_counter import RepCounter
from src.database.db import save_session, get_affected_areas
from src.compensation_detection.live_features import (
    LiveTemporalFeatureBuffer,
    extract_live_features,
)
from src.compensation_detection.lstm_inference import CompensationLSTM


# ---------------------------------
# Patient / Exercise Configuration
# ---------------------------------
patient_id = 1
exercise = "Elbow Flexion"


# ---------------------------------
# Get affected side from database
# ---------------------------------
affected_areas = get_affected_areas(patient_id)

if not affected_areas:
    raise RuntimeError(
        f"No affected area found for patient {patient_id}."
    )

affected_side = affected_areas[0][1].upper()

if affected_side not in ("LEFT", "RIGHT"):
    raise RuntimeError(
        f"Invalid affected side in database: {affected_side}"
    )


# ---------------------------------
# Open USB Webcam
# ---------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)


if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")


# ---------------------------------
# MediaPipe Pose Initialization
# ---------------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ---------------------------------
# Rep Counter
# ---------------------------------
rep_counter = RepCounter()

reps = 0
stage = "up"

max_angle = 0
min_angle = 180

# ---------------------------------
# Compensation Detection
# ---------------------------------
compensation_buffer = LiveTemporalFeatureBuffer(
    window_size=20,
    rolling_window=5,
)

compensation_model = CompensationLSTM()

compensation_label = None
compensation_confidence = None
compensation_probabilities = None


# ---------------------------------
# Window
# ---------------------------------
window_name = "Stroke Rehab - Pose Estimation"

cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)


# ---------------------------------
# Console information
# ---------------------------------
print("=" * 50)
print(" Stroke Rehab - Pose Estimation")
print(f" Patient ID: {patient_id}")
print(f" Affected side: {affected_side}")
print(f" Exercise: {exercise}")
print(" Press Q or ESC to quit")
print("=" * 50)


# ---------------------------------
# Main Loop
# ---------------------------------
while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read frame.")
        break


    elbow_angle = None
    # -------------------------------------------------
    # IMPORTANT:
    # Process the ORIGINAL camera frame.
    # Do NOT flip before MediaPipe.
    # -------------------------------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:

        lm = results.pose_landmarks.landmark

        # ---------------------------------------------
        # Compensation Detection
        # ---------------------------------------------
        required_landmarks = [
            11, 12, 13, 14, 15, 16, 23, 24
        ]

        landmarks_visible = all(
            lm[index].visibility >= 0.5
            for index in required_landmarks
        )

        if landmarks_visible:

            base_features = extract_live_features(
                results
            )

            compensation_buffer.update(
                base_features
            )

            if compensation_buffer.ready:

                compensation_window = (
                    compensation_buffer.get_window()
                )

                # Get probabilities for all classes
                compensation_probabilities = (
                    compensation_model.predict_proba(
                        compensation_window
                    )
                )

                # Get winning class
                (
                    compensation_label,
                    compensation_confidence,
                    _
                ) = compensation_model.predict(
                    compensation_window
                )

        else:

            compensation_buffer.reset()
            compensation_label = None
            compensation_confidence = None
            compensation_probabilities = None

        # ---------------------------------------------
        # Draw full pose on ORIGINAL frame
        # ---------------------------------------------
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(
                color=(0, 255, 0),
                thickness=2,
                circle_radius=3
            ),
            mp_draw.DrawingSpec(
                color=(255, 255, 255),
                thickness=2
            )
        )

        h, w, _ = frame.shape

        # ---------------------------------------------
        # Select affected arm from database
        #
        # MediaPipe:
        # LEFT  = 11, 13, 15
        # RIGHT = 12, 14, 16
        # ---------------------------------------------
        if affected_side == "LEFT":
            shoulder = lm[11]
            elbow = lm[13]
            wrist = lm[15]
        else:
            shoulder = lm[12]
            elbow = lm[14]
            wrist = lm[16]

        # ---------------------------------------------
        # Visibility check
        # ---------------------------------------------
        if (
            shoulder.visibility < 0.5
            or elbow.visibility < 0.5
            or wrist.visibility < 0.5
        ):
            cv2.putText(
                frame,
                "Keep affected arm visible",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:

            # -----------------------------------------
            # Convert landmarks to pixel coordinates
            # -----------------------------------------
            sx = int(shoulder.x * w)
            sy = int(shoulder.y * h)

            ex = int(elbow.x * w)
            ey = int(elbow.y * h)

            wx = int(wrist.x * w)
            wy = int(wrist.y * h)

            shoulder_point = (sx, sy)
            elbow_point = (ex, ey)
            wrist_point = (wx, wy)

            # -----------------------------------------
            # Calculate elbow angle
            # -----------------------------------------
            elbow_angle = calculate_angle(
                shoulder_point,
                elbow_point,
                wrist_point
            )

            # -----------------------------------------
            # ROM tracking
            # -----------------------------------------
            max_angle = max(max_angle, elbow_angle)
            min_angle = min(min_angle, elbow_angle)

            # -----------------------------------------
            # Repetition counting
            # -----------------------------------------
            reps, stage = rep_counter.update(elbow_angle)

            # -----------------------------------------
            # Highlight affected arm joints
            # -----------------------------------------
            cv2.circle(
                frame,
                shoulder_point,
                8,
                (0, 0, 255),
                -1
            )

            cv2.circle(
                frame,
                elbow_point,
                8,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                frame,
                wrist_point,
                8,
                (0, 255, 255),
                -1
            )

            
    

    else:

        compensation_buffer.reset()
        compensation_label = None
        compensation_confidence = None
        compensation_probabilities = None

        cv2.putText(
            frame,
            "No pose detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # -------------------------------------------------
    # Mirror ONLY for display
    #
    # MediaPipe processed the original frame above.
    # This gives us correct physical LEFT/RIGHT logic
    # while keeping the UI natural like a selfie camera.
    # -------------------------------------------------
    # -------------------------------------------------
    # Mirror ONLY the camera/skeleton display
    # -------------------------------------------------
    display_frame = cv2.flip(frame, 1)

    # -------------------------------------------------
    # Draw UI AFTER mirroring so text stays readable
    # -------------------------------------------------
    cv2.rectangle(
        display_frame,
        (10, 10),
        (430, 380),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display_frame,
        f"Affected: {affected_side}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    angle_text = (
        f"Angle: {elbow_angle:.1f} deg"
        if elbow_angle is not None
        else "Angle: --"
    )

    cv2.putText(
        display_frame,
        angle_text,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"Reps: {reps}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        f"Stage: {stage.upper()}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ---------------------------------------------
    # Compensation UI
    # ---------------------------------------------
    if compensation_label is None:

        cv2.putText(
            display_frame,
            "Compensation: --",
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2
        )

        cv2.putText(
            display_frame,
            "Confidence: collecting...",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2
        )

    else:

        cv2.putText(
            display_frame,
            f"Compensation: {compensation_label}",
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2
        )

        cv2.putText(
            display_frame,
            f"Confidence: {compensation_confidence * 100:.1f}%",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2
        )

        # -----------------------------------------
        # DEBUG: show all class probabilities
        # -----------------------------------------

        probability_names = [
            "No Comp",
            "Shoulder",
            "Trunk",
            "Lean",
        ]

        for i, name in enumerate(probability_names):

            probability = (
                compensation_probabilities[i] * 100
            )

            cv2.putText(
                display_frame,
                f"{name}: {probability:.1f}%",
                (20, 255 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    cv2.imshow(window_name, display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:

        save_session(
            patient_id,
            exercise,
            reps,
            max_angle,
            min_angle
        )

        print("\nSession Saved Successfully!")
        print(f"Patient ID: {patient_id}")
        print(f"Affected side: {affected_side}")
        print(f"Exercise: {exercise}")
        print(f"Reps: {reps}")
        print(f"ROM: {min_angle:.1f}° - {max_angle:.1f}°")

 

    # Close if window is manually closed
    if cv2.getWindowProperty(
        window_name,
        cv2.WND_PROP_VISIBLE
    ) < 1:
        break


# ---------------------------------
# Cleanup
# ---------------------------------
cap.release()
pose.close()
cv2.destroyAllWindows()