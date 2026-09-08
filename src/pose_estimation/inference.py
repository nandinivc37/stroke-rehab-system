import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Webcam started! Press Q to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Mirror the camera (selfie view)
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        ls = landmarks[11]   # Left shoulder
        le = landmarks[13]   # Left elbow
        lw = landmarks[15]   # Left wrist

        print(
            f"Shoulder ({ls.x:.2f}, {ls.y:.2f})  "
            f"Elbow ({le.x:.2f}, {le.y:.2f})  "
            f"Wrist ({lw.x:.2f}, {lw.y:.2f})"
        )

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Stroke Rehab - Pose Debug", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()