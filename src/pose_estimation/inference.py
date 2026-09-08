import cv2
import mediapipe as mp

# -----------------------------
# Initialize MediaPipe Pose
# -----------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Open USB Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("====================================")
print("  Stroke Rehab - Pose Estimation")
print("  Press Q to quit")
print("====================================")

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror the camera for natural interaction
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose estimation
    results = pose.process(rgb)

    if results.pose_landmarks:

        # Draw skeleton
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(255,255,255), thickness=2)
        )

        h, w, _ = frame.shape

        print("\n--------- FRAME ---------")

        for idx, landmark in enumerate(results.pose_landmarks.landmark):

            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = round(landmark.z, 3)

            print(f"Landmark {idx:02d}: ({x:3d}, {y:3d}, {z})")

            # Draw landmark number beside each point
            cv2.putText(
                frame,
                str(idx),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 255),
                1
            )

    cv2.imshow("Stroke Rehab - Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()