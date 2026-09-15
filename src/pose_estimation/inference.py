import cv2
import mediapipe as mp

from src.utils.angles import calculate_angle

from src.action_recognition.rep_counter import RepCounter



# ---------------------------------
# Open USB Webcam
# ---------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

# if not cap.isOpened():
#     raise RuntimeError("Cannot open webcam")

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


#ceating object of rep_counter class
rep_counter = RepCounter()


print("=" * 45)
print(" Stroke Rehab - Pose Estimation")
print(" Press Q or ESC to quit")
print("=" * 45)

window_name = "Stroke Rehab - Pose Estimation"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

while True:

    success, frame = cap.read()

    #print(success, frame.mean() if success else "NO FRAME")

    if not success:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Pose estimation
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)


    
    # RAW CAMERA BLOCK
    # cv2.putText(
    #     frame,
    #     "RAW CAMERA",
    #     (30, 50),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0, 255, 0),
    #     2
    # )

    if results.pose_landmarks:

        # Draw skeleton
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
        lm = results.pose_landmarks.landmark

        #check wrist visibility
        left = lm[15]
        right = lm[16]
        
        print(
            f"LEFT wrist: {left.visibility:.2f} | "
            f"RIGHT wrist: {right.visibility:.2f}"
        )

        # -------- Left Arm --------
        shoulder = lm[11]
        elbow = lm[13]
        wrist = lm[15]

        sx, sy = int(shoulder.x * w), int(shoulder.y * h)
        ex, ey = int(elbow.x * w), int(elbow.y * h)
        wx, wy = int(wrist.x * w), int(wrist.y * h)

        # Highlight left arm joints
        cv2.circle(frame, (sx, sy), 8, (0, 0, 255), -1)      # Shoulder
        cv2.circle(frame, (ex, ey), 8, (255, 0, 0), -1)      # Elbow
        cv2.circle(frame, (wx, wy), 8, (0, 255, 255), -1)    # Wrist

        # Create 2D points
        shoulder_point = (sx, sy)
        elbow_point = (ex, ey)
        wrist_point = (wx, wy)

        # Calculate elbow flexion angle
        elbow_angle = calculate_angle(
            shoulder_point,
            elbow_point,
            wrist_point
        )

        reps, stage = rep_counter.update(elbow_angle)

        # Keep text inside the camera frame
        sx_text = min(sx + 10, w - 120), max(sy - 10, 20)
        ex_text = min(ex + 10, w - 120), max(ey - 10, 20)
        wx_text = min(wx + 10, w - 120), max(wy - 10, 20)

        # Console output
        print(
            f"Elbow Angle: {elbow_angle}° | "
            f"Shoulder: ({sx}, {sy}) "
            f"Elbow: ({ex}, {ey}) "
            f"Wrist: ({wx}, {wy})"
        )

        # Display coordinates on screen
        cv2.putText(frame, f"S:{sx},{sy}", sx_text,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

        cv2.putText(frame, f"E:{ex},{ey}", ex_text,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 2)

        cv2.putText(frame, f"W:{wx},{wy}", wx_text,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,100,255), 2)

        cv2.putText(
            frame,
            f"{elbow_angle} deg",
            (ex + 15, ey + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Reps: {reps}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Stage: {stage.upper()}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )


    # Resize before displaying (fills window)
    cv2.imshow(window_name, frame)

    key = cv2.waitKeyEx(1)

    if key in [ord("q"), ord("Q"), 27]:
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()