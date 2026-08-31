import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from mediapipe.python.solutions import face_mesh as mp_face_mesh_module

face_mesh = mp_face_mesh_module.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.21
WINDOW_SECONDS = 30  # rolling window for PERCLOS calculation

# Each entry: (timestamp, is_closed)
frame_history = deque()


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(landmarks, eye_indices, w, h):
    points = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = points
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    now = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            is_closed = avg_ear < EAR_THRESHOLD
            frame_history.append((now, is_closed))

            # drop entries older than the rolling window
            while frame_history and now - frame_history[0][0] > WINDOW_SECONDS:
                frame_history.popleft()

            closed_count = sum(1 for _, closed in frame_history if closed)
            total_count = len(frame_history)
            perclos = (closed_count / total_count) * 100 if total_count > 0 else 0

            color = (0, 255, 0)
            if perclos > 40:
                color = (0, 0, 255)
            elif perclos > 20:
                color = (0, 165, 255)

            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(frame, "Face Not Detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("PERCLOS", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()