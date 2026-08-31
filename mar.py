import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions import face_mesh as mp_face_mesh_module

face_mesh = mp_face_mesh_module.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Mouth landmark indices: top lip center, bottom lip center, left corner, right corner
MOUTH = [13, 14, 78, 308]

MAR_THRESHOLD = 0.6  # tune this based on your own yawn readings


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_mar(landmarks, w, h):
    top = (landmarks[13].x * w, landmarks[13].y * h)
    bottom = (landmarks[14].x * w, landmarks[14].y * h)
    left = (landmarks[78].x * w, landmarks[78].y * h)
    right = (landmarks[308].x * w, landmarks[308].y * h)

    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)

    return vertical / horizontal


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

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            mar = calculate_mar(landmarks, w, h)
            status = "Yawning" if mar > MAR_THRESHOLD else "Normal"
            color = (0, 0, 255) if status == "Yawning" else (0, 255, 0)

            cv2.putText(frame, f"MAR: {mar:.3f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Mouth: {status}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # visual reference points
            for idx in MOUTH:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
    else:
        cv2.putText(frame, "Face Not Detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("MAR - Mouth Aspect Ratio", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()