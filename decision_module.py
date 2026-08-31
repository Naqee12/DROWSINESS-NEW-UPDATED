import cv2
import mediapipe as mp
import numpy as np
import time
from alert import play_alert
from collections import deque
from mediapipe.python.solutions import face_mesh as mp_face_mesh_module

face_mesh = mp_face_mesh_module.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- Landmark indices ----------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]
NOSE_TIP, CHIN = 1, 152
LEFT_EYE_CORNER, RIGHT_EYE_CORNER = 33, 263
LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER = 61, 291

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

# ---------- Thresholds (tune these based on your Phase 5-9 testing) ----------
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.6
PERCLOS_THRESHOLD = 40.0
PITCH_DOWN_THRESHOLD = -15
EAR_CONSEC_FRAMES = 15   # ~0.5s at 30fps, sustained closure
PITCH_CONSEC_FRAMES = 20 # sustained head-down

WINDOW_SECONDS = 30


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(landmarks, eye_indices, w, h):
    points = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = points
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4))


def calculate_mar(landmarks, w, h):
    top = (landmarks[13].x * w, landmarks[13].y * h)
    bottom = (landmarks[14].x * w, landmarks[14].y * h)
    left = (landmarks[78].x * w, landmarks[78].y * h)
    right = (landmarks[308].x * w, landmarks[308].y * h)
    return euclidean(top, bottom) / euclidean(left, right)


def get_head_pose(landmarks, w, h):
    image_points = np.array([
        (landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h),
        (landmarks[CHIN].x * w, landmarks[CHIN].y * h),
        (landmarks[LEFT_EYE_CORNER].x * w, landmarks[LEFT_EYE_CORNER].y * h),
        (landmarks[RIGHT_EYE_CORNER].x * w, landmarks[RIGHT_EYE_CORNER].y * h),
        (landmarks[LEFT_MOUTH_CORNER].x * w, landmarks[LEFT_MOUTH_CORNER].y * h),
        (landmarks[RIGHT_MOUTH_CORNER].x * w, landmarks[RIGHT_MOUTH_CORNER].y * h)
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    _, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
    pitch, yaw, roll = euler_angles.flatten()
    return pitch, yaw, roll


# ---------- State tracking ----------
frame_history = deque()   # for PERCLOS
ear_closed_frames = 0
pitch_down_frames = 0
blink_count = 0
blink_closed_frames = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    now = time.time()

    status = "No Face"
    status_color = (0, 0, 255)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(landmarks, w, h)
        pitch, yaw, roll = get_head_pose(landmarks, w, h)

        # Blink counting
        is_closed = avg_ear < EAR_THRESHOLD
        if is_closed:
            blink_closed_frames += 1
        else:
            if blink_closed_frames >= 2:
                blink_count += 1
            blink_closed_frames = 0

        # PERCLOS (rolling window)
        frame_history.append((now, is_closed))
        while frame_history and now - frame_history[0][0] > WINDOW_SECONDS:
            frame_history.popleft()
        closed_count = sum(1 for _, c in frame_history if c)
        perclos = (closed_count / len(frame_history)) * 100 if frame_history else 0

        # Sustained EAR closure
        ear_closed_frames = ear_closed_frames + 1 if is_closed else 0

        # Sustained head-down
        pitch_down_frames = pitch_down_frames + 1 if pitch < PITCH_DOWN_THRESHOLD else 0

        # Yawn
        is_yawning = mar > MAR_THRESHOLD

        # ---------- Decision fusion ----------
        drowsy = (
            ear_closed_frames >= EAR_CONSEC_FRAMES
            or perclos > PERCLOS_THRESHOLD
            or pitch_down_frames >= PITCH_CONSEC_FRAMES
        )

        status = "DROWSY" if drowsy else "NORMAL"
        status_color = (0, 0, 255) if drowsy else (0, 255, 0)

        if drowsy:
            play_alert()

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f} {'(Yawn)' if is_yawning else ''}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Status: {status}", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)
    if drowsy:
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 255), -1)
    cv2.putText(frame, "WARNING: DRIVER IS DROWSY!", (int(w*0.1), 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Decision Module", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()