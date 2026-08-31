import cv2
import numpy as np
import time
import csv
import os
from collections import deque
from mediapipe.python.solutions import face_mesh as mp_face_mesh_module
from alert import play_alert

face_mesh = mp_face_mesh_module.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------- Landmarks ----------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
NOSE_TIP, CHIN = 1, 152
LEFT_EYE_CORNER, RIGHT_EYE_CORNER = 33, 263
LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER = 61, 291

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

# ---------- Thresholds ----------
EAR_THRESHOLD = 0.21      # overwritten after calibration
MAR_THRESHOLD = 0.6
PITCH_DOWN_THRESHOLD = -15
EAR_CONSEC_FRAMES = 15
PITCH_CONSEC_FRAMES = 20
WINDOW_SECONDS = 30
ALERT_COOLDOWN = 5
CALIBRATION_SECONDS = 8

# ---------- Fusion score weights ----------
W_EAR = 0.4
W_PERCLOS = 0.3
W_PITCH = 0.2
W_YAWN = 0.1
FUSION_THRESHOLD = 0.5


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
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    _, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
    pitch, yaw, roll = euler_angles.flatten()
    return pitch, yaw, roll


def draw_dashboard(frame, w, h, ear, mar, perclos, blink_count, status, status_color, is_yawning, fusion_score):
    overlay = frame.copy()
    panel_h = 175
    cv2.rectangle(overlay, (0, 0), (280, panel_h), (30, 30, 30), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    cv2.putText(frame, f"EAR: {ear:.2f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"MAR: {mar:.2f}" + (" (Yawn)" if is_yawning else ""),
                (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Blinks: {blink_count}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Fusion Score: {fusion_score:.2f}", (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Status: {status}", (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    if status == "DROWSY":
        cv2.rectangle(frame, (0, h - 45), (w, h), (0, 0, 255), -1)
        cv2.putText(frame, "WARNING: DRIVER IS DROWSY!", (int(w * 0.15), h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def run_calibration(cap):
    ear_samples = []
    pitch_samples = []
    start_time = time.time()

    while time.time() - start_time < CALIBRATION_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            ear_samples.append((left_ear + right_ear) / 2.0)

            pitch, yaw, roll = get_head_pose(landmarks, w, h)
            pitch_samples.append(pitch)

        remaining = int(CALIBRATION_SECONDS - (time.time() - start_time))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame_disp = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.putText(frame_disp, "CALIBRATING...", (int(w*0.25), int(h*0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame_disp, "Keep eyes open, look at camera", (int(w*0.15), int(h*0.53)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_disp, f"Time left: {remaining}s", (int(w*0.35), int(h*0.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Driver Drowsiness Detection System", frame_disp)
        cv2.waitKey(1)

    baseline_ear = np.mean(ear_samples) if ear_samples else 0.28
    baseline_pitch = np.mean(pitch_samples) if pitch_samples else 0.0
    return baseline_ear, baseline_pitch


# ---------- CSV setup ----------
os.makedirs("output", exist_ok=True)
os.makedirs("output/snapshots", exist_ok=True)
log_path = os.path.join("output", "session_log.csv")
log_file = open(log_path, mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["timestamp", "elapsed_sec", "ear", "mar", "perclos", "pitch", "fusion_score", "blink_count", "status"])

# ---------- Run calibration ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting calibration...")
baseline_ear, baseline_pitch = run_calibration(cap)
EAR_THRESHOLD = baseline_ear * 0.75
PITCH_DOWN_THRESHOLD = baseline_pitch - 15

print(f"Calibration done. Baseline EAR: {baseline_ear:.3f} -> Threshold: {EAR_THRESHOLD:.3f}")
print(f"Baseline Pitch: {baseline_pitch:.1f} -> Down Threshold: {PITCH_DOWN_THRESHOLD:.1f}")

# ---------- State ----------
frame_history = deque()
ear_closed_frames = 0
pitch_down_frames = 0
blink_count = 0
blink_closed_frames = 0
last_alert_time = 0
session_start = time.time()

# ---------- Summary tracking ----------
total_frames = 0
drowsy_frames = 0
drowsy_events = 0
was_drowsy_last_frame = False
perclos_samples = []
yawn_count = 0
was_yawning_last_frame = False

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
    ear = mar = perclos = 0
    pitch = 0
    is_yawning = False
    fusion_score = 0

    if results.multi_face_landmarks:
        total_frames += 1
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(landmarks, w, h)
        pitch, yaw, roll = get_head_pose(landmarks, w, h)

        is_closed = ear < EAR_THRESHOLD
        if is_closed:
            blink_closed_frames += 1
        else:
            if blink_closed_frames >= 2:
                blink_count += 1
            blink_closed_frames = 0

        frame_history.append((now, is_closed))
        while frame_history and now - frame_history[0][0] > WINDOW_SECONDS:
            frame_history.popleft()
        closed_count = sum(1 for _, c in frame_history if c)
        perclos = (closed_count / len(frame_history)) * 100 if frame_history else 0
        perclos_samples.append(perclos)

        ear_closed_frames = ear_closed_frames + 1 if is_closed else 0
        pitch_down_frames = pitch_down_frames + 1 if pitch < PITCH_DOWN_THRESHOLD else 0
        is_yawning = mar > MAR_THRESHOLD

        if is_yawning and not was_yawning_last_frame:
            yawn_count += 1
        was_yawning_last_frame = is_yawning

        # ---------- Weighted fusion score ----------
        ear_risk = min(ear_closed_frames / EAR_CONSEC_FRAMES, 1.0)
        perclos_risk = min(perclos / 100.0, 1.0)
        pitch_risk = min(pitch_down_frames / PITCH_CONSEC_FRAMES, 1.0)
        yawn_risk = 1.0 if is_yawning else 0.0

        fusion_score = (W_EAR * ear_risk) + (W_PERCLOS * perclos_risk) + (W_PITCH * pitch_risk) + (W_YAWN * yawn_risk)
        drowsy = fusion_score >= FUSION_THRESHOLD

        status = "DROWSY" if drowsy else "NORMAL"
        status_color = (0, 0, 255) if drowsy else (0, 255, 0)

        if drowsy:
            drowsy_frames += 1
            if not was_drowsy_last_frame:
                drowsy_events += 1
                snapshot_name = f"drowsy_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                snapshot_path = os.path.join("output", "snapshots", snapshot_name)
                cv2.imwrite(snapshot_path, frame)
                print(f"Snapshot saved: {snapshot_path}")
            if (now - last_alert_time) > ALERT_COOLDOWN:
                play_alert()
                last_alert_time = now
        was_drowsy_last_frame = drowsy

    # ---------- Log every frame ----------
    csv_writer.writerow([
        time.strftime("%Y-%m-%d %H:%M:%S"),
        round(now - session_start, 2),
        round(ear, 4), round(mar, 4), round(perclos, 2),
        round(pitch, 2), round(fusion_score, 3), blink_count, status
    ])

    draw_dashboard(frame, w, h, ear, mar, perclos, blink_count, status, status_color, is_yawning, fusion_score)

    cv2.imshow("Driver Drowsiness Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()

# ---------- Session summary ----------
session_duration = time.time() - session_start
avg_perclos = np.mean(perclos_samples) if perclos_samples else 0
max_perclos = np.max(perclos_samples) if perclos_samples else 0
drowsy_percent = (drowsy_frames / total_frames * 100) if total_frames else 0

summary_lines = [
    "===== SESSION SUMMARY =====",
    f"Duration: {session_duration:.1f} sec",
    f"Total Frames Analyzed: {total_frames}",
    f"Drowsy Events (distinct triggers): {drowsy_events}",
    f"Frames Drowsy: {drowsy_frames} ({drowsy_percent:.1f}% of session)",
    f"Average PERCLOS: {avg_perclos:.1f}%",
    f"Max PERCLOS: {max_perclos:.1f}%",
    f"Total Blinks: {blink_count}",
    f"Total Yawns: {yawn_count}",
    f"Calibration Baseline EAR: {baseline_ear:.3f}",
    f"EAR Threshold Used: {EAR_THRESHOLD:.3f}",
    "============================"
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

summary_path = os.path.join("output", "session_summary.txt")
with open(summary_path, "w") as f:
    f.write(summary_text)

print(f"\nSession log saved to: {log_path}")
print(f"Session summary saved to: {summary_path}")