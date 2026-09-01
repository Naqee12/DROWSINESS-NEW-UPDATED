import cv2
import numpy as np
import time
import os
import json
import csv
import threading
from collections import deque

from flask import Flask, Response, jsonify, request, render_template, send_file
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python import BaseOptions
from mediapipe import Image, ImageFormat
from ultralytics import YOLO

from config_loader import load_config
from notifier import send_drowsy_alert
import face_id

CFG = load_config()

WINDOW_SECONDS = CFG["WINDOW_SECONDS"]
CALIBRATION_SECONDS = CFG["CALIBRATION_SECONDS"]
PITCH_BASELINE_OFFSET = CFG["PITCH_BASELINE_OFFSET"]
ALERT_COOLDOWN = CFG["ALERT_COOLDOWN"]
W_EAR = CFG["FUSION_WEIGHTS"]["EAR"]
W_PERCLOS = CFG["FUSION_WEIGHTS"]["PERCLOS"]
W_PITCH = CFG["FUSION_WEIGHTS"]["PITCH"]
W_YAWN = CFG["FUSION_WEIGHTS"]["YAWN"]

# FaceLandmarker uses 478 landmarks (468 face mesh + 10 iris)
# The first 468 are compatible with the old FaceMesh indices
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

PROFILES_PATH = "profiles.json"
YAWN_WINDOW_SECONDS = 60
YAWN_ESCALATION_COUNT = 2
YAWN_WARNING_COOLDOWN = 45
PHONE_DETECTION_INTERVAL = 10
CELL_PHONE_CLASS_ID = 67

SEVERITY_COLORS = {
    "NORMAL": "#43a047",
    "MILD FATIGUE": "#fdd835",
    "DROWSY": "#fb8c00",
    "CRITICAL": "#e53935"
}

RECOG_INTERVAL_FRAMES = 15


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
    
    pitch = np.arctan2(-rotation_matrix[2, 0], 
                       np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)) * 180.0 / np.pi
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180.0 / np.pi
    roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]) * 180.0 / np.pi
    return pitch, yaw, roll


def get_brightness(frame):
    return cv2.mean(frame)[0]


def enhance_low_light(frame, clahe_obj):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe_obj.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def get_severity(fusion_score, mild_t, drowsy_t, critical_t):
    if fusion_score >= critical_t:
        return "CRITICAL"
    elif fusion_score >= drowsy_t:
        return "DROWSY"
    elif fusion_score >= mild_t:
        return "MILD FATIGUE"
    else:
        return "NORMAL"


# ==================== Shared runtime state ====================
lock = threading.Lock()

RUNTIME = {
    "mar_threshold": CFG["MAR_THRESHOLD"],
    "ear_consec_seconds": CFG["EAR_CONSEC_SECONDS"],
    "pitch_consec_seconds": CFG["PITCH_CONSEC_SECONDS"],
    "ear_baseline_ratio": CFG["EAR_BASELINE_RATIO"],
    "mild_threshold": 0.3,
    "drowsy_threshold": 0.5,
    "critical_threshold": 0.75,
    "phone_detection_enabled": True,
    "ground_truth_drowsy": False,
    "recalibrate_requested": False,
}

METRICS = {
    "calibrating": True, "calib_remaining": CALIBRATION_SECONDS,
    "ear": 0.0, "mar": 0.0, "perclos": 0.0, "pitch": 0.0, "fusion": 0.0,
    "severity": "NORMAL", "blink_count": 0, "yawn_count": 0,
    "phone_detected": False, "new_alert_event": None, "fatigue_warning": "",
    "drowsy_events": 0,
    "driver_recognized": "Unknown", "driver_confidence": 0.0,
    "driver_profile_loaded": False
}

latest_jpeg = None
log_rows = []
phone_model = None  # loaded lazily
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))  # Created once
MAX_LOG_ROWS = 5000  # Prevent unbounded memory growth


def read_profiles():
    if os.path.exists(PROFILES_PATH):
        try:
            with open(PROFILES_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def create_face_landmarker():
    """Create and configure FaceLandmarker for video mode."""
    model_path = os.path.join(
        os.path.dirname(__import__('mediapipe').__file__),
        'tasks', 'python', 'vision', 'face_landmarker.task'
    )
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=FaceLandmarkerOptions.running_mode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def detection_loop():
    global latest_jpeg, phone_model

    face_landmarker = create_face_landmarker()

    # Use DirectShow backend on Windows for better reliability
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Reset recognition state at start of new session
    face_id.reset_recognition_state()

    calibrating = True
    calib_start_time = time.time()
    ear_samples, pitch_samples = [], []
    baseline_ear, baseline_pitch = 0.28, 0.0
    ear_threshold, pitch_down_threshold = 0.21, -15
    recog_frame_counter = 0
    recognized_driver = "Unknown"
    driver_confidence = 0.0
    profile_loaded = False

    frame_history = deque()
    ear_closed_since = None
    pitch_down_since = None
    blink_count = 0
    blink_closed_frames = 0
    was_yawning_last_frame = False
    yawn_count = 0
    yawn_timestamps = deque()
    last_yawn_warning_time = 0
    drowsy_events = 0
    last_severity = "NORMAL"
    last_alert_time = 0
    notification_sent = False
    frame_counter = 0
    phone_detected = False
    session_start = time.time()
    log_cleanup_counter = 0
    LOG_CLEANUP_INTERVAL = 300  # Clean up log rows every 300 frames (~10 seconds at 30 FPS)
    target_fps = 30
    frame_time = 1.0 / target_fps
    loop_start_time = time.time()
    timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        with lock:
            if RUNTIME["recalibrate_requested"]:
                calibrating = True
                calib_start_time = time.time()
                ear_samples, pitch_samples = [], []
                # Reset recognition state on recalibration
                face_id.reset_recognition_state()
                recognized_driver = "Unknown"
                driver_confidence = 0.0
                profile_loaded = False
                with lock:
                    METRICS["driver_recognized"] = "Unknown"
                    METRICS["driver_confidence"] = 0.0
                    METRICS["driver_profile_loaded"] = False
                RUNTIME["recalibrate_requested"] = False

        if get_brightness(frame) < 80:
            frame = enhance_low_light(frame, clahe)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        now = time.time()

        if calibrating:
            elapsed = now - calib_start_time
            remaining = max(0, int(CALIBRATION_SECONDS - elapsed))

            if results.face_landmarks:
                landmarks = results.face_landmarks[0]
                le = calculate_ear(landmarks, LEFT_EYE, w, h)
                re = calculate_ear(landmarks, RIGHT_EYE, w, h)
                ear_samples.append((le + re) / 2.0)
                pitch, _, _ = get_head_pose(landmarks, w, h)
                pitch_samples.append(pitch)

                recog_frame_counter += 1
                if recog_frame_counter % RECOG_INTERVAL_FRAMES == 0:
                    name, conf, recognized = face_id.recognize_driver(frame)
                    if recognized:
                        recognized_driver = name
                        driver_confidence = conf
                        with lock:
                            METRICS["driver_recognized"] = name
                            METRICS["driver_confidence"] = conf

            cv2.putText(frame, "CALIBRATING...", (int(w * 0.2), int(h * 0.45)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"Keep eyes open - {remaining}s left", (int(w * 0.08), int(h * 0.55)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if recognized_driver != "Unknown":
                cv2.putText(frame, f"Driver: {recognized_driver} ({driver_confidence:.1f})", (int(w * 0.08), int(h * 0.65)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Driver: Unknown (enroll later)", (int(w * 0.08), int(h * 0.65)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            with lock:
                METRICS["calibrating"] = True
                METRICS["calib_remaining"] = remaining
                METRICS["driver_recognized"] = recognized_driver
                METRICS["driver_confidence"] = driver_confidence
            # Minimal lock scope - calibration metrics updated separately

            if elapsed >= CALIBRATION_SECONDS:
                baseline_ear = np.mean(ear_samples) if ear_samples else 0.28
                baseline_pitch = np.mean(pitch_samples) if pitch_samples else 0.0
                with lock:
                    ear_threshold = baseline_ear * RUNTIME["ear_baseline_ratio"]
                pitch_down_threshold = baseline_pitch - PITCH_BASELINE_OFFSET

                if recognized_driver != "Unknown" and not profile_loaded:
                    profiles = read_profiles()
                    if recognized_driver in profiles:
                        p = profiles[recognized_driver]
                        baseline_ear = p.get("baseline_ear", baseline_ear)
                        baseline_pitch = p.get("baseline_pitch", baseline_pitch)
                        ear_threshold = p.get("ear_threshold", ear_threshold)
                        pitch_down_threshold = p.get("pitch_down_threshold", pitch_down_threshold)
                        profile_loaded = True
                        with lock:
                            METRICS["driver_profile_loaded"] = True

                calibrating = False

            _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            with lock:
                latest_jpeg = buf.tobytes()

            # Frame rate limiting during calibration
            loop_elapsed = time.time() - loop_start_time
            if loop_elapsed < frame_time:
                time.sleep(frame_time - loop_elapsed)
            loop_start_time = time.time()

            continue

        with lock:
            rt = dict(RUNTIME)

        frame_counter += 1
        if rt["phone_detection_enabled"]:
            if phone_model is None:
                phone_model = YOLO("yolo26n.pt")
            if frame_counter % PHONE_DETECTION_INTERVAL == 0:
                phone_results = phone_model(frame, verbose=False, classes=[CELL_PHONE_CLASS_ID], conf=0.35, imgsz=320)
                phone_detected = False
                for r in phone_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        phone_detected = True
                if not phone_detected:
                    pass  # Phone not detected this frame
        else:
            phone_detected = False

        ear = mar = perclos = pitch = fusion_score = 0.0
        severity = "NORMAL"
        is_yawning = False
        fatigue_warning = ""
        new_event = None

        face_detected = False
        if results.face_landmarks:
            face_detected = True
            landmarks = results.face_landmarks[0]
            le = calculate_ear(landmarks, LEFT_EYE, w, h)
            re = calculate_ear(landmarks, RIGHT_EYE, w, h)
            ear = (le + re) / 2.0
            mar = calculate_mar(landmarks, w, h)
            pitch, _, _ = get_head_pose(landmarks, w, h)

            if not profile_loaded:
                recog_frame_counter += 1
                if recog_frame_counter % RECOG_INTERVAL_FRAMES == 0:
                    name, conf, recognized = face_id.recognize_driver(frame)
                    if recognized:
                        recognized_driver = name
                        driver_confidence = conf
                        with lock:
                            METRICS["driver_recognized"] = name
                            METRICS["driver_confidence"] = conf
                        profiles = read_profiles()
                        if recognized_driver in profiles:
                            p = profiles[recognized_driver]
                            baseline_ear = p.get("baseline_ear", baseline_ear)
                            baseline_pitch = p.get("baseline_pitch", baseline_pitch)
                            ear_threshold = p.get("ear_threshold", ear_threshold)
                            pitch_down_threshold = p.get("pitch_down_threshold", pitch_down_threshold)
                            profile_loaded = True
                            with lock:
                                METRICS["driver_profile_loaded"] = True

            is_closed = ear < ear_threshold
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

            if is_closed:
                if ear_closed_since is None:
                    ear_closed_since = now
            else:
                ear_closed_since = None
            ear_closed_duration = (now - ear_closed_since) if ear_closed_since else 0.0

            if pitch < pitch_down_threshold:
                if pitch_down_since is None:
                    pitch_down_since = now
            else:
                pitch_down_since = None
            pitch_down_duration = (now - pitch_down_since) if pitch_down_since else 0.0

            is_yawning = mar > rt["mar_threshold"]
            if is_yawning and not was_yawning_last_frame:
                yawn_count += 1
                yawn_timestamps.append(now)
            was_yawning_last_frame = is_yawning

            while yawn_timestamps and now - yawn_timestamps[0] > YAWN_WINDOW_SECONDS:
                yawn_timestamps.popleft()

            if (len(yawn_timestamps) >= YAWN_ESCALATION_COUNT and
                    (now - last_yawn_warning_time) > YAWN_WARNING_COOLDOWN):
                fatigue_warning = "Fatigue building: multiple yawns detected"
                last_yawn_warning_time = now

            ear_risk = min(ear_closed_duration / rt["ear_consec_seconds"], 1.0)
            perclos_risk = min(perclos / 100.0, 1.0)
            pitch_risk = min(pitch_down_duration / rt["pitch_consec_seconds"], 1.0)
            yawn_risk = 1.0 if is_yawning else 0.0
            fusion_score = (W_EAR * ear_risk) + (W_PERCLOS * perclos_risk) + (W_PITCH * pitch_risk) + (W_YAWN * yawn_risk)

            severity = get_severity(fusion_score, rt["mild_threshold"], rt["drowsy_threshold"], rt["critical_threshold"])

            if severity in ("DROWSY", "CRITICAL"):
                if last_severity not in ("DROWSY", "CRITICAL") and (now - last_alert_time) > ALERT_COOLDOWN:
                    last_alert_time = now
                    drowsy_events += 1
                    new_event = severity

                    trigger_count = CFG["NOTIFICATIONS"].get("DROWSY_EVENT_TRIGGER_COUNT", 3)
                    send_once = CFG["NOTIFICATIONS"].get("SEND_ONCE_PER_SESSION", True)
                    should_notify = drowsy_events >= trigger_count and (not send_once or not notification_sent)
                    if should_notify:
                        os.makedirs("output/snapshots", exist_ok=True)
                        snap_path = os.path.join("output", "snapshots", f"{severity.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                        cv2.imwrite(snap_path, frame)
                        threading.Thread(target=send_drowsy_alert,
                                          args=(CFG, drowsy_events, now - session_start, snap_path),
                                          daemon=True).start()
                        notification_sent = True

                bar_color = (0, 0, 255) if severity == "CRITICAL" else (0, 100, 255)
                cv2.rectangle(frame, (0, h - 40), (w, h), bar_color, -1)
                cv2.putText(frame, f"WARNING: {severity}!", (int(w * 0.15), h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            last_severity = severity

            log_rows.append([
                time.strftime("%Y-%m-%d %H:%M:%S"), round(now - session_start, 2),
                round(ear, 4), round(mar, 4), round(perclos, 2), round(pitch, 2),
                round(fusion_score, 3), severity, blink_count, phone_detected,
                int(rt["ground_truth_drowsy"])
            ])
            if len(log_rows) > MAX_LOG_ROWS:
                log_rows.pop(0)

        # Periodic log cleanup regardless of face detection
        log_cleanup_counter += 1
        if log_cleanup_counter >= LOG_CLEANUP_INTERVAL:
            log_cleanup_counter = 0
            if len(log_rows) > MAX_LOG_ROWS:
                del log_rows[:-MAX_LOG_ROWS]

        if phone_detected:
            cv2.rectangle(frame, (0, 0), (w, 35), (0, 140, 255), -1)
            cv2.putText(frame, "WARNING: Phone Detected!", (int(w * 0.2), 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        metrics_update = {
                "calibrating": False, "ear": ear, "mar": mar, "perclos": perclos,
                "pitch": pitch, "fusion": fusion_score, "severity": severity,
                "blink_count": blink_count, "yawn_count": yawn_count,
                "phone_detected": phone_detected, "new_alert_event": new_event,
                "fatigue_warning": fatigue_warning, "drowsy_events": drowsy_events,
                "baseline_ear": baseline_ear, "ear_threshold": ear_threshold,
                "baseline_pitch": baseline_pitch, "pitch_down_threshold": pitch_down_threshold,
                "driver_recognized": recognized_driver, "driver_confidence": driver_confidence,
                "driver_profile_loaded": profile_loaded
            }
        with lock:
            METRICS.update(metrics_update)

        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        with lock:
            latest_jpeg = buf.tobytes()

        # Frame rate limiting at end of loop to account for full processing time
        loop_elapsed = time.time() - loop_start_time
        if loop_elapsed < frame_time:
            time.sleep(frame_time - loop_elapsed)
        loop_start_time = time.time()


app = Flask(__name__)


def gen_frames():
    while True:
        with lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/metrics")
def metrics_route():
    with lock:
        return jsonify(METRICS)


@app.route("/settings", methods=["POST"])
def settings_route():
    data = request.json or {}
    with lock:
        for key in ["mar_threshold", "ear_consec_seconds", "pitch_consec_seconds",
                    "ear_baseline_ratio", "mild_threshold", "drowsy_threshold", "critical_threshold"]:
            if key in data:
                RUNTIME[key] = float(data[key])
        if "phone_detection_enabled" in data:
            RUNTIME["phone_detection_enabled"] = bool(data["phone_detection_enabled"])
        if "ground_truth_drowsy" in data:
            RUNTIME["ground_truth_drowsy"] = bool(data["ground_truth_drowsy"])
    return jsonify({"ok": True})


@app.route("/recalibrate", methods=["POST"])
def recalibrate_route():
    with lock:
        RUNTIME["recalibrate_requested"] = True
    return jsonify({"ok": True})


@app.route("/save_profile", methods=["POST"])
def save_profile_route():
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "No name given"})
    with lock:
        m = dict(METRICS)
    profiles = read_profiles()
    profiles[name] = {
        "baseline_ear": m.get("baseline_ear", 0.28),
        "baseline_pitch": m.get("baseline_pitch", 0.0),
        "ear_threshold": m.get("ear_threshold", 0.21),
        "pitch_down_threshold": m.get("pitch_down_threshold", -15)
    }
    with open(PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=2)
    # Also save to face_id calibration store for auto-load on recognition
    face_id.save_driver_calibration(name, {
        "baseline_ear": m.get("baseline_ear", 0.28),
        "baseline_pitch": m.get("baseline_pitch", 0.0),
        "ear_threshold": m.get("ear_threshold", 0.21),
        "pitch_down_threshold": m.get("pitch_down_threshold", -15)
    })
    return jsonify({"ok": True})


@app.route("/enroll_driver", methods=["POST"])
def enroll_driver_route():
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "error": "No name given"})

    def enroll_task():
        success = face_id.enroll_driver(name, num_samples=20)
        return success

    thread = threading.Thread(target=enroll_task, daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": f"Enrollment started for {name}"})


@app.route("/known_drivers", methods=["GET"])
def known_drivers_route():
    drivers = face_id.get_known_drivers()
    return jsonify({"drivers": drivers})


@app.route("/download_csv")
def download_csv():
    csv_path = os.path.join("output", "web_session_log.csv")
    os.makedirs("output", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "elapsed_sec", "ear", "mar", "perclos", "pitch",
                          "fusion_score", "severity", "blink_count", "phone_detected", "ground_truth"])
        writer.writerows(log_rows)
    return send_file(csv_path, as_attachment=True)


@app.route("/")
def index():
    return render_template("dashboard.html")


if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)