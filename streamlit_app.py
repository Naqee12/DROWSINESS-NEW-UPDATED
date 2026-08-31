import streamlit as st
import cv2
import numpy as np
import time
import os
import json
import csv
import threading
from collections import deque

from mediapipe.python.solutions import face_mesh as mp_face_mesh_module
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import streamlit.components.v1 as components

from config_loader import load_config
from notifier import send_drowsy_alert

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")

CFG = load_config()

DEFAULT_MAR_THRESHOLD = CFG["MAR_THRESHOLD"]
DEFAULT_EAR_CONSEC_SECONDS = CFG["EAR_CONSEC_SECONDS"]
DEFAULT_PITCH_CONSEC_SECONDS = CFG["PITCH_CONSEC_SECONDS"]
WINDOW_SECONDS = CFG["WINDOW_SECONDS"]
CALIBRATION_SECONDS = CFG["CALIBRATION_SECONDS"]
DEFAULT_EAR_BASELINE_RATIO = CFG["EAR_BASELINE_RATIO"]
PITCH_BASELINE_OFFSET = CFG["PITCH_BASELINE_OFFSET"]
W_EAR = CFG["FUSION_WEIGHTS"]["EAR"]
W_PERCLOS = CFG["FUSION_WEIGHTS"]["PERCLOS"]
W_PITCH = CFG["FUSION_WEIGHTS"]["PITCH"]
W_YAWN = CFG["FUSION_WEIGHTS"]["YAWN"]

DEFAULT_MILD_THRESHOLD = 0.3
DEFAULT_DROWSY_THRESHOLD = 0.5
DEFAULT_CRITICAL_THRESHOLD = 0.75

PHONE_DETECTION_INTERVAL = 10
CELL_PHONE_CLASS_ID = 67

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
ALERT_COOLDOWN = CFG["ALERT_COOLDOWN"]

SEVERITY_COLORS = {
    "NORMAL": "#43a047",
    "MILD FATIGUE": "#fdd835",
    "DROWSY": "#fb8c00",
    "CRITICAL": "#e53935"
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


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


def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean()


def get_severity(fusion_score, mild_t, drowsy_t, critical_t):
    if fusion_score >= critical_t:
        return "CRITICAL"
    elif fusion_score >= drowsy_t:
        return "DROWSY"
    elif fusion_score >= mild_t:
        return "MILD FATIGUE"
    else:
        return "NORMAL"


class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh_module.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.phone_model = None  # loaded lazily only if phone detection is enabled

        self.lock = threading.Lock()

        # ---------- Tunable thresholds (set from sidebar each rerun) ----------
        self.mar_threshold = DEFAULT_MAR_THRESHOLD
        self.ear_consec_seconds = DEFAULT_EAR_CONSEC_SECONDS
        self.pitch_consec_seconds = DEFAULT_PITCH_CONSEC_SECONDS
        self.ear_baseline_ratio = DEFAULT_EAR_BASELINE_RATIO
        self.mild_threshold = DEFAULT_MILD_THRESHOLD
        self.drowsy_threshold = DEFAULT_DROWSY_THRESHOLD
        self.critical_threshold = DEFAULT_CRITICAL_THRESHOLD
        self.phone_detection_enabled = False

        # ---------- Calibration ----------
        self.calibrating = True
        self.calib_start_time = time.time()
        self.ear_samples = []
        self.pitch_samples = []
        self.baseline_ear = 0.28
        self.baseline_pitch = 0.0
        self.ear_threshold = 0.21
        self.pitch_down_threshold = -15

        # ---------- Detection state ----------
        self.frame_history = deque()
        self.ear_closed_since = None
        self.pitch_down_since = None
        self.blink_count = 0
        self.blink_closed_frames = 0
        self.was_yawning_last_frame = False
        self.yawn_count = 0
        self.yawn_timestamps = deque()
        self.last_yawn_warning_time = 0
        self.drowsy_events = 0
        self.last_severity = "NORMAL"
        self.last_alert_time = 0
        self.notification_sent = False
        self.frame_counter = 0
        self.phone_detected = False
        self.ground_truth_drowsy = False
        self.session_start = time.time()

        # ---------- Output shared with main thread ----------
        self.metrics = {
            "calibrating": True, "calib_remaining": CALIBRATION_SECONDS,
            "ear": 0.0, "mar": 0.0, "perclos": 0.0, "pitch": 0.0, "fusion": 0.0,
            "severity": "NORMAL", "blink_count": 0, "yawn_count": 0,
            "phone_detected": False, "new_alert_event": None, "fatigue_warning": ""
        }
        self.log_rows = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        if get_brightness(img) < 80:
            img = enhance_low_light(img)

        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        # ---------- Calibration ----------
        if self.calibrating:
            elapsed = now - self.calib_start_time
            remaining = max(0, int(CALIBRATION_SECONDS - elapsed))

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                le = calculate_ear(landmarks, LEFT_EYE, w, h)
                re = calculate_ear(landmarks, RIGHT_EYE, w, h)
                self.ear_samples.append((le + re) / 2.0)
                pitch, _, _ = get_head_pose(landmarks, w, h)
                self.pitch_samples.append(pitch)

            cv2.putText(img, "CALIBRATING...", (int(w * 0.2), int(h * 0.45)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(img, f"Keep eyes open - {remaining}s left", (int(w * 0.08), int(h * 0.55)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            with self.lock:
                self.metrics["calibrating"] = True
                self.metrics["calib_remaining"] = remaining

            if elapsed >= CALIBRATION_SECONDS:
                self.baseline_ear = np.mean(self.ear_samples) if self.ear_samples else 0.28
                self.baseline_pitch = np.mean(self.pitch_samples) if self.pitch_samples else 0.0
                self.ear_threshold = self.baseline_ear * self.ear_baseline_ratio
                self.pitch_down_threshold = self.baseline_pitch - PITCH_BASELINE_OFFSET
                self.calibrating = False

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # ---------- Phone detection (optional, throttled) ----------
        self.frame_counter += 1
        if self.phone_detection_enabled:
            if self.phone_model is None:
                self.phone_model = YOLO("yolo26n.pt")
            if self.frame_counter % PHONE_DETECTION_INTERVAL == 0:
                results_phone = self.phone_model(img, verbose=False, classes=[CELL_PHONE_CLASS_ID], conf=0.4, imgsz=320)
                self.phone_detected = False
                for r in results_phone:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        self.phone_detected = True
        else:
            self.phone_detected = False

        # ---------- Core detection ----------
        ear = mar = perclos = pitch = fusion_score = 0.0
        severity = "NORMAL"
        is_yawning = False
        fatigue_warning = ""

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            le = calculate_ear(landmarks, LEFT_EYE, w, h)
            re = calculate_ear(landmarks, RIGHT_EYE, w, h)
            ear = (le + re) / 2.0
            mar = calculate_mar(landmarks, w, h)
            pitch, _, _ = get_head_pose(landmarks, w, h)

            is_closed = ear < self.ear_threshold
            if is_closed:
                self.blink_closed_frames += 1
            else:
                if self.blink_closed_frames >= 2:
                    self.blink_count += 1
                self.blink_closed_frames = 0

            self.frame_history.append((now, is_closed))
            while self.frame_history and now - self.frame_history[0][0] > WINDOW_SECONDS:
                self.frame_history.popleft()
            closed_count = sum(1 for _, c in self.frame_history if c)
            perclos = (closed_count / len(self.frame_history)) * 100 if self.frame_history else 0

            if is_closed:
                if self.ear_closed_since is None:
                    self.ear_closed_since = now
            else:
                self.ear_closed_since = None
            ear_closed_duration = (now - self.ear_closed_since) if self.ear_closed_since else 0.0

            if pitch < self.pitch_down_threshold:
                if self.pitch_down_since is None:
                    self.pitch_down_since = now
            else:
                self.pitch_down_since = None
            pitch_down_duration = (now - self.pitch_down_since) if self.pitch_down_since else 0.0

            is_yawning = mar > self.mar_threshold
            if is_yawning and not self.was_yawning_last_frame:
                self.yawn_count += 1
                self.yawn_timestamps.append(now)
            self.was_yawning_last_frame = is_yawning

            while self.yawn_timestamps and now - self.yawn_timestamps[0] > YAWN_WINDOW_SECONDS:
                self.yawn_timestamps.popleft()

            if (len(self.yawn_timestamps) >= YAWN_ESCALATION_COUNT and
                    (now - self.last_yawn_warning_time) > YAWN_WARNING_COOLDOWN):
                fatigue_warning = "Fatigue building: multiple yawns detected"
                self.last_yawn_warning_time = now
            elif (now - self.last_yawn_warning_time) <= YAWN_WARNING_COOLDOWN:
                with self.lock:
                    fatigue_warning = self.metrics.get("fatigue_warning", "")

            ear_risk = min(ear_closed_duration / self.ear_consec_seconds, 1.0)
            perclos_risk = min(perclos / 100.0, 1.0)
            pitch_risk = min(pitch_down_duration / self.pitch_consec_seconds, 1.0)
            yawn_risk = 1.0 if is_yawning else 0.0
            fusion_score = (W_EAR * ear_risk) + (W_PERCLOS * perclos_risk) + (W_PITCH * pitch_risk) + (W_YAWN * yawn_risk)

            severity = get_severity(fusion_score, self.mild_threshold, self.drowsy_threshold, self.critical_threshold)

            new_event = None
            if severity in ("DROWSY", "CRITICAL"):
                if self.last_severity not in ("DROWSY", "CRITICAL") and (now - self.last_alert_time) > ALERT_COOLDOWN:
                    self.last_alert_time = now
                    self.drowsy_events += 1
                    new_event = severity

                    trigger_count = CFG["NOTIFICATIONS"].get("DROWSY_EVENT_TRIGGER_COUNT", 3)
                    send_once = CFG["NOTIFICATIONS"].get("SEND_ONCE_PER_SESSION", True)
                    should_notify = self.drowsy_events >= trigger_count and (not send_once or not self.notification_sent)
                    if should_notify:
                        snap_path = os.path.join("output", "snapshots", f"{severity.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                        os.makedirs(os.path.dirname(snap_path), exist_ok=True)
                        cv2.imwrite(snap_path, img)
                        threading.Thread(target=send_drowsy_alert,
                                          args=(CFG, self.drowsy_events, now - self.session_start, snap_path),
                                          daemon=True).start()
                        self.notification_sent = True

                bar_color = (0, 0, 255) if severity == "CRITICAL" else (0, 100, 255)
                cv2.rectangle(img, (0, h - 40), (w, h), bar_color, -1)
                cv2.putText(img, f"WARNING: {severity}!", (int(w * 0.15), h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            self.last_severity = severity

            self.log_rows.append([
                time.strftime("%Y-%m-%d %H:%M:%S"), round(now - self.session_start, 2),
                round(ear, 4), round(mar, 4), round(perclos, 2), round(pitch, 2),
                round(fusion_score, 3), severity, self.blink_count, self.phone_detected,
                int(self.ground_truth_drowsy)
            ])

            with self.lock:
                self.metrics.update({
                    "calibrating": False, "ear": ear, "mar": mar, "perclos": perclos,
                    "pitch": pitch, "fusion": fusion_score, "severity": severity,
                    "blink_count": self.blink_count, "yawn_count": self.yawn_count,
                    "phone_detected": self.phone_detected, "new_alert_event": new_event,
                    "fatigue_warning": fatigue_warning
                })

        if self.phone_detected:
            cv2.rectangle(img, (0, 0), (w, 35), (0, 140, 255), -1)
            cv2.putText(img, "WARNING: Phone Detected!", (int(w * 0.2), 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def read_profiles():
    if os.path.exists(PROFILES_PATH):
        try:
            with open(PROFILES_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


# ================= UI =================

st.title("Driver Drowsiness Detection System")
st.caption("Browser-based demo — uses YOUR webcam, processed live on the server.")

with st.sidebar:
    st.header("Driver Profile")
    driver_name = st.text_input("Driver name")
    col1, col2 = st.columns(2)
    load_clicked = col1.button("Load Profile")
    save_clicked = col2.button("Save Profile")

    st.header("Settings")
    mild_t = st.slider("Mild Fatigue Threshold", 0.1, 0.5, DEFAULT_MILD_THRESHOLD, 0.01)
    drowsy_t = st.slider("Drowsy Threshold", 0.3, 0.8, DEFAULT_DROWSY_THRESHOLD, 0.01)
    critical_t = st.slider("Critical Threshold", 0.5, 0.95, DEFAULT_CRITICAL_THRESHOLD, 0.01)
    ear_ratio = st.slider("EAR Baseline Ratio", 0.5, 0.9, DEFAULT_EAR_BASELINE_RATIO, 0.01)
    ear_seconds = st.slider("EAR Closed Duration (sec)", 0.2, 2.0, DEFAULT_EAR_CONSEC_SECONDS, 0.1)
    mar_t = st.slider("Yawn (MAR) Threshold", 0.3, 0.9, DEFAULT_MAR_THRESHOLD, 0.01)

    st.header("Features")
    phone_enabled = st.checkbox("Phone Detection (adds latency)", value=False)

    st.header("Testing")
    ground_truth = st.toggle("Mark: I am currently drowsy (for labeling)", value=False)

ctx = webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    ctx.video_processor.mild_threshold = mild_t
    ctx.video_processor.drowsy_threshold = drowsy_t
    ctx.video_processor.critical_threshold = critical_t
    ctx.video_processor.ear_baseline_ratio = ear_ratio
    ctx.video_processor.ear_consec_seconds = ear_seconds
    ctx.video_processor.mar_threshold = mar_t
    ctx.video_processor.phone_detection_enabled = phone_enabled
    ctx.video_processor.ground_truth_drowsy = ground_truth

    if ctx.video_processor.baseline_ear:
        ctx.video_processor.ear_threshold = ctx.video_processor.baseline_ear * ear_ratio

    if load_clicked and driver_name:
        profiles = read_profiles()
        if driver_name in profiles:
            p = profiles[driver_name]
            ctx.video_processor.baseline_ear = p["baseline_ear"]
            ctx.video_processor.baseline_pitch = p["baseline_pitch"]
            ctx.video_processor.ear_threshold = p["ear_threshold"]
            ctx.video_processor.pitch_down_threshold = p["pitch_down_threshold"]
            ctx.video_processor.calibrating = False
            st.sidebar.success(f"Profile '{driver_name}' loaded")
        else:
            st.sidebar.warning(f"No profile found for '{driver_name}'")

    if save_clicked and driver_name:
        profiles = read_profiles()
        profiles[driver_name] = {
            "baseline_ear": ctx.video_processor.baseline_ear,
            "baseline_pitch": ctx.video_processor.baseline_pitch,
            "ear_threshold": ctx.video_processor.ear_threshold,
            "pitch_down_threshold": ctx.video_processor.pitch_down_threshold,
        }
        with open(PROFILES_PATH, "w") as f:
            json.dump(profiles, f, indent=2)
        st.sidebar.success(f"Profile '{driver_name}' saved")

status_placeholder = st.empty()
metrics_placeholder = st.empty()
alert_audio_placeholder = st.empty()
download_placeholder = st.empty()

last_alert_severity = None

while ctx.state.playing:
    if ctx.video_processor:
        with ctx.video_processor.lock:
            m = dict(ctx.video_processor.metrics)

        if m["calibrating"]:
            status_placeholder.info(f"Calibrating... {m['calib_remaining']}s left. Keep eyes open, look at the camera.")
        else:
            color = SEVERITY_COLORS.get(m["severity"], "white")
            status_placeholder.markdown(
                f"<h2 style='color:{color}'>Status: {m['severity']}</h2>", unsafe_allow_html=True
            )

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("EAR", f"{m['ear']:.3f}")
            c2.metric("MAR", f"{m['mar']:.3f}")
            c3.metric("PERCLOS", f"{m['perclos']:.1f}%")
            c4.metric("Fusion Score", f"{m['fusion']:.2f}")
            c5.metric("Blinks", m["blink_count"])
            c6.metric("Yawns", m["yawn_count"])

            if m["phone_detected"]:
                st.warning("Phone detected!")
            if m["fatigue_warning"]:
                st.warning(m["fatigue_warning"])

            if m["new_alert_event"] and m["new_alert_event"] != last_alert_severity:
                last_alert_severity = m["new_alert_event"]
                phrase = ("You seem drowsy. Please stay alert."
                          if m["new_alert_event"] == "DROWSY"
                          else "Critical! Pull over now if you need to rest!")
                alert_audio_placeholder.empty()
                with alert_audio_placeholder:
                    components.html(f"""
                        <script>
                        var msg = new SpeechSynthesisUtterance("{phrase}");
                        window.speechSynthesis.speak(msg);
                        var ctx = new (window.AudioContext || window.webkitAudioContext)();
                        var osc = ctx.createOscillator();
                        osc.frequency.value = 880;
                        osc.connect(ctx.destination);
                        osc.start();
                        setTimeout(function() {{ osc.stop(); }}, 400);
                        </script>
                    """, height=0)

        if ctx.video_processor.log_rows:
            csv_lines = ["timestamp,elapsed_sec,ear,mar,perclos,pitch,fusion_score,severity,blink_count,phone_detected,ground_truth"]
            for row in ctx.video_processor.log_rows[-2000:]:
                csv_lines.append(",".join(str(x) for x in row))
            csv_data = "\n".join(csv_lines)
            download_placeholder.download_button("Download session CSV", csv_data, file_name="session_log.csv", mime="text/csv")

    time.sleep(0.3)
else:
    st.info("Click START above and allow camera access to begin.")
