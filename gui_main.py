import cv2
import numpy as np
import time
import csv
import os
import json
import threading
from collections import deque
from mediapipe.python.solutions import face_mesh as mp_face_mesh_module
from ultralytics import YOLO
from alert import play_alert
from config_loader import load_config
from voice_assistant import speak, listen_once
from notifier import send_drowsy_alert
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

face_mesh = mp_face_mesh_module.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,   # off on purpose: faster, and EAR/MAR landmarks don't need iris refinement
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

phone_model = YOLO("yolo26n.pt")
CELL_PHONE_CLASS_ID = 67
PHONE_DETECTION_INTERVAL = 10

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

CFG = load_config()

MAR_THRESHOLD = CFG["MAR_THRESHOLD"]
EAR_CONSEC_SECONDS = CFG["EAR_CONSEC_SECONDS"]
PITCH_CONSEC_SECONDS = CFG["PITCH_CONSEC_SECONDS"]
WINDOW_SECONDS = CFG["WINDOW_SECONDS"]
ALERT_COOLDOWN = CFG["ALERT_COOLDOWN"]
CALIBRATION_SECONDS = CFG["CALIBRATION_SECONDS"]
EAR_BASELINE_RATIO = CFG["EAR_BASELINE_RATIO"]
PITCH_BASELINE_OFFSET = CFG["PITCH_BASELINE_OFFSET"]

W_EAR = CFG["FUSION_WEIGHTS"]["EAR"]
W_PERCLOS = CFG["FUSION_WEIGHTS"]["PERCLOS"]
W_PITCH = CFG["FUSION_WEIGHTS"]["PITCH"]
W_YAWN = CFG["FUSION_WEIGHTS"]["YAWN"]

# ---------- Severity thresholds ----------
MILD_THRESHOLD = 0.3
DROWSY_THRESHOLD = 0.5
CRITICAL_THRESHOLD = 0.75

# ---------- Yawn escalation ----------
YAWN_WINDOW_SECONDS = 60
YAWN_ESCALATION_COUNT = 2
YAWN_WARNING_COOLDOWN = 45

PROFILES_PATH = "profiles.json"


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


def get_severity(fusion_score):
    if fusion_score >= CRITICAL_THRESHOLD:
        return "CRITICAL"
    elif fusion_score >= DROWSY_THRESHOLD:
        return "DROWSY"
    elif fusion_score >= MILD_THRESHOLD:
        return "MILD FATIGUE"
    else:
        return "NORMAL"


SEVERITY_COLORS = {
    "NORMAL": "#43a047",
    "MILD FATIGUE": "#fdd835",
    "DROWSY": "#fb8c00",
    "CRITICAL": "#e53935"
}


class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection System")
        self.root.configure(bg="#1e1e1e")

        self.video_label = tk.Label(root, bg="black")
        self.video_label.grid(row=0, column=0, rowspan=12, padx=10, pady=10)

        label_style = {"bg": "#1e1e1e", "fg": "white", "font": ("Segoe UI", 12), "anchor": "w"}

        # ---------- Driver profile controls ----------
        profile_frame = tk.Frame(root, bg="#1e1e1e")
        profile_frame.grid(row=0, column=1, sticky="w", padx=10, pady=(10, 4))
        tk.Label(profile_frame, text="Driver Name:", bg="#1e1e1e", fg="white", font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w")
        self.name_entry = tk.Entry(profile_frame, width=18)
        self.name_entry.grid(row=0, column=1, padx=5)
        tk.Button(profile_frame, text="Load Profile", command=self.load_profile, bg="#37474f", fg="white").grid(row=0, column=2, padx=3)
        tk.Button(profile_frame, text="Save Profile", command=self.save_profile, bg="#37474f", fg="white").grid(row=0, column=3, padx=3)

        self.status_label = tk.Label(root, text="Status: --", **{**label_style, "font": ("Segoe UI", 16, "bold")})
        self.ear_label = tk.Label(root, text="EAR: --", **label_style)
        self.mar_label = tk.Label(root, text="MAR: --", **label_style)
        self.perclos_label = tk.Label(root, text="PERCLOS: --", **label_style)
        self.fusion_label = tk.Label(root, text="Fusion Score: --", **label_style)
        self.blink_label = tk.Label(root, text="Blinks: --", **label_style)
        self.yawn_label = tk.Label(root, text="Yawns: --", **label_style)
        self.phone_label = tk.Label(root, text="Phone: --", **label_style)
        self.fatigue_warning_label = tk.Label(root, text="", **{**label_style, "fg": "#fdd835"})

        for i, lbl in enumerate([self.status_label, self.ear_label, self.mar_label,
                                  self.perclos_label, self.fusion_label,
                                  self.blink_label, self.yawn_label, self.phone_label,
                                  self.fatigue_warning_label]):
            lbl.grid(row=i + 1, column=1, sticky="w", padx=10, pady=4)

        # ---------- Buttons row ----------
        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.grid(row=10, column=1, pady=(20, 4), sticky="w")

        self.start_btn = tk.Button(btn_frame, text="Start", width=10, bg="#2e7d32", fg="white", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="Stop", width=10, bg="#c62828", fg="white", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.truth_btn = tk.Button(btn_frame, text="Mark: Awake", width=14, bg="#455a64", fg="white",
                                    command=self.toggle_ground_truth)
        self.truth_btn.grid(row=0, column=2, padx=5)

        self.settings_btn = tk.Button(btn_frame, text="Settings", width=10, bg="#37474f", fg="white",
                                       command=self.open_settings)
        self.settings_btn.grid(row=0, column=3, padx=5)

        # ---------- Feature toggles (default OFF for performance) ----------
        self.phone_detect_var = tk.BooleanVar(value=False)
        self.voice_var = tk.BooleanVar(value=False)

        toggle_frame = tk.Frame(root, bg="#1e1e1e")
        toggle_frame.grid(row=11, column=1, sticky="w", padx=10, pady=(4, 10))

        tk.Checkbutton(toggle_frame, text="Phone Detection", variable=self.phone_detect_var,
                       bg="#1e1e1e", fg="white", selectcolor="#37474f").grid(row=0, column=0, sticky="w", padx=(0, 15))
        tk.Checkbutton(toggle_frame, text="Voice Assistant", variable=self.voice_var,
                       bg="#1e1e1e", fg="white", selectcolor="#37474f").grid(row=0, column=1, sticky="w")

        # ---------- Live graph ----------
        self.fig = Figure(figsize=(4.2, 2.6), dpi=90, facecolor="#1e1e1e")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#2a2a2a")
        self.ax.tick_params(colors="white", labelsize=7)
        self.ax.set_title("EAR & Fusion Score (last 30s)", color="white", fontsize=9)
        self.ax.set_ylim(0, 1)
        self.line_ear, = self.ax.plot([], [], color="#42a5f5", label="EAR", linewidth=1.2)
        self.line_fusion, = self.ax.plot([], [], color="#ef5350", label="Fusion", linewidth=1.2)
        self.ax.legend(loc="upper right", fontsize=7, facecolor="#2a2a2a", labelcolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=12, column=1, padx=10, pady=10, sticky="w")
        self.canvas.draw()

        self.graph_times = deque()
        self.graph_ear = deque()
        self.graph_fusion = deque()
        self.graph_start = 0
        self.ground_truth_drowsy = False

        # ---------- Internal state ----------
        self.cap = None
        self.running = False
        self.calibrating = False
        self.calib_start_time = 0
        self.ear_samples = []
        self.pitch_samples = []
        self.baseline_ear = 0.28
        self.baseline_pitch = 0.0
        self.EAR_THRESHOLD = 0.21
        self.PITCH_DOWN_THRESHOLD = -15
        self.profile_loaded = False

        self.frame_history = deque()
        self.ear_closed_since = None
        self.pitch_down_since = None
        self.blink_count = 0
        self.blink_closed_frames = 0
        self.last_alert_time = 0
        self.was_yawning_last_frame = False
        self.yawn_count = 0
        self.drowsy_events = 0
        self.perclos_values = []

        self.last_severity = "NORMAL"
        self.yawn_timestamps = deque()
        self.last_yawn_warning_time = 0

        self.frame_counter = 0
        self.phone_detected = False

        self.notification_sent_this_session = False

        self._fps_counter = 0
        self._fps_last_time = time.time()

        os.makedirs("output/snapshots", exist_ok=True)
        os.makedirs("output/sessions", exist_ok=True)
        self.log_file = None
        self.csv_writer = None

    # ---------- Driver profiles ----------
    def _read_profiles(self):
        if os.path.exists(PROFILES_PATH):
            try:
                with open(PROFILES_PATH, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def load_profile(self):
        name = self.name_entry.get().strip()
        if not name:
            self.status_label.config(text="Status: Enter a name first")
            return
        profiles = self._read_profiles()
        if name in profiles:
            p = profiles[name]
            self.baseline_ear = p["baseline_ear"]
            self.baseline_pitch = p["baseline_pitch"]
            self.EAR_THRESHOLD = p["ear_threshold"]
            self.PITCH_DOWN_THRESHOLD = p["pitch_down_threshold"]
            self.profile_loaded = True
            self.status_label.config(text=f"Status: Profile '{name}' loaded")
        else:
            self.status_label.config(text=f"Status: No profile found for '{name}'")

    def save_profile(self):
        name = self.name_entry.get().strip()
        if not name:
            self.status_label.config(text="Status: Enter a name first")
            return
        profiles = self._read_profiles()
        profiles[name] = {
            "baseline_ear": self.baseline_ear,
            "baseline_pitch": self.baseline_pitch,
            "ear_threshold": self.EAR_THRESHOLD,
            "pitch_down_threshold": self.PITCH_DOWN_THRESHOLD
        }
        with open(PROFILES_PATH, "w") as f:
            json.dump(profiles, f, indent=2)
        self.status_label.config(text=f"Status: Profile '{name}' saved")

    def toggle_ground_truth(self):
        self.ground_truth_drowsy = not self.ground_truth_drowsy
        label = "Drowsy" if self.ground_truth_drowsy else "Awake"
        self.truth_btn.config(text=f"Mark: {label}",
                               bg="#c62828" if self.ground_truth_drowsy else "#455a64")

    # ---------- Voice assistant ----------
    def start_voice_listener(self):
        def _listen_loop():
            import speech_recognition as sr_module
            recognizer = sr_module.Recognizer()
            try:
                with sr_module.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
            except OSError:
                return
            while self.running:
                command = listen_once(timeout=3, phrase_time_limit=3)
                if command and "awake" in command:
                    self.drowsy_events = 0
                    self.notification_sent_this_session = False
                    speak("Okay, glad you're awake. Stay safe.")
                time.sleep(0.5)
        threading.Thread(target=_listen_loop, daemon=True).start()

    # ---------- Settings panel ----------
    def open_settings(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Settings")
        settings_win.configure(bg="#1e1e1e")
        settings_win.geometry("360x420")

        def make_slider(parent, label, from_, to_, resolution, current_value, row):
            tk.Label(parent, text=label, bg="#1e1e1e", fg="white", font=("Segoe UI", 10)).grid(
                row=row, column=0, sticky="w", padx=10, pady=(10, 0))
            var = tk.DoubleVar(value=current_value)
            slider = tk.Scale(parent, from_=from_, to=to_, resolution=resolution, orient="horizontal",
                               variable=var, bg="#1e1e1e", fg="white", troughcolor="#37474f",
                               highlightthickness=0, length=300)
            slider.grid(row=row + 1, column=0, padx=10)
            return var

        global MILD_THRESHOLD, DROWSY_THRESHOLD, CRITICAL_THRESHOLD
        global EAR_BASELINE_RATIO, EAR_CONSEC_SECONDS, MAR_THRESHOLD

        mild_var = make_slider(settings_win, "Mild Fatigue Threshold", 0.1, 0.5, 0.01, MILD_THRESHOLD, 0)
        drowsy_var = make_slider(settings_win, "Drowsy Threshold", 0.3, 0.8, 0.01, DROWSY_THRESHOLD, 2)
        critical_var = make_slider(settings_win, "Critical Threshold", 0.5, 0.95, 0.01, CRITICAL_THRESHOLD, 4)
        ear_ratio_var = make_slider(settings_win, "EAR Baseline Ratio", 0.5, 0.9, 0.01, EAR_BASELINE_RATIO, 6)
        ear_seconds_var = make_slider(settings_win, "EAR Closed Duration (sec)", 0.2, 2.0, 0.1, EAR_CONSEC_SECONDS, 8)
        mar_var = make_slider(settings_win, "Yawn (MAR) Threshold", 0.3, 0.9, 0.01, MAR_THRESHOLD, 10)

        def apply_settings():
            global MILD_THRESHOLD, DROWSY_THRESHOLD, CRITICAL_THRESHOLD
            global EAR_BASELINE_RATIO, EAR_CONSEC_SECONDS, MAR_THRESHOLD

            MILD_THRESHOLD = mild_var.get()
            DROWSY_THRESHOLD = drowsy_var.get()
            CRITICAL_THRESHOLD = critical_var.get()
            EAR_BASELINE_RATIO = ear_ratio_var.get()
            EAR_CONSEC_SECONDS = ear_seconds_var.get()
            MAR_THRESHOLD = mar_var.get()

            if self.baseline_ear:
                self.EAR_THRESHOLD = self.baseline_ear * EAR_BASELINE_RATIO

            settings_win.destroy()

        tk.Button(settings_win, text="Apply", bg="#2e7d32", fg="white", width=15,
                  command=apply_settings).grid(row=12, column=0, pady=20)

    # ---------- Start / Stop ----------
    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="Status: Cannot open webcam")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Only warm up the phone model if it will actually be used this session
        if self.phone_detect_var.get():
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            phone_model(dummy_frame, verbose=False)

        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        session_log_path = os.path.join("output", "sessions", f"session_{self.session_id}.csv")
        self.log_file = open(session_log_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["timestamp", "elapsed_sec", "ear", "mar", "perclos", "pitch",
                                   "fusion_score", "severity", "blink_count", "phone_detected", "ground_truth"])

        self.running = True
        self.calibrating = not self.profile_loaded
        self.calib_start_time = time.time()
        self.ear_samples = []
        self.pitch_samples = []
        self.session_start = time.time()
        self.graph_start = self.session_start
        self.graph_times.clear()
        self.graph_ear.clear()
        self.graph_fusion.clear()
        self.frame_counter = 0
        self.phone_detected = False
        self.perclos_values = []
        self.yawn_timestamps.clear()
        self.last_severity = "NORMAL"
        self.notification_sent_this_session = False
        self.ear_closed_since = None
        self.pitch_down_since = None
        self._fps_counter = 0
        self._fps_last_time = time.time()

        if self.voice_var.get():
            self.start_voice_listener()

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.update_frame()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.log_file:
            self.log_file.close()
        self.save_session_summary()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Status: Stopped")

    def save_session_summary(self):
        if not hasattr(self, "session_id"):
            return
        duration = time.time() - self.session_start if hasattr(self, "session_start") else 0
        avg_perclos = np.mean(self.perclos_values) if self.perclos_values else 0
        max_perclos = np.max(self.perclos_values) if self.perclos_values else 0
        blink_rate = (self.blink_count / (duration / 60)) if duration > 0 else 0

        history_path = os.path.join("output", "session_history.csv")
        file_exists = os.path.exists(history_path)
        with open(history_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["session_id", "date", "duration_sec", "avg_perclos", "max_perclos",
                                  "total_blinks", "blink_rate_per_min", "total_yawns", "drowsy_events",
                                  "baseline_ear", "ear_threshold"])
            writer.writerow([self.session_id, time.strftime("%Y-%m-%d %H:%M:%S"), round(duration, 1),
                              round(avg_perclos, 2), round(max_perclos, 2), self.blink_count,
                              round(blink_rate, 2), self.yawn_count, self.drowsy_events,
                              round(self.baseline_ear, 4), round(self.EAR_THRESHOLD, 4)])
        print(f"Session summary appended to: {history_path}")

    # ---------- Main frame loop ----------
    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()

        # FPS counter (cheap, always on for diagnostics)
        self._fps_counter += 1
        if time.time() - self._fps_last_time >= 1.0:
            print(f"[fps] {self._fps_counter} frames/sec")
            self._fps_counter = 0
            self._fps_last_time = time.time()

        if not ret:
            self.root.after(10, self.update_frame)
            return

        if get_brightness(frame) < 80:
            frame = enhance_low_light(frame)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        now = time.time()

        # ---------- Calibration phase ----------
        if self.calibrating:
            elapsed = now - self.calib_start_time
            remaining = int(CALIBRATION_SECONDS - elapsed)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                le = calculate_ear(landmarks, LEFT_EYE, w, h)
                re = calculate_ear(landmarks, RIGHT_EYE, w, h)
                self.ear_samples.append((le + re) / 2.0)
                pitch, _, _ = get_head_pose(landmarks, w, h)
                self.pitch_samples.append(pitch)

            cv2.putText(frame, "CALIBRATING...", (int(w * 0.2), int(h * 0.45)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"Keep eyes open - {remaining}s left", (int(w * 0.1), int(h * 0.55)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.status_label.config(text=f"Status: Calibrating ({remaining}s)")

            if elapsed >= CALIBRATION_SECONDS:
                self.baseline_ear = np.mean(self.ear_samples) if self.ear_samples else 0.28
                self.baseline_pitch = np.mean(self.pitch_samples) if self.pitch_samples else 0.0
                self.EAR_THRESHOLD = self.baseline_ear * EAR_BASELINE_RATIO
                self.PITCH_DOWN_THRESHOLD = self.baseline_pitch - PITCH_BASELINE_OFFSET
                self.calibrating = False

            self.display_frame(frame)
            self.root.after(10, self.update_frame)
            return

        # ---------- Phone detection (opt-in, throttled) ----------
        self.frame_counter += 1
        if self.phone_detect_var.get() and self.frame_counter % PHONE_DETECTION_INTERVAL == 0:
            phone_results = phone_model(frame, verbose=False, classes=[CELL_PHONE_CLASS_ID], conf=0.4, imgsz=320)
            self.phone_detected = False
            for r in phone_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    self.phone_detected = True
        elif not self.phone_detect_var.get():
            self.phone_detected = False

        # ---------- Core detection ----------
        status = "No Face"
        ear = mar = perclos = pitch = fusion_score = 0
        is_yawning = False
        severity = "NORMAL"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            le = calculate_ear(landmarks, LEFT_EYE, w, h)
            re = calculate_ear(landmarks, RIGHT_EYE, w, h)
            ear = (le + re) / 2.0
            mar = calculate_mar(landmarks, w, h)
            pitch, _, _ = get_head_pose(landmarks, w, h)

            is_closed = ear < self.EAR_THRESHOLD
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
            self.perclos_values.append(perclos)

            if is_closed:
                if self.ear_closed_since is None:
                    self.ear_closed_since = now
            else:
                self.ear_closed_since = None
            ear_closed_duration = (now - self.ear_closed_since) if self.ear_closed_since else 0.0

            if pitch < self.PITCH_DOWN_THRESHOLD:
                if self.pitch_down_since is None:
                    self.pitch_down_since = now
            else:
                self.pitch_down_since = None
            pitch_down_duration = (now - self.pitch_down_since) if self.pitch_down_since else 0.0

            is_yawning = mar > MAR_THRESHOLD
            if is_yawning and not self.was_yawning_last_frame:
                self.yawn_count += 1
                self.yawn_timestamps.append(now)
            self.was_yawning_last_frame = is_yawning

            while self.yawn_timestamps and now - self.yawn_timestamps[0] > YAWN_WINDOW_SECONDS:
                self.yawn_timestamps.popleft()

            if (len(self.yawn_timestamps) >= YAWN_ESCALATION_COUNT and
                    (now - self.last_yawn_warning_time) > YAWN_WARNING_COOLDOWN):
                self.fatigue_warning_label.config(text="\u26a0 Fatigue building: multiple yawns detected")
                speak("You've yawned a few times. Fatigue may be building.")
                self.last_yawn_warning_time = now
            elif (now - self.last_yawn_warning_time) > YAWN_WARNING_COOLDOWN:
                self.fatigue_warning_label.config(text="")

            ear_risk = min(ear_closed_duration / EAR_CONSEC_SECONDS, 1.0)
            perclos_risk = min(perclos / 100.0, 1.0)
            pitch_risk = min(pitch_down_duration / PITCH_CONSEC_SECONDS, 1.0)
            yawn_risk = 1.0 if is_yawning else 0.0
            fusion_score = (W_EAR * ear_risk) + (W_PERCLOS * perclos_risk) + (W_PITCH * pitch_risk) + (W_YAWN * yawn_risk)

            severity = get_severity(fusion_score)
            status = severity

            if self.frame_counter % 15 == 0:
                print(f"[debug] EAR={ear:.3f} thresh={self.EAR_THRESHOLD:.3f} | "
                      f"ear_closed_duration={ear_closed_duration:.2f}s/{EAR_CONSEC_SECONDS}s | "
                      f"perclos={perclos:.1f}% | pitch_down_duration={pitch_down_duration:.2f}s/{PITCH_CONSEC_SECONDS}s | "
                      f"fusion={fusion_score:.3f} | severity={severity}")

            if severity in ("DROWSY", "CRITICAL"):
                if self.last_severity not in ("DROWSY", "CRITICAL"):
                    self.drowsy_events += 1
                    snap_path = os.path.join("output", "snapshots", f"{severity.lower()}_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(snap_path, frame)

                    trigger_count = CFG["NOTIFICATIONS"].get("DROWSY_EVENT_TRIGGER_COUNT", 3)
                    send_once = CFG["NOTIFICATIONS"].get("SEND_ONCE_PER_SESSION", True)
                    should_notify = self.drowsy_events >= trigger_count and (not send_once or not self.notification_sent_this_session)
                    if should_notify:
                        threading.Thread(target=send_drowsy_alert,
                                          args=(CFG, self.drowsy_events, now - self.session_start, snap_path),
                                          daemon=True).start()
                        self.notification_sent_this_session = True

                if (now - self.last_alert_time) > ALERT_COOLDOWN:
                    play_alert()
                    self.last_alert_time = now
                    if severity == "DROWSY":
                        speak("You seem drowsy. Please stay alert.")
                    elif severity == "CRITICAL":
                        speak("Critical! Pull over now if you need to rest!")

                bar_color = (0, 0, 255) if severity == "CRITICAL" else (0, 100, 255)
                cv2.rectangle(frame, (0, h - 40), (w, h), bar_color, -1)
                cv2.putText(frame, f"WARNING: {severity}!", (int(w * 0.15), h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            elif severity == "MILD FATIGUE" and self.last_severity == "NORMAL":
                speak("You seem a little tired.")

            self.last_severity = severity

        if self.phone_detected:
            cv2.rectangle(frame, (0, 0), (w, 35), (0, 140, 255), -1)
            cv2.putText(frame, "WARNING: Phone Detected!", (int(w * 0.2), 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ---------- Graph update (throttled redraw) ----------
        if results.multi_face_landmarks:
            self.graph_times.append(now - self.graph_start)
            self.graph_ear.append(ear)
            self.graph_fusion.append(fusion_score)
            while self.graph_times and self.graph_times[0] < (now - self.graph_start) - 30:
                self.graph_times.popleft()
                self.graph_ear.popleft()
                self.graph_fusion.popleft()

            self.line_ear.set_data(list(self.graph_times), list(self.graph_ear))
            self.line_fusion.set_data(list(self.graph_times), list(self.graph_fusion))
            if self.graph_times:
                self.ax.set_xlim(max(0, self.graph_times[0]), max(30, self.graph_times[-1]))

            if self.frame_counter % 5 == 0:
                self.canvas.draw_idle()

        # ---------- Logging ----------
        self.csv_writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), round(now - self.session_start, 2),
            round(ear, 4), round(mar, 4), round(perclos, 2), round(pitch, 2),
            round(fusion_score, 3), severity, self.blink_count, self.phone_detected,
            int(self.ground_truth_drowsy)
        ])

        # ---------- Dashboard update ----------
        color = SEVERITY_COLORS.get(status, "white")
        self.status_label.config(text=f"Status: {status}", fg=color)
        self.ear_label.config(text=f"EAR: {ear:.3f}")
        self.mar_label.config(text=f"MAR: {mar:.3f}" + (" (Yawn)" if is_yawning else ""))
        self.perclos_label.config(text=f"PERCLOS: {perclos:.1f}%")
        self.fusion_label.config(text=f"Fusion Score: {fusion_score:.2f}")
        self.blink_label.config(text=f"Blinks: {self.blink_count}")
        self.yawn_label.config(text=f"Yawns: {self.yawn_count}")
        self.phone_label.config(text=f"Phone: {'Detected' if self.phone_detected else 'Not Detected'}",
                                 fg="#ff9800" if self.phone_detected else "white")

        self.display_frame(frame)
        self.root.after(10, self.update_frame)

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)


if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessApp(root)
    root.mainloop()
