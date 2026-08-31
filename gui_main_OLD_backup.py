import cv2
import numpy as np
import time
import csv
import os
import threading
from notifier import send_drowsy_alert
from collections import deque
from mediapipe.python.solutions import face_mesh as mp_face_mesh_module
from ultralytics import YOLO
from alert import play_alert
from config_loader import load_config
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from voice_assistant import speak, listen_once
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

face_mesh = mp_face_mesh_module.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)

phone_model = YOLO("yolov8n.pt")
CELL_PHONE_CLASS_ID = 67
PHONE_DETECTION_INTERVAL = 10  # run phone detection every N frames

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
EAR_CONSEC_FRAMES = CFG["EAR_CONSEC_FRAMES"]
PITCH_CONSEC_FRAMES = CFG["PITCH_CONSEC_FRAMES"]
WINDOW_SECONDS = CFG["WINDOW_SECONDS"]
ALERT_COOLDOWN = CFG["ALERT_COOLDOWN"]
CALIBRATION_SECONDS = CFG["CALIBRATION_SECONDS"]
EAR_BASELINE_RATIO = CFG["EAR_BASELINE_RATIO"]
PITCH_BASELINE_OFFSET = CFG["PITCH_BASELINE_OFFSET"]

W_EAR = CFG["FUSION_WEIGHTS"]["EAR"]
W_PERCLOS = CFG["FUSION_WEIGHTS"]["PERCLOS"]
W_PITCH = CFG["FUSION_WEIGHTS"]["PITCH"]
W_YAWN = CFG["FUSION_WEIGHTS"]["YAWN"]
FUSION_THRESHOLD = CFG["FUSION_THRESHOLD"]


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
    """Improves visibility in dim lighting using CLAHE on the luminance channel."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def get_brightness(frame):
    """Returns average brightness (0-255) to decide if enhancement is needed."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean()

class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Drowsiness Detection System")
        self.root.configure(bg="#1e1e1e")

        # ---------- Video display ----------
        self.video_label = tk.Label(root, bg="black")
        self.video_label.grid(row=0, column=0, rowspan=10, padx=10, pady=10)

        # ---------- Stat labels ----------
        label_style = {"bg": "#1e1e1e", "fg": "white", "font": ("Segoe UI", 12), "anchor": "w"}
        self.status_label = tk.Label(root, text="Status: --", **{**label_style, "font": ("Segoe UI", 16, "bold")})
        self.ear_label = tk.Label(root, text="EAR: --", **label_style)
        self.mar_label = tk.Label(root, text="MAR: --", **label_style)
        self.perclos_label = tk.Label(root, text="PERCLOS: --", **label_style)
        self.fusion_label = tk.Label(root, text="Fusion Score: --", **label_style)
        self.blink_label = tk.Label(root, text="Blinks: --", **label_style)
        self.yawn_label = tk.Label(root, text="Yawns: --", **label_style)
        self.phone_label = tk.Label(root, text="Phone: --", **label_style)
        self.notification_sent_this_session = False

        for i, lbl in enumerate([self.status_label, self.ear_label, self.mar_label,
                                  self.perclos_label, self.fusion_label,
                                  self.blink_label, self.yawn_label, self.phone_label]):
            lbl.grid(row=i, column=1, sticky="w", padx=10, pady=4)

        # ---------- Buttons ----------
        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.grid(row=9, column=1, pady=20, sticky="w")

        self.start_btn = tk.Button(btn_frame, text="Start", width=10, bg="#2e7d32", fg="white",
                                    command=self.start)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = tk.Button(btn_frame, text="Stop", width=10, bg="#c62828", fg="white",
                                   command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=5)

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
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=10, column=1, padx=10, pady=10, sticky="w")

        self.graph_times = deque()
        self.graph_ear = deque()
        self.graph_fusion = deque()
        self.graph_start = 0

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

        self.frame_history = deque()
        self.ear_closed_frames = 0
        self.pitch_down_frames = 0
        self.blink_count = 0
        self.blink_closed_frames = 0
        self.last_alert_time = 0
        self.was_drowsy_last_frame = False
        self.was_yawning_last_frame = False
        self.yawn_count = 0
        self.drowsy_events = 0

        self.frame_counter = 0
        self.phone_detected = False

        os.makedirs("output/snapshots", exist_ok=True)
        self.log_file = None
        self.csv_writer = None

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="Status: Cannot open webcam")
            return

        # Warm up YOLO phone model with a dummy inference so first real call isn't slow
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        phone_model(dummy_frame, verbose=False)

        os.makedirs(os.path.join("output", "sessions"), exist_ok=True)
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.session_log_path = os.path.join("output", "sessions", f"session_{self.session_id}.csv")
        self.log_file = open(self.session_log_path, mode="w", newline="")
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["timestamp", "elapsed_sec", "ear", "mar", "perclos", "pitch",
                                   "fusion_score", "blink_count", "status", "phone_detected"])

        self.running = True
        self.calibrating = True
        self.calib_start_time = time.time()
        self.perclos_values = []
        self.blink_start_count = self.blink_count  # baseline in case app wasn't restarted
        self.ear_samples = []
        self.pitch_samples = []
        self.session_start = time.time()
        self.graph_start = self.session_start
        self.graph_times.clear()
        self.graph_ear.clear()
        self.graph_fusion.clear()
        self.frame_counter = 0
        self.phone_detected = False
        self.notification_sent_this_session = False
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
            return  # never actually started a session

        duration = time.time() - self.session_start if hasattr(self, "session_start") else 0
        avg_perclos = np.mean(self.perclos_values) if getattr(self, "perclos_values", []) else 0
        max_perclos = np.max(self.perclos_values) if getattr(self, "perclos_values", []) else 0
        blink_rate_per_min = (self.blink_count / (duration / 60)) if duration > 0 else 0

        history_path = os.path.join("output", "session_history.csv")
        file_exists = os.path.exists(history_path)

        with open(history_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "session_id", "date", "duration_sec", "avg_perclos", "max_perclos",
                    "total_blinks", "blink_rate_per_min", "total_yawns", "drowsy_events",
                    "baseline_ear", "ear_threshold"
                ])
            writer.writerow([
                self.session_id,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                round(duration, 1),
                round(avg_perclos, 2),
                round(max_perclos, 2),
                self.blink_count,
                round(blink_rate_per_min, 2),
                self.yawn_count,
                self.drowsy_events,
                round(self.baseline_ear, 4),
                round(self.EAR_THRESHOLD, 4)
            ])

        print(f"Session summary appended to: {history_path}")

    def start_voice_listener(self):
        def _listen_loop():
            import speech_recognition as sr_module
            recognizer = sr_module.Recognizer()
            with sr_module.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)

            while self.running:
                command = listen_once(timeout=3, phrase_time_limit=3)
                if command and "awake" in command:
                    self.drowsy_events = 0
                    self.notification_sent_this_session = False
                    speak("Okay, glad you're awake. Stay safe.")
                time.sleep(0.5)
        threading.Thread(target=_listen_loop, daemon=True).start()    

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return
        if get_brightness(frame) < 60:
            frame = enhance_low_light(frame)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        now = time.time()

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

            cv2.putText(frame, "CALIBRATING...", (int(w*0.2), int(h*0.45)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"Keep eyes open - {remaining}s left", (int(w*0.1), int(h*0.55)),
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

        # ---------- Phone detection (throttled) ----------
        self.frame_counter += 1
        if self.frame_counter % PHONE_DETECTION_INTERVAL == 0:
            phone_results = phone_model(frame, verbose=False, classes=[CELL_PHONE_CLASS_ID], conf=0.4, imgsz=320)
            self.phone_detected = False
            for r in phone_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, "Phone", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    self.phone_detected = True

        # ---------- Normal detection loop ----------
        status = "No Face"
        ear = mar = perclos = pitch = fusion_score = 0
        is_yawning = False

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

            self.ear_closed_frames = self.ear_closed_frames + 1 if is_closed else 0
            self.pitch_down_frames = self.pitch_down_frames + 1 if pitch < self.PITCH_DOWN_THRESHOLD else 0
            is_yawning = mar > MAR_THRESHOLD

            if is_yawning and not self.was_yawning_last_frame:
                self.yawn_count += 1
            self.was_yawning_last_frame = is_yawning

            ear_risk = min(self.ear_closed_frames / EAR_CONSEC_FRAMES, 1.0)
            perclos_risk = min(perclos / 100.0, 1.0)
            pitch_risk = min(self.pitch_down_frames / PITCH_CONSEC_FRAMES, 1.0)
            yawn_risk = 1.0 if is_yawning else 0.0
            fusion_score = (W_EAR * ear_risk) + (W_PERCLOS * perclos_risk) + (W_PITCH * pitch_risk) + (W_YAWN * yawn_risk)
            drowsy = fusion_score >= FUSION_THRESHOLD
            status = "DROWSY" if drowsy else "NORMAL"

            if drowsy:
                if not self.was_drowsy_last_frame:
                    self.drowsy_events += 1
                    snap_path = os.path.join("output", "snapshots", f"drowsy_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(snap_path, frame)

                    trigger_count = CFG["NOTIFICATIONS"].get("DROWSY_EVENT_TRIGGER_COUNT", 3)
                    send_once = CFG["NOTIFICATIONS"].get("SEND_ONCE_PER_SESSION", True)

                    should_notify = self.drowsy_events >= trigger_count and (
                        not send_once or not self.notification_sent_this_session
                    )

                    if should_notify:
                        threading.Thread(
                            target=send_drowsy_alert,
                            args=(CFG, self.drowsy_events, now - self.session_start, snap_path),
                            daemon=True
                        ).start()
                        self.notification_sent_this_session = True

                if (now - self.last_alert_time) > ALERT_COOLDOWN:
                    play_alert()
                    self.last_alert_time = now

                    if self.drowsy_events == 1:
                        speak("You seem drowsy. Please stay alert.")
                    elif self.drowsy_events >= 2:
                        speak("Warning! Pull over if you need to rest!")
            self.was_drowsy_last_frame = drowsy

            if drowsy:
                cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 255), -1)
                cv2.putText(frame, "WARNING: DRIVER IS DROWSY!", (int(w*0.15), h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.phone_detected:
            cv2.rectangle(frame, (0, 0), (w, 35), (0, 140, 255), -1)
            cv2.putText(frame, "WARNING: Phone Detected!", (int(w*0.2), 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
            if self.frame_counter % 3 == 0:
                self.canvas.draw_idle()

        self.csv_writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), round(now - self.session_start, 2),
            round(ear, 4), round(mar, 4), round(perclos, 2), round(pitch, 2),
            round(fusion_score, 3), self.blink_count, status, self.phone_detected
        ])

        color = "#e53935" if status == "DROWSY" else ("#43a047" if status == "NORMAL" else "#fdd835")
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