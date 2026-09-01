"""
face_id.py
Driver face recognition module for DriveGuard AI.

Uses OpenCV's LBPH face recognizer (opencv-contrib-python) instead of
dlib/face_recognition to avoid Windows build-tool issues.

Design goals:
- Fully decoupled from the MediaPipe drowsiness pipeline in web_app.py.
- Cheap enough to run periodically (not every frame).
- Simple on-disk storage (JSON + a single .yml model file), no database.

Install:
    pip install opencv-contrib-python

Expected directory layout (created automatically if missing):
    known_faces/<driver_name>/*.jpg      -> enrollment samples
    models/face_recognizer.yml           -> trained LBPH model
    models/labels.json                   -> {"0": "Alice", "1": "Bob", ...}
    models/driver_calibration.json       -> {"Alice": {"ear_threshold": 0.21, ...}, ...}
"""

import os
import json
import time
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "face_recognizer.yml")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
CALIBRATION_PATH = os.path.join(MODELS_DIR, "driver_calibration.json")

FACE_SIZE = (200, 200)          # all training/inference crops resized to this
CONFIDENCE_THRESHOLD = 70.0     # LBPH: LOWER distance = better match. Above this -> "Unknown"
RECOGNIZE_EVERY_N_FRAMES = 15   # only run recognition this often, protect FPS

_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_HAAR_PATH)

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_labels(labels):
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)


def _load_calibration():
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH, "r") as f:
            return json.load(f)
    return {}


def save_driver_calibration(driver_name, calibration_dict):
    """
    Call this from your existing recalibration flow to persist a driver's
    thresholds (EAR/MAR/PERCLOS/pitch) so they auto-load next time they're
    recognized.
    """
    cal = _load_calibration()
    cal[driver_name] = calibration_dict
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(cal, f, indent=2)


def get_driver_calibration(driver_name):
    """Returns saved thresholds for a driver, or None if not yet calibrated."""
    return _load_calibration().get(driver_name)


def get_known_drivers():
    """Returns a list of enrolled driver names."""
    labels = _load_labels()
    return list(set(labels.values()))


def _detect_face_box(frame_bgr):
    """
    Returns (x, y, w, h) of the largest detected face, or None.
    Uses Haar cascade — swap this out to reuse a MediaPipe-derived box
    instead if you'd rather not run a second detector.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    # pick the largest face (closest to camera / driver seat)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces[0]


def _crop_and_prep(frame_bgr, box):
    x, y, w, h = box
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, FACE_SIZE)
    face = cv2.equalizeHist(face)  # normalize lighting variation
    return face


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------
def enroll_driver(driver_name, camera_index=0, num_samples=20, delay_sec=0.15):
    """
    Captures `num_samples` face crops of `driver_name` from the webcam,
    saves them, and retrains the LBPH model on the full known_faces set.

    Run this as a one-off enrollment flow (e.g. triggered from a dashboard
    "Enroll new driver" button), not inside the main detection loop.
    """
    driver_dir = os.path.join(KNOWN_FACES_DIR, driver_name)
    os.makedirs(driver_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera for enrollment.")

    captured = 0
    existing = len(os.listdir(driver_dir))
    try:
        while captured < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            box = _detect_face_box(frame)
            if box is not None:
                face = _crop_and_prep(frame, box)
                fname = os.path.join(driver_dir, f"{existing + captured:03d}.jpg")
                cv2.imwrite(fname, face)
                captured += 1
            time.sleep(delay_sec)
    finally:
        cap.release()

    train_model()
    return captured


def train_model():
    """
    Retrains the LBPH recognizer on everything currently in known_faces/
    and overwrites the saved model + labels file.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}  # name -> int id
    next_id = 0

    for driver_name in sorted(os.listdir(KNOWN_FACES_DIR)):
        driver_dir = os.path.join(KNOWN_FACES_DIR, driver_name)
        if not os.path.isdir(driver_dir):
            continue
        if driver_name not in label_map:
            label_map[driver_name] = next_id
            next_id += 1
        label_id = label_map[driver_name]

        for fname in os.listdir(driver_dir):
            fpath = os.path.join(driver_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, FACE_SIZE)
            faces.append(img)
            labels.append(label_id)

    if not faces:
        raise RuntimeError("No enrolled faces found — run enroll_driver() first.")

    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL_PATH)

    # label file stores id -> name (inverse of label_map) as strings for JSON
    id_to_name = {str(v): k for k, v in label_map.items()}
    _save_labels(id_to_name)

    return id_to_name


# ---------------------------------------------------------------------------
# Recognition (runtime)
# ---------------------------------------------------------------------------
_recognizer = None
_labels_cache = None
_frame_counter = 0
_last_result = ("Unknown", 0.0)


def _ensure_model_loaded():
    global _recognizer, _labels_cache
    if _recognizer is None:
        if not os.path.exists(MODEL_PATH):
            _recognizer = False  # sentinel: no model trained yet
            return
        _recognizer = cv2.face.LBPHFaceRecognizer_create()
        _recognizer.read(MODEL_PATH)
        _labels_cache = _load_labels()


def recognize_driver(frame_bgr, force=False):
    """
    Returns (driver_name: str, confidence: float, recognized: bool).

    - driver_name is "Unknown" if no match / no model / no face found.
    - confidence is the raw LBPH distance (lower = better match); 0.0 if N/A.
    - Only runs actual inference every RECOGNIZE_EVERY_N_FRAMES calls
      (or when force=True) to protect FPS; otherwise returns the cached
      last result so callers can call this every frame safely.
    """
    global _frame_counter, _last_result

    _frame_counter += 1
    if not force and (_frame_counter % RECOGNIZE_EVERY_N_FRAMES != 0):
        name, conf = _last_result
        return name, conf, name != "Unknown"

    _ensure_model_loaded()
    if _recognizer is False:
        _last_result = ("Unknown", 0.0)
        return "Unknown", 0.0, False

    box = _detect_face_box(frame_bgr)
    if box is None:
        _last_result = ("Unknown", 0.0)
        return "Unknown", 0.0, False

    face = _crop_and_prep(frame_bgr, box)
    label_id, confidence = _recognizer.predict(face)  # lower confidence = better match

    if confidence <= CONFIDENCE_THRESHOLD:
        name = _labels_cache.get(str(label_id), "Unknown")
    else:
        name = "Unknown"

    _last_result = (name, float(confidence))
    return name, float(confidence), name != "Unknown"


def reset_recognition_state():
    """Call at the start of a new session so a previous driver isn't carried over."""
    global _frame_counter, _last_result
    _frame_counter = 0
    _last_result = ("Unknown", 0.0)