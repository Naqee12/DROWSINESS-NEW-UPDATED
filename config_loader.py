import json
import os

DEFAULT_CONFIG = {
    "NOTIFICATIONS": {
        "ENABLED": False,
        "DROWSY_EVENT_TRIGGER_COUNT": 3,
        "SENDER_EMAIL": "",
        "SENDER_APP_PASSWORD": "",
        "RECIPIENT_EMAIL": "",
        "RECIPIENT_SMS_GATEWAY": "",
        "SEND_ONCE_PER_SESSION": True
    },
    "EAR_THRESHOLD": 0.25,
    
    "MAR_THRESHOLD": 0.6,
    "EAR_CONSEC_SECONDS": 0.5,
    "PITCH_CONSEC_SECONDS": 0.7,
    "WINDOW_SECONDS": 30,
    "ALERT_COOLDOWN": 5,
    "CALIBRATION_SECONDS": 8,
    "EAR_BASELINE_RATIO": 0.75,
    "PITCH_BASELINE_OFFSET": 15,
    "FUSION_WEIGHTS": {"EAR": 0.4, "PERCLOS": 0.3, "PITCH": 0.2, "YAWN": 0.1},
    "FUSION_THRESHOLD": 0.5
}


def load_config(path="config.json"):
    if not os.path.exists(path):
        print(f"[config] '{path}' not found, using default settings.")
        return DEFAULT_CONFIG

    try:
        with open(path, "r") as f:
            user_config = json.load(f)
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)
        print(f"[config] Loaded settings from '{path}'.")
        return config
    except (json.JSONDecodeError, OSError) as e:
        print(f"[config] Failed to load '{path}' ({e}), using default settings.")
        return DEFAULT_CONFIG