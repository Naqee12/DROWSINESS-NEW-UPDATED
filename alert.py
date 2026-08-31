import winsound
import threading

_alert_playing = False


def play_alert():
    """Plays a beep alarm in a background thread so it doesn't freeze the video loop."""
    global _alert_playing
    if _alert_playing:
        return  # avoid stacking multiple overlapping alarms

    def _beep():
        global _alert_playing
        _alert_playing = True
        for _ in range(3):
            winsound.Beep(1000, 400)  # frequency=1000Hz, duration=400ms
        _alert_playing = False

    threading.Thread(target=_beep, daemon=True).start()