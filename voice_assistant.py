import pyttsx3
import speech_recognition as sr
import threading
import queue

# ---------- Text-to-Speech ----------
_tts_engine = pyttsx3.init()
_tts_engine.setProperty("rate", 165)
_speak_queue = queue.Queue()
_speak_lock = threading.Lock()


def _tts_worker():
    while True:
        text = _speak_queue.get()
        if text is None:
            break
        with _speak_lock:
            _tts_engine.say(text)
            _tts_engine.runAndWait()


_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()


def speak(text):
    """Queue a message to be spoken without blocking the main app."""
    _speak_queue.put(text)


# ---------- Voice Command Listening ----------
_recognizer = sr.Recognizer()
_mic = None

try:
    _mic = sr.Microphone()
except OSError:
    print("[voice_assistant] No microphone detected.")


def listen_once(timeout=4, phrase_time_limit=4):
    if _mic is None:
        return None

    try:
        with _mic as source:
            audio = _recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        text = _recognizer.recognize_google(audio)
        return text.lower()
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"[voice_assistant] Speech recognition service error: {e}")
        return None


if __name__ == "__main__":
    speak("Voice assistant test. Can you hear me?")
    import time
    time.sleep(3)
    print("Say something...")
    result = listen_once()
    print(f"You said: {result}")