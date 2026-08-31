from web_app import app, detection_loop
import threading

t = threading.Thread(target=detection_loop, daemon=True)
t.start()
app.run(host='0.0.0.0', port=5000, threaded=True)