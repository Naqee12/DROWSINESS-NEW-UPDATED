from flask import Flask
import threading
import time
import requests

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello'

def background_task():
    i = 0
    while True:
        i += 1
        if i % 100000 == 0:
            print('Background task running...')
        time.sleep(0.001)

t = threading.Thread(target=background_task, daemon=True)
t.start()

print('Starting Flask...')
app.run(host='0.0.0.0', port=5004, threaded=True)