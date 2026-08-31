import cv2
import numpy as np
import time
from ultralytics import YOLO

print("=" * 60)
print("REAL CAMERA PHONE DETECTION TEST")
print("=" * 60)

phone_model = YOLO("yolo26n.pt")
print(f"Model loaded: yolo26n.pt")
print(f"Class 67 = {phone_model.names[67]}")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nStarting camera test...")
print("TEST 1: No phone in front of camera (10 seconds)")
print("TEST 2: Hold smartphone clearly in front of camera (10 seconds)")  
print("TEST 3: Remove smartphone (10 seconds)")
print("Press 'q' to quit early\n")

frame_counter = 0
phone_detected = False
test_phase = 1
phase_start = time.time()
last_print_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.05)
        continue

    frame_counter += 1

    if frame_counter % 3 == 0:
        results = phone_model(frame, verbose=False, classes=[67], conf=0.20, imgsz=640)
        
        phone_detected = False
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, f"Phone: {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                phone_detected = True
                now = time.time()
                if now - last_print_time > 1.0:
                    print(f"PHONE DETECTED")
                    print(f"  confidence: {conf:.3f}")
                    print(f"  bounding box: ({x1}, {y1}, {x2}, {y2})")
                    last_print_time = now

        if not phone_detected:
            now = time.time()
            if now - last_print_time > 2.0:
                print("NO PHONE DETECTED")
                last_print_time = now

    elapsed = time.time() - phase_start
    
    if test_phase == 1:
        cv2.putText(frame, "TEST 1: NO PHONE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {10 - int(elapsed)}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if elapsed > 10:
            test_phase = 2
            phase_start = time.time()
            print("\n=== TEST 2: HOLD PHONE IN FRONT OF CAMERA ===")
    elif test_phase == 2:
        cv2.putText(frame, "TEST 2: PHONE VISIBLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Time: {10 - int(elapsed)}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if elapsed > 10:
            test_phase = 3
            phase_start = time.time()
            print("\n=== TEST 3: REMOVE PHONE ===")
    elif test_phase == 3:
        cv2.putText(frame, "TEST 3: PHONE REMOVED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Time: {10 - int(elapsed)}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if elapsed > 10:
            print("\n=== TEST COMPLETE ===")
            break

    cv2.putText(frame, "Press 'q' to quit", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Phone Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("TEST RESULTS SUMMARY")
print("=" * 60)
print("Class ID: 67")
print("Model: yolo26n.pt")
print("No-phone result: Check terminal output above")
print("Phone-visible result: Check terminal output above") 
print("Phone confidence: Check terminal output above")
print("Phone-removed result: Check terminal output above")
print("Bounding box visible: Check video window above")
print("Dashboard status: Would show ON/OFF based on METRICS update")
print("PASS/FAIL: Review terminal output above")
print("=" * 60)