import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # general COCO model, includes 'cell phone' class

CELL_PHONE_CLASS_ID = 67  # COCO class index for "cell phone"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, classes=[CELL_PHONE_CLASS_ID], conf=0.4)

    phone_detected = False
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            phone_detected = True

    if phone_detected:
        cv2.putText(frame, "WARNING: Phone Detected!", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Phone Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()