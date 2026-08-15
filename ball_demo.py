import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "football.mp4"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, imgsz=640, verbose=False)[0]

    for box in results.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # COCO class 32 = sports ball
        if class_id == 32:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"BALL {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Ball Detection Demo", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press Q to quit
    if key == ord("q"):
        break

    # Press SPACE to pause
    if key == 32:
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()