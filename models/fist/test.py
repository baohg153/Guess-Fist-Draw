import cv2
import torch
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # YOLO inference (single frame)
    results = model(frame, conf=0.4, device=device)

    # Draw bbox 
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
