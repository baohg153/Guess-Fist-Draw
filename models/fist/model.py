import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import torch

def main():
    model = YOLO("yolov8n.pt")  

    model.train(
        data="data/fist/data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        patience=5,
        exist_ok=True,
        save=True,
        workers=2
    )

if __name__ == "__main__":
    main()  