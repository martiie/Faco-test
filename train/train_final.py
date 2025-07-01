from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.train(data="/data/final_data/data.yaml", epochs=100, imgsz=640)