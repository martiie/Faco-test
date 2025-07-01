from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="/data/labeled_500/data.yaml", epochs=100, imgsz=640)