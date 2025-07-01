import os
import shutil
from glob import glob
from ultralytics import YOLO
import random


model = YOLO("/model/best.pt")

input_dir = "/label_this/images"
label_output_dir = "/label_this/labels"
os.makedirs(label_output_dir, exist_ok=True)

# Predict แบบ batch ถ้าไม่ทำจะช้าและใช้ memory เยอะมาก
image_paths = sorted(glob(os.path.join(input_dir, "*.jpg")))
batch_size = 100

for i in range(0, len(image_paths), batch_size):
    batch = image_paths[i:i + batch_size]
    print(f"Predicting images {i} to {i + len(batch) - 1}")

    model.predict(
        source=batch,
        save=False,
        save_txt=True,
        project="/label_this",
        name="labels",                   
        exist_ok=True,
        imgsz=640,
        conf=0.4
    )
