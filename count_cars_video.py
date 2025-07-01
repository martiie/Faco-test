import threading, queue
import cv2
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import numpy as np


model = YOLO("models/final_model.pt")

font = ImageFont.truetype("THSarabun.ttf", 28)
class_names = ['รถยนต์', 'มอเตอร์ไซค์', 'รถกระบะ', 'รถกระบะ', 'รถบรรทุก', 'รถตู้']
class_colors = {
    'รถยนต์': (255, 255, 0),
    'มอเตอร์ไซค์': (0, 0, 255),
    'รถกระบะ': (0, 255, 0),
    'รถบรรทุก': (0, 255, 255),
    'รถตู้': (255, 0, 0),
}

roi = (100, 460, 1000, 720)  # (x_min, y_min, x_max, y_max)

frame_queue = queue.Queue(maxsize=5)
counted_ids = set()
class_counts = {}

cap = cv2.VideoCapture("data/Cars/MB4 IN 15.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("counted_output.mp4", fourcc, fps, (width, height))

def read_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame_queue.put(frame)

reader_thread = threading.Thread(target=read_frames, daemon=True)
reader_thread.start()

while True:
    frame = frame_queue.get()
    if frame is None:
        break
    
    x_min, y_min, x_max, y_max = roi
    roi_frame = frame[y_min:y_max, x_min:x_max]

    results = model.track(roi_frame, persist=True, conf=0.4, imgsz=576)
    boxes = results[0].boxes

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=3)
    draw.text((x_min, y_min - 30), "ROI", font=font, fill=(255, 0, 0))

    if boxes.id is not None:
        ids = boxes.id.int().tolist()
        classes = boxes.cls.int().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.tolist()

        for box, class_id, track_id, conf in zip(xyxy, classes, ids, confidences):
            x1, y1, x2, y2 = map(int, box)

            x1 += x_min
            x2 += x_min
            y1 += y_min
            y2 += y_min

            class_name = class_names[class_id]
            color = class_colors.get(class_name, (0, 255, 0))

            if track_id not in counted_ids:
                counted_ids.add(track_id)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            draw.rectangle([x1, y1, x2, y2], outline=color[::-1], width=2)
            label = f"{class_name} {conf*100:.1f}%" #ID:{track_id} 
            draw.text((x1 + 5, y1 - 30), label, font=font, fill=color[::-1])

    y = 10
    for cls, count in class_counts.items():
        draw.text((10, y), f"{cls}: {count}", font=font, fill=(255, 255, 0))
        y += 35

    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    out.write(annotated)

cap.release()
out.release()

