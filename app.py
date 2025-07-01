import threading, queue
import cv2
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os

# ===== โหลดวิดีโอ =====
video_dir = "data/Cars"
video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
if not video_files:
    print("ไม่พบไฟล์วิดีโอในโฟลเดอร์ data")
    exit()

video_path = os.path.join(video_dir, video_files[0])
print(f"เปิดไฟล์วิดีโอ: {video_path}")

# ===== โหลดโมเดล YOLO =====
model = YOLO("models/final_model.pt")

# ===== ฟอนต์และสี =====
font = ImageFont.truetype("THSarabun.ttf", 28)
class_names = ['รถยนต์', 'มอเตอร์ไซค์', 'รถกระบะ', 'รถกระบะ', 'รถบรรทุก', 'รถตู้']
class_colors = {
    'รถยนต์': (255, 255, 0),
    'มอเตอร์ไซค์': (0, 0, 255),
    'รถกระบะ': (0, 255, 0),
    'รถบรรทุก': (0, 255, 255),
    'รถตู้': (255, 0, 0),
}

# ===== ROI แบบสี่เหลี่ยมคางหมู =====
roi_polygon = np.array([[
    (260, 700),   # ล่างซ้าย ชิดขอบถนนมากขึ้น
    (510, 460),   # บนซ้าย
    (740, 460),   # บนขวา
    (1100, 700)   # ล่างขวา ลึกเข้าไปถึงโค้งถนน
]], dtype=np.int32)

roi = (260, 460, 1100, 720) # (x_min, y_min, x_max, y_max)

frame_queue = queue.Queue(maxsize=5)
counted_ids = set()
class_counts = {}

cap = cv2.VideoCapture(video_path)

def read_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put(None)
            break
        frame_queue.put(frame)

reader_thread = threading.Thread(target=read_frames, daemon=True)
reader_thread.start()

skip = 1
frame_index = 0

counted_ids = set()
class_counts = {}
track_id_to_class = {}

while True:
    frame = frame_queue.get()
    if frame is None:
        break

    frame_index += 1
    if frame_index % skip != 0:
        continue
    
    x_min, y_min, x_max, y_max = roi
    roi_frame = frame[y_min:y_max, x_min:x_max]

    # ตรวจจับใน roi_frame
    results = model.track(roi_frame, persist=True, conf=0.4, imgsz=640)
    boxes = results[0].boxes

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # วาด ROI polygon (พิกัดเต็ม)
    roi_pts = roi_polygon[0].tolist()
    draw.line(roi_pts + [roi_pts[0]], fill=(255, 0, 0), width=3)
    draw.text((roi_pts[0][0], roi_pts[0][1] - 30), "ROI", font=font, fill=(255, 0, 0))

    def point_in_polygon(point, polygon):
        return cv2.pointPolygonTest(polygon, point, False) >= 0


    if boxes.id is not None:
        ids = boxes.id.int().tolist()
        classes = boxes.cls.int().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.tolist()

        for box, class_id, track_id, conf in zip(xyxy, classes, ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            # แปลงพิกัด bbox และ center กลับเป็นพิกัดเต็มภาพ
            x1_full = x1 + x_min
            y1_full = y1 + y_min
            x2_full = x2 + x_min
            y2_full = y2 + y_min
            center_x = int((x1_full + x2_full) / 2)
            center_y = int((y1_full + y2_full) / 2)

            if not point_in_polygon((center_x, center_y), roi_polygon[0]):
                continue

            class_name = class_names[class_id]
            color = class_colors.get(class_name, (0, 255, 0))

            if track_id not in track_id_to_class:
                # เจอ track_id ใหม่
                track_id_to_class[track_id] = class_name
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            else:
                old_class = track_id_to_class[track_id]
                if old_class != class_name:
                    # ประเภทเปลี่ยน - ลดนับคลาสเก่า เพิ่มนับคลาสใหม่
                    class_counts[old_class] = max(class_counts.get(old_class, 1) - 1, 0)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    track_id_to_class[track_id] = class_name

            # วาดกรอบและ label ด้วยพิกัดเต็ม
            draw.rectangle([x1_full, y1_full, x2_full, y2_full], outline=color[::-1], width=2)
            label = f"ID:{track_id} {class_name} {conf*100:.1f}%"
            draw.text((x1_full + 5, y1_full - 30), label, font=font, fill=color[::-1])

    y = 10
    for cls, count in class_counts.items():
        draw.text((10, y), f"{cls}: {count}", font=font, fill=(255, 255, 0))
        y += 35

    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow("Show", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
