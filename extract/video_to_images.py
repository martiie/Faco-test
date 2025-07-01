import cv2
import os

def extract_frames(video_path, output_dir, fps_interval=1):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * fps_interval)

    count = 0
    saved = 0
    basename = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            filename = f"{basename}_frame_{saved:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"[{basename}] ดึงเฟรมแล้วทั้งหมด {saved} ภาพ")

video_files = ["/data/Cars/MB4 IN 13.mp4", "/data/Cars/MB4 IN 14.mp4", "/data/Cars/MB4 IN 15.mp4"]

for v in video_files:
    extract_frames(v, output_dir="frames", fps_interval=1)
