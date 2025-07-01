import os
import random
import shutil

# ที่เก็บภาพทั้งหมด (ยังไม่มี label)
all_image_dir = "/data/remaining_images"
subset_dir = "/data/label_this/images"
os.makedirs(subset_dir, exist_ok=True)

# สุ่ม 1500
all_images = [f for f in os.listdir(all_image_dir) if f.endswith(".jpg")]
sample_images = random.sample(all_images, 1500)

# คัดลอกไปโฟลเดอร์สำหรับ labeling
for img_name in sample_images:
    shutil.copy(os.path.join(all_image_dir, img_name), os.path.join(subset_dir, img_name))

print("เสร็แล้วว")