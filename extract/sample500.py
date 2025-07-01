import random
import shutil
import os

all_images_dir = "/data/frames/"
sample_output_dir = "/data/sample_500/"
num_samples = 500

output_dir = "/data/remaining_images/"
os.makedirs(output_dir, exist_ok=True)

sample_images = set(os.listdir(sample_output_dir))
all_images = set(os.listdir(all_images_dir))

remaining_images = all_images - sample_images

for img in remaining_images:
    shutil.copy(os.path.join(all_images_dir, img), os.path.join(output_dir, img))
    

output_dir = "/data/remaining_images/"
os.makedirs(output_dir, exist_ok=True)

# รายชื่อไฟล์ (ใช้เฉพาะชื่อ ไม่รวม path)
sample_images = set(os.listdir(sample_output_dir))
all_images = set(os.listdir(all_images_dir))

# หาภาพที่เหลือ
remaining_images = all_images - sample_images

# คัดลอกภาพที่เหลือไปยัง output_dir
for img in remaining_images:
    shutil.copy(os.path.join(all_images_dir, img), os.path.join(output_dir, img))

print("เสร็จแล้ววว")