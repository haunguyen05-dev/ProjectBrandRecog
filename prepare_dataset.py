import os
import shutil
import random
import yaml
from PIL import Image

RAW_DATASET = "raw_dataset"          # dataset thô ban đầu
YOLO_DATASET = "yolo_dataset"        # dataset output
TRAIN_SPLIT = 0.8                    # 80% train – 20% val

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")  # nhiều định dạng ảnh

# ======================================================
# 1. Lấy danh sách thương hiệu (tên class từ folder)
# ======================================================
classes = sorted([d for d in os.listdir(RAW_DATASET) 
                  if os.path.isdir(os.path.join(RAW_DATASET, d))])

class_to_id = {cls: i for i, cls in enumerate(classes)}
print("Classes:", class_to_id)

# ======================================================
# 2. Tạo cấu trúc folder YOLO + từng class
# ======================================================
for split in ["train", "val"]:
    for cls in classes:
        os.makedirs(f"{YOLO_DATASET}/images/{split}/{cls}", exist_ok=True)
        os.makedirs(f"{YOLO_DATASET}/labels/{split}/{cls}", exist_ok=True)

# ======================================================
# 3. Xử lý dataset
# ======================================================
for cls in classes:
    cls_folder = os.path.join(RAW_DATASET, cls)

    images = [
        f for f in os.listdir(cls_folder)
        if f.lower().endswith(EXTENSIONS)
    ]

    random.shuffle(images)
    train_len = int(len(images) * TRAIN_SPLIT)

    train_imgs = images[:train_len]
    val_imgs = images[train_len:]

    def process_images(img_list, split):
        for img_name in img_list:
            src = os.path.join(cls_folder, img_name)

            # Copy ảnh vào thư mục class tương ứng
            dst_img = os.path.join(YOLO_DATASET, "images", split, cls, img_name)
            shutil.copy(src, dst_img)

            # Lấy kích thước ảnh
            img = Image.open(src)
            w, h = img.size

            # Tạo label YOLO (bbox full ảnh)
            label_name = img_name.rsplit(".", 1)[0] + ".txt"
            label_path = os.path.join(YOLO_DATASET, "labels", split, cls, label_name)

            with open(label_path, "w") as f:
                f.write(f"{class_to_id[cls]} 0.5 0.5 1 1")

    process_images(train_imgs, "train")
    process_images(val_imgs, "val")

print("✔ Đã chuyển toàn bộ ảnh sang YOLO theo từng class!")

# ======================================================
# 4. Tạo file data.yaml
# ======================================================
data_yaml = {
    "path": YOLO_DATASET,
    "train": "images/train",
    "val": "images/val",
    "nc": len(classes),
    "names": classes
}

with open(f"{YOLO_DATASET}/data.yaml", "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, allow_unicode=True)

print("✔ Tạo xong file data.yaml!")
print("📄 Nội dung YAML:")
print(data_yaml)
