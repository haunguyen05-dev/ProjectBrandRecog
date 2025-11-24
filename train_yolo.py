# train.py
from ultralytics import YOLO
import torch

# ===============================
# 1. Chọn model YOLOv8 pretrain
# ===============================
# Gợi ý:
# - yolov8n.pt → train nhanh (test)
# - yolov8s.pt → dùng thực tế
# - yolov8m.pt → chính xác cao hơn
model_name = "yolov8s.pt"

# ===============================
# 2. Dataset YAML
# ===============================
# Tự động nhận dạng nhiều class
data_yaml = "yolo_dataset/data.yaml"

# ===============================
# 3. Kiểm tra GPU hoặc CPU
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Đang dùng device:", device)

# ===============================
# 4. Tham số train
# ===============================
train_params = {
    "data": data_yaml,
    "imgsz": 640,         # YOLO khuyến nghị 640
    "epochs": 100,         # train chuẩn
    "batch": 8,           # có thể tăng nếu GPU mạnh
    "device": device,
    "project": "runs/train",
    "name": "brand_yolo",
    "exist_ok": True,
    "workers": 2,         # giảm lỗi trên Windows
}

# ===============================
# 5. Load model
# ===============================
model = YOLO(model_name)

# ===============================
# 6. Train
# ===============================
model.train(**train_params)

print("🎉 Training hoàn tất!")
