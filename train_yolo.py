from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

def main():
    # ===============================
    # 1. Chọn model YOLOv8 pretrain
    # ===============================
    model_name = "yolov8s.pt"

    # ===============================
    # 2. Dataset YAML
    # ===============================
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
        "imgsz": 640,
        "epochs": 100,
        "batch": 6,       # Windows + GPU VRAM 3050 nên giảm
        "device": device,
        "project": "runs/train",
        "name": "brand_yolo",
        "exist_ok": True,
        "workers": 0,     # Windows: tránh crash multiprocessing
        "amp": False,     # tắt FP16 để an toàn
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

if __name__ == "__main__":
    freeze_support()  # bắt buộc trên Windows khi dùng multiprocessing
    main()
