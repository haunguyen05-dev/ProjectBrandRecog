import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageTk, ImageFont
import numpy as np
from ultralytics import YOLO
import cv2
import threading

# ==================== CONFIG ====================
MODEL_PATH = "runs/train/brand_yolo/weights/last.pt"
CONF_THRESHOLD = 0.46  # Confidence thấp hơn này sẽ coi là "Unknown"

model = YOLO(MODEL_PATH)
running_cam = False
img_result_tk = None
img_orig_tk = None

# ==================== SCALE ẢNH ====================
def scale_image(img, max_w, max_h):
    w, h = img.size
    scale = min(max_w / w, max_h / h)
    if scale < 1:
        scale = 1
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)

# ==================== LOAD IMAGE ====================
def load_image():
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not path:
        return None
    img = Image.open(path).convert("RGB")
    img = img.resize((360, 360), Image.LANCZOS)
    return img

# ==================== NHẬN DIỆN ẢNH FILE ====================
def recognize_logo():
    global img_orig_tk, img_result_tk, running_cam
    if running_cam:
        status_label.config(text="⚡ Camera đang chạy, không thể chọn ảnh.")
        return

    img_orig = load_image()
    if img_orig is None:
        status_label.config(text="❌ Chưa chọn ảnh.")
        return

    img_result = img_orig.copy()
    status_label.config(text="🔄 Đang nhận diện...")
    root.update_idletasks()

    try:
        results = model.predict(source=np.array(img_orig), conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes
        names = results[0].names
        draw = ImageDraw.Draw(img_result)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        if len(boxes) == 0:
            status_label.config(text="⚡ Không nhận diện logo")
        else:
            best_box = max(boxes, key=lambda b: float(b.conf[0].cpu().numpy()))
            conf = float(best_box.conf[0].cpu().numpy())
            
            if conf < CONF_THRESHOLD:
                label = "Unknown"
                status_label.config(text="⚡ Không nhận diện logo")
            else:
                cls_id = int(best_box.cls[0].cpu().numpy())
                label = names.get(cls_id, "Unknown")
                status_label.config(text=f"✅ Nhận diện: {label} ({conf*100:.1f}%)")

            xyxy = best_box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            draw.text((x1, max(0, y1 - 30)), f"{label} ({conf*100:.1f}%)", fill="red", font=font)

        img_orig_display = scale_image(img_orig, 480, 480)
        img_result_display = scale_image(img_result, 480, 480)
        img_orig_tk = ImageTk.PhotoImage(img_orig_display)
        img_result_tk = ImageTk.PhotoImage(img_result_display)
        label_orig.config(image=img_orig_tk)
        label_result.config(image=img_result_tk)

    except Exception as e:
        status_label.config(text=f"❌ Lỗi: {e}")

# ==================== NHẬN DIỆN CAMERA ====================
def start_camera():
    global running_cam, img_orig_tk
    if running_cam:
        return

    running_cam = True
    img_orig_tk = None
    label_orig.config(image=None)
    status_label.config(text="🎥 Camera đang chạy...")
    threading.Thread(target=cam_loop, daemon=True).start()

def stop_camera():
    global running_cam, cap
    running_cam = False  # dừng luồng cam
    try:
        cap.release()     # giải phóng camera
    except:
        pass
    status_label.config(text="⚡ Camera dừng.")

def cam_loop():
    global running_cam, img_result_tk, cap

    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            break
    else:
        status_label.config(text="❌ Không mở được camera nào!")
        running_cam = False
        return

    try:
        while running_cam:
            ret, frame = cap.read()
            if not ret:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            results = model.predict(source=np.array(img_pil), conf=CONF_THRESHOLD, verbose=False)
            boxes = results[0].boxes
            names = results[0].names
            draw = ImageDraw.Draw(img_pil)

            if len(boxes) > 0:
                best_box = max(boxes, key=lambda b: float(b.conf[0].cpu().numpy()))
                conf = float(best_box.conf[0].cpu().numpy())
                
                if conf < CONF_THRESHOLD:
                    label = "Unknown"
                    status_label.config(text="⚡ Không nhận diện logo")
                else:
                    cls_id = int(best_box.cls[0].cpu().numpy())
                    label = names.get(cls_id, "Unknown")
                    status_label.config(text=f"✅ {label} ({conf*100:.1f}%)")
                
                xyxy = best_box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, max(0, y1 - 20)), f"{label} ({conf*100:.1f}%)", fill="red")
            else:
                status_label.config(text="⚡ Không nhận diện logo")

            img_disp = scale_image(img_pil, 480, 480)
            img_result_tk = ImageTk.PhotoImage(img_disp)
            label_result.config(image=img_result_tk)

            root.update_idletasks()
            root.update()

    finally:
        try:
            cap.release()
        except:
            pass
        running_cam = False
        status_label.config(text="⚡ Camera dừng.")


def cam_loop():
    global running_cam, img_result_tk

    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            break
    else:
        status_label.config(text="❌ Không mở được camera nào!")
        running_cam = False
        return

    try:
        while running_cam:
            ret, frame = cap.read()
            if not ret:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            results = model.predict(source=np.array(img_pil), conf=CONF_THRESHOLD, verbose=False)
            boxes = results[0].boxes
            names = results[0].names
            draw = ImageDraw.Draw(img_pil)

            if len(boxes) > 0:
                best_box = max(boxes, key=lambda b: float(b.conf[0].cpu().numpy()))
                conf = float(best_box.conf[0].cpu().numpy())
                
                if conf < CONF_THRESHOLD:
                    label = "Unknown"
                    status_label.config(text="⚡ Không nhận diện logo")
                else:
                    cls_id = int(best_box.cls[0].cpu().numpy())
                    label = names.get(cls_id, "Unknown")
                    status_label.config(text=f"✅ {label} ({conf*100:.1f}%)")
                
                xyxy = best_box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                draw.text((x1, max(0, y1 - 20)), f"{label} ({conf*100:.1f}%)", fill="red")
            else:
                status_label.config(text="⚡ Không nhận diện logo")

            img_disp = scale_image(img_pil, 480, 480)
            img_result_tk = ImageTk.PhotoImage(img_disp)
            label_result.config(image=img_result_tk)

            root.update_idletasks()
            root.update()

    finally:
        cap.release()
        running_cam = False
        status_label.config(text="⚡ Camera dừng.")

# ==================== UI ====================
root = tk.Tk()
root.title("📸 Nhận Diện Logo - AI YOLOv8")
root.geometry("1200x650")
root.configure(bg="#f3f4f6")

header = tk.Frame(root, bg="#4f46e5", height=65)
header.pack(fill="x")
tk.Label(header, text="Nhận Diện Logo - AI YOLOv8", bg="#4f46e5", fg="white",
         font=("Segoe UI", 20, "bold")).pack(pady=10)

left_panel = tk.Frame(root, bg="#eef2ff", width=250)
left_panel.pack(side="left", fill="y")

tk.Button(left_panel, text="🖼 Chọn ảnh & Nhận diện", command=recognize_logo,
          font=("Segoe UI", 12, "bold"), bg="#4f46e5", fg="white", relief="flat",
          padx=10, pady=8).pack(pady=20, padx=20, fill="x")
tk.Button(left_panel, text="📹 Bật Camera", command=start_camera,
          font=("Segoe UI", 12, "bold"), bg="#10b981", fg="white", relief="flat",
          padx=10, pady=8).pack(pady=10, padx=20, fill="x")
tk.Button(left_panel, text="⏹ Dừng Camera", command=stop_camera,
          font=("Segoe UI", 12, "bold"), bg="#ef4444", fg="white", relief="flat",
          padx=10, pady=8).pack(pady=10, padx=20, fill="x")

status_label = tk.Label(left_panel, text="⚡ Sẵn sàng.", bg="#eef2ff", wraplength=220,
                        justify="left", font=("Segoe UI", 10))
status_label.pack(padx=20, pady=20)

display = tk.Frame(root, bg="white")
display.pack(side="left", expand=True, fill="both")

label_orig = tk.Label(display, bg="white")
label_orig.pack(side="left", padx=10, pady=10)
label_result = tk.Label(display, bg="white")
label_result.pack(side="left", padx=10, pady=10)

root.mainloop()
