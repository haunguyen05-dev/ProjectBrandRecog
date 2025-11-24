import cv2
import os
import random
import numpy as np
from pathlib import Path


# ==============================
# CONFIG
# ==============================
RAW_DATASET = "raw_dataset"     # thư mục gốc chứa các thư mục thương hiệu
AUG_PER_IMAGE = 10              # số ảnh muốn tạo thêm cho mỗi ảnh gốc
# ==============================


def rotate_image(image):
    angle = random.randint(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))


def blur_image(image):
    return cv2.GaussianBlur(image, (7, 7), 0)


def darken_image(image):
    factor = random.uniform(0.6, 0.9)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def brighten_image(image):
    factor = random.uniform(1.1, 1.4)
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def add_noise(image):
    noise = np.random.randint(0, 30, image.shape, dtype='uint8')
    return cv2.add(image, noise)


def crop_partial(image):
    h, w = image.shape[:2]
    x1 = random.randint(0, int(w * 0.3))
    y1 = random.randint(0, int(h * 0.3))
    x2 = random.randint(int(w * 0.7), w)
    y2 = random.randint(int(h * 0.7), h)
    return image[y1:y2, x1:x2]


def random_augment(image):
    funcs = [
        rotate_image,
        blur_image,
        darken_image,
        brighten_image,
        add_noise,
        crop_partial
    ]
    return random.choice(funcs)(image)


def augment_brand_folder(brand_path):
    img_paths = [p for p in Path(brand_path).glob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]

    print(f"📸 {brand_path.name}: {len(img_paths)} ảnh gốc")

    for img_path in img_paths:
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"⚠ Không đọc được ảnh: {img_path}")
            continue

        img_name = img_path.stem

        for i in range(AUG_PER_IMAGE):
            aug = random_augment(image)
            out_path = img_path.parent / f"{img_name}_aug_{i}.jpg"
            cv2.imwrite(str(out_path), aug)

        print(f"✔ {img_name}: tạo {AUG_PER_IMAGE} ảnh mới")


def process_all():
    brand_folders = [f for f in Path(RAW_DATASET).iterdir() if f.is_dir()]

    print(f"🔍 Tìm thấy {len(brand_folders)} thương hiệu cần augment\n")

    for brand in brand_folders:
        print(f"=== 🏷 Đang augment: {brand.name} ===")
        augment_brand_folder(brand)

    print("\n🎉 XONG! Tất cả ảnh augment đã được bỏ vào chính folder tương ứng.")


if __name__ == "__main__":
    process_all()
