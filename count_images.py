import os
from pathlib import Path

RAW_DATASET = "raw_dataset"   # thư mục chứa các folder thương hiệu

VALID_EXT = [".jpg", ".jpeg", ".png"]


def count_images_in_brand(brand_path):
    return len([
        f for f in Path(brand_path).glob("*.*")
        if f.suffix.lower() in VALID_EXT
    ])


def count_all():
    raw_path = Path(RAW_DATASET)

    if not raw_path.exists():
        print(f"❌ Không tìm thấy thư mục: {RAW_DATASET}")
        return

    brand_folders = [f for f in raw_path.iterdir() if f.is_dir()]

    print(f"📦 Tìm thấy {len(brand_folders)} thương hiệu:")
    print("────────────────────────────────────────")

    total = 0

    for brand in brand_folders:
        count = count_images_in_brand(brand)
        total += count
        print(f"🏷 {brand.name}: {count} ảnh")

    print("────────────────────────────────────────")
    print(f"📊 Tổng số ảnh trong toàn bộ dataset: {total}")


if __name__ == "__main__":
    count_all()
