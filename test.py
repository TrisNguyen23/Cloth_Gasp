import cv2
import numpy as np
import os
from pathlib import Path

def generate_mask(image_path, output_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 50])
    upper = np.array([179, 200, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), mask)
    print(f"Đã tạo mask: {output_path.name}")

def process_folder(image_folder, mask_folder):
    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)
    os.makedirs(mask_folder, exist_ok=True)

    image_paths = list(image_folder.glob("*.png"))
    if not image_paths:
        print("Không tìm thấy ảnh PNG trong thư mục.")
        return

    for image_path in image_paths:
        output_path = mask_folder / f"{image_path.stem}_mask.png"
        generate_mask(image_path, output_path)

if __name__ == "__main__":
    process_folder("./images", "./masks")
