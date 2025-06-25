import cv2
import numpy as np
import os
import csv
from pathlib import Path
import random

def farthest_point_sampling(points, k):
    selected = [random.choice(points)]
    for _ in range(1, k):
        dists = np.array([min(np.linalg.norm(p - s) for s in selected) for p in points])
        next_point = points[np.argmax(dists)]
        selected.append(next_point)
    return np.array(selected)

def process_mask(mask_path, k=4):
    mask = cv2.imread(str(mask_path), 0)
    if mask is None:
        print(f"Không đọc được: {mask_path.name}")
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Không có contour trong: {mask_path.name}")
        return None

    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2)

    if len(points) < k:
        print(f"Quá ít điểm để lấy {k} điểm: {mask_path.name}")
        return None

    sampled = farthest_point_sampling(points, k)
    coords = sampled.flatten().tolist()
    return [mask_path.name] + coords

def generate_labels(mask_dir, output_csv, k=4):
    mask_dir = Path(mask_dir)
    rows = [["filename", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]]

    for mask_path in mask_dir.glob("*.png"):
        result = process_mask(mask_path, k)
        if result:
            rows.append(result)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Đã tạo file label: {output_csv} (tổng {len(rows) - 1} ảnh)")

if __name__ == "__main__":
    generate_labels(mask_dir="./masks", output_csv="./dataset/labels.csv", k=4)
