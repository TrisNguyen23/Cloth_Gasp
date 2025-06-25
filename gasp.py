import cv2
import numpy as np
import random
import os
from pathlib import Path

def farthest_point_sampling(points, k):
    selected = [random.choice(points)]
    for _ in range(1, k):
        dists = np.array([min(np.linalg.norm(p - s) for s in selected) for p in points])
        next_point = points[np.argmax(dists)]
        selected.append(next_point)
    return np.array(selected)

def process_mask(mask_path, output_path, k=4):
    mask = cv2.imread(str(mask_path), 0)
    if mask is None:
        print(f"Không đọc được mask: {mask_path}")
        return

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Không tìm thấy contour trong: {mask_path}")
        return

    contour = max(contours, key=cv2.contourArea)
    contour_points = contour.reshape(-1, 2)

    if len(contour_points) < k:
        print(f"Không đủ điểm contour để lấy {k} điểm: {mask_path}")
        return

    selected = farthest_point_sampling(contour_points, k)

    # Đánh dấu màu từng điểm
    color_map = [
        (0, 255, 0),     
        (255, 0, 0),    
        (0, 255, 255),   
        (255, 0, 255),   
    ]

    output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for i, pt in enumerate(selected):
        pt = tuple(pt.astype(int))
        color = color_map[i % len(color_map)]
        cv2.circle(output, pt, 8, color, -1)
        cv2.putText(output, f"P{i+1}", (pt[0] + 6, pt[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), output)
    print(f"Lưu ảnh FPS: {output_path.name}")

def process_folder(mask_folder, result_folder, k=4):
    mask_folder = Path(mask_folder)
    result_folder = Path(result_folder)
    os.makedirs(result_folder, exist_ok=True)

    mask_paths = list(mask_folder.glob("*.png"))
    if not mask_paths:
        print("Không có file mask PNG nào trong thư mục.")
        return

    for mask_path in mask_paths:
        output_path = result_folder / f"{mask_path.stem}_fps.png"
        process_mask(mask_path, output_path, k=k)

if __name__ == "__main__":
    process_folder("./masks", "./results", k=4)
