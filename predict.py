# predict.py
import torch
import cv2
import numpy as np
import os
from grasp_model import GraspPointNet
import torch.nn.functional as F

def load_model(weights_path, device):
    model = GraspPointNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def predict_grasp(model, image_path, device, save_result=True):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi đọc ảnh: {image_path}")
        return

    # Đảm bảo ảnh là 3 kênh
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Chuyển thành tensor
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        output = model(img_tensor)[0].cpu().numpy()

    print("✅ Output shape:", output.shape)
    print("✅ Output:", output)

    if output.shape[0] != 8:
        print("❌ Output model không đúng số lượng điểm (phải là 8 giá trị). Kiểm tra model.")
        return

    # Scale lại tọa độ nếu cần
    output[0::2] *= 224
    output[1::2] *= 224

    # Vẽ điểm
    vis_img = cv2.resize(img, (224, 224))
    colors = [(0,255,0), (255,0,0), (0,255,255), (255,0,255)]
    for i in range(4):
        x, y = int(output[2*i]), int(output[2*i+1])
        cv2.circle(vis_img, (x, y), 6, colors[i], -1)
        cv2.putText(vis_img, f"P{i+1}", (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)

    if save_result:
        output_path = os.path.splitext(image_path)[0] + "_predict.png"
        cv2.imwrite(output_path, vis_img)
        print(f"✅ Đã lưu ảnh kết quả tại: {output_path}")


# ======================
if __name__ == "__main__":
    image_path = "./test/pattern1_test.png"
    weights_path = "./grasp_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, device)

    predict_grasp(model, image_path, device)
