import torch
from torch.utils.data import DataLoader
from grasp_dataset import GraspDataset
from grasp_model import GraspPointNet
import torch.nn as nn
import torch.optim as optim

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GraspDataset(csv_file="./dataset/labels.csv", image_dir="./masks", transform=None)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = GraspPointNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            preds = model(imgs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "grasp_model.pth")
    print("Model đã lưu tại grasp_model.pth")

if __name__ == "__main__":
    train()
