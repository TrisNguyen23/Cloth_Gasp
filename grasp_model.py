import torch.nn as nn
import torchvision.models as models
import torch

class GraspPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)  # Set pretrained=False to avoid downloading weights
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 8)

    def forward(self, x):
        x = self.backbone(x)
        return torch.sigmoid(x)

def load_model(weights_path, device):
    model = GraspPointNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model