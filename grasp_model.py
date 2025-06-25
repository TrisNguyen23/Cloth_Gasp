import torch.nn as nn
import torchvision.models as models

class GraspPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 8)

    def forward(self, x):
        return self.backbone(x)
