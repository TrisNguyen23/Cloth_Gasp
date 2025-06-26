# grasp_model.py
import torch.nn as nn
import torchvision.models as models
import torch

class GraspPointNet(nn.Module):
    def __init__(self):
        super(GraspPointNet, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 8)

    def forward(self, x):
        return self.backbone(x)


