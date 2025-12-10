import torch.nn as nn
import torchvision.models as models

class ImageModel(nn.Module):
    """
    ResNet18 backbone generating 512-D feature vector.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)
