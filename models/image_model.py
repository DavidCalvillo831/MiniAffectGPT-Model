import torch.nn as nn
import torchvision.models as models

class ImageModel(nn.Module):
    """
    ResNet18 backbone generating 512-D feature vector.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        #using RESNET18 architecture ***note for paper
        self.backbone = models.resnet18(pretrained=pretrained)
         # remove final fc layer
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        # x shape: [batch, 3, 224, 224]
        # output shape: [batch, 512]
        return self.backbone(x)

