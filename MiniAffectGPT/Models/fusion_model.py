import torch
import torch.nn as nn

class FusionModel(nn.Module):
    """
    Fusion of image + text → emotion classification + intensity regression
    """
    def __init__(self, img_dim=512, txt_dim=768, hidden=512, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(img_dim + txt_dim, hidden)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        self.emotion_head = nn.Linear(hidden, num_classes)
        self.intensity_head = nn.Linear(hidden, 1)

    def forward(self, img_feat, txt_feat):
        x = torch.cat([img_feat, txt_feat], dim=1)
        x = self.relu(self.fc(x))
        x = self.dropout(x)

        emo = self.emotion_head(x)
        intensity = self.intensity_head(x).squeeze(1)
        return emo, intensity
