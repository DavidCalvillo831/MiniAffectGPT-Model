import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import random
import torchvision.transforms as transforms

from models.image_model import ImageModel
from models.text_model import TextModel
from models.fusion_model import FusionModel

class MultimodalDataset(Dataset):
    """
    Loads: image, text (synthetic), emotion label, intensity
    """
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        emotion = int(row["emotion"])
        intensity = float(row["intensity"])

        # Simple dummy text until you pair real text
        text = "This is an emotional moment."

        return img, text, emotion, intensity

def train_one_epoch(model, image_encoder, text_encoder, loader, optimizer, device):
    model.train()
    image_encoder.train()
    text_encoder.train()

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    total = 0
    for img, text, emo, inten in loader:
        img = img.to(device)
        emo = emo.to(device)
        inten = inten.to(device)

        img_feat = image_encoder(img)
        txt_feat = text_encoder([text[0]])
        txt_feat = txt_feat.to(device)

        emo_pred, inten_pred = model(img_feat, txt_feat)

        loss = ce(emo_pred, emo) + 0.5 * mse(inten_pred, inten)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_ds = MultimodalDataset("data/fer_train.csv", transform)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    image_encoder = ImageModel(pretrained=True).to(device)
    text_encoder = TextModel().to(device)
    fusion_model = FusionModel().to(device)

    optimizer = torch.optim.Adam(
        list(fusion_model.parameters()) +
        list(image_encoder.parameters()) +
        list(text_encoder.parameters()),
        lr=1e-4
    )

    for epoch in range(3):
        loss = train_one_epoch(
            fusion_model, image_encoder, text_encoder,
            train_loader, optimizer, device
        )
        print(f"Epoch {epoch+1}: loss={loss:.4f}")

    torch.save({
        "fusion": fusion_model.state_dict(),
        "image": image_encoder.state_dict(),
        "text": text_encoder.state_dict()
    }, "results/best.pth")

if __name__ == "__main__":
    main()

