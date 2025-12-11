import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from tqdm import tqdm

from models.image_model import ImageModel
from models.text_model import TextModel
from models.fusion_model import FusionModel

class MultimodalDataset(Dataset):
    """
    Loads: image, text, emotion label, intensity
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

        # Get text from CSV if available, otherwise use default
        text = row.get("text", "A person showing emotion")

        return img, text, emotion, intensity

def train_one_epoch(model, image_encoder, text_encoder, loader, optimizer, device):
    model.train()
    image_encoder.train()
    text_encoder.eval()  # Keep text encoder frozen typically
    
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    total_loss = 0
    for img, texts, emo, inten in tqdm(loader, desc="Training"):
        img = img.to(device)
        emo = emo.to(device)
        inten = inten.to(device)

        # Process image features
        img_feat = image_encoder(img)
        
        # Process text features - handle batch properly
        with torch.no_grad():  # Freeze text encoder
            txt_feat = text_encoder(list(texts))
        txt_feat = txt_feat.to(device)

        # Fusion prediction
        emo_pred, inten_pred = model(img_feat, txt_feat)

        # Combined loss
        loss = ce(emo_pred, emo) + 0.5 * mse(inten_pred, inten)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation for training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    train_ds = MultimodalDataset("data/fer_train.csv", transform)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

    # Initialize models
    image_encoder = ImageModel(pretrained=True).to(device)
    text_encoder = TextModel().to(device)
    fusion_model = FusionModel().to(device)

    # Only optimize fusion model and image encoder
    optimizer = torch.optim.Adam(
        list(fusion_model.parameters()) + list(image_encoder.parameters()),
        lr=1e-4
    )

    # Training loop
    num_epochs = 10
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        loss = train_one_epoch(
            fusion_model, image_encoder, text_encoder,
            train_loader, optimizer, device
        )
        print(f"Epoch {epoch+1}/{num_epochs}: loss={loss:.4f}")
        
        # Save best model
        if loss < best_loss:
            best_loss = loss
            torch.save({
                "fusion": fusion_model.state_dict(),
                "image": image_encoder.state_dict(),
                "text": text_encoder.state_dict(),
                "epoch": epoch,
                "loss": loss
            }, "results/best.pth")
            print(f"✓ Saved best model (loss: {loss:.4f})")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main()