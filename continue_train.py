import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from models.image_model import ImageModel
from models.text_model import TextModel
from models.fusion_model import FusionModel
from train import MultimodalDataset

def continue_training(start_epoch=6, num_epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    train_ds = MultimodalDataset("data/fer_train.csv", transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    
    val_ds = MultimodalDataset("data/fer_val.csv", transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # Load models
    image_encoder = ImageModel(pretrained=False).to(device)
    text_encoder = TextModel().to(device)
    fusion_model = FusionModel().to(device)

    # Load checkpoint
    ckpt = torch.load("results/best.pth", map_location=device)
    image_encoder.load_state_dict(ckpt["image"])
    text_encoder.load_state_dict(ckpt["text"])
    fusion_model.load_state_dict(ckpt["fusion"])
    print(f"✓ Loaded checkpoint from epoch {start_epoch}\n")

    # Optimizer
    optimizer = torch.optim.Adam(
        list(fusion_model.parameters()) + list(image_encoder.parameters()),
        lr=1e-4
    )

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Train
        fusion_model.train()
        image_encoder.train()
        text_encoder.eval()
        
        train_loss = 0
        for img, texts, emo, inten in tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+num_epochs}"):
            img = img.to(device)
            emo = emo.to(device)
            inten = inten.float().to(device)

            img_feat = image_encoder(img)
            with torch.no_grad():
                txt_feat = text_encoder(list(texts)).to(device)

            emo_pred, inten_pred = fusion_model(img_feat, txt_feat)
            loss = ce(emo_pred, emo) + 0.5 * mse(inten_pred, inten)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        # Validation
        fusion_model.eval()
        image_encoder.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for img, texts, emo, inten in val_loader:
                img = img.to(device)
                emo = emo.to(device)
                inten = inten.float().to(device)

                img_feat = image_encoder(img)
                txt_feat = text_encoder(list(texts)).to(device)
                emo_pred, inten_pred = fusion_model(img_feat, txt_feat)
                
                loss = ce(emo_pred, emo) + 0.5 * mse(inten_pred, inten)
                val_loss += loss.item()
                
                correct += (emo_pred.argmax(dim=1) == emo).sum().item()
                total += emo.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f} ({val_acc*100:.2f}%)")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "fusion": fusion_model.state_dict(),
                "image": image_encoder.state_dict(),
                "text": text_encoder.state_dict(),
                "epoch": epoch + 1,
                "loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, "results/best.pth")
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                "fusion": fusion_model.state_dict(),
                "image": image_encoder.state_dict(),
                "text": text_encoder.state_dict(),
                "epoch": epoch + 1,
                "loss": train_loss
            }, f"results/checkpoint_epoch_{epoch+1}.pth")
            print(f"  ✓ Saved checkpoint at epoch {epoch+1}")

    print(f"\n✅ Training complete! Best val_loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    continue_training(start_epoch=6, num_epochs=30)
