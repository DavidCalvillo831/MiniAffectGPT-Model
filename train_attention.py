import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.image_model import ImageModel
from models.text_model import TextModel
from models.attention_fusion import AttentionFusionModel
from train import MultimodalDataset

def train_with_attention(num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading datasets...")
    train_ds = MultimodalDataset("data/fer_train.csv", transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    
    val_ds = MultimodalDataset("data/fer_val.csv", transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples\n")

    # Initialize models
    image_encoder = ImageModel(pretrained=False).to(device)
    text_encoder = TextModel().to(device)
    fusion_model = AttentionFusionModel().to(device)

    # Load previous image encoder (transfer learning)
    if os.path.exists("results/best.pth"):
        print("Loading previous image encoder...")
        ckpt = torch.load("results/best.pth", map_location=device)
        image_encoder.load_state_dict(ckpt["image"])
        print("✓ Loaded\n")
    
    optimizer = torch.optim.Adam(
        list(fusion_model.parameters()) + list(image_encoder.parameters()),
        lr=5e-4
    )
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    ce_loss = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0

    print("="*60)
    print("TRAINING WITH CROSS-MODAL ATTENTION")
    print("="*60)
    
    for epoch in range(num_epochs):
        # Train
        fusion_model.train()
        image_encoder.train()
        text_encoder.eval()
        
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for img, texts, emo, inten in pbar:
            img = img.to(device)
            emo = emo.to(device)

            img_feat = image_encoder(img)
            with torch.no_grad():
                txt_feat = text_encoder(list(texts)).to(device)

            emo_pred = fusion_model(img_feat, txt_feat)
            loss = ce_loss(emo_pred, emo)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (emo_pred.argmax(dim=1) == emo).sum().item()
            train_total += emo.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        fusion_model.eval()
        image_encoder.eval()
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for img, texts, emo, inten in val_loader:
                img = img.to(device)
                emo = emo.to(device)

                img_feat = image_encoder(img)
                txt_feat = text_encoder(list(texts)).to(device)
                emo_pred = fusion_model(img_feat, txt_feat)
                
                loss = ce_loss(emo_pred, emo)
                val_loss += loss.item()
                
                val_correct += (emo_pred.argmax(dim=1) == emo).sum().item()
                val_total += emo.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  LR: {current_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            torch.save({
                "fusion": fusion_model.state_dict(),
                "image": image_encoder.state_dict(),
                "text": text_encoder.state_dict(),
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "val_loss": val_loss
            }, "results/best_attention.pth")
            
            print(f"  ✓ NEW BEST! Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        print("-"*60)

    print(f"\n✅ DONE! Best Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print("Model saved to: results/best_attention.pth")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    train_with_attention(num_epochs=20)
