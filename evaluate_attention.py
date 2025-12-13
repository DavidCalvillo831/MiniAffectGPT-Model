import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from models.image_model import ImageModel
from models.text_model import TextModel
from models.attention_fusion import AttentionFusionModel
from train import MultimodalDataset

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def evaluate_attention(checkpoint_path="results/best_attention.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = MultimodalDataset("data/fer_test.csv", transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    print(f"Loaded {len(test_ds)} test samples\n")

    image_encoder = ImageModel(pretrained=False).to(device)
    text_encoder = TextModel().to(device)
    fusion = AttentionFusionModel().to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    image_encoder.load_state_dict(ckpt["image"])
    text_encoder.load_state_dict(ckpt["text"])
    fusion.load_state_dict(ckpt["fusion"])
    
    print(f"✓ Loaded from epoch {ckpt.get('epoch', '?')}\n")

    image_encoder.eval()
    text_encoder.eval()
    fusion.eval()

    all_preds = []
    all_true = []
    total_loss = 0
    ce = nn.CrossEntropyLoss()

    print("Evaluating...")
    with torch.no_grad():
        for img, texts, emo, inten in tqdm(test_loader):
            img = img.to(device)
            emo = emo.to(device)

            img_feat = image_encoder(img)
            txt_feat = text_encoder(list(texts)).to(device)
            emo_pred = fusion(img_feat, txt_feat)

            loss = ce(emo_pred, emo)
            total_loss += loss.item()

            all_preds.extend(emo_pred.argmax(dim=1).cpu().numpy())
            all_true.extend(emo.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    acc = accuracy_score(all_true, all_preds)
    f1_macro = f1_score(all_true, all_preds, average='macro')
    f1_weighted = f1_score(all_true, all_preds, average='weighted')
    
    print("\n" + "="*60)
    print("RESULTS (ATTENTION MODEL)")
    print("="*60)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 (Macro): {f1_macro:.4f}")
    print(f"F1 (Weighted): {f1_weighted:.4f}")
    print("\n" + classification_report(all_true, all_preds, target_names=EMOTIONS, digits=4, zero_division=0))

    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Confusion Matrix (Attention)')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix_attention.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion_matrix_attention.png")
    
    torch.save({'accuracy': acc, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted}, 
               'results/results_attention.pth')
    print("="*60)
    
    return acc, f1_macro

if __name__ == "__main__":
    evaluate_attention()
