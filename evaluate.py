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
from models.fusion_model import FusionModel
from train import MultimodalDataset

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def evaluate(checkpoint_path="results/best.pth", data_path="data/fer_test.csv"):
    """
    Evaluate model on test set
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Test transform (no augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_ds = MultimodalDataset(data_path, transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
    print(f"Loaded {len(test_ds)} test samples\n")

    # Initialize models
    image_encoder = ImageModel(pretrained=False).to(device)
    text_encoder = TextModel().to(device)
    fusion = FusionModel().to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    image_encoder.load_state_dict(ckpt["image"])
    text_encoder.load_state_dict(ckpt["text"])
    fusion.load_state_dict(ckpt["fusion"])
    print(f"✓ Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
    
    loss_val = ckpt.get('loss', 'unknown')
    if isinstance(loss_val, (int, float)):
        print(f"  Training loss: {loss_val:.4f}\n")
    else:
        print(f"  Training loss: {loss_val}\n")

    # Set to evaluation mode
    image_encoder.eval()
    text_encoder.eval()
    fusion.eval()

    # Collect predictions
    all_emo_preds = []
    all_emo_true = []
    all_inten_preds = []
    all_inten_true = []

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    total_loss = 0

    print("Evaluating...")
    with torch.no_grad():
        for img, texts, emo, inten in tqdm(test_loader):
            img = img.to(device)
            emo = emo.to(device)
            inten = inten.to(device)

            # Forward pass
            img_feat = image_encoder(img)
            txt_feat = text_encoder(list(texts)).to(device)
            emo_pred, inten_pred = fusion(img_feat, txt_feat)

            # Calculate loss
            loss = ce(emo_pred, emo) + 0.5 * mse(inten_pred, inten)
            total_loss += loss.item()

            # Store predictions
            all_emo_preds.extend(emo_pred.argmax(dim=1).cpu().numpy())
            all_emo_true.extend(emo.cpu().numpy())
            all_inten_preds.extend(inten_pred.cpu().numpy())
            all_inten_true.extend(inten.cpu().numpy())

    avg_loss = total_loss / len(test_loader)

    # Convert to numpy
    all_emo_preds = np.array(all_emo_preds)
    all_emo_true = np.array(all_emo_true)
    all_inten_preds = np.array(all_inten_preds)
    all_inten_true = np.array(all_inten_true)

    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nAverage Loss: {avg_loss:.4f}")
    
    # Emotion classification metrics
    accuracy = accuracy_score(all_emo_true, all_emo_preds)
    f1_macro = f1_score(all_emo_true, all_emo_preds, average='macro')
    f1_weighted = f1_score(all_emo_true, all_emo_preds, average='weighted')
    
    print(f"\n--- Emotion Classification ---")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")

    # Intensity regression metrics
    mse_inten = np.mean((all_inten_true - all_inten_preds) ** 2)
    mae_inten = np.mean(np.abs(all_inten_true - all_inten_preds))
    
    print(f"\n--- Intensity Regression ---")
    print(f"MSE: {mse_inten:.4f}")
    print(f"MAE: {mae_inten:.4f}")

    # Detailed classification report
    print(f"\n--- Per-Class Performance ---")
    print(classification_report(all_emo_true, all_emo_preds, 
                                target_names=EMOTIONS, 
                                digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_emo_true, all_emo_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to results/confusion_matrix.png")
    
    # Save results
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'mse_intensity': mse_inten,
        'mae_intensity': mae_inten,
        'avg_loss': avg_loss
    }
    
    torch.save(results, 'results/evaluation_results.pth')
    print(f"✓ Results saved to results/evaluation_results.pth")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    evaluate()
