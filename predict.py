import torch
from PIL import Image
import torchvision.transforms as transforms
import sys

from models.image_model import ImageModel
from models.text_model import TextModel
from models.fusion_model import FusionModel

# Emotion labels for FER2013
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def predict(image_path, text, checkpoint_path="results/best.pth"):
    """
    Predict emotion and intensity from image and text
    
    Args:
        image_path: Path to image file
        text: Text description
        checkpoint_path: Path to model checkpoint
    
    Returns:
        emotion_name: Predicted emotion label
        emotion_idx: Emotion index
        intensity: Predicted intensity score
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Same transform as training (without augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Load and transform image
    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None

    # Initialize models
    image_encoder = ImageModel(pretrained=False).to(device)
    text_encoder = TextModel().to(device)
    fusion = FusionModel().to(device)

    # Load checkpoint
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        image_encoder.load_state_dict(ckpt["image"])
        text_encoder.load_state_dict(ckpt["text"])
        fusion.load_state_dict(ckpt["fusion"])
        print(f"✓ Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None

    # Set to evaluation mode
    image_encoder.eval()
    text_encoder.eval()
    fusion.eval()

    # Make prediction
    with torch.no_grad():
        img_feat = image_encoder(img)
        txt_feat = text_encoder([text])
        txt_feat = txt_feat.to(device)

        emo_logits, inten = fusion(img_feat, txt_feat)

    emotion_idx = emo_logits.argmax().item()
    emotion_name = EMOTIONS[emotion_idx]
    intensity = float(inten.item())

    return emotion_name, emotion_idx, intensity

if __name__ == "__main__":
    # Example usage - properly handle command line arguments
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
        text = sys.argv[2] if len(sys.argv) >= 3 else "A person showing emotion"
    else:
        print("Usage: python predict.py <image_path> [text_description]")
        print("Example: python predict.py data/image.jpg 'A happy person'")
        sys.exit(1)
    
    print(f"\nPredicting emotion for: {image_path}")
    print(f"Text: {text}\n")
    
    emotion_name, emotion_idx, intensity = predict(image_path, text)
    
    if emotion_name is not None:
        print("=" * 50)
        print(f"Predicted Emotion: {emotion_name} (class {emotion_idx})")
        print(f"Intensity Score: {intensity:.2f}")
        print("=" * 50)
    else:
        print("Prediction failed. Check errors above.")