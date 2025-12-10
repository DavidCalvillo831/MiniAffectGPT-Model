import torch
from PIL import Image
import torchvision.transforms as transforms

from models.image_model import ImageModel
from models.text_model import TextModel
from models.fusion_model import FusionModel

def predict(image_path, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    image_encoder = ImageModel(pretrained=False).to(device)
    text_encoder = TextModel().to(device)
    fusion = FusionModel().to(device)

    ckpt = torch.load("results/best.pth", map_location=device)
    image_encoder.load_state_dict(ckpt["image"])
    text_encoder.load_state_dict(ckpt["text"])
    fusion.load_state_dict(ckpt["fusion"])

    img_feat = image_encoder(img)
    txt_feat = text_encoder([text])

    emo, inten = fusion(img_feat, txt_feat)

    return emo.argmax().item(), float(inten.item())

if __name__ == "__main__":
    emo, intensity = predict("data/FER2013/images/img_0.jpg", "I feel great today!")
    print("Emotion:", emo)
    print("Intensity:", intensity)
