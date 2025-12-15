# MiniAffectGPT: Efficient Multimodal Emotion Recognition

**Cross-modal attention fusion with BLIP-generated captions for facial emotion recognition**

ECE 4990 Final Project - Cal Poly Pomona, December 2025  
Author: David Calvillo

## Results

| Dataset | Samples | Accuracy | F1 (Macro) | Training |
|---------|---------|----------|------------|----------|
| FER2013 | 6,929 | **63.60%** | 0.5824 | Trained |
| CK+ | 186 | **82.26%** | 0.5307 | Zero-shot transfer |

### Per-Class Performance (FER2013)

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Happy | 0.8709 | 0.8518 | 0.8612 |
| Surprise | 0.7144 | 0.7873 | 0.7491 |
| Neutral | 0.6162 | 0.5757 | 0.5953 |
| Angry | 0.6071 | 0.5256 | 0.5634 |
| Sad | 0.4565 | 0.6099 | 0.5222 |
| Fear | 0.4745 | 0.3738 | 0.4182 |
| Disgust | 0.4235 | 0.3243 | 0.3673 |

## Architecture
```
Input Image (224x224x3)
    ↓
ResNet18 (pretrained ImageNet) → 512-D features
    ↓                                ↓
    ↓                          Cross-Modal
    ↓                          Attention
    ↓                                ↑
BLIP Caption → DistilBERT (frozen) → 768-D features
    ↓
Fusion Layer (512-D)
    ↓
Emotion Classification (7 classes)
```

**Key Components:**
- **Image Encoder**: ResNet18 pretrained on ImageNet
- **Text Encoder**: DistilBERT-base-uncased (frozen)
- **Caption Generation**: BLIP (Salesforce/blip-image-captioning-base)
- **Fusion**: Cross-modal attention mechanism (main novelty)

## Reproduction Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Google Colab or local setup

### Step 1: Clone Repository
```bash
git clone https://github.com/DavidCalvillo831/MiniAffectGPT-Model.git
cd MiniAffectGPT-Model
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Datasets

**FER2013:**
1. Download from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
2. Extract to `/content/archive/` (or update paths in CSVs)
3. Should have structure: `archive/train/{emotion}/` and `archive/test/{emotion}/`

**CK+ (optional, for validation):**
1. Download from Kaggle: https://www.kaggle.com/datasets/shawon10/ckplus
2. Extract and run preprocessing (see below)

### Step 4: Generate BLIP Captions
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Load BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to('cuda')

# Generate captions for each image
def generate_caption(img_path):
    image = Image.open(img_path).convert('RGB')
    inputs = processor(image, return_tensors="pt").to('cuda')
    out = model.generate(**inputs, max_length=30)
    return processor.decode(out[0], skip_special_tokens=True)

# Process all splits (train, val, test)
for split in ['train', 'val', 'test']:
    # Load image paths and labels from FER2013
    # Generate captions
    # Save to data/fer_{split}.csv
```

See `data/sample_format.csv` for required CSV format.

### Step 5: Train Model
```bash
python train_attention.py
```

**Training parameters:**
- Epochs: 20
- Batch size: 32
- Learning rate: 5e-4
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)

**Expected training time:** ~40 minutes on T4 GPU

### Step 6: Evaluate
```bash
# FER2013 test set
python evaluate_attention.py

# CK+ zero-shot (after preparing CK+ CSVs)
python evaluate_ck.py
```

### Step 7: Predict on Custom Images
```bash
python predict.py /path/to/image.jpg "mouth wide open, eyes bright"
```

## File Structure
```
MiniAffectGPT-Model/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── train_attention.py           # Main training script
├── evaluate_attention.py        # Evaluation with metrics
├── predict.py                   # Single image prediction
├── models/
│   ├── attention_fusion.py     # Cross-modal attention (NOVELTY)
│   ├── image_model.py          # ResNet18 wrapper
│   └── text_model.py           # DistilBERT wrapper
└── data/
    └── sample_format.csv        # Example CSV format
```

## Technical Details

### Cross-Modal Attention Mechanism

Our main contribution is the attention-based fusion:
```python
# Image features as Query
Q = W_q @ img_features  # [batch, 512]

# Text features as Key and Value
K = W_k @ text_features  # [batch, 512]
V = W_v @ text_features  # [batch, 512]

# Compute attention
attention = softmax(Q @ K.T / sqrt(d_k))
attended_text = attention @ V

# Fuse with image features
fused = concat([img_features, attended_text])
```

This allows the model to adaptively weight textual features based on visual content.

### BLIP Caption Examples

Generated descriptions for facial expressions:
- Happy: "mouth corners turned up (smiling), eyes wide, eyebrows neutral"
- Angry: "mouth neutral or closed, eyes narrow, eyebrows furrowed"
- Surprise: "mouth wide open, eyes wide, eyebrows raised"

These descriptions enable multimodal learning without manual annotation.

## Results Analysis

**Why CK+ accuracy (82%) > FER2013 (64%)?**

This demonstrates that training on challenging in-the-wild data (FER2013) yields models that generalize exceptionally well to controlled lab conditions (CK+). The 18% improvement on CK+ reflects:
1. Cleaner images (lab-controlled vs in-the-wild)
2. Better lighting and image quality
3. Less occlusion and noise
4. Model learned robust, generalizable features

## Citation
```bibtex
@misc{calvillo2025miniaffectgpt,
  title={MiniAffectGPT: Efficient Multimodal Emotion Recognition with Cross-Modal Attention},
  author={Calvillo, David},
  year={2025},
  institution={California State Polytechnic University, Pomona},
  course={ECE 4990 - Deep Learning}
}
```

## Acknowledgments

- FER2013 dataset: Goodfellow et al.
- CK+ dataset: Lucey et al.
- BLIP model: Salesforce Research
- ResNet: He et al.
- DistilBERT: Sanh et al.
