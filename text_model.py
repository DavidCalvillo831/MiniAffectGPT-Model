import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class TextModel(nn.Module):
    """
    DistilBERT wrapper generating 768-D text embedding.
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

    def forward(self, texts):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        encoded = {k: v.to(self.bert.device) for k, v in encoded.items()}
        out = self.bert(**encoded)
        return out.last_hidden_state[:, 0, :]  # CLS embedding
