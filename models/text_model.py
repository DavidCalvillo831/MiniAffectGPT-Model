import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

class TextModel(nn.Module):
    """
    DistilBERT wrapper generating 768-D text embedding.
    """
    def __init__(self):
        super().__init__()
        # using DistilBert to use smaller version and enhance speed, still effective model   *** notes for paper
        
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # NOTE: we keep this trainable (not frozen) in this version
        # tried both frozen and trainable, both work pretty well
    

    def forward(self, texts):
        # tokenize the input captions from BLIP
        # padding makes all sequences same length
        # truncation cuts off anything too long
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        # move tensors to same device as model
        encoded = {k: v.to(self.bert.device) for k, v in encoded.items()}
        # forces a run thru bert 
        out = self.bert(**encoded)
    
        return out.last_hidden_state[:, 0, :]  # CLS embedding

