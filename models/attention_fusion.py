import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionFusionModel(nn.Module):
    """
    Cross-Modal Attention Fusion for Emotion Recognition
    
    This is your MAIN NOVELTY!
    Image features act as Query, Text features provide Key/Value
    """
    def __init__(self, img_dim=512, txt_dim=768, hidden=512, num_classes=7):
        super().__init__()
        
        # Cross-modal attention
        self.query_proj = nn.Linear(img_dim, hidden)
        self.key_proj = nn.Linear(txt_dim, hidden)
        self.value_proj = nn.Linear(txt_dim, hidden)
        
        # Fusion
        self.fusion_fc = nn.Linear(img_dim + hidden, hidden)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
        # Output head - EMOTION ONLY (no intensity!)
        self.emotion_head = nn.Linear(hidden, num_classes)
    
    def forward(self, img_feat, txt_feat):
        batch_size = img_feat.size(0)
        
        # Attention mechanism
        Q = self.query_proj(img_feat)      # [B, hidden]
        K = self.key_proj(txt_feat)        # [B, hidden]
        V = self.value_proj(txt_feat)      # [B, hidden]
        
        # Compute attention scores for each sample in batch
        # Use unsqueeze to make dimensions compatible for batch matrix multiply
        attn_scores = torch.sum(Q * K, dim=1, keepdim=True)  # [B, 1]
        attn_scores = attn_scores / math.sqrt(Q.size(-1))
        attn_weights = F.softmax(attn_scores, dim=0)  # [B, 1]
        
        # Apply attention weights to values
        attended_text = attn_weights * V  # [B, hidden]
        
        # Fuse features
        fused = torch.cat([img_feat, attended_text], dim=1)  # [B, img_dim + hidden]
        x = self.relu(self.fusion_fc(fused))
        x = self.dropout(x)
        
        # Only emotion classification (NO INTENSITY)
        emotion_logits = self.emotion_head(x)
        
        return emotion_logits
