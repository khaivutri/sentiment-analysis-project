import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class EnhancedBERTModel(nn.Module):
    """
    Enhanced model with attention and additional features
    """
    def __init__(self, num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = 768
        
        # Multi-head self-attention for better feature extraction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Deeper classifier with residual connections
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(hidden_size // 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 4)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use last hidden states for attention
        sequence_output = outputs.last_hidden_state
        
        # Apply self-attention
        attn_output, _ = self.attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Global average pooling
        masked_output = attn_output * attention_mask.unsqueeze(-1)
        sum_output = masked_output.sum(dim=1)
        lengths = attention_mask.sum(dim=1, keepdim=True).float()
        pooled = sum_output / lengths
        
        # Classification head with residual connections
        x = self.dropout(pooled)
        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x)
        
        return logits
