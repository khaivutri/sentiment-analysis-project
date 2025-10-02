"""
ELMo Transformer Model Implementation
Alternative approach using transformer-based embeddings instead of traditional ELMo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer, AutoModel


class TransformerELMoAlternative:
    """
    Transformer-based alternative to ELMo embeddings
    """
    def __init__(self):
        # Move the tokenizer and model initialization here
        model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def get_embeddings(self, sentences, device):
        """
        Get embeddings for a list of sentences
        
        Args:
            sentences: List of input sentences
            device: Target device (CPU/GPU)
            
        Returns:
            torch.Tensor of embeddings
        """
        # Use its own tokenizer
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            self.model.to(device)  # Ensure the model is on the correct device
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
        return embeddings


class AdditiveAttention(nn.Module):
    """
    Additive attention mechanism
    """
    def __init__(self, hidden_dim: int, attn_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply additive attention
        
        Args:
            H: Hidden states [B, T, D_hidden]
            mask: Optional attention mask
            
        Returns:
            Tuple of (context vector, attention weights)
        """
        # H: [B, T, D_hidden]
        scores = self.v(torch.tanh(self.proj(H))).squeeze(-1)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        alpha = F.softmax(scores, dim=-1)  # [B, T]
        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # [B, D_hidden]
        return context, alpha


class GaussianNoise(nn.Module):
    """
    Gaussian noise layer for regularization
    """
    def __init__(self, sigma=0.3):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma != 0:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


class ELMoTransformerModel(nn.Module):
    """
    ELMo Transformer-based sentiment classification model
    """
    def __init__(
        self,
        num_classes: int,
        elmo_dim: int = 1024,
        lstm_hidden: int = 150,
        lstm_layers: int = 1,
        dropout: float = 0.25
    ):
        super().__init__()
        # Instantiate the new self-contained class
        self.transformer_elmo = TransformerELMoAlternative()

        # BiLSTM layer
        self.encoder = nn.LSTM(
            input_size=elmo_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Attention layer
        self.attn = AdditiveAttention(hidden_dim=2 * lstm_hidden)

        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 150),
            nn.ReLU(),
            GaussianNoise(sigma=0.3),
            nn.Dropout(dropout),
            nn.Linear(150, num_classes)
        )

    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the ELMo Transformer model
        
        Args:
            texts: List of input text strings
            
        Returns:
            Dictionary containing:
                - logits: Classification logits
                - attn: Attention weights
        """
        # Get ELMo embeddings using the new class
        # Get the device from the model itself to ensure consistency
        device = next(self.parameters()).device
        elmo_reps = self.transformer_elmo.get_embeddings(texts, device)

        # Create a mask based on the ELMo embeddings (assuming padding tokens have zero vectors)
        # This is a simple and common way to create a mask for a single embedding model
        mask = (elmo_reps.sum(dim=-1) != 0).long()

        # Pass through BiLSTM
        H, _ = self.encoder(elmo_reps)

        # Apply Attention
        context, alpha = self.attn(H, mask=mask)

        # Classify
        logits = self.classifier(context)

        return {"logits": logits, "attn": alpha}
