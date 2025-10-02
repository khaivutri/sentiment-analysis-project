import torch
import torch.nn.functional as F
from .preprocessing import intelligent_preprocess  # ✅ Use the actual function
from transformers import AutoTokenizer
import logging

# Global variables for expensive initialization
_stanza_nlp = None
_pos_vocab = None

def predict_sentiment_elmo_transformer(model, text, device):
    """
    Prediction function for ELMo Transformer model
    
    Args:
        model: ELMo Transformer model instance
        text: Input text to analyze
        device: Device to run inference on
        
    Returns:
        Dictionary containing prediction results
    """
    model.eval()

    processed_text = intelligent_preprocess(text)  # ✅ Fixed

    with torch.no_grad():
        # ELMo Transformer model expects a list of texts
        outputs = model([processed_text])
        logits = outputs["logits"]
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "probabilities": {
            "class_0": float(probabilities[0][0]),
            "class_1": float(probabilities[0][1]),
            "class_2": float(probabilities[0][2]),
            "class_3": float(probabilities[0][3])
        },
        "input_text": text,
        "processed_text": processed_text,
        "attention_weights": outputs.get("attn", None)
    }

