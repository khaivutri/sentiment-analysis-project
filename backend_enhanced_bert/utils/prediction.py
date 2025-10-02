import torch
import torch.nn.functional as F
from .preprocessing import intelligent_preprocess 
from transformers import AutoTokenizer
import logging
import stanza

# Global variables for expensive initialization
_stanza_nlp = None
_pos_vocab = None


def predict_sentiment_enhanced_bert(model, text, tokenizer, device, max_length=128):
    """Internal prediction function"""
    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
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
        "processed_text": text
    }

