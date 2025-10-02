import torch
import torch.nn.functional as F
from .preprocessing import intelligent_preprocess  # ✅ Use the actual function
from transformers import AutoTokenizer
import logging
import stanza

# Global variables for expensive initialization
_stanza_nlp = None
_pos_vocab = None


def predict_sentiment_elmo_bert(model, text, tokenizer, device, max_length=64):
    """
    Prediction function for ELMo+BERT model
    
    Args:
        model: ELMo+BERT model instance
        text: Input text to analyze
        tokenizer: BERT tokenizer
        device: Device to run inference on
        max_length: Maximum sequence length (default: 64 for ELMo+BERT)
        
    Returns:
        Dictionary containing prediction results
    """
    model.eval()

    processed_text = intelligent_preprocess(text)  # ✅ Fixed

    # Prepare input in the format expected by ELMo+BERT model
    encoding = tokenizer(
        [processed_text],  # Note: needs to be a list for batch processing
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create batch dictionary as expected by DICETModel
    batch = {
        'sentences': [processed_text],  # ELMo needs the raw sentences
        'input_ids': encoding['input_ids'].to(device),
        'attention_mask': encoding['attention_mask'].to(device),
        'token_type_ids': encoding['token_type_ids'].to(device)
    }
    
    with torch.no_grad():
        outputs = model(batch)
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
    }