import torch
from transformers import BertTokenizer
from models import ELMoTransformerModel
from utils import DriveModelLoader


def setup_device():
    """Setup and return the best available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("📱 Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")
    return device


def load_elmo_transformer_model(device):
    """Load ELMo Transformer model from Google Drive"""
    print("🏗️ Initializing ELMo Transformer model architecture...")
    model = ELMoTransformerModel(num_classes=4)
    
    print("📦 Loading ELMo Transformer model weights from Google Drive...")
    try:
        drive_loader = DriveModelLoader()
        model_path = drive_loader.download_elmo_transformer_model()
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ ELMo Transformer model weights loaded successfully from {model_path}")
        
    except Exception as e:
        print(f"❌ Error loading ELMo Transformer model: {str(e)}")
        raise
    
    model = model.to(device)
    model.eval()
    print(f"🎯 ELMo Transformer model ready on {device}")
    
    return model


