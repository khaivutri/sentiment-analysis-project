"""
Twitter Sentiment Analysis API
A FastAPI server for sentiment analysis using multiple model architectures
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time

# Import our custom modules
from utils import (
    predict_sentiment_enhanced_bert, 
    setup_device,
    load_tokenizer,
    load_enhanced_bert_model,
)

# ============== MODEL INITIALIZATION WITH GRACEFUL DEGRADATION ==============
# Setup device and tokenizer
print("🚀 Setting up device and tokenizer...")
device = setup_device()
tokenizer = load_tokenizer()
print("✅ Device and tokenizer setup complete")

# Initialize models dictionary
models = {}
failed_models = {}

# Model loader configuration
model_loaders = {
    "enhanced-bert": load_enhanced_bert_model
}

print("🚀 Loading models with graceful degradation...")
print("=" * 50)

# Try to load each model, continue on failures
for model_name, loader_func in model_loaders.items():
    try:
        print(f"📦 Loading {model_name} model...")
        start_time = time.time()
        
        models[model_name] = loader_func(device)
        
        load_time = time.time() - start_time
        print(f"✅ {model_name} loaded successfully! ({load_time:.2f}s)")
        
    except Exception as e:
        print(f"❌ {model_name} failed to load: {str(e)}")
        models[model_name] = None
        failed_models[model_name] = str(e)

# Summary of loaded models
loaded_count = len([m for m in models.values() if m is not None])
total_count = len(models)

print("=" * 50)
print(f"🎯 MODEL LOADING SUMMARY: {loaded_count}/{total_count} models loaded successfully")

if loaded_count > 0:
    available_models = [name for name, model in models.items() if model is not None]
    print(f"✅ Available models: {', '.join(available_models)}")
    
if failed_models:
    print(f"❌ Failed models: {', '.join(failed_models.keys())}")

print("🚀 Starting FastAPI server...")

# Sentiment mapping
SENTIMENT_LABELS = {
    0: "positive",
    1: "negative", 
    2: "neutral",
    3: "irrelevant"
}

# ============== FASTAPI APP SETUP ==============
app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="Enhanced BERT model for sentiment analysis with graceful degradation",
    version="2.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== REQUEST/RESPONSE MODELS ==============
class TextInput(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict
    processed_text: str
    inference_time: float
    model_used: str

class ModelStatusResponse(BaseModel):
    total_models: int
    loaded_models: int
    available_models: list
    failed_models: dict

# ============== API ENDPOINTS ==============
@app.get("/")
def read_root():
    """Root endpoint with API information"""
    available_models = [name for name, model in models.items() if model is not None]
    
    return {
        "message": "Twitter Sentiment Analysis API with Graceful Degradation", 
        "device": str(device),
        "status": "ready" if available_models else "degraded",
        "models_status": {
            "total": len(models),
            "loaded": len(available_models),
            "available": available_models,
            "failed": list(failed_models.keys()) if failed_models else []
        },
        "model_descriptions": {
            "enhanced-bert": "Enhanced BERT with Attention",
            "elmo-bert": "ELMo + BERT Combined Architecture",
            "elmo-transformer": "ELMo Transformer Alternative", 
            "5-embedding": "Five Embedding Architecture"
        },
        "endpoints": {
            "predict_any_model": "POST /predict/{model_name}",
            "predict_default": "POST /predict (Enhanced BERT if available)",
            "model_status": "GET /models/status",
            "available_models": available_models
        },
        "version": "2.1.0",
        "features": [
            "Graceful model loading degradation",
            "Model availability status",
            "Inference timing metrics",
            "Advanced error handling",
            "4-class sentiment classification",
            "Multiple model architectures",
            "RESTful API with path parameters"
        ]
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    available_count = len([m for m in models.values() if m is not None])
    return {
        "status": "healthy" if available_count > 0 else "unhealthy", 
        "models_loaded": available_count,
        "total_models": len(models),
        "device": str(device)
    }

@app.get("/models/status", response_model=ModelStatusResponse)
def get_models_status():
    """Get detailed status of all models"""
    available_models = [name for name, model in models.items() if model is not None]
    
    return ModelStatusResponse(
        total_models=len(models),
        loaded_models=len(available_models),
        available_models=available_models,
        failed_models=failed_models
    )

@app.post("/predict/{model_name}", response_model=SentimentResponse)
async def predict_sentiment(model_name: str, input_data: TextInput):
    """
    Predict sentiment using the specified model
    
    Args:
        model_name: Name of the model to use (enhanced-bert, elmo-bert, elmo-transformer, 5-embedding)
        input_data: TextInput containing the text to analyze
        
    Returns:
        SentimentResponse with prediction results
    """
    # Validate model name
    valid_models = ["enhanced-bert", "elmo-bert", "elmo-transformer", "5-embedding"]
    if model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model name. Available models: {', '.join(valid_models)}"
        )
    
    # Check if model is loaded
    if models.get(model_name) is None:
        available_models = [name for name, model in models.items() if model is not None]
        error_msg = f"Model '{model_name}' is not available."
        
        if model_name in failed_models:
            error_msg += f" Loading failed with: {failed_models[model_name]}"
        
        if available_models:
            error_msg += f" Try these available models: {', '.join(available_models)}"
        
        raise HTTPException(
            status_code=503,
            detail=error_msg
        )
    
    try:
        # Start timing
        start_time = time.time()
        
        # Route to appropriate prediction function
        if model_name == "enhanced-bert":
            raw_result = predict_sentiment_enhanced_bert(
                models[model_name], input_data.text, tokenizer, device
            )
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Map the raw class to sentiment labels
        predicted_class = raw_result["predicted_class"]
        sentiment = SENTIMENT_LABELS[predicted_class]
        
        # Return structured response with timing
        return SentimentResponse(
            text=input_data.text,
            sentiment=sentiment,
            confidence=round(raw_result["confidence"], 4),
            probabilities={
                "positive": round(raw_result["probabilities"]["class_0"], 4),
                "negative": round(raw_result["probabilities"]["class_1"], 4),
                "neutral": round(raw_result["probabilities"]["class_2"], 4),
                "irrelevant": round(raw_result["probabilities"]["class_3"], 4)
            },
            processed_text=raw_result["processed_text"],
            inference_time=round(inference_time, 4),
            model_used=model_name
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"{model_name} prediction error: {str(e)}"
        )

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment_default(input_data: TextInput):
    """
    Default prediction endpoint with fallback logic
    Priority: enhanced-bert > elmo-bert > elmo-transformer > 5-embedding
    """
    # Define fallback order (best to worst)
    fallback_order = ["enhanced-bert", "elmo-bert", "elmo-transformer", "5-embedding"]
    
    # Find first available model
    for model_name in fallback_order:
        if models.get(model_name) is not None:
            return await predict_sentiment(model_name, input_data)
    
    # No models available
    raise HTTPException(
        status_code=503,
        detail="No models are currently available. Please check /models/status for details."
    )

# ============== SERVER STARTUP ==============
if __name__ == "__main__":
    if len([m for m in models.values() if m is not None]) == 0:
        print("⚠️ WARNING: No models loaded successfully!")
        print("⚠️ API will start but all prediction requests will fail")
        print("⚠️ Check your model files and Google Drive connectivity")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8888,
        log_level="info"
    )