# Twitter Sentiment Analysis with Enhanced BERT

A complete end-to-end sentiment analysis system using Enhanced BERT model with FastAPI backend and React frontend. This project classifies tweets into 4 categories: Positive, Negative, Neutral, and Irrelevant.

## 🚀 Features

- **Enhanced BERT Model**: Custom BERT architecture with multi-head attention and deeper classification layers
- **ELMo+BERT Model**: Combined architecture using ELMo embeddings and BERT for enhanced performance
- **4-Class Classification**: Positive, Negative, Neutral, Irrelevant sentiment detection
- **Google Drive Integration**: Automatic model download from cloud storage
- **FastAPI Backend**: High-performance REST API with CORS support
- **React Frontend**: Modern, responsive user interface with real-time predictions
- **Docker Support**: Fully containerized deployment
- **Intelligent Preprocessing**: Advanced text preprocessing for social media content
- **Real-time Predictions**: Fast sentiment analysis with confidence scores
- **Multiple Model Architectures**: Support for both Enhanced BERT and ELMo+BERT models

## 📁 Project Structure

```
twitter_sentiment_bert_only/
├── backend/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── enhanced_bert.py          # Enhanced BERT model architecture
│   │   ├── elmo_bert.py              # ELMo+BERT combined model architecture
│   │   └── elmo_bert.ipynb           # Jupyter notebook with model development
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── drive_loader.py           # Google Drive API integration
│   │   ├── model_loader.py           # Model loading and initialization
│   │   ├── prediction.py             # Prediction pipeline
│   │   └── preprocessing.py          # Text preprocessing utilities
│   ├── server.py                     # FastAPI application
│   ├── requirements.txt              # Python dependencies
│   ├── Dockerfile                    # Docker configuration
│   ├── .env.example                  # Environment variables template
│   └── .gitignore                    # Git ignore rules
├── frontend/
│   ├── src/
│   │   ├── App.tsx                   # Main React sentiment analyzer UI
│   │   ├── components/ui/            # Reusable UI components
│   │   └── ...
│   ├── package.json                  # Node.js dependencies
│   ├── tailwind.config.ts            # Tailwind CSS configuration
│   └── ...
└── README.md
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.12+
- Node.js 18+
- Docker (optional)
- Google Drive API credentials

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/VoMinhKhoii/hcmut-project-cuoi-khoa.git
   cd hcmut-project-cuoi-khoa/backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Google Drive API**
   - Get Google Drive API credentials from Google Cloud Console
   - Create a `.env` file in the backend directory:
   ```env
   GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here
   REFRESH_TOKEN=your_refresh_token_here
   CLIENT_ID=your_client_id_here
   CLIENT_SECRET=your_client_secret_here
   ```

5. **Run the server**
   ```bash
   python server.py
   # Or with uvicorn
   uvicorn server:app --host 0.0.0.0 --port 8888
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd ../frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Access the application**
   - Frontend: http://localhost:8080
   - Backend API: http://localhost:8888
   - API Documentation: http://localhost:8888/docs

### Docker Setup (Recommended for Production)

1. **Build and run with Docker**
   ```bash
   cd backend
   docker build -t sentiment-api .
   docker run -p 8888:8888 sentiment-api
   ```

2. **Run frontend separately**
   ```bash
   cd frontend
   npm run dev
   ```

## 🔧 API Usage

### Predict Sentiment (Enhanced BERT)

**POST** `/predict`

```json
{
  "text": "I love this product! It works amazing!"
}
```

**Response:**
```json
{
  "text": "I love this product! It works amazing!",
  "sentiment": "positive",
  "confidence": 0.9245,
  "probabilities": {
    "positive": 0.9245,
    "negative": 0.0123,
    "neutral": 0.0532,
    "irrelevant": 0.0100
  },
  "processed_text": "i love this product it works amazing"
}
```

### Predict Sentiment (ELMo+BERT)

**POST** `/predict-elmo-bert`

Uses the combined ELMo+BERT architecture for potentially enhanced performance:

```json
{
  "text": "This movie is absolutely fantastic!"
}
```

**Response:** Same format as above, but uses the ELMo+BERT model for prediction.

### Health Check

**GET** `/`

Returns API status and model information.

## 🧠 Model Architecture

### Enhanced BERT Model

The Enhanced BERT model includes:

- **Base BERT**: `bert-base-uncased` as the foundation
- **Multi-head Attention**: 8-head self-attention mechanism
- **Deep Classification**: 3-layer fully connected network with residual connections
- **Layer Normalization**: Batch normalization for stable training
- **Dropout**: Regularization to prevent overfitting

### ELMo+BERT Combined Model

The ELMo+BERT model (DICET architecture) combines:

- **ELMo Embeddings**: Contextualized word representations from TensorFlow Hub
- **BERT Representations**: Pre-trained BERT embeddings
- **Feature Fusion**: Concatenation of ELMo (1024-dim) and BERT (768-dim) features
- **BiLSTM Encoder**: Bidirectional LSTM for sequence modeling (256 hidden units)
- **Additive Attention**: Attention mechanism for feature weighting
- **Classification Head**: Fully connected layers with dropout and ReLU activation
- **Gaussian Noise**: Regularization during training (σ=0.3)

## 📊 Performance

- **Accuracy**: ~85-90% on validation set
- **Classes**: 4-class classification (Positive, Negative, Neutral, Irrelevant)
- **Inference Speed**: <100ms per prediction
- **Model Size**: ~110MB (BERT base)

## 🔄 Text Preprocessing Pipeline

1. URL replacement with tokens
2. Mention handling (@user → @USER)
3. Hashtag extraction and processing
4. Emoji conversion to descriptive text
5. Punctuation normalization
6. Contraction expansion
7. Case normalization with emphasis detection

## 🌐 Frontend Features

- **Real-time Analysis**: Instant sentiment prediction
- **Interactive UI**: Clean, modern interface built with React and Tailwind CSS
- **Confidence Visualization**: Progress bars showing prediction confidence
- **Example Texts**: Pre-loaded examples for quick testing
- **Error Handling**: Graceful error messages and loading states
- **Responsive Design**: Works on desktop and mobile devices

## 📝 Environment Variables

Create a `.env` file in the backend directory:

```env
# Google Drive Configuration
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_containing_model
REFRESH_TOKEN=your_google_refresh_token
CLIENT_ID=your_google_client_id
CLIENT_SECRET=your_google_client_secret
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build the image
docker build -t sentiment-api ./backend

# Run the container
docker run -p 8888:8888 --env-file ./backend/.env sentiment-api
```

### Cloud Deployment

The application is ready for deployment on:
- Google Cloud Run
- AWS ECS
- Azure Container Instances
- Heroku

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Vo Minh Khoi** - Initial work - [VoMinhKhoii](https://github.com/VoMinhKhoii)

## 🙏 Acknowledgments

- HCMUT for the project opportunity
- Hugging Face for the BERT model
- FastAPI and React communities for excellent frameworks
- Google Drive API for model storage solution

## 📧 Contact

For questions or support, please contact:
- Email: your.email@example.com
- GitHub: [@VoMinhKhoii](https://github.com/VoMinhKhoii)

---

**Built with ❤️ at Ho Chi Minh City University of Technology**
