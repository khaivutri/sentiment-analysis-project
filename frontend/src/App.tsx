import React, { useState } from 'react';
import { AlertCircle, Send, Loader2, TrendingUp, TrendingDown, Minus, HelpCircle, ChevronDown, Check } from 'lucide-react';

// TypeScript interfaces
interface SentimentResult {
  text: string;
  sentiment: string;
  confidence: number;
  probabilities: {
    positive: number;
    negative: number;
    neutral: number;
    irrelevant: number;
  };
  processed_text: string;
  inference_time?: number;
  model_used?: string;
}

interface ApiError {
  error: string;
  detail?: string;
}

// API configuration - Different base URLs for different models
const MODEL_APIS = {
  'enhanced-bert': 'http://localhost:8888',
  'elmo-bert': 'http://localhost:8889',
  'elmo-transformer': 'http://localhost:8890',
  '5-embedding': 'https://96b8d52563f2.ngrok-free.app'  // ✅ Special ngrok URL
};

// Model descriptions
const MODEL_INFO = {
  'enhanced-bert': {
    name: 'Enhanced BERT',
    description: 'BERT with attention mechanism',
    color: 'blue',
    port: 8888
  },
  'elmo-bert': {
    name: 'ELMo + BERT',
    description: 'Combined ELMo and BERT embeddings',
    color: 'purple',
    port: 8889
  },
  'elmo-transformer': {
    name: 'ELMo Transformer',
    description: 'Transformer with ELMo embeddings',
    color: 'indigo',
    port: 8890
  },
  '5-embedding': {
    name: '5-Embedding',
    description: 'Multi-embedding architecture',
    color: 'green',
    port: 'ngrok'  // ✅ Special indicator for ngrok
  }
};

const SentimentAnalyzer: React.FC = () => {
  const [inputText, setInputText] = useState<string>('');
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<string>('enhanced-bert');
  const [dropdownOpen, setDropdownOpen] = useState<boolean>(false);
  
  // All models are available from the start
  const availableModels = ['enhanced-bert', 'elmo-bert', 'elmo-transformer', '5-embedding'];
  
  const [examples] = useState<string[]>([
    "Not gonna lie, didn't think I wouldn't dislike Borderlands this much.",
    "Oh great, another Facebook update that nobody asked for. Just what we needed!",
    "@Meta fix your platform #FacebookDown #Frustrated #TechFail",
    "Twitter's new feature rollout includes: 1) Extended video uploads 2) Better threading UI 3) Improved search functionality. Some users like it, others don't. Time will tell if these changes stick."
  ]);

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownOpen && !(event.target as Element).closest('.dropdown-container')) {
        setDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [dropdownOpen]);

  // Get API URL for the selected model
  const getApiUrl = (modelName: string) => {
    return MODEL_APIS[modelName as keyof typeof MODEL_APIS] || MODEL_APIS['enhanced-bert'];
  };

  // Get the complete API endpoint URL
  const getApiEndpoint = (modelName: string) => {
    const baseUrl = getApiUrl(modelName);
    
    // Special case for 5-embedding: uses direct /predict endpoint
    if (modelName === '5-embedding') {
      return `${baseUrl}/predict`;
    }
    
    // For other models: use /predict/{modelName} pattern
    return `${baseUrl}/predict/${modelName}`;
  };

  // Add this function to handle both string and number sentiment responses
  const normalizeSentiment = (result: any) => {
    // First check if sentiment field exists (Enhanced BERT format)
    if (result.sentiment && typeof result.sentiment === 'string') {
      return result.sentiment.toLowerCase();
    }
    
    // Then check predicted_class field (5-Embedding format)
    if (result.predicted_class) {
      if (typeof result.predicted_class === 'string') {
        return result.predicted_class.toLowerCase(); // Convert "Neutral" -> "neutral"
      }
      
      // Fallback: if predicted_class is a number, map it
      if (typeof result.predicted_class === 'number') {
        const SENTIMENT_LABELS = {
          0: "positive",
          1: "negative", 
          2: "neutral",
          3: "irrelevant"
        };
        return SENTIMENT_LABELS[result.predicted_class as keyof typeof SENTIMENT_LABELS] || "unknown";
      }
    }
    
    return "unknown";
  };

  // Analyze sentiment with dynamic API URL
  const analyzeSentiment = async (text?: string) => {
    const textToAnalyze = text || inputText;
    
    if (!textToAnalyze.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      // Get the complete API endpoint for the selected model
      const apiEndpoint = getApiEndpoint(selectedModel);
      
      console.log(`🎯 Making request to: ${apiEndpoint}`);
      console.log(`📡 Model: ${selectedModel} on ${MODEL_INFO[selectedModel as keyof typeof MODEL_INFO].port}`);

      // Special headers for ngrok
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      };

      // Add ngrok bypass header for 5-embedding model
      if (selectedModel === '5-embedding') {
        headers['ngrok-skip-browser-warning'] = 'true';
      }

      const response = await fetch(apiEndpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify({ text: textToAnalyze }),
      });

      const data = await response.json();
      console.log(data);
      
      if (!response.ok) {
        // Handle different error formats
        const errorMessage = data.detail || data.error || `HTTP error! status: ${response.status}`;
        throw new Error(errorMessage);
      }

      if ('error' in data) {
        throw new Error(data.error);
      }

      // Normalize the sentiment regardless of API response format
      const normalizedSentiment = normalizeSentiment(data);

      // Add model information to the result
      const resultWithMeta = {
        ...data,
        sentiment: normalizedSentiment,  // ✅ Add normalized sentiment
        model_used: selectedModel,
        api_url: getApiUrl(selectedModel)
      };

      setResult(resultWithMeta as SentimentResult);
      if (text) {
        setInputText(text);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to analyze sentiment';
      console.error(`❌ Error with ${selectedModel}:`, err);
      
      // Add helpful error context
      const portInfo = selectedModel === '5-embedding' ? 'ngrok tunnel' : `port ${MODEL_INFO[selectedModel as keyof typeof MODEL_INFO].port}`;
      setError(`${errorMessage} (Model: ${MODEL_INFO[selectedModel as keyof typeof MODEL_INFO].name} on ${portInfo})`);
    } finally {
      setLoading(false);
    }
  };

  // Get sentiment icon
  const getSentimentIcon = (sentiment: string) => {
    const sentimentLower = sentiment.toLowerCase();
    switch (sentimentLower) {
      case 'positive':
        return <TrendingUp className="w-6 h-6 text-green-500" />;
      case 'negative':
        return <TrendingDown className="w-6 h-6 text-red-500" />;
      case 'neutral':
        return <Minus className="w-6 h-6 text-gray-500" />;
      case 'irrelevant':
        return <HelpCircle className="w-6 h-6 text-blue-500" />;
      default:
        return null;
    }
  };

  // Get sentiment color
  const getSentimentColor = (sentiment: string) => {
    const sentimentLower = sentiment.toLowerCase();
    switch (sentimentLower) {
      case 'positive':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'negative':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'neutral':
        return 'bg-gray-100 text-gray-800 border-gray-300';
      case 'irrelevant':
        return 'bg-blue-100 text-blue-800 border-blue-300';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Format confidence percentage
  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  // Capitalize first letter
  const capitalizeFirst = (str: string) => {
    return str.charAt(0).toUpperCase() + str.slice(1);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 pt-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Multi-Model Sentiment Analyzer
          </h1>
          <p className="text-gray-600">
            Compare 4 different AI architectures
          </p>
          <p className="text-sm text-gray-500 mt-2">
            All {availableModels.length} models available for testing
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          {/* Input Section */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Enter text to analyze:
            </label>
            <div className="relative">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 min-h-[120px] resize-none"
                placeholder="Type or paste your text here..."
                disabled={loading}
              />
              <div className="absolute bottom-3 right-3 flex gap-2">
                {/* Model Selector Dropdown */}
                <div className="dropdown-container relative">
                  <button
                    onClick={() => setDropdownOpen(!dropdownOpen)}
                    className="bg-gray-100 text-gray-700 rounded-lg px-3 py-2 hover:bg-gray-200 transition-colors flex items-center gap-2 text-sm font-medium"
                    disabled={loading}
                  >
                    {MODEL_INFO[selectedModel as keyof typeof MODEL_INFO]?.name || selectedModel}
                    {/* <span className="text-xs text-gray-500">
                      :{MODEL_INFO[selectedModel as keyof typeof MODEL_INFO]?.port}
                    </span> */}
                    <ChevronDown className={`w-4 h-4 transition-transform ${dropdownOpen ? 'rotate-180' : ''}`} />
                  </button>
                  
                  {dropdownOpen && (
                    <div className="absolute bottom-full mb-2 right-0 w-96 bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden">
                      <div className="py-1">
                        {availableModels.map((model) => (
                          <button
                            key={model}
                            onClick={() => {
                              setSelectedModel(model);
                              setDropdownOpen(false);
                            }}
                            className={`w-full px-4 py-3 text-left hover:bg-gray-50 flex items-start gap-3 ${
                              selectedModel === model ? 'bg-blue-50' : ''
                            }`}
                          >
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-sm">
                                  {MODEL_INFO[model as keyof typeof MODEL_INFO]?.name || model}
                                </span>
                                {/* <span className="text-xs bg-gray-200 px-2 py-1 rounded">
                                  :{MODEL_INFO[model as keyof typeof MODEL_INFO]?.port}
                                </span> */}
                                {selectedModel === model && (
                                  <Check className="w-4 h-4 text-blue-500" />
                                )}
                              </div>
                              <span className="text-xs text-gray-500">
                                {MODEL_INFO[model as keyof typeof MODEL_INFO]?.description}
                              </span>
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Analyze Button */}
                <button
                  onClick={() => analyzeSentiment()}
                  disabled={loading || !inputText.trim()}
                  className="bg-blue-500 text-white rounded-lg px-4 py-2 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      Analyze
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Example Buttons */}
          <div className="mb-6">
            <p className="text-sm text-gray-600 mb-2">Try an example:</p>
            <div className="flex flex-wrap gap-2">
              {examples.map((example, idx) => (
                <button
                  key={idx}
                  onClick={() => analyzeSentiment(example)}
                  className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-1.5 rounded-full transition-colors"
                  disabled={loading}
                >
                  {example.substring(0, 30)}...
                </button>
              ))}
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
              <div>
                <p className="text-red-800 font-medium">Error</p>
                <p className="text-red-600 text-sm">{error}</p>
              </div>
            </div>
          )}

          {/* Results Section */}
          {result && (
            <div className="space-y-4 animate-in fade-in duration-500">
              {/* Main Result */}
              <div className="border-t pt-4">
                <h3 className="text-lg font-semibold mb-3">Analysis Result</h3>
                
                <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg border-2 ${getSentimentColor(result.sentiment)}`}>
                  {getSentimentIcon(result.sentiment)}
                  <span className="font-bold text-lg">{capitalizeFirst(result.sentiment)}</span>
                  <span className="text-sm opacity-75">
                    ({formatPercentage(result.confidence)} confident)
                  </span>
                </div>

                {result.model_used && (
                  <p className="text-xs text-gray-500 mt-2">
                    Analyzed by {MODEL_INFO[result.model_used as keyof typeof MODEL_INFO]?.name || result.model_used} 
                    {result.inference_time && ` in ${result.inference_time.toFixed(2)}s`}
                    {' '}({result.model_used === '5-embedding' ? 'ngrok tunnel' : `Port: ${MODEL_INFO[result.model_used as keyof typeof MODEL_INFO]?.port}`})
                  </p>
                )}
              </div>

              {/* Confidence Scores */}
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">All Scores:</h4>
                <div className="space-y-2">
                  {result.probabilities && Object.entries(result.probabilities).map(([classKey, score]) => {
                    // Map class_X to sentiment names
                    const sentimentMap: { [key: string]: string } = {
                      'class_0': 'positive',
                      'class_1': 'negative', 
                      'class_2': 'neutral',
                      'class_3': 'irrelevant'
                    };
                    const sentiment = sentimentMap[classKey] || classKey;
                    
                    return (
                      <div key={classKey} className="flex items-center gap-2">
                        <span className="w-20 text-sm text-gray-600">{capitalizeFirst(sentiment)}:</span>
                        <div className="flex-1 bg-gray-200 rounded-full h-6 relative">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              sentiment === 'positive' ? 'bg-green-500' :
                              sentiment === 'negative' ? 'bg-red-500' :
                              sentiment === 'neutral' ? 'bg-gray-500' :
                              'bg-blue-500'
                            }`}
                            style={{ width: `${(score as number) * 100}%` }}
                          />
                          <span className="absolute right-2 top-0 h-full flex items-center text-xs text-gray-700">
                            {formatPercentage(score as number)}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Processed Text (Debug) */}
              <details className="border-t pt-3">
                <summary className="text-sm text-gray-500 cursor-pointer hover:text-gray-700">
                  View processed text
                </summary>
                <div className="mt-2 space-y-2">
                  <div>
                    <p className="text-xs font-mono bg-gray-100 p-2 rounded">
                      {result.processed_text}
                    </p>
                  </div>
                </div>
              </details>
            </div>
          )}
        </div>

        {/* Info Card */}
        <div className="bg-white/50 rounded-lg p-4 text-sm text-gray-600">
          <p className="font-semibold mb-1">🚀 Model Architecture Comparison:</p>
          <ul className="space-y-1 text-s">
            <li>• <strong>Enhanced BERT:</strong> BERT with attention mechanism - balanced speed/accuracy</li>
            <li>• <strong>ELMo + BERT:</strong> Combines contextual ELMo with BERT - high context awareness</li>
            <li>• <strong>ELMo Transformer:</strong> Alternative transformer with ELMo - experimental architecture</li>
            <li>• <strong>5-Embedding:</strong> BERT + ELMo + GloVe + POS + Character - most comprehensive</li>
            <li className="pt-1 text-gray-500">🎯 Most models use: baseUrl/predict/modelName</li>
            <li className="text-gray-500">🌐 5-Embedding uses: ngrok-url/predict (special deployment)</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default SentimentAnalyzer;