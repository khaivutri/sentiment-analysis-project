import React, { useState } from 'react';
import { AlertCircle, Send, Loader2, TrendingUp, TrendingDown, Minus, HelpCircle } from 'lucide-react';

// TypeScript interfaces
interface SentimentResult {
  text: string;
  processed_text: string;
  predicted_sentiment: 'Positive' | 'Negative' | 'Neutral' | 'Irrelevant';
  confidence: number;
  all_scores: {
    Positive: number;
    Negative: number;
    Neutral: number;
    Irrelevant: number;
  };
}

interface ApiError {
  error: string;
}

// API configuration
const API_URL = 'http://localhost:5000';

const SentimentAnalyzer: React.FC = () => {
  const [inputText, setInputText] = useState<string>('');
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');
  const [examples] = useState<string[]>([
    "I absolutely love this product! Best purchase ever! 🎉",
    "This is terrible. Complete waste of money. Never buying again.",
    "It's okay, nothing special but does the job.",
    "The weather is nice today"
  ]);

  // Analyze sentiment
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
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: textToAnalyze }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: SentimentResult | ApiError = await response.json();
      
      if ('error' in data) {
        throw new Error(data.error);
      }

      setResult(data as SentimentResult);
      if (text) {
        setInputText(text);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze sentiment');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Get sentiment icon
  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'Positive':
        return <TrendingUp className="w-6 h-6 text-sentiment-positive" />;
      case 'Negative':
        return <TrendingDown className="w-6 h-6 text-sentiment-negative" />;
      case 'Neutral':
        return <Minus className="w-6 h-6 text-sentiment-neutral" />;
      case 'Irrelevant':
        return <HelpCircle className="w-6 h-6 text-sentiment-irrelevant" />;
      default:
        return null;
    }
  };

  // Get sentiment CSS class
  const getSentimentClass = (sentiment: string) => {
    switch (sentiment) {
      case 'Positive':
        return 'sentiment-card sentiment-positive';
      case 'Negative':
        return 'sentiment-card sentiment-negative';
      case 'Neutral':
        return 'sentiment-card sentiment-neutral';
      case 'Irrelevant':
        return 'sentiment-card sentiment-irrelevant';
      default:
        return 'sentiment-card';
    }
  };

  // Get progress bar class
  const getProgressBarClass = (sentiment: string) => {
    switch (sentiment) {
      case 'Positive':
        return 'progress-bar-positive';
      case 'Negative':
        return 'progress-bar-negative';
      case 'Neutral':
        return 'progress-bar-neutral';
      case 'Irrelevant':
        return 'progress-bar-irrelevant';
      default:
        return 'bg-muted';
    }
  };

  // Format confidence percentage
  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className="min-h-screen p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <header className="text-center mb-8 pt-8">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            🎭 Sentiment Analyzer
          </h1>
          <p className="text-muted-foreground text-lg">
            Powered by Enhanced BERT Model
          </p>
        </header>

        {/* Main Card */}
        <main className="bg-card rounded-xl shadow-lg p-6 mb-6 border border-border">
          {/* Input Section */}
          <section className="mb-6">
            <label className="block text-sm font-medium text-foreground mb-2">
              Enter text to analyze:
            </label>
            <div className="relative">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full p-3 border border-input rounded-lg focus:ring-2 focus:ring-primary focus:border-primary min-h-[120px] resize-none bg-background text-foreground placeholder:text-muted-foreground"
                placeholder="Type or paste your text here..."
                disabled={loading}
              />
              <button
                onClick={() => analyzeSentiment()}
                disabled={loading || !inputText.trim()}
                className="absolute bottom-3 right-3 bg-primary text-primary-foreground rounded-lg px-4 py-2 hover:bg-primary-hover disabled:bg-muted disabled:text-muted-foreground disabled:cursor-not-allowed transition-colors flex items-center gap-2"
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
          </section>

          {/* Example Buttons */}
          <section className="mb-6">
            <p className="text-sm text-muted-foreground mb-2">Try an example:</p>
            <div className="flex flex-wrap gap-2">
              {examples.map((example, idx) => (
                <button
                  key={idx}
                  onClick={() => analyzeSentiment(example)}
                  className="text-xs bg-muted hover:bg-accent text-muted-foreground hover:text-accent-foreground px-3 py-1.5 rounded-full transition-colors disabled:opacity-50"
                  disabled={loading}
                >
                  {example.substring(0, 30)}...
                </button>
              ))}
            </div>
          </section>

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-destructive/10 border border-destructive/20 rounded-lg flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-destructive mt-0.5" />
              <div>
                <p className="text-destructive font-medium">Error</p>
                <p className="text-destructive/80 text-sm">{error}</p>
              </div>
            </div>
          )}

          {/* Results Section */}
          {result && (
            <div className="space-y-4 animate-in fade-in duration-500">
              {/* Main Result */}
              <div className="border-t border-border pt-4">
                <h3 className="text-lg font-semibold mb-3 text-foreground">Analysis Result</h3>
                
                <div className={getSentimentClass(result.predicted_sentiment)}>
                  {getSentimentIcon(result.predicted_sentiment)}
                  <span>{result.predicted_sentiment}</span>
                  <span className="text-sm opacity-75">
                    ({formatPercentage(result.confidence)} confident)
                  </span>
                </div>
              </div>

              {/* Confidence Scores */}
              <div>
                <h4 className="text-sm font-medium text-foreground mb-2">All Scores:</h4>
                <div className="space-y-2">
                  {Object.entries(result.all_scores).map(([sentiment, score]) => (
                    <div key={sentiment} className="flex items-center gap-2">
                      <span className="w-20 text-sm text-muted-foreground">{sentiment}:</span>
                      <div className="flex-1 bg-muted rounded-full h-6 relative">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${getProgressBarClass(sentiment)}`}
                          style={{ width: `${score * 100}%` }}
                        />
                        <span className="absolute right-2 top-0 h-full flex items-center text-xs text-foreground">
                          {formatPercentage(score)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Processed Text (Debug) */}
              <details className="border-t border-border pt-3">
                <summary className="text-sm text-muted-foreground cursor-pointer hover:text-foreground">
                  View processed text (debug)
                </summary>
                <p className="mt-2 text-xs font-mono bg-muted p-2 rounded text-muted-foreground">
                  {result.processed_text}
                </p>
              </details>
            </div>
          )}
        </main>

        {/* Info Card */}
        <aside className="bg-card-glass backdrop-blur-sm rounded-lg p-4 text-sm text-muted-foreground border border-border/50">
          <p className="font-semibold mb-1 text-foreground">ℹ️ About this model:</p>
          <ul className="space-y-1 text-xs">
            <li>• Enhanced BERT model trained on Twitter sentiment data</li>
            <li>• Classifies text into: Positive, Negative, Neutral, or Irrelevant</li>
            <li>• Handles emojis, hashtags, and informal language</li>
            <li>• Best with English text under 280 characters</li>
          </ul>
        </aside>
      </div>
    </div>
  );
};

export default SentimentAnalyzer;