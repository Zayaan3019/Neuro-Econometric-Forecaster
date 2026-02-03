"""
Sentiment Analysis Module using FinBERT.

This module wraps the FinBERT model (fine-tuned BERT for financial sentiment)
to convert news headlines into quantitative sentiment scores for the hybrid model.
"""

from typing import List, Dict, Optional, Union
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

from app.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analyzer for financial text.
    
    FinBERT is a BERT model fine-tuned on financial news and analyst reports,
    specifically trained to classify sentiment in financial contexts.
    
    Output Format:
        For each input text, returns probabilities for:
        - Positive: Bullish sentiment
        - Negative: Bearish sentiment
        - Neutral: No clear directional bias
    
    Reference:
        Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained
        Language Models." arXiv:1908.10063
    """
    
    def __init__(
        self,
        model_name: str = Config.FINBERT_MODEL,
        device: Optional[torch.device] = None,
        batch_size: int = Config.SENTIMENT_BATCH_SIZE
    ):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name: Hugging Face model identifier (default: ProsusAI/finbert).
            device: Computation device (CUDA/CPU). Auto-detects if None.
            batch_size: Number of texts to process simultaneously.
        """
        self.model_name = model_name
        self.device = device if device else Config.DEVICE
        self.batch_size = batch_size
        
        logger.info(f"Loading FinBERT model: {model_name}")
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load FinBERT tokenizer and model from Hugging Face.
        
        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Label mapping (FinBERT specific)
            self.label_map = {
                0: "positive",
                1: "negative",
                2: "neutral"
            }
            
            logger.info(f"FinBERT loaded successfully on {self.device}")
        
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def analyze(
        self,
        texts: Union[str, List[str]],
        return_probabilities: bool = True
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Analyze sentiment of financial text(s).
        
        Args:
            texts: Single text string or list of texts.
            return_probabilities: If True, return probability distribution.
                                 If False, return only the dominant label.
        
        Returns:
            Dictionary or list of dictionaries with sentiment scores.
            Format: {'positive': 0.7, 'negative': 0.1, 'neutral': 0.2}
        
        Mathematical Foundation:
            For input text X, FinBERT outputs logits L ∈ ℝ³
            Probabilities: P = softmax(L) = exp(L_i) / Σ(exp(L_j))
            
            Positive sentiment score can be interpreted as the probability
            that market-moving news will drive prices upward.
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        # Filter out empty texts
        texts = [t if t else "No content" for t in texts]
        
        all_results = []
        
        # Process in batches for efficiency
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self._process_batch(batch_texts, return_probabilities)
            all_results.extend(batch_results)
        
        return all_results[0] if single_input else all_results
    
    def _process_batch(
        self,
        texts: List[str],
        return_probabilities: bool
    ) -> List[Dict[str, float]]:
        """
        Process a batch of texts through FinBERT.
        
        Args:
            texts: List of text strings.
            return_probabilities: Whether to return full probability distribution.
        
        Returns:
            List of sentiment dictionaries.
        """
        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass (no gradient computation)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Convert logits to probabilities
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
        
        # Parse results
        results = []
        for probs in probabilities:
            if return_probabilities:
                result = {
                    "positive": float(probs[0]),
                    "negative": float(probs[1]),
                    "neutral": float(probs[2])
                }
            else:
                # Return only dominant label
                dominant_idx = np.argmax(probs)
                result = {
                    "label": self.label_map[dominant_idx],
                    "confidence": float(probs[dominant_idx])
                }
            
            results.append(result)
        
        return results
    
    def compute_sentiment_score(
        self,
        texts: Union[str, List[str]],
        aggregation: str = 'mean'
    ) -> float:
        """
        Compute a single numerical sentiment score.
        
        Args:
            texts: Single text or list of texts.
            aggregation: Method to aggregate multiple texts ('mean', 'median', 'weighted').
        
        Returns:
            Sentiment score in range [-1, 1].
            - Positive values indicate bullish sentiment
            - Negative values indicate bearish sentiment
            - Values near 0 indicate neutral sentiment
        
        Mathematical Formulation:
            Score = P(positive) - P(negative)
            
            This formulation:
            - Ignores neutral probability (it's implicit)
            - Produces a bipolar scale suitable for regression targets
        """
        sentiments = self.analyze(texts, return_probabilities=True)
        
        if isinstance(sentiments, dict):
            sentiments = [sentiments]
        
        # Convert to bipolar scores
        scores = [s['positive'] - s['negative'] for s in sentiments]
        
        if aggregation == 'mean':
            return float(np.mean(scores))
        elif aggregation == 'median':
            return float(np.median(scores))
        elif aggregation == 'weighted':
            # Weight by confidence (distance from neutral)
            weights = [abs(s['positive'] - s['negative']) for s in sentiments]
            if sum(weights) == 0:
                return 0.0
            return float(np.average(scores, weights=weights))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    def batch_analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'headlines',
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Apply sentiment analysis to a DataFrame with text data.
        
        Args:
            df: DataFrame containing text column.
            text_column: Name of column with text data.
            show_progress: Whether to display progress bar.
        
        Returns:
            DataFrame with added sentiment columns:
            - sentiment_positive
            - sentiment_negative
            - sentiment_neutral
            - sentiment_score (bipolar: -1 to 1)
        
        Usage:
            This is the primary method for integrating sentiment with OHLCV data.
            The output can be merged with price data on the date index.
        """
        if text_column not in df.columns:
            logger.warning(f"Column '{text_column}' not found. Returning neutral sentiment.")
            df['sentiment_score'] = 0.0
            df['sentiment_positive'] = 0.33
            df['sentiment_negative'] = 0.33
            df['sentiment_neutral'] = 0.34
            return df
        
        texts = df[text_column].fillna("No content").tolist()
        
        all_sentiments = []
        iterator = range(0, len(texts), self.batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Analyzing sentiment", unit="batch")
        
        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            batch_sentiments = self.analyze(batch_texts, return_probabilities=True)
            all_sentiments.extend(batch_sentiments)
        
        # Convert to DataFrame columns
        df['sentiment_positive'] = [s['positive'] for s in all_sentiments]
        df['sentiment_negative'] = [s['negative'] for s in all_sentiments]
        df['sentiment_neutral'] = [s['neutral'] for s in all_sentiments]
        df['sentiment_score'] = [s['positive'] - s['negative'] for s in all_sentiments]
        
        logger.info(f"Sentiment analysis complete. Mean score: {df['sentiment_score'].mean():.3f}")
        
        return df
    
    def get_market_sentiment_regime(self, score: float) -> str:
        """
        Classify market sentiment into discrete regimes.
        
        Args:
            score: Sentiment score in range [-1, 1].
        
        Returns:
            Regime label: 'strongly_bearish', 'bearish', 'neutral', 'bullish', 'strongly_bullish'.
        
        Application:
            Can be used by the fusion network to adjust gating weights.
            During extreme sentiment regimes, the model may trust neural predictions more.
        """
        if score < -0.5:
            return "strongly_bearish"
        elif score < -0.2:
            return "bearish"
        elif score > 0.5:
            return "strongly_bullish"
        elif score > 0.2:
            return "bullish"
        else:
            return "neutral"


class SentimentFeatureEngineer:
    """
    Advanced sentiment feature engineering for time-series modeling.
    
    Creates derived features from raw sentiment scores that capture
    temporal dynamics and sentiment momentum.
    """
    
    @staticmethod
    def add_temporal_features(df: pd.DataFrame, sentiment_col: str = 'sentiment_score') -> pd.DataFrame:
        """
        Add temporal sentiment features.
        
        Args:
            df: DataFrame with sentiment_score column.
            sentiment_col: Name of sentiment column.
        
        Returns:
            DataFrame with additional features:
            - sentiment_ma7: 7-day moving average (short-term trend)
            - sentiment_ma21: 21-day moving average (medium-term trend)
            - sentiment_momentum: Rate of change in sentiment
            - sentiment_volatility: Rolling standard deviation
        
        Rationale:
            Raw sentiment can be noisy. Moving averages capture persistent
            sentiment shifts, while momentum captures acceleration in sentiment change.
        """
        df = df.copy()
        
        if sentiment_col not in df.columns:
            logger.warning(f"Sentiment column '{sentiment_col}' not found.")
            return df
        
        # Moving averages
        df['sentiment_ma7'] = df[sentiment_col].rolling(window=7, min_periods=1).mean()
        df['sentiment_ma21'] = df[sentiment_col].rolling(window=21, min_periods=1).mean()
        
        # Sentiment momentum (rate of change)
        df['sentiment_momentum'] = df[sentiment_col].diff(periods=5)
        
        # Sentiment volatility
        df['sentiment_volatility'] = df[sentiment_col].rolling(window=14, min_periods=1).std()
        
        # Sentiment divergence (current vs moving average)
        df['sentiment_divergence'] = df[sentiment_col] - df['sentiment_ma21']
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        logger.info("Added 5 temporal sentiment features")
        
        return df
