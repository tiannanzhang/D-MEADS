"""
SentimentAnalyzer: Extracts sentiment from news headlines using FinBERT.

This module handles:
1. FinBERT sentiment analysis for news headlines
2. Batch processing for efficiency
3. Support for pre-cleaned news data

Author: D-MEADS Team
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyzes sentiment of financial text using FinBERT models.

    Designed for pre-cleaned news data from NewsDataBuilder.
    """

    def __init__(self,
                 finbert_model: str = "ProsusAI/finbert",
                 device: Optional[str] = None):
        """
        Initialize SentimentAnalyzer with FinBERT model.

        Args:
            finbert_model: Hugging Face model ID for news sentiment
            device: Device to run models on ('cuda', 'cpu', or None for auto)
        """
        self.finbert_model_name = finbert_model

        # Auto-detect device (prioritize CUDA > MPS > CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Initialize models (lazy loading)
        self.finbert_model = None
        self.finbert_tokenizer = None

    def _load_finbert(self):
        """Lazy load FinBERT model and tokenizer."""
        if self.finbert_model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                logger.info(f"Loading FinBERT model: {self.finbert_model_name}")
                self.finbert_tokenizer = AutoTokenizer.from_pretrained(self.finbert_model_name)
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                    self.finbert_model_name
                )
                self.finbert_model.to(self.device)
                self.finbert_model.eval()
                logger.info("FinBERT model loaded successfully")

            except ImportError:
                logger.error("transformers library not installed. Install with: pip install transformers")
                raise
            except Exception as e:
                logger.error(f"Error loading FinBERT model: {e}")
                raise

    def analyze_headlines(self,
                          headlines: List[str],
                          batch_size: int = 32) -> np.ndarray:
        """
        Analyze sentiment of news headlines using FinBERT.

        Args:
            headlines: List of news headline strings
            batch_size: Batch size for processing

        Returns:
            Array of sentiment scores in range [-1, 1]
            where -1 = very negative, 0 = neutral, 1 = very positive
        """
        if not headlines:
            return np.array([])

        self._load_finbert()

        sentiments = []

        with torch.no_grad():
            for i in range(0, len(headlines), batch_size):
                batch = headlines[i:i + batch_size]

                # Tokenize
                inputs = self.finbert_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get predictions
                outputs = self.finbert_model(**inputs)
                logits = outputs.logits

                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)

                # FinBERT outputs: [negative, neutral, positive]
                # Convert to sentiment score: -1 to 1
                # sentiment = (positive - negative)
                negative_probs = probs[:, 0].cpu().numpy()
                neutral_probs = probs[:, 1].cpu().numpy()
                positive_probs = probs[:, 2].cpu().numpy()

                # Calculate sentiment score as: positive - negative
                batch_sentiments = positive_probs - negative_probs

                sentiments.extend(batch_sentiments)

        return np.array(sentiments)

    def analyze_news_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame of news data.

        Supports both 'headline' and 'Article_title' column names.

        Args:
            news_df: DataFrame with 'headline' or 'Article_title' column

        Returns:
            DataFrame with added 'sentiment' column
        """
        # Check for headline column (flexible naming)
        headline_col = None
        if 'headline' in news_df.columns:
            headline_col = 'headline'
        elif 'Article_title' in news_df.columns:
            headline_col = 'Article_title'
        else:
            raise ValueError("news_df must have 'headline' or 'Article_title' column")

        if len(news_df) == 0:
            news_df['sentiment'] = []
            return news_df

        headlines = news_df[headline_col].fillna('').tolist()
        sentiments = self.analyze_headlines(headlines)

        news_df = news_df.copy()
        news_df['sentiment'] = sentiments

        logger.info(f"Analyzed sentiment for {len(news_df)} headlines")
        logger.info(f"  Mean sentiment: {sentiments.mean():.3f}")
        logger.info(f"  Sentiment range: [{sentiments.min():.3f}, {sentiments.max():.3f}]")

        # Show sentiment distribution
        positive = (sentiments > 0.1).sum()
        neutral = ((sentiments >= -0.1) & (sentiments <= 0.1)).sum()
        negative = (sentiments < -0.1).sum()

        logger.info(f"  Distribution: {positive} positive, {neutral} neutral, {negative} negative")

        return news_df


if __name__ == "__main__":
    # Example usage with cleaned news data
    from preprocessing.NewsDataBuilder import NewsDataBuilder

    # Load cleaned news data
    builder = NewsDataBuilder()
    available_tickers = builder.get_available_tickers()

    if len(available_tickers) == 0:
        print("No news data available. Please ensure cleaned news files exist in data/news/")
        print("Expected files:")
        print("  - data/news/final_tsla_news_cleaned.csv")
        print("  - data/news/final_intc_news_cleaned.csv")
    else:
        print(f"Available tickers: {available_tickers}")

        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer()

        # Analyze news for first available ticker
        ticker = available_tickers[0]
        print(f"\n{'='*80}")
        print(f"Analyzing sentiment for {ticker}")
        print(f"{'='*80}\n")

        news_df = builder.load_news_data(ticker)

        if len(news_df) > 0:
            # Analyze sentiment
            news_df = analyzer.analyze_news_dataframe(news_df)

            # Show results
            print("\nSample Results:")
            print(news_df[['timestamp', 'headline', 'sentiment']].head(10))

            # Show most positive and negative headlines
            print(f"\n{'='*80}")
            print("Most Positive Headlines:")
            print(f"{'='*80}")
            top_positive = news_df.nlargest(5, 'sentiment')[['timestamp', 'headline', 'sentiment']]
            for idx, row in top_positive.iterrows():
                print(f"\nSentiment: {row['sentiment']:+.3f}")
                print(f"Date: {row['timestamp']}")
                print(f"Headline: {row['headline'][:100]}")

            print(f"\n{'='*80}")
            print("Most Negative Headlines:")
            print(f"{'='*80}")
            top_negative = news_df.nsmallest(5, 'sentiment')[['timestamp', 'headline', 'sentiment']]
            for idx, row in top_negative.iterrows():
                print(f"\nSentiment: {row['sentiment']:+.3f}")
                print(f"Date: {row['timestamp']}")
                print(f"Headline: {row['headline'][:100]}")
        else:
            print(f"No news data found for {ticker}")
