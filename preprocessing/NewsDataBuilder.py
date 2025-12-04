"""
NewsDataBuilder: Loads and preprocesses cleaned news data for LOB modeling.

This module handles:
1. Loading pre-cleaned news data from CSV files
2. Data organization and formatting for sentiment analysis

Author: D-MEADS Team
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsDataBuilder:
    """
    Loads and processes cleaned news data from CSV files.

    The news data has been pre-cleaned to remove:
    - Promotional content and advertisements
    - Non-influential articles
    - Duplicate entries
    """

    def __init__(self, data_dir: str = "data/news"):
        """
        Initialize NewsDataBuilder.

        Args:
            data_dir: Root directory containing news CSV files
        """
        self.data_dir = Path(data_dir)

        # Mapping of tickers to their cleaned data files
        self.data_files = {
            'TSLA': self.data_dir / 'final_tsla_news_cleaned.csv',
            'INTC': self.data_dir / 'final_intc_news_cleaned.csv',
        }

        logger.info(f"NewsDataBuilder initialized with data directory: {self.data_dir}")

    def load_news_data(self,
                       ticker: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load cleaned news data for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'TSLA', 'INTC')
            start_date: Optional start date in 'YYYY-MM-DD' format
            end_date: Optional end date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with columns: [timestamp, headline, url, ticker, sentiment, publisher, influence_reason]
        """
        ticker = ticker.upper()

        if ticker not in self.data_files:
            logger.error(f"No cleaned data available for ticker: {ticker}")
            logger.info(f"Available tickers: {list(self.data_files.keys())}")
            return pd.DataFrame(columns=['timestamp', 'headline', 'url', 'ticker'])

        data_file = self.data_files[ticker]

        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return pd.DataFrame(columns=['timestamp', 'headline', 'url', 'ticker'])

        logger.info(f"Loading news data for {ticker} from {data_file}")

        # Load CSV
        df = pd.read_csv(data_file)

        # Standardize column names
        df = df.rename(columns={
            'Date': 'timestamp',
            'Article_title': 'headline',
            'Url': 'url',
            'Publisher': 'publisher',
            'Article': 'article',
            'influence_reason': 'influence_reason'
        })

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Add ticker column if not present
        if 'ticker' not in df.columns:
            df['ticker'] = ticker

        # Filter by date range if provided
        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            # Handle timezone-aware timestamps
            if df['timestamp'].dt.tz is not None:
                start_dt = start_dt.tz_localize('UTC')
            df = df[df['timestamp'] >= start_dt]

        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            # Handle timezone-aware timestamps
            if df['timestamp'].dt.tz is not None:
                end_dt = end_dt.tz_localize('UTC')
            df = df[df['timestamp'] <= end_dt]

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} news articles for {ticker}")
        if len(df) > 0:
            logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"  Publishers: {df['publisher'].nunique()} unique")

        return df

    def get_available_tickers(self) -> List[str]:
        """
        Get list of tickers with available news data.

        Returns:
            List of ticker symbols
        """
        available = []
        for ticker, filepath in self.data_files.items():
            if filepath.exists():
                available.append(ticker)

        return available

    def get_data_summary(self, ticker: str) -> Dict:
        """
        Get summary statistics for a ticker's news data.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with summary statistics
        """
        df = self.load_news_data(ticker)

        if len(df) == 0:
            return {
                'ticker': ticker,
                'total_articles': 0,
                'date_range': None,
                'publishers': 0
            }

        return {
            'ticker': ticker,
            'total_articles': len(df),
            'date_range': (df['timestamp'].min(), df['timestamp'].max()),
            'publishers': df['publisher'].nunique(),
            'publishers_list': df['publisher'].value_counts().head(5).to_dict(),
            'articles_per_day': len(df) / (df['timestamp'].max() - df['timestamp'].min()).days if len(df) > 1 else 0,
            'influence_types': df.get('influence_reason', pd.Series()).value_counts().head(5).to_dict() if 'influence_reason' in df.columns else {}
        }


if __name__ == "__main__":
    # Example usage
    builder = NewsDataBuilder()

    # Show available tickers
    available = builder.get_available_tickers()
    print(f"Available tickers: {available}")

    # Load news for Tesla
    if 'TSLA' in available:
        print("\n" + "="*80)
        print("TESLA (TSLA) News Data")
        print("="*80)

        tsla_news = builder.load_news_data('TSLA')
        print(f"\nLoaded {len(tsla_news)} articles")

        # Show summary
        summary = builder.get_data_summary('TSLA')
        print("\nSummary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        # Show sample articles
        print("\nSample Articles:")
        print(tsla_news[['timestamp', 'headline', 'publisher']].head(5))

    # Load news for Intel
    if 'INTC' in available:
        print("\n" + "="*80)
        print("INTEL (INTC) News Data")
        print("="*80)

        intc_news = builder.load_news_data('INTC')
        print(f"\nLoaded {len(intc_news)} articles")

        # Show summary
        summary = builder.get_data_summary('INTC')
        print("\nSummary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        # Show sample articles
        print("\nSample Articles:")
        print(intc_news[['timestamp', 'headline', 'publisher']].head(5))
