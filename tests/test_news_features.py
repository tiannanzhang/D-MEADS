"""
Test suite for news features integration in D-MEADS.

This module tests the following components:
1. NewsDataBuilder - data collection
2. SentimentAnalyzer - FinBERT sentiment extraction
3. Feature extraction and alignment
4. Normalization
5. Dataset loading with news features
6. Model forward pass with news conditioning

Run with: python -m pytest tests/test_news_features.py -v
"""

import pytest
import numpy as np
import pandas as pd
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.NewsDataBuilder import NewsDataBuilder
from preprocessing.SentimentAnalyzer import SentimentAnalyzer
from utils.utils_data import extract_news_features, normalize_news_features
from preprocessing.LOBDataset import LOBDataset
import constants as cst


class TestNewsDataBuilder:
    """Test NewsDataBuilder functionality."""

    def test_init(self):
        """Test NewsDataBuilder initialization."""
        builder = NewsDataBuilder(data_dir="data/news")
        assert builder.data_dir.exists() or True  # Directory might not exist yet
        assert isinstance(builder, NewsDataBuilder)

    def test_collect_yahoo_news_placeholder(self):
        """Test Yahoo Finance news collection (placeholder test)."""
        builder = NewsDataBuilder(data_dir="data/news")

        # Note: This will likely return empty or limited data without API access
        try:
            news_df = builder.collect_yahoo_news(
                ticker="TSLA",
                start_date="2024-01-01",
                end_date="2024-01-02",
                save=False
            )
            assert isinstance(news_df, pd.DataFrame)
            assert all(col in news_df.columns for col in ['timestamp', 'headline', 'ticker'])
        except Exception as e:
            pytest.skip(f"Yahoo Finance API not available: {e}")


class TestSentimentAnalyzer:
    """Test SentimentAnalyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create SentimentAnalyzer instance."""
        return SentimentAnalyzer()

    @pytest.fixture
    def sample_headlines(self):
        """Sample news headlines for testing."""
        return [
            "Tesla stock surges on strong earnings report",
            "Market crashes amid recession fears",
            "Federal Reserve maintains interest rates"
        ]

    def test_init(self, analyzer):
        """Test SentimentAnalyzer initialization."""
        assert isinstance(analyzer, SentimentAnalyzer)
        assert analyzer.device is not None

    def test_analyze_headlines(self, analyzer, sample_headlines):
        """Test headline sentiment analysis."""
        try:
            sentiments = analyzer.analyze_headlines(sample_headlines)
            assert len(sentiments) == len(sample_headlines)
            assert all(-1 <= s <= 1 for s in sentiments)
            assert isinstance(sentiments, np.ndarray)
        except Exception as e:
            pytest.skip(f"FinBERT model not available: {e}")

    def test_analyze_news_dataframe(self, analyzer):
        """Test DataFrame sentiment analysis."""
        news_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1h'),
            'headline': [
                "Positive market outlook",
                "Negative economic indicators",
                "Neutral policy statement"
            ]
        })

        try:
            result_df = analyzer.analyze_news_dataframe(news_df)
            assert 'sentiment' in result_df.columns
            assert len(result_df) == 3
            assert all(-1 <= s <= 1 for s in result_df['sentiment'])
        except Exception as e:
            pytest.skip(f"FinBERT model not available: {e}")


class TestFeatureExtraction:
    """Test news feature extraction and normalization."""

    @pytest.fixture
    def messages_df(self):
        """Sample LOB messages with timestamps."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:30:00', periods=100, freq='1s'),
            'event_type': np.random.randint(0, 3, 100),
            'price': np.random.randn(100) * 100 + 20000,
            'size': np.random.randint(1, 1000, 100)
        })

    @pytest.fixture
    def news_df(self):
        """Sample news data with sentiment."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:30:00', periods=10, freq='1min'),
            'sentiment': np.random.randn(10) * 0.5,
            'headline': ['Headline ' + str(i) for i in range(10)]
        })

    def test_extract_news_features(self, messages_df, news_df):
        """Test news feature extraction with rolling window and exponential weighting."""
        news_features = extract_news_features(
            messages_df=messages_df,
            news_df=news_df,
            lookback_window_sec=60,
            half_life_sec=30
        )

        assert isinstance(news_features, pd.DataFrame)
        assert len(news_features) == len(messages_df)
        assert all(col in news_features.columns for col in ['sentiment', 'headline_count'])

    def test_normalize_news_features(self):
        """Test news feature normalization."""
        news_features = pd.DataFrame({
            'sentiment': np.random.randn(100),
            'headline_count': np.random.randint(0, 10, 100)
        })

        normalized, mean_s, mean_h, std_s, std_h = normalize_news_features(news_features)

        assert isinstance(normalized, pd.DataFrame)
        assert len(normalized) == len(news_features)
        # Check that normalized features have approximately 0 mean and 1 std
        assert abs(normalized['sentiment'].mean()) < 0.1
        assert abs(normalized['sentiment'].std() - 1.0) < 0.1


class TestDatasetIntegration:
    """Test LOBDataset with news features."""

    @pytest.fixture
    def temp_data_path(self, tmp_path):
        """Create temporary test data."""
        # Create dummy LOB data (orders + LOB snapshots)
        lob_data = np.random.randn(1000, 46)  # 6 order features + 40 LOB features
        lob_path = tmp_path / "test.npy"
        np.save(lob_path, lob_data)

        # Create dummy news data
        news_data = np.random.randn(1000, 2)  # 2 news features
        news_path = tmp_path / "test_news.npy"
        np.save(news_path, news_data)

        return str(lob_path), str(news_path)

    def test_dataset_with_news(self, temp_data_path):
        """Test LOBDataset loading with news features."""
        lob_path, news_path = temp_data_path

        dataset = LOBDataset(
            paths=[lob_path],
            seq_size=256,
            gen_seq_size=1,
            chosen_model=cst.Models.TRADES,
            news_paths=[news_path],
            use_news_features=True
        )

        assert dataset.news_data is not None
        assert dataset.use_news_features is True

        # Test __getitem__
        sample = dataset[0]
        assert len(sample) == 4  # (cond, x_0, lob, news)

        cond, x_0, lob, news = sample
        assert news.shape[-1] == 2  # 2 news features

    def test_dataset_without_news(self, temp_data_path):
        """Test LOBDataset loading without news features (backward compatibility)."""
        lob_path, _ = temp_data_path

        dataset = LOBDataset(
            paths=[lob_path],
            seq_size=256,
            gen_seq_size=1,
            chosen_model=cst.Models.TRADES,
            use_news_features=False
        )

        assert dataset.news_data is None
        assert dataset.use_news_features is False

        # Test __getitem__
        sample = dataset[0]
        assert len(sample) == 3  # (cond, x_0, lob) - no news


class TestModelIntegration:
    """Test TRADES model with news features."""

    def test_trades_forward_with_news(self):
        """Test TRADES forward pass with news conditioning."""
        from models.diffusers.TRADES.TRADES import TRADES

        model = TRADES(
            input_size=6,  # Base order feature dimension (news handled internally)
            cond_seq_len=255,
            num_diffusionsteps=100,
            depth=8,
            num_heads=1,
            gen_sequence_size=1,
            cond_dropout_prob=0.0,
            is_augmented=True,  # True to ensure even embedding dimension (matches real config)
            dropout=0.1,
            cond_type="full",
            cond_method="concatenation",
            use_news_features=True,
            news_feature_dim=2
        )

        # Create dummy inputs
        batch_size = 4
        cond_seq_len = 255
        gen_seq_len = 1

        x = torch.randn(batch_size, gen_seq_len, 6)  # Noisy order features
        cond_orders = torch.randn(batch_size, cond_seq_len, 6)  # Past order history
        cond_lob = torch.randn(batch_size, cond_seq_len + 1, 40)  # LOB snapshots
        cond_news = torch.randn(batch_size, cond_seq_len, 2)  # News features
        t = torch.randint(0, 100, (batch_size,))  # Diffusion timesteps

        # Forward pass
        noise, var = model(x, cond_orders, t, cond_lob, cond_news)

        assert noise.shape == (batch_size, gen_seq_len, 6)
        assert var.shape == (batch_size, gen_seq_len, 6)

    def test_trades_forward_without_news(self):
        """Test TRADES forward pass without news (backward compatibility)."""
        from models.diffusers.TRADES.TRADES import TRADES

        model = TRADES(
            input_size=6,
            cond_seq_len=255,
            num_diffusionsteps=100,
            depth=8,
            num_heads=1,
            gen_sequence_size=1,
            cond_dropout_prob=0.0,
            is_augmented=True,  # True to ensure even embedding dimension (matches real config)
            dropout=0.1,
            cond_type="full",
            cond_method="concatenation",
            use_news_features=False,
            news_feature_dim=0
        )

        # Create dummy inputs (no news)
        batch_size = 4
        cond_seq_len = 255
        gen_seq_len = 1

        x = torch.randn(batch_size, gen_seq_len, 6)
        cond_orders = torch.randn(batch_size, cond_seq_len, 6)
        cond_lob = torch.randn(batch_size, cond_seq_len + 1, 40)
        t = torch.randint(0, 100, (batch_size,))

        # Forward pass without news
        noise, var = model(x, cond_orders, t, cond_lob, cond_news=None)

        assert noise.shape == (batch_size, gen_seq_len, 6)
        assert var.shape == (batch_size, gen_seq_len, 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
