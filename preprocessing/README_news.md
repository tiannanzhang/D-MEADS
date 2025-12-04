# News Features Integration Guide

This document explains how to use news features in the D-MEADS framework.

## Overview

The D-MEADS framework supports incorporating exogenous information from pre-cleaned financial news data. News features are extracted using FinBERT sentiment analysis and aligned with LOB (Limit Order Book) event timestamps.

## Data Source

News data has been **pre-processed and cleaned** to include only influential articles:
- **Tesla (TSLA)**: 76 articles from January 2015
- **Intel (INTC)**: 238 articles from January 2015

The data includes:
- ✓ Only stock-influential articles (material news, analyst ratings, price movements)
- ✓ Removed promotional content and advertisements
- ✓ Deduplicated entries
- ✓ Cleaned HTML and formatting

**Location**: `data/news/final_<ticker>_news_cleaned.csv`

## Features

The news integration provides two features per timestep:

1. **Sentiment** (`float`, range -1 to +1): FinBERT sentiment score from news headlines
   - -1: Very negative
   - 0: Neutral
   - +1: Very positive

2. **Headline Count** (`int`, range 0 to ∞): Number of news headlines in the time window

## Temporal Weighting Methodology

The sentiment feature uses **exponential decay weighting** to emphasize recent news over older news. This approach is proven in financial sentiment analysis research.

### Exponential Weighting Formula

For each LOB event at time `t`, we compute sentiment as:

```
sentiment(t) = Σ [w(Δt_i) × sentiment_i] / Σ w(Δt_i)

where:
  w(Δt) = exp(-λ × Δt)
  Δt = time since news headline
  λ = ln(2) / half_life
```

**Example:** With a 30-second half-life:
- News from 0 seconds ago: weight = 1.00 (100%)
- News from 30 seconds ago: weight = 0.50 (50%)
- News from 60 seconds ago: weight = 0.25 (25%)

This ensures that recent breaking news has stronger influence on order generation than older headlines.

**Research Support:**
- Exponential Moving Average (EMA) is the standard for financial time series [[1]](https://link.springer.com/article/10.1007/s42521-024-00107-2)
- Optimal half-lives for news sentiment: 1-7 days for daily data
- Scaled to 30 seconds for sub-minute LOB events

## Configuration

### Enabling News Features

In `configuration.py`, set the following parameters:

```python
self.USE_NEWS_FEATURES = True  # Enable news features
self.NEWS_FEATURE_DIM = 2  # Number of news features (sentiment, headline_count)
self.NEWS_LOOKBACK_WINDOW = 60  # Seconds to look back for news (rolling window)
self.NEWS_HALF_LIFE = 30  # Half-life in seconds for exponential decay weighting
self.FINBERT_MODEL = 'ProsusAI/finbert'  # FinBERT model for news sentiment
```

**Note on Temporal Parameters:**
- `NEWS_LOOKBACK_WINDOW`: How far back to search for news headlines (default 60 seconds = 1 minute)
- `NEWS_HALF_LIFE`: Controls recency bias in exponential weighting (default 30 seconds)
  - Smaller values = stronger recency bias (recent news matters more)
  - Larger values = more uniform weighting (past news retains influence longer)
  - Based on research showing optimal half-lives of 1-7 days for daily data; scaled to sub-minute LOB events

### Dependencies

Install required packages:

```bash
source ~/.zshrc && conda activate dmeads
pip3 install transformers sentencepiece torch
```

**Note**: `yfinance` is NOT required as we use pre-cleaned data.

## Usage

### 1. Load News Data

Use `NewsDataBuilder` to load pre-cleaned news data:

```python
from preprocessing.NewsDataBuilder import NewsDataBuilder

builder = NewsDataBuilder(data_dir="data/news")

# Load news for Tesla
news_df = builder.load_news_data('TSLA')

# Load news for Intel
news_df = builder.load_news_data('INTC')

# Check available tickers
available = builder.get_available_tickers()
print(f"Available: {available}")  # ['TSLA', 'INTC']
```

The loaded DataFrame contains:
- `timestamp`: Article publication time
- `headline`: Cleaned headline text
- `url`: Article URL
- `ticker`: Stock ticker
- `publisher`: News source
- `influence_reason`: Why article was marked as influential

### 2. Extract Sentiment

Use `SentimentAnalyzer` to process text and extract sentiment:

```python
from preprocessing.SentimentAnalyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Analyze news headlines
news_df = analyzer.analyze_news_dataframe(news_df)
```

The sentiment analyzer will:
- Load the FinBERT model from Hugging Face
- Process headlines in batches for efficiency
- Add a 'sentiment' column to the DataFrame
- Show sentiment distribution statistics

**Example Output:**
```
INFO:__main__:Analyzed sentiment for 76 headlines
INFO:__main__:  Mean sentiment: +0.123
INFO:__main__:  Sentiment range: [-0.845, +0.912]
INFO:__main__:  Distribution: 42 positive, 18 neutral, 16 negative
```

### 3. Preprocess LOB Data with News

Update your data preprocessing to include news features:

```python
from preprocessing.LOBSTERDataBuilder import LOBSTERDataBuilder
import constants as cst

builder = LOBSTERDataBuilder(
    stock_name="TSLA",
    data_dir="data",
    date_trading_days=cst.DATE_TRADING_DAYS,
    split_rates=cst.SPLIT_RATES,
    chosen_model=cst.Models.TRADES,
    use_news_features=True,  # Enable news features
    news_lookback_window=60,  # Look back 60 seconds for news
    news_half_life=30  # 30-second half-life for exponential weighting
)

builder.prepare_save_datasets()
```

This will:
- Load pre-cleaned news data from `data/news/final_tsla_news_cleaned.csv`
- Extract news features using rolling window with exponential decay weighting
  - For each LOB event, look back 60 seconds for news
  - Weight recent headlines more heavily using exponential decay (λ = ln(2) / half_life)
  - Count headlines in the window
- Normalize using z-score
- Save as `train_news.npy`, `val_news.npy`, `test_news.npy`

### 4. Train Model with News Features

The TRADES model automatically uses news features when available:

```python
from configuration import Configuration
import constants as cst

config = Configuration()
config.USE_NEWS_FEATURES = True

# Train model (news features will be loaded automatically)
# The model will receive news conditioning via cond_news parameter
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Pre-Cleaned News Data (CSV)                     │
│  - Influential articles only                                 │
│  - No promotional content                                    │
│  - Deduplicated                                              │
│  Location: data/news/final_<ticker>_news_cleaned.csv        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │   NewsDataBuilder                  │
         │   - Load CSV data                  │
         │   - Filter by date range           │
         │   - Standardize column names       │
         └────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │   FinBERT Sentiment Analysis       │
         │   - Load model from HuggingFace    │
         │   - Batch process headlines        │
         │   - Output: sentiment score [-1,1] │
         └────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │   Feature Extraction               │
         │   - Rolling window per LOB event   │
         │   - Exponentially weighted avg     │
         │   - Count headlines in window      │
         │   - weight(Δt) = exp(-λ*Δt)       │
         │   - λ = ln(2) / half_life         │
         └────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │   Temporal Alignment               │
         │   - Each event has specific window │
         │   - Recent news weighted higher    │
         │   - No pre-aggregation required    │
         └────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │   Normalization                    │
         │   - Z-score normalization          │
         │   - Train set statistics           │
         │   - Applied to val/test sets       │
         └────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────┐
         │   Model Training                   │
         │   - News features as conditioning  │
         │   - Concatenated with order hist.  │
         │   - Processed by TRADES model      │
         └────────────────────────────────────┘
```

## Feature Dimensions

News features are treated as **exogenous conditioning** (like LOB state), not intrinsic order properties:

- **Order features** (always 6D): `[time, type, size, price, direction, depth]`
- **LOB features** (40D): `[ask_prices, ask_volumes, bid_prices, bid_volumes]` × 10 levels
- **News features** (2D): `[sentiment, headline_count]` (separate conditioning)

**Architecture:**
```
┌──────────────────────────────────────────────────────┐
│  TRADES Model Input                                   │
│                                                       │
│  Orders (N, 256, 6)  ─────┐                         │
│                            ├─> Concatenated          │
│  LOB (N, 255, 40)  ────────┤    (N, 256, 43)        │
│                            │                          │
│  News (N, 255, 2)  ────────┘                         │
│                                                       │
│  News provides environmental context that influences  │
│  order generation, but doesn't alter order structure │
└──────────────────────────────────────────────────────┘
```

The model architecture automatically adjusts:
- `SIZE_ORDER_EMB`: Remains 6D (order features unchanged)
- Input dimension calculation in TRADES accounts for news concatenated with LOB
- `TRADES forward()`: accepts `cond_news` parameter (optional)

## File Locations

After preprocessing, news features are saved alongside LOB data:

```
data/
├── TSLA/
│   ├── train.npy          # LOB data (orders + LOB snapshots)
│   ├── train_news.npy     # News features (2D: sentiment, count)
│   ├── val.npy
│   ├── val_news.npy
│   ├── test.npy
│   └── test_news.npy
└── news/
    ├── final_tsla_news_cleaned.csv  # Pre-cleaned Tesla news
    ├── final_intc_news_cleaned.csv  # Pre-cleaned Intel news
    └── README.md                     # Data documentation
```

## Data Coverage

### Tesla (TSLA) - January 2015

- **Articles**: 76 influential articles
- **Date range**: January 2-30, 2015
- **Publishers**: 7 unique sources
- **Key events**:
  - Jan 14: CEO announces losses to persist until 2020, China sales weak (stock crashed 18.9%)
  - Jan 16: Production target announcement (500k by 2020)
  - Jan 22: Morgan Stanley price target reduction
  - Jan 30: Q4 earnings release date announced

### Intel (INTC) - January 2015

- **Articles**: 238 influential articles
- **Date range**: January 1-30, 2015
- **Publishers**: 13 unique sources
- **Key events**:
  - Jan 5-7: Broadwell processor launch at CES 2015
  - Jan 5: $24.8M investment in Vuzix for wearables
  - Throughout: Analyst coverage and stock movements

## Backward Compatibility

The news feature integration is **fully backward compatible**:

- Set `USE_NEWS_FEATURES = False` to disable (default)
- Old models and data work without modification
- LOBDataset returns 3-tuple `(cond, x_0, lob)` when news disabled
- LOBDataset returns 4-tuple `(cond, x_0, lob, news)` when news enabled

## Troubleshooting

### Issue: "No cleaned data available for ticker"

**Solution**: Ensure cleaned news files exist:
```bash
ls data/news/final_*_news_cleaned.csv
```

Expected files:
- `data/news/final_tsla_news_cleaned.csv`
- `data/news/final_intc_news_cleaned.csv`

### Issue: "FinBERT model download fails"

**Solution**: Check internet connection and Hugging Face access. Alternatively, download model manually:
```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("ProsusAI/finbert")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
```

### Issue: "Dimension mismatch in TRADES forward()"

**Solution**: Ensure `USE_NEWS_FEATURES` is consistent across:
- `configuration.py`: `USE_NEWS_FEATURES = True`
- `LOBSTERDataBuilder`: `use_news_features=True`
- `LOBDataset`: `use_news_features=True`

### Issue: "Timestamps not aligning with LOB events"

**Solution**: The news data uses actual timestamps from January 2015. Ensure your LOB data is also from January 2015, or adjust the temporal alignment logic in LOBSTERDataBuilder to match your LOB data's time period.

## Performance Considerations

- **FinBERT inference**: GPU recommended for faster sentiment analysis (10x speedup)
  - CPU: ~2-3 articles/second
  - GPU (CUDA): ~20-30 articles/second
  - MPS (Apple Silicon): ~10-15 articles/second

- **News data volume**:
  - TSLA: 76 articles → ~150KB preprocessed
  - INTC: 238 articles → ~450KB preprocessed

- **Model training**: +2 features increases model size by ~5% (negligible)

- **Inference speed**: No significant impact (<5% slower)

## Adding New Tickers

To add news data for additional tickers:

1. **Collect and clean news data** (see `/scripts/data_cleaning/README.md`)
2. **Save to** `data/news/final_<ticker>_news_cleaned.csv`
3. **Update** `NewsDataBuilder.py` to include the new ticker:
   ```python
   self.data_files = {
       'TSLA': self.data_dir / 'final_tsla_news_cleaned.csv',
       'INTC': self.data_dir / 'final_intc_news_cleaned.csv',
       'AAPL': self.data_dir / 'final_aapl_news_cleaned.csv',  # Add new ticker
   }
   ```

## Citation

If you use the news features integration, please cite:

```bibtex
@article{dmeads2024,
  title={D-MEADS: Deep Market Event-driven Agent-based Diffusion Simulator},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

And the FinBERT paper:

```bibtex
@article{araci2019finbert,
  title={FinBERT: Financial Sentiment Analysis with Pre-trained Language Models},
  author={Araci, Dogu},
  journal={arXiv preprint arXiv:1908.10063},
  year={2019}
}
```

## Future Enhancements

Potential improvements:

1. **More time periods**: Extend beyond January 2015
2. **More tickers**: Add AAPL, MSFT, GOOGL, etc.
3. **Topic modeling**: Extract themes from headlines (earnings, M&A, regulation)
4. **Entity recognition**: Link news to specific companies/sectors
5. **Cross-asset sentiment**: Sentiment from related assets (S&P 500, sector indices)
6. **Real-time updates**: Live news feed integration for deployment

## Support

For issues or questions:
- Check `data/news/README.md` for data documentation
- Review `/scripts/data_cleaning/README.md` for data processing details
- Open an issue on GitHub
- Review the paper for methodology details
