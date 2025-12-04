# News Data Directory

This directory contains financial news articles for stock market analysis and prediction.

## 📁 Directory Contents

### Final Cleaned Datasets (Ready for Use)

These are the final, cleaned, and filtered datasets ready for model training:

- **`final_tsla_news_cleaned.csv`** (76 articles)
  - Tesla (TSLA) influential news from January 2015
  - Combined from NASDAQ external data and JSON sources
  - Cleaned of all promotional content
  - Only articles that could influence stock price

- **`final_intc_news_cleaned.csv`** (238 articles)
  - Intel (INTC) influential news from January 2015
  - Combined from NASDAQ external data and JSON sources
  - Cleaned of all promotional content
  - Only articles that could influence stock price

### Raw Source Files

#### Large CSV Datasets

- **`All_external.csv`** (5.3 GB, 13M+ rows)
  - Comprehensive news dataset from 2009-2020
  - 6,619 unique stock symbols
  - 1,142 publishers
  - Coverage: Aug 2009 - Jun 2020
  - Note: 75% of rows missing stock symbols, 90% missing full article text

- **`nasdaq_exteral_data.csv`**
  - NASDAQ-specific external news data
  - Coverage: 2017-2023
  - More recent data than All_external.csv

#### JSON Source Files

- **`news_data_TSLA.json`** (42 articles)
  - Tesla news from various sources
  - Fields: author, content, created_at, headline, id, images, source, summary, symbols, updated_at, url

- **`news_data_INTC.json`** (37 articles)
  - Intel news from various sources
  - Same structure as TSLA JSON

### Intermediate Processing Files

#### TSLA Processing Chain
1. `nasdaq_tsla_mentions_jan2015.csv` (119 articles) - Raw keyword search results
2. `nasdaq_tsla_jan2015_influential_cleaned.csv` (100 articles) - After cleaning + filtering
3. `final_tsla_news_cleaned.csv` (76 articles) - **FINAL** (combined with JSON, deduped)

#### INTC Processing Chain
1. `nasdaq_intc_mentions_jan2015.csv` (556 articles) - Raw keyword search results
2. `nasdaq_intc_jan2015_influential_cleaned.csv` (249 articles) - After cleaning + filtering
3. `final_intc_news_cleaned.csv` (238 articles) - **FINAL** (combined with JSON, deduped)

#### Other Filtered Data
- `csco_jan2015.csv` - Cisco (CSCO) data for January 2015
- `pcln_jan2015.csv` - Priceline (PCLN) data for January 2015

## 📊 Data Statistics

### Coverage Summary

| Ticker | Final Articles | Date Range | Publishers | Source Mix |
|--------|---------------|------------|------------|------------|
| TSLA   | 76            | Jan 2-30, 2015 | 7 | 57 CSV + 19 JSON |
| INTC   | 238           | Jan 1-30, 2015 | 13 | 227 CSV + 22 JSON |

### Data Quality Metrics

**TSLA Final Dataset:**
- Influential articles: 100% (filtered)
- Promotional content: 0% (removed)
- Duplicate articles: 0 (removed 43 duplicates)
- Articles with content: High (JSON sources have full content)

**INTC Final Dataset:**
- Influential articles: 100% (filtered)
- Promotional content: 0% (removed)
- Duplicate articles: 0 (removed 33 duplicates)
- Articles with content: High (JSON sources have full content)

## 🔄 Data Processing Pipeline

### Step 1: Keyword Search
Extract articles mentioning ticker symbols or company names from large datasets.

**Example:** Search for "TSLA" or "Tesla" in headlines and article content.

### Step 2: Ad Removal & Cleaning
Remove promotional content using pattern matching:
- Zacks Investment Research ads
- Motley Fool advertisements
- "Free Stock Analysis Report" lists
- "Top X Most-Read Stories" sections
- Engagement prompts ("What do you think?", "Register now", etc.)
- HTML tags (from JSON sources)
- Footer disclaimers and legal text

### Step 3: Influential Article Filtering
Keep only articles that could influence stock prices:

**Inclusion Criteria:**
- ✓ Ticker/company in article title
- ✓ Earnings, revenue, guidance announcements
- ✓ Product launches and releases
- ✓ Analyst ratings and price targets
- ✓ CEO statements and major announcements
- ✓ Stock price movement tracking
- ✓ Articles with substantial coverage (3+ mentions + material keywords)
- ✓ Articles tagged with the stock symbol

**Exclusion Criteria:**
- ✗ Passing mentions in unrelated articles
- ✗ Generic industry news without specific company focus
- ✗ Promotional/advertisement content
- ✗ Top stories lists featuring multiple unrelated stocks

### Step 4: Deduplication & Combination
- Combine CSV and JSON sources
- Remove duplicates by URL
- Sort by date
- Standardize column format

## 📋 Column Definitions

### Standard CSV Format

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Date` | datetime | Publication date/time | 2015-01-14 09:30:00+00:00 |
| `Article_title` | string | Headline | "Tesla CEO Announces Production Targets" |
| `Stock_symbol` | string | Ticker symbol | TSLA |
| `Url` | string | Article URL | https://www.nasdaq.com/articles/... |
| `Publisher` | string | Source/publisher | Benzinga, Nasdaq, Zacks |
| `Author` | string | Article author | May be empty |
| `Article` | string | Full article text | Main content (cleaned) |
| `Summary` | string | Article summary | May be empty |
| `influence_reason` | string | Why article is influential | "Material news in title" |

## 🎯 Use Cases

### 1. Stock Price Prediction
Use news sentiment and content to predict stock movements:
```python
import pandas as pd

# Load data
tsla = pd.read_csv('data/news/final_tsla_news_cleaned.csv')
tsla['Date'] = pd.to_datetime(tsla['Date'])

# Combine with stock price data
# Analyze sentiment impact on next-day returns
```

### 2. Event Study Analysis
Study market reaction to specific events:
```python
# Major events in January 2015
# TSLA: Jan 14 - CEO announces losses to persist until 2020, China sales weak
# INTC: Jan 6 - Broadwell processor launch at CES 2015

# Measure abnormal returns around these dates
```

### 3. Sentiment Analysis
Train sentiment models on influential news:
```python
# Articles are pre-filtered for influence
# Can be labeled with stock price direction for supervised learning
```

## 🔍 Key Events Covered

### Tesla (TSLA) - January 2015
- **Jan 2:** Roadster upgrade announcement
- **Jan 13-14:** **MAJOR** - Weak Q4 China sales, losses to persist until 2020
- **Jan 14:** Stock crashed 18.9% in pre-market trading
- **Jan 15:** Analysis of stock fall, Model X promises
- **Jan 16:** Production target announcement (500k by 2020)
- **Jan 22:** Morgan Stanley price target reduction
- **Jan 30:** Q4 earnings release date announced

### Intel (INTC) - January 2015
- **Jan 5-7:** Broadwell fifth-generation processor launch at CES 2015
- **Jan 5:** $24.8M investment in Vuzix for wearables market
- **Jan 6-9:** CES 2015 product announcements
- Throughout: Analyst coverage and stock price movements
- PC market dynamics and semiconductor industry analysis

## 📚 Data Sources

1. **NASDAQ External Data**
   - Large-scale financial news aggregation
   - Multiple publishers (Benzinga, Zacks, Seeking Alpha, etc.)
   - Coverage: 2009-2023

2. **JSON News API**
   - Structured news data with full content
   - High-quality article text
   - Rich metadata (author, source, images, etc.)

## ⚙️ Processing Scripts

All processing scripts are located in `/scripts/data_cleaning/`:

- `clean_news_articles.py` - Remove promotional content
- `filter_influential_articles.py` - Filter for influential articles
- `combine_news_data.py` - Combine JSON and CSV sources

See `/scripts/data_cleaning/README.md` for usage instructions.

## 📝 Data Quality Notes

### Strengths
- ✓ Cleaned of all promotional content
- ✓ Filtered for stock-influential articles only
- ✓ No duplicates
- ✓ Standardized format
- ✓ Rich metadata (publisher, date, URL)
- ✓ Covers major market events

### Limitations
- Limited to January 2015 time period
- Only two tickers (TSLA, INTC) fully processed
- Some articles may lack full text (especially older CSV data)
- Author field often empty in CSV sources
- Publisher diversity varies (TSLA: 7, INTC: 13)

### Missing Data Patterns

**CSV Sources:**
- ~90% missing full article text (only titles/URLs)
- ~91% missing author information
- 100% missing article summaries (Lsa, Luhn, Textrank, Lexrank columns empty)

**JSON Sources:**
- Complete article text ✓
- Author information ✓
- Summaries included ✓

## 🚀 Quick Start

```python
import pandas as pd

# Load final datasets
tsla = pd.read_csv('data/news/final_tsla_news_cleaned.csv', parse_dates=['Date'])
intc = pd.read_csv('data/news/final_intc_news_cleaned.csv', parse_dates=['Date'])

# Basic stats
print(f"TSLA articles: {len(tsla)}")
print(f"Date range: {tsla['Date'].min()} to {tsla['Date'].max()}")
print(f"Publishers: {tsla['Publisher'].nunique()}")

# Filter by date
jan_14_news = tsla[tsla['Date'].dt.date == pd.to_datetime('2015-01-14').date()]
print(f"Articles on Jan 14, 2015: {len(jan_14_news)}")

# Most common influence reasons
print("\nInfluence breakdown:")
print(tsla['influence_reason'].value_counts().head())
```

## 📄 Citation

If using this data, please note:
- Original data sources: NASDAQ, Benzinga, Zacks, and other financial news publishers
- Processing: D-MEADS project data cleaning pipeline
- Time period: January 2015
- Tickers: TSLA (Tesla Motors, Inc.), INTC (Intel Corporation)

## 🔄 Updating the Data

To process additional tickers or time periods:

```bash
# 1. Search for keyword mentions
python scripts/search_mentions.py <ticker> <start_date> <end_date>

# 2. Clean articles
python scripts/data_cleaning/clean_news_articles.py input.csv cleaned.csv

# 3. Filter for influential
python scripts/data_cleaning/filter_influential_articles.py \
    cleaned.csv influential.csv <ticker> <company_name>

# 4. Combine with existing data (if needed)
python scripts/data_cleaning/combine_news_data.py
```

## 📞 Questions?

For questions about the data processing pipeline, see:
- `/scripts/data_cleaning/README.md` - Processing scripts documentation
- Project main README - Overall project documentation
