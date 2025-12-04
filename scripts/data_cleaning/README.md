# Data Cleaning Scripts

This folder contains scripts for cleaning and filtering financial news articles.

## Scripts

### 1. `clean_news_articles.py`

Removes promotional content, advertisements, and engagement prompts from news articles.

**What it removes:**
- Zacks Investment Research promotional content
- Motley Fool advertisements
- "Free Stock Analysis Report" lists
- "Top X Most-Read Stories" lists
- Engagement prompts ("What do you think?", "Register for account", etc.)
- Video tags, image credits, editorial notes
- Footer content and disclaimers

**Usage:**
```bash
python clean_news_articles.py <input_csv> <output_csv>
```

**Example:**
```bash
python clean_news_articles.py nasdaq_tsla_mentions_jan2015.csv nasdaq_tsla_mentions_jan2015_cleaned.csv
```

### 2. `filter_influential_articles.py`

Filters articles to keep only those that would likely influence stock price.

**What it keeps:**
- Articles with ticker/company in title
- Earnings, revenue, guidance announcements
- Product launches and releases
- Analyst upgrades/downgrades and price targets
- CEO statements and major announcements
- Stock price movement tracking
- Articles with substantial coverage (multiple mentions with material keywords)
- Articles tagged with the stock symbol

**Usage:**
```bash
python filter_influential_articles.py <input_csv> <output_csv> <ticker> [company_name]
```

**Examples:**
```bash
# Tesla
python filter_influential_articles.py \
    nasdaq_tsla_mentions_jan2015_cleaned.csv \
    nasdaq_tsla_jan2015_influential.csv \
    TSLA Tesla

# Intel
python filter_influential_articles.py \
    nasdaq_intc_mentions_jan2015_cleaned.csv \
    nasdaq_intc_jan2015_influential.csv \
    INTC Intel
```

## Typical Workflow

1. **Clean articles** - Remove promotional content:
   ```bash
   python clean_news_articles.py raw_data.csv cleaned_data.csv
   ```

2. **Filter for influential** - Keep only stock-moving articles:
   ```bash
   python filter_influential_articles.py \
       cleaned_data.csv \
       influential_data.csv \
       TICKER CompanyName
   ```

## Example: January 2015 TSLA and INTC Data

### Tesla (TSLA)
```bash
# Clean
python clean_news_articles.py \
    data/news/nasdaq_tsla_mentions_jan2015.csv \
    data/news/nasdaq_tsla_mentions_jan2015_cleaned.csv

# Filter
python filter_influential_articles.py \
    data/news/nasdaq_tsla_mentions_jan2015_cleaned.csv \
    data/news/nasdaq_tsla_jan2015_influential_cleaned.csv \
    TSLA Tesla
```

**Results:**
- Original: 119 articles
- Cleaned: 119 articles (370 ad patterns removed)
- Influential: 100 articles (16% reduction)

### Intel (INTC)
```bash
# Clean
python clean_news_articles.py \
    data/news/nasdaq_intc_mentions_jan2015.csv \
    data/news/nasdaq_intc_mentions_jan2015_cleaned.csv

# Filter
python filter_influential_articles.py \
    data/news/nasdaq_intc_mentions_jan2015_cleaned.csv \
    data/news/nasdaq_intc_jan2015_influential_cleaned.csv \
    INTC Intel
```

**Results:**
- Original: 556 articles
- Cleaned: 556 articles (1,542 ad patterns removed)
- Influential: 249 articles (55% reduction)

## Output

Both scripts output:
- Cleaned/filtered CSV file
- Statistics summary
- Verification of removed content
- Sample articles

## Requirements

```bash
pip install pandas
```

## Notes

- All patterns are case-insensitive
- The cleaning process preserves the original CSV structure
- Both scripts can be customized by editing the pattern lists
- For new tickers, add them to the `ticker_to_company` dictionary in `filter_influential_articles.py`
