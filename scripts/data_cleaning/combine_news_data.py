"""
Process JSON news files, clean them, filter for influential articles,
and combine with existing CSV data.
"""

import pandas as pd
import json
import re
from datetime import datetime

# Import cleaning functions
import sys
sys.path.append('.')

def get_ad_patterns():
    """Define all ad patterns to remove"""
    return [
        # Zacks ads
        r'Want the latest recommendations from Zacks.*?(?:>>|report)',
        r'To read this article on Zacks\.com click here.*?(?:\n|\.)',
        r'Zacks Investment Research',
        r'[A-Z\s]+\([A-Z]+\):\s*Free Stock Analysis Report',

        # Motley Fool ads
        r'Warren Buffett\'s worst auto.*?(?:nightmare|threat).*?(?:\.|investors)',
        r'Try any of our Foolish newsletter services.*?(?:days|\.).*?\.',
        r'The Motley Fool recommends.*?(?:\.|owns shares).*?\.',
        r'The Motley Fool owns shares of.*?\.',
        r'We Fools may not all hold the same opinions.*?\.',

        # Generic promotional
        r'Click to get this free report.*?(?:>>|\n)',
        r'Get 7 Best Stocks for.*?(?:Days|>>)',
        r'Download 7 Best Stocks.*?(?:\n|\.)',
        r'Click here for.*?(?:free report|more information)',

        # Metadata/footer
        r'The views and opinions expressed herein.*?(?:Nasdaq|author).*?\.',
        r'VIDEO:.*?(?:\n|$)',
        r'\[Ed\'s note:.*?\]',
        r'Read more:.*?(?:\n|$)',
        r'Read on to see why.*?(?:\n|\.)',
        r'Be sure to check back here at Fool\.com.*?(?:\n|\.)',
        r'Image source:.*?(?:\n|\.)',
        r'Source:.*?Image.*?(?:\n|\.)',
        r'\^[A-Z]+\s+data by YCharts.*?(?:\n|\.)',

        # HTML tags
        r'<[^>]+>',
        r'&nbsp;',
        r'&amp;',
        r'&lt;',
        r'&gt;',

        # Multi-line stock report lists
        r'(?:[A-Z][A-Z\s&\.]+\s*\([A-Z]+\):\s*Free Stock Analysis Report\s*)+',

        # Top stories/stock picks lists
        r'Top \d+ Most-Read Stories.*',
        r'Top Five.*?Stories.*',
        r'Top-Rated Stock Picks.*',
        r'Stock of the Week:.*',
        r'Stock of the Day:.*',

        # Engagement/comment prompts
        r'Comment:.*',
        r'What do you think\?.*',
        r'Do you like the stock picks.*',
        r'If you want to get in on the fun.*',
        r'Be sure to leave a note.*',
        r'Give us your take in the comments.*',
        r'just register for an account.*',
        r'register for an account with Nasdaq\.com.*',
        r'start rating stocks today.*',

        # Community/welcome sections
        r'Welcome to the latest installment of community stock picks.*',
        r'Bullish Stocks: Here\'s What You Said:.*',
        r'Here\'s What You Said:.*',
        r'community stock picks and overview.*',
    ]

def clean_text(text):
    """Remove ad patterns and HTML from text"""
    if pd.isna(text) or text == '' or text == 'nan':
        return ''

    cleaned = str(text)
    patterns = get_ad_patterns()

    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Remove multiple consecutive newlines/whitespace
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned.strip()

def json_to_dataframe(json_file, ticker):
    """Convert JSON file to DataFrame with standardized columns"""
    print(f"\nProcessing {json_file}...")

    with open(json_file, 'r') as f:
        data = json.load(f)

    print(f"  Found {len(data)} articles")

    # Convert to DataFrame with standardized column names
    rows = []
    for item in data:
        row = {
            'Date': item.get('created_at', ''),
            'Article_title': item.get('headline', ''),
            'Stock_symbol': ticker,
            'Url': item.get('url', ''),
            'Publisher': item.get('source', ''),
            'Author': item.get('author', ''),
            'Article': item.get('content', ''),
            'Summary': item.get('summary', ''),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    return df

def is_influential(row, ticker, company_name):
    """Determine if article would influence stock"""
    title = str(row['Article_title']).lower()
    article = str(row['Article']).lower()
    ticker_lower = ticker.lower()
    company_lower = company_name.lower()

    # Check title
    if ticker_lower in title or company_lower in title:
        # Material news
        if any(kw in title for kw in [
            'earnings', 'quarter', 'revenue', 'profit', 'loss', 'sales',
            'delivery', 'production', 'upgrade', 'downgrade', 'target',
            'analyst', 'ceo', 'recall', 'lawsuit', 'acquisition', 'deal',
            'product', 'launch', 'unveil'
        ]):
            return True, "Material news in title"

        # Stock movement
        if any(kw in title for kw in ['movers', 'stock', 'shares', 'drops', 'falls', 'rises']):
            return True, "Stock movement"

        # Featured
        if ticker_lower in title[:50] or company_lower in title[:50]:
            return True, f"{ticker} featured in title"

    # Check content
    if article and article != 'nan':
        mention_count = article.count(ticker_lower) + article.count(company_lower)

        if mention_count >= 3:
            material_kw = sum(1 for kw in ['earnings', 'revenue', 'quarter', 'ceo', 'sales'] if kw in article)
            if material_kw >= 2:
                return True, f"Substantial coverage ({mention_count} mentions)"

    return False, ""

def process_and_combine(ticker, company_name):
    """Process JSON and CSV files, combine into final dataset"""

    print(f"\n{'='*100}")
    print(f"Processing {ticker} ({company_name})")
    print(f"{'='*100}")

    # 1. Load and process JSON file
    json_file = f'data/news/news_data_{ticker}.json'
    json_df = json_to_dataframe(json_file, ticker)

    print(f"  Cleaning JSON articles...")
    json_df['Article'] = json_df['Article'].apply(clean_text)
    json_df['Article_title'] = json_df['Article_title'].apply(clean_text)
    json_df['Summary'] = json_df['Summary'].apply(clean_text)

    # Filter JSON for influential
    print(f"  Filtering JSON for influential articles...")
    json_influential = []
    for idx, row in json_df.iterrows():
        is_inf, reason = is_influential(row, ticker, company_name)
        if is_inf:
            json_influential.append(idx)

    json_df_influential = json_df.iloc[json_influential].copy()
    print(f"  JSON: {len(json_df)} → {len(json_df_influential)} influential")

    # 2. Load existing cleaned CSV
    csv_file = f'data/news/nasdaq_{ticker.lower()}_jan2015_influential_cleaned.csv'
    print(f"\n  Loading existing CSV: {csv_file}")
    csv_df = pd.read_csv(csv_file)
    csv_df['Date'] = pd.to_datetime(csv_df['Date'], errors='coerce')
    print(f"  CSV articles: {len(csv_df)}")

    # 3. Combine
    print(f"\n  Combining datasets...")

    # Ensure same columns
    all_columns = set(csv_df.columns) | set(json_df_influential.columns)
    for col in all_columns:
        if col not in csv_df.columns:
            csv_df[col] = None
        if col not in json_df_influential.columns:
            json_df_influential[col] = None

    # Combine
    combined_df = pd.concat([csv_df, json_df_influential], ignore_index=True)

    # Remove duplicates by URL
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['Url'], keep='first')
    removed_dupes = initial_count - len(combined_df)

    # Sort by date
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)

    print(f"  Combined: {len(csv_df)} CSV + {len(json_df_influential)} JSON = {initial_count}")
    print(f"  Removed {removed_dupes} duplicates")
    print(f"  Final: {len(combined_df)} articles")

    # 4. Save final dataset
    output_file = f'data/news/final_{ticker.lower()}_news_cleaned.csv'
    combined_df.to_csv(output_file, index=False)
    print(f"\n  ✓ Saved to: {output_file}")

    # 5. Statistics
    print(f"\n  Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    print(f"  Publishers: {combined_df['Publisher'].nunique()} unique")

    return combined_df

# Main execution
if __name__ == "__main__":
    print("="*100)
    print("COMBINING NEWS DATA FOR TSLA AND INTC")
    print("="*100)

    # Process TSLA
    tsla_df = process_and_combine('TSLA', 'Tesla')

    # Process INTC
    intc_df = process_and_combine('INTC', 'Intel')

    print(f"\n{'='*100}")
    print("FINAL SUMMARY")
    print(f"{'='*100}")
    print(f"TSLA: {len(tsla_df)} influential articles")
    print(f"INTC: {len(intc_df)} influential articles")
    print(f"\nFinal files created:")
    print(f"  - data/news/final_tsla_news_cleaned.csv")
    print(f"  - data/news/final_intc_news_cleaned.csv")
    print(f"\n{'='*100}")
    print("DONE!")
