"""
Filter news articles to keep only those that would influence stock price.

Usage:
    python filter_influential_articles.py <input_csv> <output_csv> <ticker>

Example:
    python filter_influential_articles.py nasdaq_tsla_mentions.csv tsla_influential.csv TSLA
"""

import pandas as pd
import sys

def is_influential_article(row, ticker, company_name):
    """Determine if an article would influence the stock"""

    title = str(row['Article_title']).lower() if pd.notna(row['Article_title']) else ""
    article = str(row['Article']).lower() if pd.notna(row['Article']) else ""
    ticker_lower = ticker.lower()
    company_lower = company_name.lower()

    # Check if ticker/company in title (strong signal)
    if ticker_lower in title or company_lower in title:
        # HIGH influence: Company-specific news
        if any(keyword in title for keyword in [
            'earnings', 'quarter', 'q1', 'q2', 'q3', 'q4', 'revenue', 'profit', 'loss',
            'sales', 'delivery', 'deliveries', 'production', 'shipment',
            'upgrade', 'downgrade', 'target', 'rating', 'analyst',
            'ceo', 'recall', 'investigation', 'lawsuit',
            'acquisition', 'deal', 'partnership', 'plant', 'factory',
            'product', 'launch', 'release', 'unveil'
        ]):
            return True, "Company-specific material news in title"

        # Stock movement tracking
        if any(keyword in title for keyword in [
            'movers', 'stock', 'shares', 'drops', 'falls', 'rises', 'gains',
            'tumbles', 'soars', 'plunges', 'rallies'
        ]):
            return True, "Stock price movement tracking"

        # Featured prominently in title
        if company_lower in title[:50] or ticker_lower in title[:50]:
            return True, f"{ticker} featured in title"

    # Check article content
    if article and article != 'nan':
        # Count mentions
        mention_count = article.count(ticker_lower) + article.count(company_lower)
        word_count = len(article.split())

        if mention_count >= 3 and word_count > 100:
            # Look for material content (ticker-specific keywords can be customized)
            material_keywords = [
                ticker_lower, company_lower,
                'earnings', 'revenue', 'guidance', 'forecast',
                'ceo', 'quarter', 'sales',
            ]

            material_count = sum(1 for keyword in material_keywords if keyword in article)

            if material_count >= 2:
                return True, f"Substantial coverage ({mention_count} mentions, {material_count} material keywords)"
            elif mention_count >= 5:
                return True, f"Multiple mentions ({mention_count} times)"

    # Check if tagged with ticker symbol
    if pd.notna(row.get('Stock_symbol')) and str(row['Stock_symbol']).upper() == ticker.upper():
        return True, f"Tagged with {ticker} symbol"

    return False, ""

def filter_influential(input_csv, output_csv, ticker, company_name=None):
    """Filter articles to keep only influential ones"""

    # Infer company name from ticker if not provided
    ticker_to_company = {
        'TSLA': 'Tesla',
        'INTC': 'Intel',
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        # Add more as needed
    }

    if company_name is None:
        company_name = ticker_to_company.get(ticker.upper(), ticker)

    print(f"Filtering for {ticker} ({company_name}) influential articles...")
    print(f"Reading {input_csv}...")

    df = pd.read_csv(input_csv)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Total articles: {len(df)}")
    print("\nAnalyzing articles for stock influence...")

    influential_articles = []

    for idx, row in df.iterrows():
        is_influential, reason = is_influential_article(row, ticker, company_name)

        if is_influential:
            influential_articles.append({
                'index': idx,
                'reason': reason
            })

    print(f"\nFound {len(influential_articles)} influential articles out of {len(df)}")

    # Create influential dataframe
    influential_indices = [a['index'] for a in influential_articles]
    influential_df = df.iloc[influential_indices].copy()

    # Add influence reason
    influential_df['influence_reason'] = [a['reason'] for a in influential_articles]

    # Save to CSV
    influential_df.to_csv(output_csv, index=False)

    print(f"\nInfluential articles saved to: {output_csv}")

    print(f"\n{'='*100}")
    print("BREAKDOWN BY INFLUENCE REASON")
    print(f"{'='*100}")
    print(influential_df['influence_reason'].value_counts())

    print(f"\n{'='*100}")
    print("SAMPLE INFLUENTIAL ARTICLES (first 5)")
    print(f"{'='*100}")

    for idx, row in influential_df.head(5).iterrows():
        print(f"\nDate: {row['Date'].strftime('%Y-%m-%d')}")
        print(f"Title: {row['Article_title'][:100]}")
        print(f"Reason: {row['influence_reason']}")
        print("-" * 80)

    print(f"\n{'='*100}")
    print(f"Summary: Filtered from {len(df)} to {len(influential_df)} influential articles")
    print(f"Reduction: {(1 - len(influential_df)/len(df))*100:.1f}%")
    print(f"{'='*100}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python filter_influential_articles.py <input_csv> <output_csv> <ticker> [company_name]")
        print("\nExample:")
        print("  python filter_influential_articles.py data.csv influential.csv TSLA Tesla")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    ticker = sys.argv[3]
    company_name = sys.argv[4] if len(sys.argv) > 4 else None

    filter_influential(input_csv, output_csv, ticker, company_name)
    print("\nDONE!")
