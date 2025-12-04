"""
Clean news articles by removing promotional content, ads, and engagement prompts.

Usage:
    python clean_news_articles.py <input_csv> <output_csv>
"""

import pandas as pd
import re
import sys

def get_ad_patterns():
    """Define all ad patterns to remove from articles"""
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

        # Multi-line stock report lists
        r'(?:[A-Z][A-Z\s&\.]+\s*\([A-Z]+\):\s*Free Stock Analysis Report\s*)+',

        # Top stories/stock picks lists
        r'Top \d+ Most-Read Stories.*',
        r'Top Five.*?Stories.*',
        r'Top-Rated Stock Picks.*',
        r'\d+\.\s+[A-Z][^.]*(?:Bitcoin|Apple|Oil|iPhone).*',
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

def clean_text(text, patterns):
    """Remove ad patterns from text"""
    if pd.isna(text):
        return text

    cleaned = str(text)
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

    # Remove multiple consecutive newlines
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)

    return cleaned.strip()

def clean_articles(input_csv, output_csv):
    """Clean all articles in CSV file"""

    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)

    print(f"Total articles: {len(df)}")
    print("\nCleaning promotional/ad content...")

    patterns = get_ad_patterns()

    # Statistics
    articles_cleaned = 0
    total_removals = 0

    # Clean article content
    for idx, row in df.iterrows():
        if pd.notna(row['Article']):
            original = str(row['Article'])
            cleaned = clean_text(original, patterns)

            if cleaned != original:
                df.at[idx, 'Article'] = cleaned
                articles_cleaned += 1
                removals = sum(1 for p in patterns if re.search(p, original, re.IGNORECASE | re.DOTALL))
                total_removals += removals

    # Clean other columns
    for col in ['Article_title', 'Lsa_summary', 'Luhn_summary', 'Textrank_summary', 'Lexrank_summary']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: clean_text(x, patterns))

    # Save cleaned CSV
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*100}")
    print("CLEANING SUMMARY")
    print(f"{'='*100}")
    print(f"Articles processed: {len(df)}")
    print(f"Articles cleaned: {articles_cleaned}")
    print(f"Total ad patterns removed: {total_removals}")
    print(f"\nCleaned data saved to: {output_csv}")

    # Verification
    print(f"\n{'='*100}")
    print("VERIFICATION")
    print(f"{'='*100}")

    verification_patterns = [
        "Free Stock Analysis Report",
        "Zacks.com click here",
        "What do you think",
        "Top Five Most-Read",
        "Warren Buffett's worst",
    ]

    for pattern in verification_patterns:
        count = df.astype(str).apply(lambda x: x.str.contains(pattern, case=False, na=False)).sum().sum()
        status = "✓" if count == 0 else f"⚠ {count} found"
        print(f"{pattern:40s} → {status}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_news_articles.py <input_csv> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    clean_articles(input_csv, output_csv)
    print("\nDONE!")
