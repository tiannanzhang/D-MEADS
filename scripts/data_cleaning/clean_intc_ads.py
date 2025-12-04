import pandas as pd
import re

# Read the INTC mentions CSV
csv_path = 'data/news/nasdaq_intc_mentions_jan2015.csv'
df = pd.read_csv(csv_path)

print(f"Total INTC articles: {len(df)}")
print("\nCleaning promotional/ad content...\n")

# Common ad patterns to remove (same as TSLA cleaning)
ad_patterns = [
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

# Statistics
articles_cleaned = 0
total_removals = 0

# Clean each article
for idx, row in df.iterrows():
    if pd.notna(row['Article']):
        original_article = str(row['Article'])
        cleaned_article = original_article

        removals_in_article = 0

        # Apply each pattern
        for pattern in ad_patterns:
            matches = re.findall(pattern, cleaned_article, re.IGNORECASE | re.DOTALL)
            if matches:
                cleaned_article = re.sub(pattern, '', cleaned_article, flags=re.IGNORECASE | re.DOTALL)
                removals_in_article += len(matches)

        # Remove multiple consecutive newlines
        cleaned_article = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_article)

        # Trim whitespace
        cleaned_article = cleaned_article.strip()

        if cleaned_article != original_article:
            df.at[idx, 'Article'] = cleaned_article
            articles_cleaned += 1
            total_removals += removals_in_article

# Clean article titles and summaries too
for col in ['Article_title', 'Lsa_summary', 'Luhn_summary', 'Textrank_summary', 'Lexrank_summary']:
    if col in df.columns:
        for idx, row in df.iterrows():
            if pd.notna(row[col]):
                original = str(row[col])
                cleaned = original
                for pattern in ad_patterns:
                    cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
                cleaned = cleaned.strip()
                if cleaned != original:
                    df.at[idx, col] = cleaned

# Save cleaned CSV
output_file = 'data/news/nasdaq_intc_mentions_jan2015_cleaned.csv'
df.to_csv(output_file, index=False)

print("="*100)
print("CLEANING SUMMARY - INTC Articles")
print("="*100)
print(f"Total articles processed: {len(df)}")
print(f"Articles cleaned: {articles_cleaned}")
print(f"Total ad patterns removed: {total_removals}")
print(f"\nCleaned data saved to: {output_file}")

# Verification checks
print("\n" + "="*100)
print("VERIFICATION")
print("="*100)

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

print("\n" + "="*100)
print("DONE!")
