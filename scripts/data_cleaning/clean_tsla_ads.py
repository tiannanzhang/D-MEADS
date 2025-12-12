import pandas as pd
import re

# Read the influential articles CSV
csv_path = "data/news/nasdaq_tsla_jan2015_influential.csv"
df = pd.read_csv(csv_path)

print(f"Total articles: {len(df)}")
print("\nSearching for ad/promotional content patterns...\n")

# Common ad patterns to remove
ad_patterns = [
    # Zacks ads
    r"Want the latest recommendations from Zacks.*?(?:>>|report)",
    r"To read this article on Zacks\.com click here.*?(?:\n|\.)",
    r"Zacks Investment Research",
    r"[A-Z\s]+\([A-Z]+\):\s*Free Stock Analysis Report",
    # Motley Fool ads
    r"Warren Buffett\'s worst auto.*?(?:nightmare|threat).*?(?:\.|investors)",
    r"Try any of our Foolish newsletter services.*?(?:days|\.).*?\.",
    r"The Motley Fool recommends.*?(?:\.|owns shares).*?\.",
    r"The Motley Fool owns shares of.*?\.",
    r"We Fools may not all hold the same opinions.*?\.",
    # Generic promotional
    r"Click to get this free report.*?(?:>>|\n)",
    r"Get 7 Best Stocks for.*?(?:Days|>>)",
    r"Download 7 Best Stocks.*?(?:\n|\.)",
    r"Click here for.*?(?:free report|more information)",
    # Metadata/footer
    r"The views and opinions expressed herein.*?(?:Nasdaq|author).*?\.",
    r"VIDEO:.*?(?:\n|$)",
    r"\[Ed\'s note:.*?\]",
    r"Read more:.*?(?:\n|$)",
    r"Read on to see why.*?(?:\n|\.)",
    r"Be sure to check back here at Fool\.com.*?(?:\n|\.)",
    r"Image source:.*?(?:\n|\.)",
    r"Source:.*?Image.*?(?:\n|\.)",
    r"\^[A-Z]+\s+data by YCharts.*?(?:\n|\.)",
    # Multi-line stock report lists
    r"(?:[A-Z][A-Z\s&\.]+\s*\([A-Z]+\):\s*Free Stock Analysis Report\s*)+",
    # Top stories/stock picks lists
    r"Top \d+ Most-Read Stories.*",
    r"Top Five.*?Stories.*",
    r"Top-Rated Stock Picks.*",
    r"\d+\.\s+[A-Z][^.]*(?:Bitcoin|Apple|Oil|iPhone).*",  # Numbered list items
    r"Stock of the Week:.*",
    r"Stock of the Day:.*",
    # Engagement/comment prompts
    r"Comment:.*",
    r"What do you think\?.*",
    r"Do you like the stock picks.*",
    r"If you want to get in on the fun.*",
    r"Be sure to leave a note.*",
    r"Give us your take in the comments.*",
    r"just register for an account.*",
    r"register for an account with Nasdaq\.com.*",
    r"start rating stocks today.*",
    # Community/welcome sections
    r"Welcome to the latest installment of community stock picks.*",
    r"Bullish Stocks: Here\'s What You Said:.*",
    r"Here\'s What You Said:.*",
    r"community stock picks and overview.*",
]

# Statistics
articles_cleaned = 0
total_removals = 0

# Clean each article
for idx, row in df.iterrows():
    if pd.notna(row["Article"]):
        original_article = str(row["Article"])
        cleaned_article = original_article

        removals_in_article = 0

        # Apply each pattern
        for pattern in ad_patterns:
            matches = re.findall(pattern, cleaned_article, re.IGNORECASE | re.DOTALL)
            if matches:
                cleaned_article = re.sub(
                    pattern, "", cleaned_article, flags=re.IGNORECASE | re.DOTALL
                )
                removals_in_article += len(matches)

        # Also remove multiple consecutive newlines
        cleaned_article = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned_article)

        # Trim whitespace
        cleaned_article = cleaned_article.strip()

        if cleaned_article != original_article:
            df.at[idx, "Article"] = cleaned_article
            articles_cleaned += 1
            total_removals += removals_in_article

            if removals_in_article > 0:
                print(f"Article #{idx+1}: Removed {removals_in_article} ad pattern(s)")
                print(f"  Title: {row['Article_title'][:80]}")

# Clean article titles and summaries too
for col in [
    "Article_title",
    "Lsa_summary",
    "Luhn_summary",
    "Textrank_summary",
    "Lexrank_summary",
]:
    if col in df.columns:
        for idx, row in df.iterrows():
            if pd.notna(row[col]):
                original = str(row[col])
                cleaned = original
                for pattern in ad_patterns:
                    cleaned = re.sub(
                        pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL
                    )
                cleaned = cleaned.strip()
                if cleaned != original:
                    df.at[idx, col] = cleaned

# Save cleaned CSV
output_file = "data/news/nasdaq_tsla_jan2015_influential_cleaned.csv"
df.to_csv(output_file, index=False)

print("\n" "=" * 100)
print("CLEANING SUMMARY")
print("=" * 100)
print(f"Articles processed: {len(df)}")
print(f"Articles cleaned: {articles_cleaned}")
print(f"Total ad patterns removed: {total_removals}")
print(f"\nCleaned data saved to: {output_file}")

# Show a before/after example
print("\n" "=" * 100)
print("BEFORE/AFTER EXAMPLE")
print("=" * 100)

# Find an article that was cleaned
for idx, row in df.iterrows():
    # Re-read original for comparison
    original_df = pd.read_csv(csv_path)
    if pd.notna(row["Article"]) and pd.notna(original_df.iloc[idx]["Article"]):
        original = str(original_df.iloc[idx]["Article"])
        cleaned = str(row["Article"])
        if len(original) != len(cleaned):
            print(f"\nArticle: {row['Article_title'][:80]}")
            print(f"\nOriginal length: {len(original)} chars")
            print(f"Cleaned length: {len(cleaned)} chars")
            print(f"Removed: {len(original) - len(cleaned)} chars")

            # Show what was removed
            if len(original) - len(cleaned) < 500:
                # Find differences
                orig_lines = set(original.split("\n"))
                clean_lines = set(cleaned.split("\n"))
                removed_lines = orig_lines - clean_lines
                if removed_lines:
                    print(f"\nRemoved content:")
                    for line in list(removed_lines)[:3]:
                        print(f"  - {line[:100]}")
            break

print("\n" "=" * 100)
print("DONE!")
