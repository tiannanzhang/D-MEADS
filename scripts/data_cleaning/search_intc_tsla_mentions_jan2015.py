import pandas as pd
from datetime import datetime
import re

csv_path = 'data/news/nasdaq_exteral_data.csv'
chunk_size = 100000

print("Searching for INTC/Intel and TSLA/Tesla mentions in January 2015...")
print("=" * 80)

intc_mentions = []
tsla_mentions = []

# Keywords to search for (case insensitive)
intc_keywords = ['intc', 'intel']
tsla_keywords = ['tsla', 'tesla']

def contains_keyword(text, keywords):
    """Check if text contains any of the keywords (case insensitive)"""
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    return any(keyword in text_lower for keyword in keywords)

print("\nProcessing file in chunks...")

# Process file in chunks
total_jan2015 = 0
for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)):
    # Convert Date column to datetime
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce', utc=True)

    # Filter for January 2015
    jan_2015_mask = (chunk['Date'].dt.year == 2015) & (chunk['Date'].dt.month == 1)
    jan_2015_chunk = chunk[jan_2015_mask]
    total_jan2015 += len(jan_2015_chunk)

    if len(jan_2015_chunk) > 0:
        # Search for INTC/Intel mentions
        for idx, row in jan_2015_chunk.iterrows():
            title_match_intc = contains_keyword(row['Article_title'], intc_keywords)
            article_match_intc = contains_keyword(row['Article'], intc_keywords)

            if title_match_intc or article_match_intc:
                intc_mentions.append({
                    'row': row,
                    'title_match': title_match_intc,
                    'article_match': article_match_intc
                })

            # Search for TSLA/Tesla mentions
            title_match_tsla = contains_keyword(row['Article_title'], tsla_keywords)
            article_match_tsla = contains_keyword(row['Article'], tsla_keywords)

            if title_match_tsla or article_match_tsla:
                tsla_mentions.append({
                    'row': row,
                    'title_match': title_match_tsla,
                    'article_match': article_match_tsla
                })

    if (i + 1) % 20 == 0:
        print(f"  Processed {(i + 1) * chunk_size:,} rows...")

print(f"\nTotal articles in January 2015: {total_jan2015:,}")

print("\n" + "=" * 80)
print("SEARCH RESULTS")
print("=" * 80)

# Display INTC/Intel results
print(f"\n{'='*80}")
print(f"INTC/Intel Mentions: Found {len(intc_mentions)} articles")
print(f"{'='*80}")

if intc_mentions:
    # Count matches by type
    title_only = sum(1 for m in intc_mentions if m['title_match'] and not m['article_match'])
    article_only = sum(1 for m in intc_mentions if m['article_match'] and not m['title_match'])
    both = sum(1 for m in intc_mentions if m['title_match'] and m['article_match'])

    print(f"\nMatch breakdown:")
    print(f"  Title only: {title_only}")
    print(f"  Article only: {article_only}")
    print(f"  Both title and article: {both}")

    # Create dataframe from results
    intc_df = pd.DataFrame([m['row'] for m in intc_mentions])

    print(f"\nDate distribution:")
    print(intc_df['Date'].dt.date.value_counts().sort_index())

    print(f"\nPublishers (top 10):")
    print(intc_df['Publisher'].value_counts().head(10))

    print(f"\nStock symbols associated with these articles:")
    print(intc_df['Stock_symbol'].value_counts().head(20))

    print(f"\n{'-'*80}")
    print("Sample articles (first 10):")
    print(f"{'-'*80}")

    for i, mention in enumerate(intc_mentions[:10]):
        row = mention['row']
        print(f"\n[{i+1}] Date: {row['Date']}")
        print(f"Title: {row['Article_title']}")
        print(f"Stock Symbol: {row['Stock_symbol']}")
        print(f"Publisher: {row['Publisher']}")
        print(f"URL: {row['Url']}")
        print(f"Match: {'Title' if mention['title_match'] else ''}{' & ' if mention['title_match'] and mention['article_match'] else ''}{'Article' if mention['article_match'] else ''}")

        if pd.notna(row['Article']):
            article_text = str(row['Article'])
            # Find and show context around keyword
            for keyword in intc_keywords:
                if keyword in article_text.lower():
                    idx = article_text.lower().find(keyword)
                    start = max(0, idx - 100)
                    end = min(len(article_text), idx + 100)
                    context = article_text[start:end]
                    print(f"Context: ...{context}...")
                    break
        print(f"{'-'*40}")

    # Save to file
    output_file = 'data/news/nasdaq_intc_mentions_jan2015.csv'
    intc_df.to_csv(output_file, index=False)
    print(f"\nINTC mentions saved to: {output_file}")
else:
    print("\nNo articles found mentioning INTC or Intel")

# Display TSLA/Tesla results
print(f"\n{'='*80}")
print(f"TSLA/Tesla Mentions: Found {len(tsla_mentions)} articles")
print(f"{'='*80}")

if tsla_mentions:
    # Count matches by type
    title_only = sum(1 for m in tsla_mentions if m['title_match'] and not m['article_match'])
    article_only = sum(1 for m in tsla_mentions if m['article_match'] and not m['title_match'])
    both = sum(1 for m in tsla_mentions if m['title_match'] and m['article_match'])

    print(f"\nMatch breakdown:")
    print(f"  Title only: {title_only}")
    print(f"  Article only: {article_only}")
    print(f"  Both title and article: {both}")

    # Create dataframe from results
    tsla_df = pd.DataFrame([m['row'] for m in tsla_mentions])

    print(f"\nDate distribution:")
    print(tsla_df['Date'].dt.date.value_counts().sort_index())

    print(f"\nPublishers (top 10):")
    print(tsla_df['Publisher'].value_counts().head(10))

    print(f"\nStock symbols associated with these articles:")
    print(tsla_df['Stock_symbol'].value_counts().head(20))

    print(f"\n{'-'*80}")
    print("Sample articles (first 10):")
    print(f"{'-'*80}")

    for i, mention in enumerate(tsla_mentions[:10]):
        row = mention['row']
        print(f"\n[{i+1}] Date: {row['Date']}")
        print(f"Title: {row['Article_title']}")
        print(f"Stock Symbol: {row['Stock_symbol']}")
        print(f"Publisher: {row['Publisher']}")
        print(f"URL: {row['Url']}")
        print(f"Match: {'Title' if mention['title_match'] else ''}{' & ' if mention['title_match'] and mention['article_match'] else ''}{'Article' if mention['article_match'] else ''}")

        if pd.notna(row['Article']):
            article_text = str(row['Article'])
            # Find and show context around keyword
            for keyword in tsla_keywords:
                if keyword in article_text.lower():
                    idx = article_text.lower().find(keyword)
                    start = max(0, idx - 100)
                    end = min(len(article_text), idx + 100)
                    context = article_text[start:end]
                    print(f"Context: ...{context}...")
                    break
        print(f"{'-'*40}")

    # Save to file
    output_file = 'data/news/nasdaq_tsla_mentions_jan2015.csv'
    tsla_df.to_csv(output_file, index=False)
    print(f"\nTSLA mentions saved to: {output_file}")
else:
    print("\nNo articles found mentioning TSLA or Tesla")

print("\n" + "=" * 80)
print("Analysis complete!")
