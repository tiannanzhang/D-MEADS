import pandas as pd
from datetime import datetime

csv_path = 'data/news/nasdaq_exteral_data.csv'
chunk_size = 100000

print("Investigating NASDAQ dataset for INTC and TSLA in January 2015...")
print("=" * 80)

# First, check if file exists and get basic info
try:
    # Get basic info from first chunk
    first_chunk = pd.read_csv(csv_path, nrows=1000)
    print("\nColumn Names:")
    print(first_chunk.columns.tolist())
    print("\nData Types:")
    print(first_chunk.dtypes)
    print("\nFirst few rows:")
    print(first_chunk.head(3))
except FileNotFoundError:
    print(f"\n✗ ERROR: File not found at {csv_path}")
    print("\nLet me check what files are available in data/news/")
    import os
    if os.path.exists('data/news/'):
        print("\nFiles in data/news/:")
        for f in sorted(os.listdir('data/news/')):
            if f.endswith('.csv'):
                size = os.path.getsize(f'data/news/{f}')
                print(f"  {f} ({size / (1024**3):.2f} GB)")
    exit()

intc_all = []
tsla_all = []
intc_jan2015 = []
tsla_jan2015 = []

print("\nProcessing file in chunks...")

# Process file in chunks
for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, dtype={'Stock_symbol': str}, low_memory=False)):
    # Convert Date column to datetime
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce', utc=True)

    # Check for INTC
    intc_chunk = chunk[chunk['Stock_symbol'] == 'INTC']
    if len(intc_chunk) > 0:
        intc_all.append(intc_chunk)
        # Check for January 2015
        jan_2015_mask = (intc_chunk['Date'].dt.year == 2015) & (intc_chunk['Date'].dt.month == 1)
        intc_jan = intc_chunk[jan_2015_mask]
        if len(intc_jan) > 0:
            intc_jan2015.append(intc_jan)

    # Check for TSLA
    tsla_chunk = chunk[chunk['Stock_symbol'] == 'TSLA']
    if len(tsla_chunk) > 0:
        tsla_all.append(tsla_chunk)
        # Check for January 2015
        jan_2015_mask = (tsla_chunk['Date'].dt.year == 2015) & (tsla_chunk['Date'].dt.month == 1)
        tsla_jan = tsla_chunk[jan_2015_mask]
        if len(tsla_jan) > 0:
            tsla_jan2015.append(tsla_jan)

    if (i + 1) % 20 == 0:
        print(f"  Processed {(i + 1) * chunk_size:,} rows...")

print("\n" + "=" * 80)
print("OVERALL COVERAGE")
print("=" * 80)

# Analyze INTC overall
if intc_all:
    intc_df = pd.concat(intc_all, ignore_index=True)
    print(f"\nINTC: Found {len(intc_df)} total articles")
    print(f"  Date range: {intc_df['Date'].min()} to {intc_df['Date'].max()}")
    print(f"  Distribution by year:")
    print(intc_df['Date'].dt.year.value_counts().sort_index())
else:
    print("\nINTC: No articles found in entire dataset")

# Analyze TSLA overall
if tsla_all:
    tsla_df = pd.concat(tsla_all, ignore_index=True)
    print(f"\nTSLA: Found {len(tsla_df)} total articles")
    print(f"  Date range: {tsla_df['Date'].min()} to {tsla_df['Date'].max()}")
    print(f"  Distribution by year:")
    print(tsla_df['Date'].dt.year.value_counts().sort_index())
else:
    print("\nTSLA: No articles found in entire dataset")

print("\n" + "=" * 80)
print("JANUARY 2015 SPECIFIC COVERAGE")
print("=" * 80)

# Analyze INTC January 2015
if intc_jan2015:
    intc_jan_df = pd.concat(intc_jan2015, ignore_index=True)
    print(f"\n✓ INTC - January 2015: Found {len(intc_jan_df)} articles")
    print(f"\nBreakdown by date:")
    print(intc_jan_df['Date'].dt.date.value_counts().sort_index())

    print(f"\nPublishers (top 10):")
    print(intc_jan_df['Publisher'].value_counts().head(10))

    print(f"\nData completeness:")
    for col in intc_jan_df.columns:
        non_null = intc_jan_df[col].notna().sum()
        pct = (non_null / len(intc_jan_df)) * 100
        print(f"  {col}: {non_null}/{len(intc_jan_df)} ({pct:.1f}%)")

    print(f"\nSample articles (first 10):")
    print("-" * 80)
    for idx, row in intc_jan_df.head(10).iterrows():
        print(f"\nDate: {row['Date']}")
        print(f"Title: {row['Article_title']}")
        print(f"Publisher: {row['Publisher']}")
        print(f"URL: {row['Url']}")
        if pd.notna(row.get('Author')):
            print(f"Author: {row['Author']}")
        if pd.notna(row.get('Article')):
            article_preview = str(row['Article'])[:300] + "..." if len(str(row['Article'])) > 300 else str(row['Article'])
            print(f"Article Preview: {article_preview}")
        print("-" * 40)

    # Save to file
    output_file = 'data/news/nasdaq_intc_jan2015.csv'
    intc_jan_df.to_csv(output_file, index=False)
    print(f"\nINTC data saved to: {output_file}")
else:
    print("\n✗ INTC - January 2015: No articles found")

# Analyze TSLA January 2015
if tsla_jan2015:
    tsla_jan_df = pd.concat(tsla_jan2015, ignore_index=True)
    print(f"\n✓ TSLA - January 2015: Found {len(tsla_jan_df)} articles")
    print(f"\nBreakdown by date:")
    print(tsla_jan_df['Date'].dt.date.value_counts().sort_index())

    print(f"\nPublishers (top 10):")
    print(tsla_jan_df['Publisher'].value_counts().head(10))

    print(f"\nData completeness:")
    for col in tsla_jan_df.columns:
        non_null = tsla_jan_df[col].notna().sum()
        pct = (non_null / len(tsla_jan_df)) * 100
        print(f"  {col}: {non_null}/{len(tsla_jan_df)} ({pct:.1f}%)")

    print(f"\nSample articles (first 10):")
    print("-" * 80)
    for idx, row in tsla_jan_df.head(10).iterrows():
        print(f"\nDate: {row['Date']}")
        print(f"Title: {row['Article_title']}")
        print(f"Publisher: {row['Publisher']}")
        print(f"URL: {row['Url']}")
        if pd.notna(row.get('Author')):
            print(f"Author: {row['Author']}")
        if pd.notna(row.get('Article')):
            article_preview = str(row['Article'])[:300] + "..." if len(str(row['Article'])) > 300 else str(row['Article'])
            print(f"Article Preview: {article_preview}")
        print("-" * 40)

    # Save to file
    output_file = 'data/news/nasdaq_tsla_jan2015.csv'
    tsla_jan_df.to_csv(output_file, index=False)
    print(f"\nTSLA data saved to: {output_file}")
else:
    print("\n✗ TSLA - January 2015: No articles found")

print("\n" + "=" * 80)
print("Analysis complete!")
