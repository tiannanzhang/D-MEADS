import pandas as pd
from datetime import datetime

csv_path = 'data/news/All_external.csv'
chunk_size = 100000

print("Investigating CSCO and PCLN for January 2015...")
print("=" * 80)

csco_all = []
pcln_all = []
csco_jan2015 = []
pcln_jan2015 = []

# Process file in chunks
for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, dtype={'Stock_symbol': str})):
    # Convert Date column to datetime
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce', utc=True)

    # Check for CSCO
    csco_chunk = chunk[chunk['Stock_symbol'] == 'CSCO']
    if len(csco_chunk) > 0:
        csco_all.append(csco_chunk)
        # Check for January 2015
        jan_2015_mask = (csco_chunk['Date'].dt.year == 2015) & (csco_chunk['Date'].dt.month == 1)
        csco_jan = csco_chunk[jan_2015_mask]
        if len(csco_jan) > 0:
            csco_jan2015.append(csco_jan)

    # Check for PCLN
    pcln_chunk = chunk[chunk['Stock_symbol'] == 'PCLN']
    if len(pcln_chunk) > 0:
        pcln_all.append(pcln_chunk)
        # Check for January 2015
        jan_2015_mask = (pcln_chunk['Date'].dt.year == 2015) & (pcln_chunk['Date'].dt.month == 1)
        pcln_jan = pcln_chunk[jan_2015_mask]
        if len(pcln_jan) > 0:
            pcln_jan2015.append(pcln_jan)

    if (i + 1) % 50 == 0:
        print(f"  Processed {(i + 1) * chunk_size:,} rows...")

print("\n" + "=" * 80)
print("OVERALL COVERAGE")
print("=" * 80)

# Analyze CSCO overall
if csco_all:
    csco_df = pd.concat(csco_all, ignore_index=True)
    print(f"\nCSCO: Found {len(csco_df)} total articles")
    print(f"  Date range: {csco_df['Date'].min()} to {csco_df['Date'].max()}")
    print(f"  Distribution by year:")
    print(csco_df['Date'].dt.year.value_counts().sort_index())
else:
    print("\nCSCO: No articles found in entire dataset")

# Analyze PCLN overall
if pcln_all:
    pcln_df = pd.concat(pcln_all, ignore_index=True)
    print(f"\nPCLN: Found {len(pcln_df)} total articles")
    print(f"  Date range: {pcln_df['Date'].min()} to {pcln_df['Date'].max()}")
    print(f"  Distribution by year:")
    print(pcln_df['Date'].dt.year.value_counts().sort_index())
else:
    print("\nPCLN: No articles found in entire dataset")

print("\n" + "=" * 80)
print("JANUARY 2015 SPECIFIC COVERAGE")
print("=" * 80)

# Analyze CSCO January 2015
if csco_jan2015:
    csco_jan_df = pd.concat(csco_jan2015, ignore_index=True)
    print(f"\n✓ CSCO - January 2015: Found {len(csco_jan_df)} articles")
    print(f"\nBreakdown by date:")
    print(csco_jan_df['Date'].dt.date.value_counts().sort_index())

    print(f"\nPublishers:")
    print(csco_jan_df['Publisher'].value_counts())

    print(f"\nData completeness:")
    for col in ['Article_title', 'Url', 'Publisher', 'Author', 'Article']:
        non_null = csco_jan_df[col].notna().sum()
        pct = (non_null / len(csco_jan_df)) * 100
        print(f"  {col}: {non_null}/{len(csco_jan_df)} ({pct:.1f}%)")

    print(f"\nSample articles:")
    print("-" * 80)
    for idx, row in csco_jan_df.iterrows():
        print(f"\nDate: {row['Date']}")
        print(f"Title: {row['Article_title']}")
        print(f"Publisher: {row['Publisher']}")
        print(f"URL: {row['Url']}")
        if pd.notna(row['Author']):
            print(f"Author: {row['Author']}")
        if pd.notna(row['Article']):
            article_preview = str(row['Article'])[:300] + "..." if len(str(row['Article'])) > 300 else str(row['Article'])
            print(f"Article Preview: {article_preview}")
        print("-" * 40)

    # Save to file
    output_file = 'data/news/csco_jan2015.csv'
    csco_jan_df.to_csv(output_file, index=False)
    print(f"\nCSCO data saved to: {output_file}")
else:
    print("\n✗ CSCO - January 2015: No articles found")

# Analyze PCLN January 2015
if pcln_jan2015:
    pcln_jan_df = pd.concat(pcln_jan2015, ignore_index=True)
    print(f"\n✓ PCLN - January 2015: Found {len(pcln_jan_df)} articles")
    print(f"\nBreakdown by date:")
    print(pcln_jan_df['Date'].dt.date.value_counts().sort_index())

    print(f"\nPublishers:")
    print(pcln_jan_df['Publisher'].value_counts())

    print(f"\nData completeness:")
    for col in ['Article_title', 'Url', 'Publisher', 'Author', 'Article']:
        non_null = pcln_jan_df[col].notna().sum()
        pct = (non_null / len(pcln_jan_df)) * 100
        print(f"  {col}: {non_null}/{len(pcln_jan_df)} ({pct:.1f}%)")

    print(f"\nSample articles:")
    print("-" * 80)
    for idx, row in pcln_jan_df.iterrows():
        print(f"\nDate: {row['Date']}")
        print(f"Title: {row['Article_title']}")
        print(f"Publisher: {row['Publisher']}")
        print(f"URL: {row['Url']}")
        if pd.notna(row['Author']):
            print(f"Author: {row['Author']}")
        if pd.notna(row['Article']):
            article_preview = str(row['Article'])[:300] + "..." if len(str(row['Article'])) > 300 else str(row['Article'])
            print(f"Article Preview: {article_preview}")
        print("-" * 40)

    # Save to file
    output_file = 'data/news/pcln_jan2015.csv'
    pcln_jan_df.to_csv(output_file, index=False)
    print(f"\nPCLN data saved to: {output_file}")
else:
    print("\n✗ PCLN - January 2015: No articles found")

print("\n" + "=" * 80)
print("Analysis complete!")
