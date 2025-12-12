import pandas as pd
from datetime import datetime

csv_path = "data/news/All_external.csv"
chunk_size = 100000

print("Investigating INTC and TSLA coverage...")


intc_articles = []
tsla_articles = []
jan_2015_articles = []
target_tickers = ["INTC", "TSLA"]

# Process file in chunks
for i, chunk in enumerate(
    pd.read_csv(csv_path, chunksize=chunk_size, dtype={"Stock_symbol": str})
):
    # Convert Date column to datetime
    chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce", utc=True)

    # Check for INTC
    intc_chunk = chunk[chunk["Stock_symbol"] == "INTC"]
    if len(intc_chunk) > 0:
        intc_articles.append(intc_chunk)

    # Check for TSLA
    tsla_chunk = chunk[chunk["Stock_symbol"] == "TSLA"]
    if len(tsla_chunk) > 0:
        tsla_articles.append(tsla_chunk)

    # Check for any articles in January 2015
    jan_2015_mask = (chunk["Date"].dt.year == 2015) & (chunk["Date"].dt.month == 1)
    jan_2015_chunk = chunk[jan_2015_mask & chunk["Stock_symbol"].notna()]
    if len(jan_2015_chunk) > 0:
        jan_2015_articles.append(jan_2015_chunk)

    if (i + 1) % 50 == 0:
        print(f"  Processed {(i + 1) * chunk_size:,} rows...")

print("\n")
print("RESULTS")


# Analyze INTC
if intc_articles:
    intc_df = pd.concat(intc_articles, ignore_index=True)
    print(f"\nINTC: Found {len(intc_df)} articles")
    print(f"  Date range: {intc_df['Date'].min()} to {intc_df['Date'].max()}")
    print(f"  Dates distribution by year:")
    print(intc_df["Date"].dt.year.value_counts().sort_index())
    print(f"\n  Sample dates (first 10):")
    print(intc_df["Date"].head(10).tolist())
else:
    print("\nINTC: No articles found")

# Analyze TSLA
if tsla_articles:
    tsla_df = pd.concat(tsla_articles, ignore_index=True)
    print(f"\nTSLA: Found {len(tsla_df)} articles")
    print(f"  Date range: {tsla_df['Date'].min()} to {tsla_df['Date'].max()}")
    print(f"  Dates distribution by year:")
    print(tsla_df["Date"].dt.year.value_counts().sort_index())
    print(f"\n  Sample dates (first 10):")
    print(tsla_df["Date"].head(10).tolist())
else:
    print("\nTSLA: No articles found")

# Analyze January 2015
if jan_2015_articles:
    jan_2015_df = pd.concat(jan_2015_articles, ignore_index=True)
    print(f"\nJanuary 2015: Found {len(jan_2015_df)} articles (any ticker)")
    print(f"  Top tickers in January 2015:")
    print(jan_2015_df["Stock_symbol"].value_counts().head(20))
    print(f"\n  Sample articles (first 5):")
    for idx, row in jan_2015_df.head(5).iterrows():
        print(
            f"    {row['Date']} | {row['Stock_symbol']} | {row['Article_title'][:80]}"
        )
else:
    print("\nJanuary 2015: No articles found with stock symbols")

print("\n")
print("Analysis complete!")
