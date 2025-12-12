import pandas as pd
import numpy as np
from datetime import datetime

# Read CSV in chunks to handle large file
csv_path = "data/news/All_external.csv"
chunk_size = 100000

print("Analyzing All_external.csv...")


# First, get basic info from first chunk
first_chunk = pd.read_csv(csv_path, nrows=1000)
print("\nColumn Names:")
print(first_chunk.columns.tolist())
print("\nData Types:")
print(first_chunk.dtypes)

# Count total rows
total_rows = 0
unique_stocks = set()
dates = []
publishers = set()
authors = set()
null_counts = {col: 0 for col in first_chunk.columns}

print("\nProcessing file in chunks...")
for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
    total_rows += len(chunk)

    # Collect unique stocks
    unique_stocks.update(chunk["Stock_symbol"].dropna().unique())

    # Collect date range (sample)
    if i < 10:  # Only sample first 10 chunks for dates
        dates.extend(chunk["Date"].dropna().tolist())

    # Collect publishers and authors
    publishers.update(chunk["Publisher"].dropna().unique())
    authors.update(chunk["Author"].dropna().unique())

    # Count nulls
    for col in chunk.columns:
        null_counts[col] += chunk[col].isna().sum()

    if (i + 1) % 10 == 0:
        print(f"  Processed {(i + 1) * chunk_size:,} rows...")

print("\n")
print("SUMMARY STATISTICS")


print(f"\nTotal Rows: {total_rows:,}")
print(f"Number of Unique Stock Symbols: {len(unique_stocks)}")
print(f"Number of Unique Publishers: {len(publishers)}")
print(f"Number of Unique Authors: {len(authors)}")

# Parse and show date range
if dates:
    date_objects = []
    for d in dates[:1000]:  # Sample first 1000 dates
        try:
            date_objects.append(pd.to_datetime(d))
        except:
            pass
    if date_objects:
        print(f"\nDate Range (sampled):")
        print(f"  Earliest: {min(date_objects)}")
        print(f"  Latest: {max(date_objects)}")

print("\nNull/Empty Value Counts by Column:")
for col, count in null_counts.items():
    pct = (count / total_rows) * 100
    print(f"  {col}: {count:,} ({pct:.1f}%)")

print("\nSample Stock Symbols (first 20):")
print(sorted(list(unique_stocks))[:20])

print("\nSample Publishers (first 10):")
print(sorted(list(publishers))[:10])

print("\n")
