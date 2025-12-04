import pandas as pd
from datetime import datetime

# Read CSV in chunks and filter for INTC and TSLA in January 2015
csv_path = 'data/news/All_external.csv'
chunk_size = 100000

print("Filtering for INTC and TSLA in January 2015...")
print("=" * 80)

filtered_data = []
target_tickers = ['INTC', 'TSLA']

# Process file in chunks
for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, dtype={'Stock_symbol': str})):
    # Convert Date column to datetime
    chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce', utc=True)

    # Filter for January 2015
    jan_2015_mask = (chunk['Date'].dt.year == 2015) & (chunk['Date'].dt.month == 1)
    jan_2015_chunk = chunk[jan_2015_mask]

    # Filter for INTC and TSLA
    ticker_mask = jan_2015_chunk['Stock_symbol'].isin(target_tickers)
    filtered_chunk = jan_2015_chunk[ticker_mask]

    if len(filtered_chunk) > 0:
        filtered_data.append(filtered_chunk)
        print(f"  Found {len(filtered_chunk)} articles in chunk {i+1}")

# Combine all filtered data
if filtered_data:
    df = pd.concat(filtered_data, ignore_index=True)

    print("\n" + "=" * 80)
    print("FILTERED DATA SUMMARY")
    print("=" * 80)

    print(f"\nTotal Articles Found: {len(df)}")
    print(f"\nBreakdown by Ticker:")
    print(df['Stock_symbol'].value_counts())

    print(f"\nBreakdown by Date:")
    print(df['Date'].dt.date.value_counts().sort_index())

    print(f"\nPublishers (top 10):")
    print(df['Publisher'].value_counts().head(10))

    print(f"\nData Completeness:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = (non_null / len(df)) * 100
        print(f"  {col}: {non_null}/{len(df)} ({pct:.1f}%)")

    # Show sample articles for each ticker
    print("\n" + "=" * 80)
    print("SAMPLE ARTICLES")
    print("=" * 80)

    for ticker in target_tickers:
        ticker_df = df[df['Stock_symbol'] == ticker].sort_values('Date')
        print(f"\n{ticker} - Found {len(ticker_df)} articles")
        print("-" * 80)

        if len(ticker_df) > 0:
            for idx, row in ticker_df.head(10).iterrows():
                print(f"\nDate: {row['Date']}")
                print(f"Title: {row['Article_title']}")
                print(f"Publisher: {row['Publisher']}")
                print(f"URL: {row['Url']}")
                if pd.notna(row['Author']):
                    print(f"Author: {row['Author']}")
                if pd.notna(row['Article']):
                    article_preview = str(row['Article'])[:200] + "..." if len(str(row['Article'])) > 200 else str(row['Article'])
                    print(f"Article Preview: {article_preview}")
                print("-" * 40)

            if len(ticker_df) > 10:
                print(f"\n... and {len(ticker_df) - 10} more articles for {ticker}")

    # Save filtered data to CSV
    output_file = 'data/news/jan2015_intc_tsla_filtered.csv'
    df.to_csv(output_file, index=False)
    print(f"\n" + "=" * 80)
    print(f"Filtered data saved to: {output_file}")
    print("=" * 80)

else:
    print("\nNo articles found for INTC or TSLA in January 2015")

print("\nAnalysis complete!")
