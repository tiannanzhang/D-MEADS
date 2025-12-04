import pandas as pd

# Read the CSV
csv_path = 'data/news/nasdaq_tsla_mentions_jan2015.csv'
df = pd.read_csv(csv_path)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

print(f"Total articles: {len(df)}\n")
print("=" * 100)

# Show 5 entries at a time
batch_size = 5
total_batches = (len(df) + batch_size - 1) // batch_size

for batch_num in range(total_batches):
    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, len(df))

    print(f"\n{'#' * 100}")
    print(f"BATCH {batch_num + 1}/{total_batches} - Entries {start_idx + 1} to {end_idx}")
    print(f"{'#' * 100}\n")

    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        print(f"\n{'=' * 100}")
        print(f"ENTRY #{i + 1}")
        print(f"{'=' * 100}")
        print(f"Date: {row['Date']}")
        print(f"Title: {row['Article_title']}")
        print(f"Stock Symbol: {row['Stock_symbol']}")
        print(f"Publisher: {row['Publisher']}")
        print(f"URL: {row['Url']}")

        # Show article content preview
        if pd.notna(row['Article']):
            article = str(row['Article'])
            # Find Tesla mentions
            article_lower = article.lower()
            if 'tesla' in article_lower or 'tsla' in article_lower:
                # Show context around Tesla mention
                for keyword in ['tesla', 'tsla']:
                    if keyword in article_lower:
                        idx = article_lower.find(keyword)
                        start = max(0, idx - 150)
                        end = min(len(article), idx + 300)
                        context = article[start:end]
                        print(f"\nTesla Context:")
                        print(f"...{context}...")
                        break
            else:
                print(f"\nArticle Preview (first 300 chars):")
                print(article[:300] + "...")
        else:
            print(f"\nNo article content available")

        print(f"\n{'-' * 100}")

    print(f"\n\nEnd of batch {batch_num + 1}/{total_batches}")
    print("=" * 100)

    if batch_num < total_batches - 1:
        input("\nPress Enter to see next batch...")

print("\n\nAll entries displayed!")
