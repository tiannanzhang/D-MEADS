import pandas as pd

df = pd.read_csv('data/news/nasdaq_tsla_mentions_jan2015.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

print(f"Total articles: {len(df)}\n")

# I'll manually mark influential indices after reviewing
# Let me show all entries in a compact format for review

for i in range(len(df)):
    row = df.iloc[i]
    print(f"\n[{i+1}] {row['Date'].strftime('%Y-%m-%d')} | {row['Stock_symbol']}")
    print(f"    {row['Article_title']}")

    if pd.notna(row['Article']):
        article = str(row['Article'])
        article_lower = article.lower()

        # Find Tesla context
        for keyword in ['tesla', 'tsla']:
            if keyword in article_lower:
                idx = article_lower.find(keyword)
                start = max(0, idx - 100)
                end = min(len(article), idx + 250)
                context = article[start:end].replace('\n', ' ')
                print(f"    >> ...{context}...")
                break

    if (i + 1) % 5 == 0:
        print(f"\n{'─'*100}")
