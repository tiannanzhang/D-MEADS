import pandas as pd

# Read the cleaned INTC CSV
df = pd.read_csv('data/news/nasdaq_intc_mentions_jan2015_cleaned.csv')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

print(f"Total INTC mentions: {len(df)}")
print("\nAnalyzing articles for Intel stock influence...\n")

influential_articles = []

for idx, row in df.iterrows():
    title = str(row['Article_title']).lower() if pd.notna(row['Article_title']) else ""
    article = str(row['Article']).lower() if pd.notna(row['Article']) else ""

    is_influential = False
    reason = ""

    # Check if it's in title (strong signal)
    if 'intc' in title or 'intel' in title:
        # HIGH influence: Intel-specific news
        if any(keyword in title for keyword in [
            'earnings', 'quarter', 'q1', 'q2', 'q3', 'q4', 'revenue', 'profit', 'loss',
            'sales', 'shipment', 'processor', 'chip',
            'upgrade', 'downgrade', 'target', 'rating', 'analyst',
            'ceo', 'recall', 'investigation', 'lawsuit',
            'broadwell', 'core', 'xeon', 'atom', 'skylake',
            'acquisition', 'deal', 'partnership', 'plant', 'factory'
        ]):
            is_influential = True
            reason = "Intel-specific news in title"
        # Stock movement tracking
        elif any(keyword in title for keyword in ['movers', 'stock', 'shares', 'drops', 'falls', 'rises', 'gains']):
            is_influential = True
            reason = "Stock price movement"
        # Analyst/industry coverage that mentions Intel prominently
        elif 'intel' in title[:50]:  # Intel mentioned early in title
            is_influential = True
            reason = "Intel featured in title"

    # Check article content if title doesn't have Intel
    elif article:
        # Count Intel mentions
        intel_count = article.count('intel') + article.count('intc')
        word_count = len(article.split())

        if intel_count >= 3 and word_count > 100:
            # Check for material Intel content
            material_keywords = [
                'intel corp', 'intel corporation', 'intc',
                'broadwell', 'skylake', 'core processor', 'xeon', 'atom',
                'chip maker', 'chipmaker', 'semiconductor',
                'earnings', 'revenue', 'guidance', 'forecast',
                'ceo', 'brian krzanich',
                '14 nanometer', '14nm', '22nm',
                'pc market', 'tablet', 'mobile chip',
                'data center', 'server',
            ]

            material_count = sum(1 for keyword in material_keywords if keyword in article)

            if material_count >= 2:
                is_influential = True
                reason = f"Substantial Intel coverage ({intel_count} mentions, {material_count} material keywords)"
            elif intel_count >= 5:
                is_influential = True
                reason = f"Multiple Intel mentions ({intel_count} times)"

    # Check if tagged with INTC stock symbol
    if pd.notna(row['Stock_symbol']) and row['Stock_symbol'] == 'INTC':
        if not is_influential:
            is_influential = True
            reason = "Tagged with INTC symbol"
        else:
            reason += " + INTC tagged"

    if is_influential:
        influential_articles.append({
            'index': idx,
            'reason': reason
        })

print(f"Found {len(influential_articles)} influential articles out of {len(df)}")

# Create influential dataframe
influential_indices = [a['index'] for a in influential_articles]
influential_df = df.iloc[influential_indices].copy()

# Add influence reason
influential_df['influence_reason'] = [a['reason'] for a in influential_articles]

# Save to CSV
output_file = 'data/news/nasdaq_intc_jan2015_influential_cleaned.csv'
influential_df.to_csv(output_file, index=False)

print(f"\nInfluential articles saved to: {output_file}")

print("\n" + "="*100)
print("BREAKDOWN BY INFLUENCE REASON")
print("="*100)
print(influential_df['influence_reason'].value_counts())

print("\n" + "="*100)
print("SAMPLE INFLUENTIAL ARTICLES")
print("="*100)

# Show top 10 examples
for idx, row in influential_df.head(10).iterrows():
    print(f"\nDate: {row['Date'].strftime('%Y-%m-%d')}")
    print(f"Title: {row['Article_title'][:100]}")
    print(f"Reason: {row['influence_reason']}")
    print("-" * 80)

print("\n" + "="*100)
print(f"Summary: Filtered from {len(df)} to {len(influential_df)} influential articles")
print("="*100)
