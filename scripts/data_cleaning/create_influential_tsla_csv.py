import pandas as pd

# Read the full dataset
df = pd.read_csv("data/news/nasdaq_tsla_mentions_jan2015.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

# Influential article indices (1-based, so subtract 1 for 0-based indexing)
# Based on manual review of all 119 articles
influential_indices = [
    # Jan 2: Roadster upgrade + stock movements
    3,
    5,
    6,
    # Jan 5: Valuation concerns + stock down
    8,
    10,
    11,
    12,
    13,
    # Jan 6: Gigafactory
    14,
    15,
    # Jan 7: Patent sharing
    16,
    17,
    18,
    # Jan 8: Stock decline on oil prices
    19,
    20,
    # Jan 12-13: Auto show, China sales miss (MAJOR)
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    # Jan 14: MAJOR NEWS DAY - China sales weak, losses to 2020, stock crashes
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    # Jan 15: Stock fall analysis, Connecticut sales, China issues
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    # Jan 16: Production targets, Texas plant
    77,
    79,
    80,
    81,
    82,
    # Jan 17: Valuation analysis
    83,
    84,
    # Jan 18: Auto show analysis, autonomous tech
    86,
    87,
    # Jan 20-21: Analyst views, valuations, partnerships
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    # Jan 22: Competition, analyst price target
    98,
    99,
    # Jan 23: Valuation math
    100,
    # Jan 26: NRG partnership
    104,
    # Jan 27: P/E analysis
    107,
    # Jan 28: Stock movement
    108,
    # Jan 29: Analyst ratings, industry impact
    109,
    111,
    112,
    113,
    114,
    # Jan 30: Analyst ratings, China opportunities, earnings announcement
    116,
    117,
    119,
]

# Convert to 0-based and filter
influential_indices_0based = [i - 1 for i in influential_indices]
influential_df = df.iloc[influential_indices_0based].copy()

# Add influence reason column
influence_reasons = []

for idx in influential_indices_0based:
    row = df.iloc[idx]
    title = str(row["Article_title"]).lower()

    # Categorize the reason
    if any(
        word in title
        for word in ["movers", "morning", "futures", "pre-market", "after-hours"]
    ):
        reason = "Stock price movement tracking"
    elif any(
        word in title
        for word in ["china", "sales weak", "slump", "dives", "drop", "falls"]
    ):
        reason = "Negative news - China sales/stock decline"
    elif any(
        word in title
        for word in ["target", "analyst", "price", "valuation", "undervalued"]
    ):
        reason = "Analyst commentary/price target"
    elif any(
        word in title for word in ["loses", "losses", "0.5m", "500,000", "production"]
    ):
        reason = "Production targets/financial guidance"
    elif any(word in title for word in ["roadster", "model x", "model 3", "upgrade"]):
        reason = "Product news"
    elif any(word in title for word in ["gigafactory", "plant", "factory"]):
        reason = "Facility/expansion news"
    elif any(word in title for word in ["patent", "partnership", "teams up", "ally"]):
        reason = "Partnership/strategic moves"
    elif any(word in title for word in ["ceo", "musk", "elon"]):
        reason = "CEO statements/appearances"
    elif any(
        word in title for word in ["earnings", "financial results", "release date"]
    ):
        reason = "Earnings announcement"
    elif "auto show" in title or "detroit" in title:
        reason = "Auto show coverage"
    elif any(
        word in title for word in ["bmw", "volkswagen", "supercharger", "competition"]
    ):
        reason = "Competitive landscape"
    else:
        reason = "Tesla-specific analysis"

    influence_reasons.append(reason)

influential_df["influence_reason"] = influence_reasons

# Save to CSV
output_file = "data/news/nasdaq_tsla_jan2015_influential.csv"
influential_df.to_csv(output_file, index=False)

print(f"Filtered from {len(df)} articles to {len(influential_df)} influential articles")
print(f"\nSaved to: {output_file}")
print("\nBreakdown by influence reason:")
print(influential_df["influence_reason"].value_counts())

print("\n" "=" * 100)
print("SAMPLE INFLUENTIAL ARTICLES")
print("=" * 100)

# Show some key examples
key_dates = ["2015-01-14", "2015-01-15", "2015-01-30"]
for date in key_dates:
    date_articles = influential_df[
        influential_df["Date"].dt.strftime("%Y-%m-%d") == date
    ]
    if len(date_articles) > 0:
        print(f"\n{date} ({len(date_articles)} articles):")
        for idx, row in date_articles.head(5).iterrows():
            print(f"  - {row['Article_title']}")
            print(f"    Reason: {row['influence_reason']}")

print("\n" "=" * 100)
print("DONE!")
