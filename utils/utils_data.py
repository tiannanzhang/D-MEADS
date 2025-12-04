import pandas as pd
import numpy as np
import os

import torch
import pandas
import constants as cst

def z_score_market_features(data, mean_spread=None, mean_returns=None, mean_vol_imb=None, mean_abs_vol=None, std_spread=None, std_returns=None, std_vol_imb=None, std_abs_vol=None):
    data = data.reset_index(drop=True)
    if (mean_spread is None) or (std_spread is None):
        mean_spread = data["spread"].mean()
        std_spread = data["spread"].std()
    
    if (mean_returns is None) or (std_returns is None):
        #concatenates returns_1 and returns_5
        mean_returns = pd.concat([data["returns_1"], data["returns_50"]]).mean()
        std_returns = pd.concat([data["returns_1"], data["returns_50"]]).std()
        
    if (mean_vol_imb is None) or (std_vol_imb is None):
        mean_vol_imb = pd.concat([data["volume_imbalance_1"], data["volume_imbalance_5"]]).mean()
        std_vol_imb = pd.concat([data["volume_imbalance_1"], data["volume_imbalance_5"]]).std()
        
    if (mean_abs_vol is None) or (std_abs_vol is None):
        mean_abs_vol = pd.concat([data["absolute_volume_1"], data["absolute_volume_5"]]).mean()
        std_abs_vol = pd.concat([data["absolute_volume_1"], data["absolute_volume_5"]]).std()
    
    data["spread"] = (data["spread"] - mean_spread) / std_spread
    data["returns_1"] = (data["returns_1"] - mean_returns) / std_returns
    data["returns_50"] = (data["returns_50"] - mean_returns) / std_returns
    data["volume_imbalance_1"] = (data["volume_imbalance_1"] - mean_vol_imb) / std_vol_imb
    data["volume_imbalance_5"] = (data["volume_imbalance_5"] - mean_vol_imb) / std_vol_imb
    data["absolute_volume_1"] = (data["absolute_volume_1"] - mean_abs_vol) / std_abs_vol
    data["absolute_volume_5"] = (data["absolute_volume_5"] - mean_abs_vol) / std_abs_vol
    print()
    print("mean spread ", mean_spread)
    print("std spread ", std_spread)
    print("mean returns ", mean_returns)
    print("std returns ", std_returns)
    print("mean vol imb ", mean_vol_imb)
    print("std vol imb ", std_vol_imb)
    print("mean abs vol ", mean_abs_vol)
    print("std abs vol ", std_abs_vol)
    print(data[:10])
    print()
    return data, mean_spread, mean_returns, mean_vol_imb, mean_abs_vol, std_spread, std_returns, std_vol_imb, std_abs_vol



def normalize_order_cgan(data, mean_size=None, mean_depth=None, mean_cancel_depth=None, mean_size_100=None, std_size=None, std_depth=None, std_cancel_depth=None, std_size_100=None):
    data = data.reset_index(drop=True)
    if (mean_size is None) or (std_size is None):
        mean_size = data["size"].mean()
        std_size = data["size"].std()
    
    if (mean_depth is None) or (std_depth is None):
        mean_depth = data["depth"].mean()
        std_depth = data["depth"].std()
        
    if (mean_cancel_depth is None) or (std_cancel_depth is None):
        mean_cancel_depth = data["cancel_depth"].mean()
        std_cancel_depth = data["cancel_depth"].std()
        
    if (mean_size_100 is None) or (std_size_100 is None):
        mean_size_100 = data["quantity_100"].mean()
        std_size_100 = data["quantity_100"].std()
        
    data["size"] = (data["size"] - mean_size) / std_size
    data["depth"] = (data["depth"] - mean_depth) / std_depth
    data["cancel_depth"] = (data["cancel_depth"] - mean_cancel_depth) / std_cancel_depth
    data["quantity_100"] = (data["quantity_100"] - mean_size_100) / std_size_100
    
    data["event_type"] = data["event_type"]-1.0
    data["event_type"] = data["event_type"].replace(2, 1)
    data["event_type"] = data["event_type"].replace(3, 2)
    data["event_type"] = data["event_type"]-1.0
    # order_type = -1 -> limit order
    # order_type = 0 -> cancel order
    # order_type = 1 -> market order
    print("mean size order cgan", mean_size)
    print("std size order cgan", std_size)
    print("mean depth order cgan", mean_depth)
    print("std depth order cgan", std_depth)
    print("mean cancel depth order cgan", mean_cancel_depth)
    print("std cancel depth order cgan", std_cancel_depth)
    print("mean size 100 order cgan", mean_size_100)
    print("std size 100 order cgan", std_size_100)
    print(data[:5])
    
    return data, mean_size, mean_depth, mean_cancel_depth, mean_size_100, std_size, std_depth, std_cancel_depth, std_size_100


def z_score_orderbook(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None):
    """ DONE: remember to use the mean/std of the training set, to z-normalize the test set. """
    if (mean_size is None) or (std_size is None):
        mean_size = data.iloc[:, 1::2].stack().mean()
        std_size = data.iloc[:, 1::2].stack().std()

    #do the same thing for prices
    if (mean_prices is None) or (std_prices is None):
        mean_prices = data.iloc[:, 0::2].stack().mean()
        std_prices = data.iloc[:, 0::2].stack().std()

    # apply the z score to the original data using .loc with explicit float cast
    price_cols = data.columns[0::2]
    size_cols = data.columns[1::2]

    #apply the z score to the original data
    for col in size_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_size) / std_size

    for col in price_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_prices) / std_prices

    # check if there are null values, then raise value error
    if data.isnull().values.any():
        raise ValueError("data contains null value")

    return data, mean_size, mean_prices, std_size,  std_prices


def normalize_messages(data, mean_size=None, mean_prices=None, std_size=None,  std_prices=None, mean_time=None, std_time=None, mean_depth=None, std_depth=None):

    #apply z score to prices and size column
    if (mean_size is None) or (std_size is None):
        mean_size = data["size"].mean()
        std_size = data["size"].std()

    if (mean_prices is None) or (std_prices is None):
        mean_prices = data["price"].mean()
        std_prices = data["price"].std()

    if (mean_time is None) or (std_time is None):
        mean_time = data["time"].mean()
        std_time = data["time"].std()

    if (mean_depth is None) or (std_depth is None):
        mean_depth = data["depth"].mean()
        std_depth = data["depth"].std()

    #apply the z score to the original data
    data["time"] = (data["time"] - mean_time) / std_time
    data["size"] = (data["size"] - mean_size) / std_size
    data["price"] = (data["price"] - mean_prices) / std_prices
    data["depth"] = (data["depth"] - mean_depth) / std_depth

    # check if there are null values, then raise value error
    if data.isnull().values.any():
        raise ValueError("data contains null value")

    data["event_type"] = data["event_type"]-1.0
    data["event_type"] = data["event_type"].replace(2, 1)
    data["event_type"] = data["event_type"].replace(3, 2)
    # order_type = 0 -> limit order
    # order_type = 1 -> cancel order
    # order_type = 2 -> market order

    return data, mean_size, mean_prices, std_size,  std_prices, mean_time, std_time, mean_depth, std_depth


def load_compute_normalization_terms(stock_name, data_dir, model, n_lob_levels):
    path = "{}/{}/{}_{}_{}".format(
            data_dir,
            stock_name,
            stock_name,
            cst.DATE_TRADING_DAYS[0],
            cst.DATE_TRADING_DAYS[-1]
        )
    COLUMNS_NAMES = {"orderbook": ["sell1", "vsell1", "buy1", "vbuy1",
                                       "sell2", "vsell2", "buy2", "vbuy2",
                                       "sell3", "vsell3", "buy3", "vbuy3",
                                       "sell4", "vsell4", "buy4", "vbuy4",
                                       "sell5", "vsell5", "buy5", "vbuy5",
                                       "sell6", "vsell6", "buy6", "vbuy6",
                                       "sell7", "vsell7", "buy7", "vbuy7",
                                       "sell8", "vsell8", "buy8", "vbuy8",
                                       "sell9", "vsell9", "buy9", "vbuy9",
                                       "sell10", "vsell10", "buy10", "vbuy10"],
                         "message": ["time", "event_type", "order_id", "size", "price", "direction"]}
    
    num_trading_days = len(os.listdir(path))//2
    split_rates = cst.SPLIT_RATES
    train = int(round(num_trading_days * split_rates[0]))
    val = int(round(num_trading_days * split_rates[1])) + train
    test = int(round(num_trading_days * split_rates[2])) + val
    split_days = [train, val, test]
    split_days = [i * 2 for i in split_days]
    for i, filename in enumerate(sorted(os.listdir(path))):
        f = os.path.join(path, filename)
        if os.path.isfile(f):
            # then we create the df for the training set
            if i < split_days[0]:
                if (i % 2) == 0:
                    if i == 0:
                        train_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"])
                    else:
                        train_message = pd.read_csv(f, names=COLUMNS_NAMES["message"])

                else:
                    if i == 1:
                        train_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                        train_orderbooks, train_messages = preprocess_data([train_messages, train_orderbooks], n_lob_levels, model)
                        if (len(train_orderbooks) != len(train_messages)):
                            raise ValueError("train_orderbook length is different than train_messages")
                    else:
                        train_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                        train_orderbook, train_message = preprocess_data([train_message, train_orderbook], n_lob_levels, model)
                        train_messages = pd.concat([train_messages, train_message], axis=0)
                        train_orderbooks = pd.concat([train_orderbooks, train_orderbook], axis=0)
    if model == cst.Models.TRADES:
        train_orderbooks.loc[:, ::2] /= 100
        train_messages["price"] /= 100
        _, lob_mean_size, lob_mean_prices, lob_std_size, lob_std_prices = z_score_orderbook(train_orderbooks)
        _, mean_size, mean_prices, std_size,  std_prices, mean_time, std_time, mean_depth, std_depth = normalize_messages(train_messages)
        normalization_terms = {
            "lob": (lob_mean_size, lob_std_size, lob_mean_prices, lob_std_prices),
            "event": (mean_size, std_size, mean_prices, std_prices, mean_time, std_time, mean_depth, std_depth)
        }
        return normalization_terms
    elif model == cst.Models.CGAN:
        _, mean_spread, mean_returns, mean_vol_imb, mean_abs_vol, std_spread, std_returns, std_vol_imb, std_abs_vol = z_score_market_features(train_orderbooks)
        _, mean_size, mean_depth, mean_cancel_depth, mean_size_100, std_size, std_depth, std_cancel_depth, std_size_100 = normalize_order_cgan(train_messages)
        normalization_terms = {
            "lob": (mean_spread, std_spread, mean_returns, std_returns, mean_vol_imb, std_vol_imb, mean_abs_vol, std_abs_vol, mean_cancel_depth, std_cancel_depth, mean_size_100, std_size_100, mean_depth, std_depth, mean_size, std_size),
        }
        return normalization_terms

def reset_indexes(dataframes):
    # reset the indexes of the messages and orderbooks
    dataframes[0] = dataframes[0].reset_index(drop=True)
    dataframes[1] = dataframes[1].reset_index(drop=True)
    return dataframes


def preprocess_data(dataframes, n_lob_levels, chosen_model, trading_date=None):
    """
    Preprocess LOBSTER data.

    Args:
        dataframes: [messages_df, orderbook_df]
        n_lob_levels: Number of LOB levels to keep
        chosen_model: Model type
        trading_date: Trading date string (e.g., '2015-01-02') for timestamp reconstruction

    Returns:
        tuple: (orderbook_df, messages_df, timestamps_df)
            - timestamps_df: DataFrame with actual timestamps (if trading_date provided)
    """
    dataframes = reset_indexes(dataframes)

    # take only the first n_lob_levels levels of the orderbook and drop the others
    dataframes[1] = dataframes[1].iloc[:, :n_lob_levels * cst.LEN_LEVEL]

    # take the indexes of the dataframes that are of type
    # 2 (partial deletion), 5 (execution of a hidden limit order),
    # 6 (cross trade), 7 (trading halt) and drop it
    indexes_to_drop = dataframes[0][dataframes[0]["event_type"].isin([2, 5, 6, 7])].index
    dataframes[0] = dataframes[0].drop(indexes_to_drop)
    dataframes[1] = dataframes[1].drop(indexes_to_drop)

    dataframes = reset_indexes(dataframes)

    # drop index column in messages
    dataframes[0] = dataframes[0].drop(columns=["order_id"])

    # PRESERVE ORIGINAL TIMESTAMPS before converting to inter-arrival times
    original_times_seconds = dataframes[0]["time"].values.copy()
    timestamps_df = None

    if trading_date is not None:
        # Convert seconds from midnight to actual timestamps
        base_date = pd.to_datetime(trading_date)
        timestamps_df = pd.DataFrame({
            'timestamp': base_date + pd.to_timedelta(original_times_seconds, unit='s')
        })

    # do the difference of time row per row in messages and subsitute the values with the differences
    # Store the initial value of the "time" column
    first_time = dataframes[0]["time"].values[0]
    # Calculate the difference using diff
    dataframes[0]["time"] = dataframes[0]["time"].diff()
    # Set the first value directly
    dataframes[0].iat[0, dataframes[0].columns.get_loc("time")] = first_time - 34200
        
    # add depth column to messages
    dataframes[0]["depth"] = 0

    # we compute the depth of the orders with respect to the orderbook
    # Extract necessary columns
    prices = dataframes[0]["price"].values
    directions = dataframes[0]["direction"].values
    event_types = dataframes[0]["event_type"].values
    bid_sides = dataframes[1].iloc[:, 2::4].values
    ask_sides = dataframes[1].iloc[:, 0::4].values
    
    # Initialize depth array
    depths = np.zeros(dataframes[0].shape[0], dtype=int)

    # Compute the depth of the orders with respect to the orderbook
    for j in range(1, len(prices)):
        order_price = prices[j]
        direction = directions[j]
        event_type = event_types[j]
        
        index = j if event_type == 1 else j - 1
        
        if direction == 1:
            bid_price = bid_sides[index, 0]
            depth = (bid_price - order_price) // 100
        else:
            ask_price = ask_sides[index, 0]
            depth = (order_price - ask_price) // 100
        
        depths[j] = max(depth, 0)
    
    # Assign the computed depths back to the DataFrame
    dataframes[0]["depth"] = depths
        
    # we eliminate the first row of every dataframe because we can't deduce the depth
    dataframes[0] = dataframes[0].iloc[1:, :]
    dataframes[1] = dataframes[1].iloc[1:, :]

    # Also remove first row from timestamps if they exist
    if timestamps_df is not None:
        timestamps_df = timestamps_df.iloc[1:, :].reset_index(drop=True)

    dataframes = reset_indexes(dataframes)
    if chosen_model == cst.Models.CGAN:
        # Initialize new columns
        dataframes[0]["cancel_depth"] = 0
        dataframes[0]["quantity_100"] = dataframes[0]["size"].apply(lambda x: x // 100 if x % 100 == 0 else 0)
        dataframes[0]["quantity_type"] = dataframes[0]["size"].apply(lambda x: -1 if x % 100 == 0 else 1)

        # Calculate cancel_depth using vectorization
        cancel_mask = dataframes[0]["event_type"] == 3
        shifted_prices = dataframes[1].shift(1).fillna(method='bfill')
        price_levels = shifted_prices.iloc[:, ::2].apply(lambda row: dict(zip(row, range(0, len(row)*2, 2))), axis=1)
        dataframes[0].loc[cancel_mask, "cancel_depth"] = dataframes[0].loc[cancel_mask].apply(
            lambda row: price_levels[row.name].get(row["price"], np.nan) // 2, axis=1
        )

        # Drop unnecessary columns
        dataframes[0] = dataframes[0].drop(columns=["price", "time"])

        # Shift and fill NaN values
        dataframes[1] = dataframes[1].shift(1).fillna(0)

        # Calculate volume imbalances, absolute volumes, and spread
        lob_sizes = dataframes[1].iloc[:, 1::2]  # Even columns (size)
        lob_prices = dataframes[1].iloc[:, 0::2]  # Odd columns (price)

        # Volume imbalance for level 1
        dataframes[1]["volume_imbalance_1"] = lob_sizes.iloc[:, 1] / (lob_sizes.iloc[:, 1] + lob_sizes.iloc[:, 0])

        # Volume imbalance for levels 1-5
        best_5_asks = lob_sizes.iloc[:, 1:10:2]  # Columns 1,3,5,7,9
        best_5_bids = lob_sizes.iloc[:, 0:10:2]  # Columns 0,2,4,6,8
        dataframes[1]["volume_imbalance_5"] = best_5_asks.sum(axis=1) / (best_5_asks.sum(axis=1) + best_5_bids.sum(axis=1))

        # Absolute volumes
        dataframes[1]["absolute_volume_1"] = lob_sizes.iloc[:, 1] + lob_sizes.iloc[:, 0]
        dataframes[1]["absolute_volume_5"] = (lob_sizes.iloc[:, :10]).sum(axis=1)

        # Spread
        dataframes[1]["spread"] = lob_prices.iloc[:, 0] - lob_prices.iloc[:, 1]

        # Calculate mid prices
        mid_prices = (lob_prices.iloc[:, 0] + lob_prices.iloc[:, 1]) / 2

        # Calculate order sign imbalances and returns using rolling sums
        dataframes[0]["cumulative_direction"] = dataframes[0]["direction"].cumsum()
        dataframes[1]["order_sign_imbalance_256"] = dataframes[0]["cumulative_direction"] - dataframes[0]["cumulative_direction"].shift(256, fill_value=0)
        dataframes[1]["order_sign_imbalance_128"] = dataframes[0]["cumulative_direction"].shift(128, fill_value=0) - dataframes[0]["cumulative_direction"].shift(256, fill_value=0)

        # Returns
        dataframes[1]["returns_1"] = mid_prices.pct_change(periods=1).shift(-1)
        dataframes[1]["returns_50"] = mid_prices.pct_change(periods=50).shift(-50)

        # Trim the first 255 rows
        dataframes[0] = dataframes[0].iloc[256:].reset_index(drop=True)
        dataframes[1] = dataframes[1].iloc[256:].reset_index(drop=True)

        # Also trim timestamps if they exist (for CGAN model)
        if timestamps_df is not None:
            timestamps_df = timestamps_df.iloc[256:].reset_index(drop=True)

        # Select required columns
        dataframes[1] = dataframes[1][[
            "volume_imbalance_1", "volume_imbalance_5",
            "absolute_volume_1", "absolute_volume_5",
            "spread", "order_sign_imbalance_256",
            "order_sign_imbalance_128", "returns_1", "returns_50"
        ]]

        # Fill NaN values
        dataframes[0] = dataframes[0].fillna(0)
        dataframes[1] = dataframes[1].fillna(0)
    
    # we transform the execution of a sell limit order in a buy market order and viceversa
    dataframes[0]["direction"] = dataframes[0]["direction"] * dataframes[0]["event_type"].apply(
        lambda x: -1 if x == 4 else 1)

    # Return orderbook, messages, and optionally timestamps
    if timestamps_df is not None:
        return dataframes[1], dataframes[0], timestamps_df
    else:
        return dataframes[1], dataframes[0]


def unnormalize(x, mean, std):
    return x * std + mean


def one_hot_encoding_type(data):
    encoded_data = torch.zeros(data.shape[0], data.shape[1] + 2, dtype=torch.float32)
    encoded_data[:, 0] = data[:, 0]
    # encoding order type
    one_hot_order_type = torch.nn.functional.one_hot((data[:, 1]).to(torch.int64), num_classes=3).to(
        torch.float32)
    encoded_data[:, 1:4] = one_hot_order_type
    encoded_data[:, 4:] = data[:, 2:]
    return encoded_data


def tanh_encoding_type(data):
    data[:, 1] = torch.where(data[:, 1] == 1.0, 2.0, torch.where(data[:, 1] == 2.0, 1.0, data[:, 1]))
    data[:, 1] = data[:, 1] - 1
    return data


def extract_news_features(messages_df, news_df, lookback_window_sec=60, half_life_sec=30):
    """
    Extract and align news features with LOB event timestamps.

    This function uses proven exponential weighting for time-series sentiment analysis
    (see: https://link.springer.com/article/10.1007/s42521-024-00107-2)

    For each LOB event:
    1. Counts headlines in the past lookback_window_sec seconds (60s window)
    2. Computes exponentially weighted sentiment from ALL past news (no window restriction)
       - weight(Δt) = exp(-λ * Δt) where λ = ln(2) / half_life_sec
       - More recent headlines receive higher weight
       - Headlines far in the past contribute negligibly due to exponential decay

    Args:
        messages_df: DataFrame with LOB messages (must have 'timestamp' column)
        news_df: DataFrame with columns ['timestamp', 'sentiment']
        lookback_window_sec: Window for counting headlines (default: 60 seconds)
                            Note: Only applies to headline_count, NOT sentiment
        half_life_sec: Half-life for exponential decay weighting of sentiment (default: 30 seconds)
                      Research suggests 1-7 days for daily data; we use 30s for sub-minute LOB events

    Returns:
        DataFrame with columns: ['sentiment', 'headline_count'] aligned with messages_df
    """
    # Validate inputs
    if 'timestamp' not in messages_df.columns:
        raise ValueError("messages_df must have 'timestamp' column")

    # Ensure timestamps are datetime
    messages_df = messages_df.copy()
    messages_df['timestamp'] = pd.to_datetime(messages_df['timestamp'])

    # Initialize result arrays
    num_events = len(messages_df)
    sentiments = np.zeros(num_events)
    headline_counts = np.zeros(num_events, dtype=int)

    # If no news data, return zeros
    if len(news_df) == 0 or 'timestamp' not in news_df.columns:
        return pd.DataFrame({
            'sentiment': sentiments,
            'headline_count': headline_counts
        })

    # Prepare news data
    news_df = news_df.copy()
    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])

    # Ensure timezone compatibility
    if messages_df['timestamp'].dt.tz is not None and news_df['timestamp'].dt.tz is None:
        # Make news timestamps timezone-aware (assume UTC)
        news_df['timestamp'] = news_df['timestamp'].dt.tz_localize('UTC')
    elif messages_df['timestamp'].dt.tz is None and news_df['timestamp'].dt.tz is not None:
        # Make news timestamps timezone-naive
        news_df['timestamp'] = news_df['timestamp'].dt.tz_localize(None)

    news_df = news_df.sort_values('timestamp')

    # Compute decay constant: λ = ln(2) / half_life
    decay_lambda = np.log(2) / half_life_sec

    # Convert lookback window to timedelta
    lookback_td = pd.Timedelta(seconds=lookback_window_sec)

    # For each LOB event, compute rolling window features
    for i, event_time in enumerate(messages_df['timestamp']):
        # Find news in lookback window (for headline count only)
        window_start = event_time - lookback_td
        window_mask = (news_df['timestamp'] >= window_start) & (news_df['timestamp'] <= event_time)
        window_news = news_df[window_mask]

        # Count headlines in 60-second window
        headline_counts[i] = len(window_news)

        # Compute exponentially weighted sentiment from ALL past news (no window restriction)
        past_news_mask = news_df['timestamp'] <= event_time
        past_news = news_df[past_news_mask]

        if len(past_news) > 0 and 'sentiment' in past_news.columns:
            # Calculate time deltas in seconds
            time_deltas = (event_time - past_news['timestamp']).dt.total_seconds().values

            # Compute exponential weights: w(Δt) = exp(-λ * Δt)
            weights = np.exp(-decay_lambda * time_deltas)

            # Weighted average sentiment
            sentiments_values = past_news['sentiment'].values

            # Filter out NaN values from sentiments and corresponding weights
            valid_mask = ~np.isnan(sentiments_values)
            if np.any(valid_mask):
                valid_sentiments = sentiments_values[valid_mask]
                valid_weights = weights[valid_mask]
                weight_sum = np.sum(valid_weights)

                if weight_sum > 0:
                    sentiments[i] = np.sum(valid_weights * valid_sentiments) / weight_sum
                else:
                    sentiments[i] = 0.0
            else:
                sentiments[i] = 0.0
        else:
            sentiments[i] = 0.0

    return pd.DataFrame({
        'sentiment': sentiments,
        'headline_count': headline_counts
    })


def normalize_news_features(news_features_df, mean_sentiment=None, mean_headline_count=None,
                            std_sentiment=None, std_headline_count=None):
    """
    Z-score normalize news features.

    Args:
        news_features_df: DataFrame with columns ['sentiment', 'headline_count']
        mean_sentiment: Mean sentiment (compute from training set)
        mean_headline_count: Mean headline count (compute from training set)
        std_sentiment: Std sentiment
        std_headline_count: Std headline count

    Returns:
        Tuple: (normalized_df, mean_sentiment, mean_headline_count,
                std_sentiment, std_headline_count)
    """
    news_features_df = news_features_df.copy()

    # Compute statistics if not provided (training set)
    if mean_sentiment is None or std_sentiment is None:
        mean_sentiment = news_features_df['sentiment'].mean()
        std_sentiment = news_features_df['sentiment'].std()
        if std_sentiment == 0:
            std_sentiment = 1.0  # Avoid division by zero

    if mean_headline_count is None or std_headline_count is None:
        mean_headline_count = news_features_df['headline_count'].mean()
        std_headline_count = news_features_df['headline_count'].std()
        if std_headline_count == 0:
            std_headline_count = 1.0

    # Apply z-score normalization
    news_features_df['sentiment'] = (news_features_df['sentiment'] - mean_sentiment) / std_sentiment
    news_features_df['headline_count'] = (news_features_df['headline_count'] - mean_headline_count) / std_headline_count

    print()
    print("News Feature Normalization:")
    print(f"  mean sentiment: {mean_sentiment:.4f}")
    print(f"  std sentiment: {std_sentiment:.4f}")
    print(f"  mean headline_count: {mean_headline_count:.4f}")
    print(f"  std headline_count: {std_headline_count:.4f}")
    print()

    return news_features_df, mean_sentiment, mean_headline_count, std_sentiment, std_headline_count


def to_sparse_representation(lob, n_levels):
    if not isinstance(lob, np.ndarray):
        lob = np.array(lob)
    sparse_lob = np.zeros(n_levels * 2)
    for j in range(lob.shape[0] // 2):
        if j % 2 == 0:
            ask_price = lob[0]
            current_ask_price = lob[j*2]
            depth = (current_ask_price - ask_price) // 100
            if depth < n_levels and int(lob[j*2]) != 0:
                sparse_lob[2*int(depth)] = lob[j*2+1]
        else:
            bid_price = lob[2]
            current_bid_price = lob[j*2]
            depth = (bid_price - current_bid_price) // 100
            if depth < n_levels and int(lob[j*2]) != 0:
                sparse_lob[2*int(depth)+1] = lob[j*2+1]
    return sparse_lob
