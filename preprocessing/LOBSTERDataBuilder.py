import os
from utils.utils_data import (z_score_orderbook, normalize_messages, preprocess_data,
                               z_score_market_features, normalize_order_cgan,
                               extract_news_features, normalize_news_features)
import pandas as pd
import numpy as np
import constants as cst
import logging

logger = logging.getLogger(__name__)


class LOBSTERDataBuilder:
    def __init__(
        self,
        stock_name,
        data_dir,
        date_trading_days,
        split_rates,
        chosen_model,
        use_news_features=False,
        news_lookback_window=60,
        news_half_life=30
    ):
        self.n_lob_levels = cst.N_LOB_LEVELS
        self.data_dir = data_dir
        self.date_trading_days = date_trading_days
        self.stock_name = stock_name
        self.split_rates = split_rates
        self.dataframes = []
        self.news_dataframes = []  # Store news features
        self.timestamps = []  # Store actual timestamps for news alignment
        self.trading_dates = []  # Store trading dates for each split
        self.chosen_model = chosen_model
        self.use_news_features = use_news_features
        self.news_lookback_window = news_lookback_window
        self.news_half_life = news_half_life

    def prepare_save_datasets(self):
        path = "{}/{}/{}_{}_{}".format(
            self.data_dir,
            self.stock_name,
            self.stock_name,
            self.date_trading_days[0],
            self.date_trading_days[1],
        )

        self._prepare_dataframes(path)

        # Prepare news features if enabled
        if self.use_news_features:
            logger.info("Preparing news features...")
            self._prepare_news_data()

        path_where_to_save = "{}/{}".format(
            self.data_dir,
            self.stock_name,
        )

        self.train_set = pd.concat(self.dataframes[0], axis=1).values
        self.val_set = pd.concat(self.dataframes[1], axis=1).values
        self.test_set = pd.concat(self.dataframes[2], axis=1).values

        if self.use_news_features:
            self.train_news = self.news_dataframes[0].values
            self.val_news = self.news_dataframes[1].values
            self.test_news = self.news_dataframes[2].values

        self._save(path_where_to_save)


    def _prepare_dataframes(self, path):
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
        # Only read first 6 columns from message files (skip market participant ID column)
        self.MESSAGE_USECOLS = list(range(6))
        # Count only January 2015 files for proper splits
        all_files = os.listdir(path)
        january_files = [f for f in all_files if '2015-01-' in f]
        self.num_trading_days = len(january_files)//2
        split_days = self._split_days()
        split_days = [i * 2 for i in split_days]
        self._create_dataframes_splitted(path, split_days, COLUMNS_NAMES)
        # to conclude the preprocessing we normalize the dataframes
        if (self.chosen_model == cst.Models.CGAN):
            self._normalize_dataframes_gan()
        else:
            self._normalize_dataframes_TRADES()


    def _create_dataframes_splitted(self, path, split_days, COLUMNS_NAMES):
        """
        Create train/val/test dataframes from LOBSTER files.
        Now also extracts and preserves timestamps for news alignment.
        """
        # Helper function to extract trading date from filename
        def extract_date_from_filename(filename):
            """Extract date from LOBSTER filename format: TSLA_2015-01-02_..."""
            parts = filename.split('_')
            if len(parts) >= 2:
                return parts[1]  # e.g., '2015-01-02'
            return None

        # iterate over files in the data directory of self.STOCK_NAME
        # Filter to only process January 2015 files
        all_files = sorted(os.listdir(path))
        january_files = [f for f in all_files if '2015-01-' in f]

        logger.info(f"Total files in directory: {len(all_files)}")
        logger.info(f"January 2015 files to process: {len(january_files)}")

        for i, filename in enumerate(january_files):
            f = os.path.join(path, filename)
            print(f)
            if os.path.isfile(f):
                # Extract trading date from filename
                trading_date = extract_date_from_filename(filename)

                # then we create the df for the training set
                if i < split_days[0]:
                    if (i % 2) == 0:
                        if i == 0:
                            train_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"], usecols=self.MESSAGE_USECOLS)
                            current_train_date = trading_date
                        else:
                            train_message = pd.read_csv(f, names=COLUMNS_NAMES["message"], usecols=self.MESSAGE_USECOLS)
                            current_train_date = trading_date

                    else:
                        if i == 1:
                            train_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            result = preprocess_data([train_messages, train_orderbooks], self.n_lob_levels, self.chosen_model, trading_date=current_train_date)
                            if len(result) == 3:
                                train_orderbooks, train_messages, train_timestamps = result
                            else:
                                train_orderbooks, train_messages = result
                                train_timestamps = None
                            if (len(train_orderbooks) != len(train_messages)):
                                raise ValueError("train_orderbook length is different than train_messages")
                        else:
                            train_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            result = preprocess_data([train_message, train_orderbook], self.n_lob_levels, self.chosen_model, trading_date=current_train_date)
                            if len(result) == 3:
                                train_orderbook, train_message, message_timestamps = result
                                if train_timestamps is not None:
                                    train_timestamps = pd.concat([train_timestamps, message_timestamps], axis=0)
                            else:
                                train_orderbook, train_message = result
                            train_messages = pd.concat([train_messages, train_message], axis=0)
                            train_orderbooks = pd.concat([train_orderbooks, train_orderbook], axis=0)

                elif split_days[0] <= i < split_days[1]:  # then we are creating the df for the validation set
                    if (i % 2) == 0:
                        if (i == split_days[0]):
                            self.dataframes.append([train_messages, train_orderbooks])
                            if train_timestamps is not None:
                                self.timestamps.append(train_timestamps)
                                self.trading_dates.append('train')
                            val_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"], usecols=self.MESSAGE_USECOLS)
                            current_val_date = trading_date
                        else:
                            val_message = pd.read_csv(f, names=COLUMNS_NAMES["message"], usecols=self.MESSAGE_USECOLS)
                            current_val_date = trading_date
                    else:
                        if i == split_days[0] + 1:
                            val_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            result = preprocess_data([val_messages, val_orderbooks], self.n_lob_levels, self.chosen_model, trading_date=current_val_date)
                            if len(result) == 3:
                                val_orderbooks, val_messages, val_timestamps = result
                            else:
                                val_orderbooks, val_messages = result
                                val_timestamps = None
                            if (len(val_orderbooks) != len(val_messages)):
                                raise ValueError("val_orderbook length is different than val_messages")
                        else:
                            val_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            result = preprocess_data([val_message, val_orderbook], self.n_lob_levels, self.chosen_model, trading_date=current_val_date)
                            if len(result) == 3:
                                val_orderbook, val_message, message_timestamps = result
                                if val_timestamps is not None:
                                    val_timestamps = pd.concat([val_timestamps, message_timestamps], axis=0)
                            else:
                                val_orderbook, val_message = result
                            val_messages = pd.concat([val_messages, val_message], axis=0)
                            val_orderbooks = pd.concat([val_orderbooks, val_orderbook], axis=0)

                else:  # then we are creating the df for the test set

                    if (i % 2) == 0:
                        if (i == split_days[1]):
                            self.dataframes.append([val_messages, val_orderbooks])
                            if val_timestamps is not None:
                                self.timestamps.append(val_timestamps)
                                self.trading_dates.append('val')
                            test_messages = pd.read_csv(f, names=COLUMNS_NAMES["message"], usecols=self.MESSAGE_USECOLS)
                            current_test_date = trading_date
                        else:
                            test_message = pd.read_csv(f, names=COLUMNS_NAMES["message"], usecols=self.MESSAGE_USECOLS)
                            current_test_date = trading_date

                    else:
                        if i == split_days[1] + 1:
                            test_orderbooks = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            result = preprocess_data([test_messages, test_orderbooks], self.n_lob_levels, self.chosen_model, trading_date=current_test_date)
                            if len(result) == 3:
                                test_orderbooks, test_messages, test_timestamps = result
                            else:
                                test_orderbooks, test_messages = result
                                test_timestamps = None

                            if (len(test_orderbooks) != len(test_messages)):
                                raise ValueError("test_orderbook length is different than test_messages")
                        else:
                            test_orderbook = pd.read_csv(f, names=COLUMNS_NAMES["orderbook"])
                            result = preprocess_data([test_message, test_orderbook], self.n_lob_levels, self.chosen_model, trading_date=current_test_date)
                            if len(result) == 3:
                                test_orderbook, test_message, message_timestamps = result
                                if test_timestamps is not None:
                                    test_timestamps = pd.concat([test_timestamps, message_timestamps], axis=0)
                            else:
                                test_orderbook, test_message = result
                            test_messages = pd.concat([test_messages, test_message], axis=0)
                            test_orderbooks = pd.concat([test_orderbooks, test_orderbook], axis=0)

            else:
                raise ValueError("File {} is not a file".format(f))

        self.dataframes.append([test_messages, test_orderbooks])
        if test_timestamps is not None:
            self.timestamps.append(test_timestamps)
            self.trading_dates.append('test')


    def _normalize_dataframes_TRADES(self):
        # divide all the price, both of lob and messages, by 100
        for i in range(len(self.dataframes)):
            self.dataframes[i][0]["price"] = self.dataframes[i][0]["price"] / 100
            self.dataframes[i][1].loc[:, ::2] /= 100

        #apply z score to orderbooks
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][1], mean_size, mean_prices, std_size, std_prices = z_score_orderbook(self.dataframes[i][1])
            else:
                self.dataframes[i][1], _, _, _, _ = z_score_orderbook(self.dataframes[i][1], mean_size, mean_prices, std_size, std_prices)

        #apply z-score to size and prices of messages with the statistics of the train set
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][0], mean_size, mean_prices, std_size, std_prices, mean_time, std_time, mean_depth, std_depth = normalize_messages(self.dataframes[i][0])
            else:
                self.dataframes[i][0], _, _, _, _, _, _, _, _ = normalize_messages(self.dataframes[i][0], mean_size, mean_prices, std_size, std_prices, mean_time, std_time, mean_depth, std_depth)

    def _normalize_dataframes_gan(self):
        #apply z score to orderbooks
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][1], mean_spread, mean_returns, mean_vol_imb, mean_abs_vol, std_spread, std_returns, std_vol_imb, std_abs_vol = z_score_market_features(self.dataframes[i][1])
            else:
                self.dataframes[i][1], _, _, _, _, _, _, _, _ = z_score_market_features(self.dataframes[i][1], mean_spread, mean_returns, mean_vol_imb, mean_abs_vol, std_spread, std_returns, std_vol_imb, std_abs_vol)

        #apply z-score to size and prices of messages with the statistics of the train set
        for i in range(len(self.dataframes)):
            if (i == 0):
                self.dataframes[i][0], mean_size, mean_depth, mean_cancel_depth, mean_size_100, std_size, std_depth, std_cancel_depth, std_size_100 = normalize_order_cgan(self.dataframes[i][0])
            else:
                self.dataframes[i][0], _, _, _, _, _, _, _, _ = normalize_order_cgan(self.dataframes[i][0], mean_size, mean_depth, mean_cancel_depth, mean_size_100, std_size, std_depth, std_cancel_depth, std_size_100)

    def _prepare_news_data(self):
        """
        Prepare news features for train/val/test splits.

        Note: This is a placeholder implementation. In practice, you would:
        1. Load news data from NewsDataBuilder
        2. Process sentiment using SentimentAnalyzer
        3. Extract and align features with LOB events

        For now, this creates empty news features that can be populated later.
        """
        from preprocessing.NewsDataBuilder import NewsDataBuilder
        from preprocessing.SentimentAnalyzer import SentimentAnalyzer

        logger.warning("News feature extraction is currently a placeholder. " +
                      "Please ensure news data is collected and sentiment is analyzed before using.")

        # Initialize builders
        news_builder = NewsDataBuilder(data_dir=cst.NEWS_DATA_DIR)

        # Load news data
        try:
            news_df = news_builder.load_news_data(
                ticker=self.stock_name,
                start_date=self.date_trading_days[0],
                end_date=self.date_trading_days[1]
            )

            # If data exists and doesn't have sentiment, analyze it
            if len(news_df) > 0 and 'sentiment' not in news_df.columns:
                logger.info("Analyzing news sentiment...")
                analyzer = SentimentAnalyzer()
                news_df = analyzer.analyze_news_dataframe(news_df)

        except Exception as e:
            logger.warning(f"Could not load news data: {e}. Using placeholder zeros.")
            news_df = pd.DataFrame(columns=['timestamp', 'sentiment'])

        # Extract news features for each split
        for i, (messages_df, _) in enumerate(self.dataframes):
            # Use the preserved timestamps from preprocessing
            if i < len(self.timestamps) and self.timestamps[i] is not None:
                logger.info(f"Using preserved timestamps for {self.trading_dates[i] if i < len(self.trading_dates) else 'split ' + str(i)}")
                messages_with_timestamps = self.timestamps[i].copy()
                # DEBUG: Check timestamp range
                ts_min = messages_with_timestamps['timestamp'].min()
                ts_max = messages_with_timestamps['timestamp'].max()
                logger.info(f"  Timestamp range: {ts_min} to {ts_max}")
                logger.info(f"  Sample timestamps (first 3): {list(messages_with_timestamps['timestamp'].head(3))}")
            else:
                # Fallback: Create placeholder timestamps
                logger.warning("Timestamps not available in messages. Creating placeholder timestamps.")
                messages_with_timestamps = messages_df.copy()
                base_date = pd.to_datetime(self.date_trading_days[0]) + pd.Timedelta(hours=9, minutes=30)
                messages_with_timestamps['timestamp'] = base_date

            # Extract news features with rolling window and exponential weighting
            try:
                # DEBUG: Check news_df before extraction
                logger.info(f"  News data: {len(news_df)} entries available")
                if len(news_df) > 0:
                    logger.info(f"    News time range: {news_df['timestamp'].min()} to {news_df['timestamp'].max()}")

                news_features = extract_news_features(
                    messages_df=messages_with_timestamps,
                    news_df=news_df,
                    lookback_window_sec=self.news_lookback_window,
                    half_life_sec=self.news_half_life
                )
                logger.info(f"  Extracted news features: {len(news_features)} samples")
                logger.info(f"    Non-zero sentiments: {(news_features['sentiment'] != 0).sum()}/{len(news_features)}")
            except Exception as e:
                logger.warning(f"Error extracting news features: {e}. Using zeros.")
                news_features = pd.DataFrame({
                    'sentiment': [0.0] * len(messages_df),
                    'headline_count': [0] * len(messages_df)
                })

            # Normalize news features (training set computes stats, others use training stats)
            if i == 0:
                news_features, mean_sentiment, mean_headline_count, \
                    std_sentiment, std_headline_count = normalize_news_features(news_features)
            else:
                news_features, _, _, _, _ = normalize_news_features(
                    news_features, mean_sentiment, mean_headline_count,
                    std_sentiment, std_headline_count
                )

            self.news_dataframes.append(news_features)

        logger.info(f"Prepared news features for {len(self.news_dataframes)} splits")

    def _save(self, path_where_to_save):
        if self.chosen_model == cst.Models.CGAN:
            np.save(path_where_to_save + "/train_cgan.npy", self.train_set)
            np.save(path_where_to_save + "/val_cgan.npy", self.val_set)
            np.save(path_where_to_save + "/test_cgan.npy", self.test_set)
        else:
            np.save(path_where_to_save + "/train.npy", self.train_set)
            np.save(path_where_to_save + "/val.npy", self.val_set)
            np.save(path_where_to_save + "/test.npy", self.test_set)

        # Save news features if enabled
        if self.use_news_features:
            suffix = "_cgan" if self.chosen_model == cst.Models.CGAN else ""
            np.save(path_where_to_save + f"/train{suffix}_news.npy", self.train_news)
            np.save(path_where_to_save + f"/val{suffix}_news.npy", self.val_news)
            np.save(path_where_to_save + f"/test{suffix}_news.npy", self.test_news)
            logger.info(f"Saved news features to {path_where_to_save}")


    def _split_days(self):
        train = int(self.num_trading_days * self.split_rates[0])
        val = int(self.num_trading_days * self.split_rates[1]) + train
        test = int(self.num_trading_days * self.split_rates[2]) + val
        print(f"There are {train} days for training, {val - train} days for validation and {test - val} days for testing")
        return [train, val, test]


