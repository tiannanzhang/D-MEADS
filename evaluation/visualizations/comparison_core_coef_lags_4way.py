import os
from einops import rearrange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_compute_log_returns(file_path):
    with open(file_path, 'r') as f:
        df = pd.read_csv(f)
    df.rename(columns={'Unnamed: 0': 'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df['minute'] = df['time'].dt.floor('T')
    df = df.query("ask_price_1 < 9999999")
    df = df.query("bid_price_1 < 9999999")
    df = df.query("ask_price_1 > -9999999")
    df = df.query("bid_price_1 > -9999999")
    df = df.groupby('minute')['MID_PRICE'].first().reset_index()
    df['log_return'] = np.log(df['MID_PRICE'].shift(1) / df['MID_PRICE'])
    df.dropna(inplace=True)
    return df['log_return']

def compute_correlation_by_lag(log_returns, max_lag):
    correlations = []
    for lag in range(1, max_lag + 1, 2):
        corr = log_returns.corr(log_returns.shift(lag))
        correlations.append(corr)
    return correlations

def main(real_path, TRADES_path, finetuned_TRADES_path, cgan_path):

    log_returns_real = load_and_compute_log_returns(real_path)
    log_returns_TRADES = load_and_compute_log_returns(TRADES_path)
    log_returns_finetuned = load_and_compute_log_returns(finetuned_TRADES_path)
    log_returns_cgan = load_and_compute_log_returns(cgan_path)

    correlations_real = compute_correlation_by_lag(log_returns_real, 30)
    correlations_TRADES = compute_correlation_by_lag(log_returns_TRADES, 30)
    correlations_finetuned = compute_correlation_by_lag(log_returns_finetuned, 30)
    correlations_cgan = compute_correlation_by_lag(log_returns_cgan, 30)

    plt.plot(range(1, 31, 2), correlations_real, marker='o', linestyle='-', label='Real', color='orange')
    plt.plot(range(1, 31, 2), correlations_TRADES, marker='o', linestyle='-', label='TRADES', color='blue')
    plt.plot(range(1, 31, 2), correlations_finetuned, marker='o', linestyle='-', label='Finetuned TRADES', color='purple')
    plt.plot(range(1, 31, 2), correlations_cgan, marker='o', linestyle='-', label='CGAN', color='red')

    plt.xlabel('Lag (minutes)')
    plt.ylabel('Correlation Coefficient')
    plt.title('Log Returns Autocorrelation (4-way)')
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--')
    file_name = f"corr_coef_lag_join_4way.pdf"
    dir_path = os.path.dirname(TRADES_path)
    file_path = os.path.join(dir_path, file_name)
    plt.savefig(file_path)
    plt.close()

if __name__ == '__main__':
    main()
