# News Finetuning Pipeline for D-MEADS

This document describes the complete pipeline for finetuning TRADES models with news features using gated cross-attention.

## Overview

The news finetuning pipeline consists of two main stages:

1. **Preprocessing**: Load news data, perform sentiment analysis, and align with LOB data
2. **Finetuning**: Train gated cross-attention modules to incorporate news conditioning

## Quick Start

### Basic Usage

```bash
./finetune_news_pipeline.sh \
    --checkpoint data/checkpoints/TRADES/val_ema=2.5_epoch=25_TSLA_TRADES_seed_30.ckpt \
    --stock TSLA
```

### With Custom Parameters

```bash
./finetune_news_pipeline.sh \
    --checkpoint data/checkpoints/TRADES/val_ema=2.5_epoch=25_TSLA_TRADES_seed_30.ckpt \
    --stock TSLA \
    --learning_rate 1e-4 \
    --epochs 15 \
    --batch_size 128 \
    --patience 5
```

## Prerequisites

### 1. Environment Setup

Make sure you have the conda environment set up:

```bash
conda activate dmeads
```

### 2. Required Data

The pipeline requires:

- **LOBSTER Data**: Order book and message files in `data/<STOCK>/`
  - Files format: `<STOCK>_2015-01-<DD>_*.csv`

- **News Data**: Cleaned news CSV files in `data/news/`
  - `final_tsla_news_cleaned.csv` for TSLA
  - `final_intc_news_cleaned.csv` for INTC

- **Pretrained Checkpoint**: A TRADES checkpoint trained on LOB data
  - Located in `data/checkpoints/TRADES/`

### 3. Dependencies

Required Python packages:
- torch
- transformers (for FinBERT sentiment analysis)
- pandas
- numpy
- pytorch-lightning

Install with:
```bash
pip3 install torch transformers pandas numpy pytorch-lightning
```

## Pipeline Stages

### Stage 1: Preprocessing

The preprocessing stage:

1. **Loads News Data** using `NewsDataBuilder`
   - Reads cleaned news CSV files
   - Filters by date range matching LOB data

2. **Analyzes Sentiment** using `SentimentAnalyzer`
   - Uses FinBERT model for financial sentiment
   - Generates sentiment scores in range [-1, 1]

3. **Prepares Datasets** using `LOBSTERDataBuilder`
   - Loads and processes LOBSTER order book data
   - Aligns news features with LOB timestamps
   - Applies exponential decay based on news age
   - Splits into train/val/test sets
   - Saves as `.npy` files

**Output Files:**
```
data/<STOCK>/
├── train.npy         # LOB training data
├── val.npy           # LOB validation data
├── test.npy          # LOB test data
├── train_news.npy    # News features for training
├── val_news.npy      # News features for validation
└── test_news.npy     # News features for testing
```

**Skip Preprocessing:**

If you've already run preprocessing and have the `.npy` files:
```bash
./finetune_news_pipeline.sh \
    --checkpoint <path> \
    --stock <STOCK> \
    --skip_preprocessing
```

### Stage 2: Finetuning

The finetuning stage:

1. **Loads Pretrained Model**
   - Loads checkpoint from `--checkpoint` path
   - Enables news features in configuration

2. **Initializes Gated Cross-Attention**
   - Adds gated cross-attention modules to transformer layers
   - Initializes gates to 0 (model starts with base behavior)

3. **Freezes Base Model** (default)
   - Only trains gated cross-attention parameters
   - Preserves pretrained LOB knowledge
   - ~2-5% of total parameters are trainable

4. **Trains Model**
   - Uses lower learning rate than initial training
   - Monitors validation loss for early stopping
   - Saves top-k best checkpoints

**Output:**
```
data/checkpoints/TRADES/finetuned_gated_cross_attn/
├── epoch01_val0.1234.ckpt
├── epoch05_val0.0987.ckpt
├── epoch10_val0.0856.ckpt
├── last.ckpt
└── logs/
    ├── finetune_<timestamp>.log
    └── metrics/
```

## Command-Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--checkpoint <path>` | Path to pretrained checkpoint (.ckpt file) |
| `--stock <TSLA\|INTC>` | Stock symbol to finetune on |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--learning_rate <float>` | 1e-4 | Learning rate for finetuning |
| `--epochs <int>` | 15 | Number of training epochs |
| `--batch_size <int>` | 128 | Batch size |
| `--patience <int>` | 5 | Early stopping patience (epochs) |
| `--data_dir <path>` | data | Root data directory |
| `--output_dir <path>` | auto | Output directory for checkpoints |
| `--gradient_clip <float>` | 1.0 | Gradient clipping value |
| `--num_workers <int>` | 4 | Number of dataloader workers |
| `--skip_preprocessing` | false | Skip preprocessing step |
| `--no_freeze_base` | false | Train all parameters (not recommended) |
| `-h, --help` | - | Show help message |

## Examples

### Example 1: Basic TSLA Finetuning

```bash
./finetune_news_pipeline.sh \
    --checkpoint data/checkpoints/TRADES/val_ema=2.5_epoch=25_TSLA_TRADES_seed_30.ckpt \
    --stock TSLA
```

### Example 2: INTC with Custom Hyperparameters

```bash
./finetune_news_pipeline.sh \
    --checkpoint data/checkpoints/TRADES/val_ema=2.5_epoch=25_INTC_TRADES_seed_30.ckpt \
    --stock INTC \
    --learning_rate 5e-5 \
    --epochs 20 \
    --batch_size 64 \
    --patience 10
```

### Example 3: Skip Preprocessing

If preprocessing was already done:

```bash
./finetune_news_pipeline.sh \
    --checkpoint data/checkpoints/TRADES/val_ema=2.5_epoch=25_TSLA_TRADES_seed_30.ckpt \
    --stock TSLA \
    --skip_preprocessing
```

### Example 4: Train All Parameters

To train the entire model (not just gated cross-attention):

```bash
./finetune_news_pipeline.sh \
    --checkpoint data/checkpoints/TRADES/val_ema=2.5_epoch=25_TSLA_TRADES_seed_30.ckpt \
    --stock TSLA \
    --no_freeze_base \
    --learning_rate 5e-5
```

**Note:** Training all parameters may lead to catastrophic forgetting of LOB patterns. Use with caution.

### Example 5: Custom Output Directory

```bash
./finetune_news_pipeline.sh \
    --checkpoint data/checkpoints/TRADES/val_ema=2.5_epoch=25_TSLA_TRADES_seed_30.ckpt \
    --stock TSLA \
    --output_dir experiments/news_finetune_tsla_$(date +%Y%m%d)
```

## Hyperparameter Tuning Guide

### Learning Rate

- **Recommended**: 1e-4 to 1e-5
- **Too high**: May cause instability or catastrophic forgetting
- **Too low**: Slow convergence, may need more epochs

### Batch Size

- **Larger (128-256)**: More stable gradients, faster training
- **Smaller (32-64)**: Better generalization, lower memory usage
- Adjust based on available GPU memory

### Epochs & Patience

- **Default**: 15 epochs with patience 5
- **More conservative**: 20 epochs with patience 10
- Monitor validation loss to avoid overfitting

### Gradient Clipping

- **Default**: 1.0
- Prevents gradient explosion
- Increase if training is too stable
- Decrease if experiencing instability

## Troubleshooting

### Error: "News data file not found"

**Solution:** Ensure cleaned news CSV files exist:
```bash
ls data/news/final_tsla_news_cleaned.csv
ls data/news/final_intc_news_cleaned.csv
```

### Error: "LOB data not found"

**Solution:** Ensure LOBSTER data files exist:
```bash
ls data/TSLA/TSLA_2015-01-*
```

### Error: "Failed to activate conda environment"

**Solution:** Create and activate the environment:
```bash
conda create -n dmeads python=3.9
conda activate dmeads
pip3 install torch transformers pandas numpy pytorch-lightning
```

### Error: "Required preprocessed file not found"

**Solution:** Run without `--skip_preprocessing` or manually run preprocessing:
```bash
python3 main.py  # with appropriate config
```

### Out of Memory (OOM)

**Solution:** Reduce batch size:
```bash
./finetune_news_pipeline.sh \
    --checkpoint <path> \
    --stock <STOCK> \
    --batch_size 64  # or 32
```

### Slow Training

**Causes:**
- Sentiment analysis during preprocessing (one-time cost)
- Large batch size with limited resources
- Too many dataloader workers

**Solutions:**
- Use `--skip_preprocessing` after first run
- Reduce `--batch_size`
- Reduce `--num_workers`

## Advanced Usage

### Running Preprocessing Only

Create a custom script to run only preprocessing:

```python
from preprocessing.NewsDataBuilder import NewsDataBuilder
from preprocessing.SentimentAnalyzer import SentimentAnalyzer
from preprocessing.LOBSTERDataBuilder import LOBSTERDataBuilder
import constants as cst

# Load and analyze news
news_builder = NewsDataBuilder(data_dir=cst.NEWS_DATA_DIR)
news_df = news_builder.load_news_data('TSLA')

analyzer = SentimentAnalyzer()
news_df = analyzer.analyze_news_dataframe(news_df)

# Prepare datasets
data_builder = LOBSTERDataBuilder(
    stock_name='TSLA',
    data_dir=cst.DATA_DIR,
    date_trading_days=cst.DATE_TRADING_DAYS,
    split_rates=[0.6, 0.2, 0.2],
    chosen_model=cst.Models.TRADES,
    use_news_features=True,
    news_lookback_window=60,
    news_half_life=30
)
data_builder.prepare_save_datasets()
```

### Custom News Feature Configuration

Modify news feature extraction in `configuration.py`:

```python
config.USE_NEWS_FEATURES = True
config.NEWS_FEATURE_DIM = 2  # [sentiment, volume]
config.NEWS_LOOKBACK_WINDOW = 60  # minutes
config.NEWS_HALF_LIFE = 30  # minutes
```

### Evaluating Fine-tuned Models

After finetuning, evaluate with:

```bash
# Load checkpoint
checkpoint_path = "data/checkpoints/TRADES/finetuned_gated_cross_attn/epoch10_val0.0856.ckpt"

# Generate samples with news conditioning
python3 -c "
from models.diffusers.diffusion_engine import DiffusionEngine
from configuration import Configuration

config = Configuration()
config.USE_NEWS_FEATURES = True

model = DiffusionEngine.load_from_checkpoint(checkpoint_path, config=config)
# Run evaluation...
"
```

## Architecture Details

### Gated Cross-Attention Mechanism

Each transformer layer includes:
```
x_lob = self_attention(x_lob)  # Standard self-attention on LOB
x_news = cross_attention(x_lob, news_features)  # Cross-attention to news
gate = sigmoid(learnable_parameter)  # Learned gate (initialized to 0)
x_out = x_lob + gate * x_news  # Gated residual connection
```

**Benefits:**
- Gradual incorporation of news features
- Preserves base model if news is not informative
- Prevents catastrophic forgetting

### News Features

Two features per LOB event:
1. **Sentiment Score**: Weighted average of recent news sentiment
2. **News Volume**: Count of recent news articles

**Temporal Weighting:**
```
weight = exp(-age_minutes / half_life)
```

**Lookback Window:**
- Default: 60 minutes
- Captures recent market-moving news
- Balances relevance vs. coverage

## Performance Tips

1. **Use SSD Storage**: Faster data loading
2. **Sufficient RAM**: Datasets loaded into memory
3. **GPU Recommended**: Significantly faster than CPU
4. **Parallel Workers**: Adjust `--num_workers` based on CPU cores
5. **Mixed Precision**: Enabled automatically for compatible GPUs

## Citation

If you use this pipeline, please cite:

```bibtex
@article{dmeads2024,
  title={D-MEADS: Diffusion Models for Event-based Agent-based Simulations},
  author={Your Authors},
  journal={Your Journal},
  year={2024}
}
```

## Additional Resources

- **Main README**: `README.md`
- **Finetuning Details**: `README_finetuning.md`
- **TRADES Paper**: `TRADES.md`
- **Examples**: Run `./run_news_finetuning_examples.sh` to see all examples

## Support

For issues or questions:
1. Check this README
2. Review logs in output directory
3. Open an issue on GitHub
