#!/bin/bash
################################################################################
# News Finetuning Pipeline for D-MEADS
#
# This script runs the complete pipeline for finetuning TRADES models with
# news features:
#   1. Preprocessing: Load and process news data with sentiment analysis
#   2. Finetuning: Train gated cross-attention modules with news conditioning
#
# Usage:
#   ./finetune_news_pipeline.sh --checkpoint <path> --stock <TSLA|INTC> [options]
#
# Example:
#   ./finetune_news_pipeline.sh \
#     --checkpoint data/checkpoints/TRADES/val_ema=2.5_epoch=25_TSLA_TRADES_seed_30.ckpt \
#     --stock TSLA \
#     --learning_rate 1e-4 \
#     --epochs 15 \
#     --batch_size 128
#
# Author: D-MEADS Team
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

################################################################################
# Color codes for output
################################################################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper functions
################################################################################

print_header() {
    echo -e "\n${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ Error: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ Warning: $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

################################################################################
# Default configuration
################################################################################

CHECKPOINT=""
STOCK=""
LEARNING_RATE="1e-4"
EPOCHS="15"
BATCH_SIZE="128"
PATIENCE="5"
DATA_DIR="data"
OUTPUT_DIR=""
GRADIENT_CLIP="1.0"
NUM_WORKERS="4"
SKIP_PREPROCESSING="false"
FREEZE_BASE="true"

################################################################################
# Parse command-line arguments
################################################################################

show_usage() {
    cat << EOF
Usage: $0 --checkpoint <path> --stock <TSLA|INTC> [options]

Required Arguments:
    --checkpoint <path>         Path to pretrained checkpoint (.ckpt file)
    --stock <TSLA|INTC>        Stock symbol to finetune on

Optional Arguments:
    --learning_rate <float>     Learning rate for finetuning (default: 1e-4)
    --epochs <int>              Number of epochs (default: 15)
    --batch_size <int>          Batch size (default: 128)
    --patience <int>            Early stopping patience (default: 5)
    --data_dir <path>           Root data directory (default: data)
    --output_dir <path>         Output directory for checkpoints (default: auto)
    --gradient_clip <float>     Gradient clipping value (default: 1.0)
    --num_workers <int>         Number of dataloader workers (default: 4)
    --skip_preprocessing        Skip preprocessing step (use existing data)
    --no_freeze_base            Train all parameters (not just gated cross-attention)
    -h, --help                  Show this help message

Example:
    $0 --checkpoint data/checkpoints/TRADES/val_ema=2.5_epoch=25_TSLA_TRADES_seed_30.ckpt \\
       --stock TSLA --learning_rate 1e-4 --epochs 15

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --stock)
            STOCK="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gradient_clip)
            GRADIENT_CLIP="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --skip_preprocessing)
            SKIP_PREPROCESSING="true"
            shift
            ;;
        --no_freeze_base)
            FREEZE_BASE="false"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

################################################################################
# Validate required arguments
################################################################################

if [ -z "$CHECKPOINT" ]; then
    print_error "Missing required argument: --checkpoint"
    show_usage
    exit 1
fi

if [ -z "$STOCK" ]; then
    print_error "Missing required argument: --stock"
    show_usage
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    print_error "Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

if [[ ! "$STOCK" =~ ^(TSLA|INTC)$ ]]; then
    print_error "Invalid stock symbol: $STOCK (must be TSLA or INTC)"
    exit 1
fi

################################################################################
# Display configuration
################################################################################

print_header "News Finetuning Pipeline Configuration"

echo "Checkpoint:        $CHECKPOINT"
echo "Stock:             $STOCK"
echo "Learning Rate:     $LEARNING_RATE"
echo "Epochs:            $EPOCHS"
echo "Batch Size:        $BATCH_SIZE"
echo "Patience:          $PATIENCE"
echo "Data Directory:    $DATA_DIR"
echo "Gradient Clip:     $GRADIENT_CLIP"
echo "Num Workers:       $NUM_WORKERS"
echo "Skip Preprocessing: $SKIP_PREPROCESSING"
echo "Freeze Base Model:  $FREEZE_BASE"

if [ -n "$OUTPUT_DIR" ]; then
    echo "Output Directory:  $OUTPUT_DIR"
else
    echo "Output Directory:  <auto-generated>"
fi

################################################################################
# Step 0: Environment Setup
################################################################################

# print_header "Step 0: Environment Setup"

# print_info "Activating conda environment 'dmeads'..."
# source ~/.zshrc
# conda activate dmeads

# if [ $? -ne 0 ]; then
#     print_error "Failed to activate conda environment 'dmeads'"
#     print_info "Please create the environment with required dependencies first"
#     exit 1
# fi

# print_success "Environment activated"

# # Verify required dependencies
# print_info "Verifying dependencies..."
# python3 -c "import torch; import transformers; import pandas; import numpy" 2>/dev/null
# if [ $? -ne 0 ]; then
#     print_error "Missing required dependencies. Please install:"
#     print_info "  pip3 install torch transformers pandas numpy pytorch-lightning"
#     exit 1
# fi
# print_success "Dependencies verified"

################################################################################
# Step 1: Preprocessing (if not skipped)
################################################################################

if [ "$SKIP_PREPROCESSING" = "false" ]; then
    print_header "Step 1: Preprocessing - News Data & Sentiment Analysis"

    # Check if news data exists
    NEWS_FILE="$DATA_DIR/news/final_${STOCK,,}_news_cleaned.csv"
    if [ ! -f "$NEWS_FILE" ]; then
        print_error "News data file not found: $NEWS_FILE"
        print_info "Please ensure cleaned news CSV files exist in $DATA_DIR/news/"
        exit 1
    fi
    print_success "News data found: $NEWS_FILE"

    # Check if LOB data exists
    LOB_DIR="$DATA_DIR/$STOCK"
    if [ ! -d "$LOB_DIR" ] || [ -z "$(ls -A $LOB_DIR/${STOCK}_2015-01-* 2>/dev/null)" ]; then
        print_error "LOB data not found in: $LOB_DIR"
        print_info "Please ensure LOBSTER data files exist for $STOCK"
        exit 1
    fi
    print_success "LOB data found in: $LOB_DIR"

    # Create preprocessing script in project directory
    print_info "Creating preprocessing script..."
    PREPROCESS_SCRIPT=".preprocess_news_temp.py"

    cat > "$PREPROCESS_SCRIPT" << 'PYEOF'
"""
Preprocessing script for news features
Loads news data, analyzes sentiment, and prepares datasets
"""
import sys
import logging
from preprocessing.NewsDataBuilder import NewsDataBuilder
from preprocessing.SentimentAnalyzer import SentimentAnalyzer
from preprocessing.LOBSTERDataBuilder import LOBSTERDataBuilder
import constants as cst
from configuration import Configuration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(stock_name):
    logger.info(f"Starting preprocessing for {stock_name}")

    # Step 1: Load news data
    logger.info("Step 1: Loading news data...")
    news_builder = NewsDataBuilder(data_dir=cst.NEWS_DATA_DIR)

    if stock_name not in news_builder.get_available_tickers():
        logger.error(f"No news data available for {stock_name}")
        logger.error(f"Available tickers: {news_builder.get_available_tickers()}")
        sys.exit(1)

    news_df = news_builder.load_news_data(stock_name)
    logger.info(f"Loaded {len(news_df)} news articles")

    # Step 2: Analyze sentiment
    logger.info("Step 2: Analyzing sentiment with FinBERT...")
    analyzer = SentimentAnalyzer()
    news_df = analyzer.analyze_news_dataframe(news_df)
    logger.info(f"Sentiment analysis complete")
    logger.info(f"  Mean sentiment: {news_df['sentiment'].mean():.3f}")
    logger.info(f"  Sentiment range: [{news_df['sentiment'].min():.3f}, {news_df['sentiment'].max():.3f}]")

    # Step 3: Prepare LOB datasets with news features
    logger.info("Step 3: Preparing LOB datasets with news features...")
    config = Configuration()

    # Get stock enum
    stock_enum = getattr(cst.Stocks, stock_name)

    # Initialize LOBSTERDataBuilder with news features enabled
    data_builder = LOBSTERDataBuilder(
        stock_name=stock_name,
        data_dir=cst.DATA_DIR,
        date_trading_days=cst.DATE_TRADING_DAYS,
        split_rates=config.SPLIT_RATES,
        chosen_model=cst.Models.TRADES,
        use_news_features=True,
        news_lookback_window=240,  # 600 seconds lookback
        news_half_life=120  # 300 seconds half-life for exponential decay
    )

    logger.info("Preparing and saving datasets...")
    data_builder.prepare_save_datasets()

    logger.info("✓ Preprocessing complete!")
    logger.info(f"  Train data: {cst.DATA_DIR}/{stock_name}/train.npy")
    logger.info(f"  Val data: {cst.DATA_DIR}/{stock_name}/val.npy")
    logger.info(f"  Test data: {cst.DATA_DIR}/{stock_name}/test.npy")
    logger.info(f"  Train news: {cst.DATA_DIR}/{stock_name}/train_news.npy")
    logger.info(f"  Val news: {cst.DATA_DIR}/{stock_name}/val_news.npy")
    logger.info(f"  Test news: {cst.DATA_DIR}/{stock_name}/test_news.npy")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess_news.py <STOCK>")
        sys.exit(1)
    main(sys.argv[1])
PYEOF

    # Run preprocessing
    print_info "Running preprocessing (this may take several minutes)..."
    python3 "$PREPROCESS_SCRIPT" "$STOCK"

    PREPROCESS_EXIT_CODE=$?

    # Clean up temporary script
    rm -f "$PREPROCESS_SCRIPT"

    if [ $PREPROCESS_EXIT_CODE -ne 0 ]; then
        print_error "Preprocessing failed"
        exit 1
    fi

    print_success "Preprocessing completed successfully"

    # Verify output files exist
    REQUIRED_FILES=(
        "$DATA_DIR/$STOCK/train.npy"
        "$DATA_DIR/$STOCK/val.npy"
        "$DATA_DIR/$STOCK/train_news.npy"
        "$DATA_DIR/$STOCK/val_news.npy"
    )

    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "Expected output file not found: $file"
            exit 1
        fi
    done
    print_success "All preprocessing output files verified"
else
    print_header "Step 1: Preprocessing - SKIPPED"
    print_info "Using existing preprocessed data"

    # Verify required files exist
    REQUIRED_FILES=(
        "$DATA_DIR/$STOCK/train.npy"
        "$DATA_DIR/$STOCK/val.npy"
        "$DATA_DIR/$STOCK/train_news.npy"
        "$DATA_DIR/$STOCK/val_news.npy"
    )

    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "Required preprocessed file not found: $file"
            print_info "Run without --skip_preprocessing to generate data"
            exit 1
        fi
    done
    print_success "All required data files verified"
fi

################################################################################
# Step 2: Fine-tuning
################################################################################

# print_header "Step 2: Fine-tuning Model with News Features"

# # Build finetuning command
# FINETUNE_CMD="python3 finetune_with_news.py \
#     --checkpoint $CHECKPOINT \
#     --stock $STOCK \
#     --learning_rate $LEARNING_RATE \
#     --epochs $EPOCHS \
#     --batch_size $BATCH_SIZE \
#     --patience $PATIENCE \
#     --data_dir $DATA_DIR \
#     --gradient_clip $GRADIENT_CLIP \
#     --num_workers $NUM_WORKERS"

# if [ -n "$OUTPUT_DIR" ]; then
#     FINETUNE_CMD="$FINETUNE_CMD --output_dir $OUTPUT_DIR"
# fi

# if [ "$FREEZE_BASE" = "true" ]; then
#     FINETUNE_CMD="$FINETUNE_CMD --freeze_base"
# fi

# print_info "Running finetuning command:"
# echo "$FINETUNE_CMD"
# echo

# # Run finetuning
# eval $FINETUNE_CMD

# if [ $? -ne 0 ]; then
#     print_error "Finetuning failed"
#     exit 1
# fi

# print_success "Finetuning completed successfully"

################################################################################
# Summary
################################################################################

print_header "Pipeline Complete!"

print_success "All steps completed successfully"
echo
print_info "Next steps:"
echo "  1. Check the output directory for fine-tuned checkpoints"
echo "  2. Evaluate the model on test data"
echo "  3. Use the best checkpoint for inference with news conditioning"
echo

exit 0
