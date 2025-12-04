"""
Fine-Tuning Script for TRADES with News Features using Gated Cross-Attention

This script fine-tunes a pretrained TRADES model to incorporate news features
via gated cross-attention. Only the gated cross-attention modules are trained,
preserving the pretrained DiT backbone.

Usage:
    source ~/.zshrc && conda activate dmeads
    python3 finetune_with_news.py \
        --checkpoint data/checkpoints/TRADES/val_ema=2.5_epoch=25_TSLA_TRADES_seed_30.ckpt \
        --learning_rate 1e-4 \
        --epochs 15 \
        --batch_size 128
"""

import argparse
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
import logging
from datetime import datetime

from configuration import Configuration
from models.diffusers.diffusion_engine import DiffusionEngine
from preprocessing.LOBDataset import LOBDataset
import constants as cst
from constants import LearningHyperParameter


def parse_args():
    """Parse command-line arguments for fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune TRADES model with news features via gated cross-attention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pretrained checkpoint (.ckpt file)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning (use lower than original training)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of epochs to fine-tune"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for fine-tuning"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without improvement)"
    )

    # Data arguments
    parser.add_argument(
        "--stock",
        type=str,
        default=None,
        help="Stock symbol (default: inferred from checkpoint filename)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root data directory"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save fine-tuned checkpoints (default: auto-generated)"
    )

    # Other arguments
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (0 = disabled)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--freeze_base",
        action="store_true",
        default=True,
        help="Freeze base model (only train gated cross-attention)"
    )

    return parser.parse_args()


def infer_stock_from_checkpoint(checkpoint_path):
    """
    Infer stock symbol from checkpoint filename.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        str: Stock symbol (e.g., 'TSLA', 'INTC')
    """
    filename = os.path.basename(checkpoint_path)

    # Try to find stock name in known stocks
    for stock in cst.Stocks:
        if stock.name in filename:
            return stock.name

    raise ValueError(
        f"Could not infer stock from checkpoint filename: {filename}. "
        "Please specify --stock explicitly."
    )


def setup_data_loaders(config, stock_name, data_dir, batch_size, num_workers):
    """
    Create train and validation data loaders with news features.

    Args:
        config: Configuration object
        stock_name: Stock symbol
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        tuple: (train_loader, val_loader)
    """
    stock_dir = f"{data_dir}/{stock_name}"

    # Paths to data files
    train_path = f"{stock_dir}/train.npy"
    val_path = f"{stock_dir}/val.npy"
    train_news_path = f"{stock_dir}/train_news.npy"
    val_news_path = f"{stock_dir}/val_news.npy"

    # Check if files exist
    for path in [train_path, val_path, train_news_path, val_news_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required data file not found: {path}\n"
                f"Make sure to run preprocessing with USE_NEWS_FEATURES=True first."
            )

    # Create datasets
    train_dataset = LOBDataset(
        paths=[train_path],
        news_paths=[train_news_path],
        use_news_features=True,
        is_val=False,
        seq_size=config.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE],
        gen_seq_size=config.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE],
        chosen_model=config.CHOSEN_MODEL
    )

    val_dataset = LOBDataset(
        paths=[val_path],
        news_paths=[val_news_path],
        use_news_features=True,
        is_val=True,
        seq_size=config.HYPER_PARAMETERS[LearningHyperParameter.SEQ_SIZE],
        gen_seq_size=config.HYPER_PARAMETERS[LearningHyperParameter.MASKED_SEQ_SIZE],
        chosen_model=config.CHOSEN_MODEL,
        batch_size=batch_size,
        limit_val_batches=100  # Limit to 100 batches for faster validation
    )

    # Create data loaders
    # pin_memory is beneficial for CUDA, not necessary for MPS or CPU
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    print(f"✓ Created data loaders:")
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset):,} samples, {len(val_loader)} batches")

    return train_loader, val_loader


def freeze_base_model(model):
    """
    Freeze all parameters except gated cross-attention modules.

    Args:
        model: DiffusionEngine model
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze gated cross-attention modules
    trainable_params = 0
    total_params = 0

    if hasattr(model.diffuser, 'NN') and hasattr(model.diffuser.NN, 'layers'):
        for layer in model.diffuser.NN.layers.layers:
            if hasattr(layer, 'gated_cross_attn'):
                for param in layer.gated_cross_attn.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()

    for param in model.parameters():
        total_params += param.numel()

    percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0
    print(f"✓ Froze base model")
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)")
    print(f"  Only gated cross-attention modules will be trained")

    return trainable_params


def main():
    """Main fine-tuning function."""
    args = parse_args()

    print("=" * 80)
    print("TRADES Fine-Tuning with News Features (Gated Cross-Attention)")
    print("=" * 80)

    # Infer stock if not provided
    stock_name = args.stock if args.stock else infer_stock_from_checkpoint(args.checkpoint)
    print(f"\nStock: {stock_name}")
    print(f"Checkpoint: {args.checkpoint}")

    # Setup configuration
    print("\nSetting up configuration...")
    config = Configuration()

    # Enable news features
    config.USE_NEWS_FEATURES = True
    config.NEWS_FEATURE_DIM = 2
    config.IS_TRAINING = True
    config.IS_DATA_PREPROCESSED = True
    config.IS_WANDB = False  # Disable W&B for fine-tuning (can enable if desired)

    # Set stock
    stock_enum = getattr(cst.Stocks, stock_name)
    config.CHOSEN_STOCK = [stock_enum]

    # Update hyperparameters for fine-tuning
    config.HYPER_PARAMETERS[LearningHyperParameter.LEARNING_RATE] = args.learning_rate
    config.HYPER_PARAMETERS[LearningHyperParameter.EPOCHS] = args.epochs
    config.HYPER_PARAMETERS[LearningHyperParameter.BATCH_SIZE] = args.batch_size

    # Add fine-tuning flag
    config.IS_FINETUNING = True

    print(f"✓ Configuration ready")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")

    # Load model
    print("\n" + "=" * 80)
    print("Loading Model")
    print("=" * 80)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = DiffusionEngine.load_from_checkpoint(
        args.checkpoint,
        config=config,
        map_location=cst.DEVICE,
        strict=False  # Allow new gated cross-attention modules
    )

    print(f"✓ Loaded pretrained model")

    # CRITICAL: Properly initialize all gated cross-attention parameters
    # When loading with strict=False, new parameters get random initialization
    # which can cause NaN issues on MPS. We need to initialize them properly.
    print("\nInitializing gated cross-attention parameters...")
    if hasattr(model.diffuser, 'NN') and hasattr(model.diffuser.NN, 'layers'):
        for i, layer in enumerate(model.diffuser.NN.layers.layers):
            if hasattr(layer, 'gated_cross_attn'):
                gca = layer.gated_cross_attn
                with torch.no_grad():
                    # Initialize gate to 0 (preserves pretrained model behavior)
                    gca.gate.fill_(0.0)

                    # Initialize linear layers with Xavier uniform (stable initialization)
                    torch.nn.init.xavier_uniform_(gca.news_projection.weight)
                    torch.nn.init.zeros_(gca.news_projection.bias)
                    torch.nn.init.xavier_uniform_(gca.to_q.weight)
                    torch.nn.init.xavier_uniform_(gca.to_k.weight)
                    torch.nn.init.xavier_uniform_(gca.to_v.weight)
                    torch.nn.init.xavier_uniform_(gca.to_out.weight)

                    # Initialize layer norm (standard initialization)
                    torch.nn.init.ones_(gca.layer_norm.weight)
                    torch.nn.init.zeros_(gca.layer_norm.bias)

                print(f"  Layer {i}: initialized all gated cross-attention parameters")

    # CRITICAL: Reinitialize EMA to include new gated cross-attention parameters
    # The loaded checkpoint has EMA state for old parameters only. We need to:
    # 1. Save the old EMA shadow parameters (for pretrained parameters)
    # 2. Reinitialize EMA with ALL current parameters (old + new)
    # 3. Restore old shadow values where possible (new params start fresh)
    print("\nUpdating EMA to include new parameters...")
    try:
        # Save old EMA shadow parameters from checkpoint
        old_shadow_params = list(model.ema.shadow_params)
        old_param_names = [name for name, _ in model.named_parameters()]

        # Reinitialize EMA with ALL current parameters (old + new)
        from torch_ema import ExponentialMovingAverage
        model.ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        model.ema.to(cst.DEVICE)

        # Manually restore old shadow values for existing parameters
        # torch_ema doesn't support strict=False in load_state_dict, so we copy manually
        new_shadow_params = list(model.ema.shadow_params)
        num_old = len(old_shadow_params)
        num_new = len(new_shadow_params)

        if num_old < num_new:
            # Copy old shadow params to corresponding positions in new EMA
            with torch.no_grad():
                for i in range(num_old):
                    new_shadow_params[i].copy_(old_shadow_params[i])

            print("✓ EMA reinitialized with new parameters")
            print(f"  Old parameters ({num_old}): restored from checkpoint EMA")
            print(f"  New parameters ({num_new - num_old}): initialized from current values")
        else:
            print(f"⚠ Warning: Unexpected parameter count (old={num_old}, new={num_new})")
            print("  EMA initialized fresh for all parameters")

    except Exception as e:
        print(f"⚠ Warning: Could not update EMA state: {e}")
        print("  EMA will be reinitialized fresh for all parameters")
        import traceback
        traceback.print_exc()
        # If loading fails, at least we have fresh EMA for all params
        from torch_ema import ExponentialMovingAverage
        model.ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        model.ema.to(cst.DEVICE)

    # Freeze base model if requested
    if args.freeze_base:
        print("\nFreezing base model parameters...")
        freeze_base_model(model)

    # Create data loaders
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80 + "\n")

    train_loader, val_loader = setup_data_loaders(
        config=config,
        stock_name=stock_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = f"{cst.DIR_SAVED_MODEL}/{config.CHOSEN_MODEL}/finetuned_gated_cross_attn"

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n✓ Checkpoints will be saved to: {args.output_dir}")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="epoch{epoch:02d}_val{val_ema_loss:.4f}",
        monitor="val_ema_loss",
        mode="min",
        save_top_k=3,  # Keep top 3 checkpoints
        save_last=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_ema_loss",
        patience=args.patience,
        mode="min",
        min_delta=0.001,
        verbose=True
    )

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"{args.output_dir}/logs"
    os.makedirs(log_dir, exist_ok=True)

    # Setup file logger
    log_file = f"{log_dir}/finetune_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Starting Fine-Tuning Session")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Log file: {log_file}")

    # Setup CSV logger for metrics
    csv_logger = CSVLogger(save_dir=log_dir, name="metrics")

    # Setup trainer
    print("\n" + "=" * 80)
    print("Starting Fine-Tuning")
    print("=" * 80 + "\n")
    print(f"Logs will be saved to: {log_file}")
    print(f"Metrics will be saved to: {csv_logger.log_dir}")

    # Determine accelerator (support both CUDA and MPS)
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=csv_logger,
        accelerator=accelerator,
        devices=1,
        precision=32,
        gradient_clip_val=args.gradient_clip if args.gradient_clip > 0 else None,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Fine-tune
    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 80)
    print("Fine-Tuning Complete!")
    print("=" * 80)
    print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_ema_loss: {checkpoint_callback.best_model_score:.4f}")
    print(f"\nTo use the fine-tuned model:")
    print(f"  1. Update configuration: config.USE_NEWS_FEATURES = True")
    print(f"  2. Load checkpoint: model = DiffusionEngine.load_from_checkpoint('{checkpoint_callback.best_model_path}')")
    print(f"  3. Generate with news conditioning")


if __name__ == "__main__":
    main()
