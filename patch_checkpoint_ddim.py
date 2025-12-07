"""
Patch finetuned checkpoints to add DDIM buffers.

This script loads existing finetuned checkpoints, registers the DDIM schedule,
and re-saves them so they can be used with DDIM sampling.

Usage:
    python patch_checkpoint_ddim.py
"""

import torch
from pathlib import Path
import constants as cst
from models.diffusers.diffusion_engine import DiffusionEngine
import configuration

def patch_checkpoint(checkpoint_path):
    """
    Load checkpoint, add DDIM buffers, and re-save.

    Args:
        checkpoint_path: Path to the checkpoint file
    """
    print(f"\nPatching checkpoint: {checkpoint_path.name}")
    print("=" * 80)

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load model from checkpoint
    print("Loading model...")
    config = checkpoint["hyper_parameters"]["config"]
    config.IS_WANDB = False

    model = DiffusionEngine.load_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location='cpu',
        weights_only=False
    )

    # Get required values from existing buffers and config
    print("Computing DDIM buffers...")
    alphas_cumprod = model.diffuser.alphas_cumprod
    num_diffusionsteps = model.diffuser.num_diffusionsteps

    # Use default DDIM hyperparameters (same as baseline TRADES)
    ddim_eta = 0.0  # Deterministic DDIM
    ddim_nsteps = 1  # Fast sampling with 10 steps

    # Compute DDIM buffers (same logic as GaussianDiffusion.__init__ lines 85-104)
    tmp = num_diffusionsteps / ddim_nsteps
    t = torch.arange(0, num_diffusionsteps, tmp).long() + 1
    ddim_alpha = alphas_cumprod[t].clone()
    ddim_alpha_sqrt = torch.sqrt(ddim_alpha)
    ddim_alpha_prev = torch.cat([alphas_cumprod[0:1], alphas_cumprod[t[:-1]]])
    ddim_sqrt_one_minus_alpha = (1. - ddim_alpha) ** .5
    ddim_sigma = (ddim_eta *
                 ((1 - ddim_alpha_prev) / (1 - ddim_alpha + 1e-8) *
                  (1 - ddim_alpha / (ddim_alpha_prev + 1e-8))) ** .5)

    # Register DDIM buffers on the model
    model.diffuser.register_buffer('t', t)
    model.diffuser.register_buffer('ddim_alpha', ddim_alpha)
    model.diffuser.register_buffer('ddim_alpha_sqrt', ddim_alpha_sqrt)
    model.diffuser.register_buffer('ddim_alpha_prev', ddim_alpha_prev)
    model.diffuser.register_buffer('ddim_sqrt_one_minus_alpha', ddim_sqrt_one_minus_alpha)
    model.diffuser.register_buffer('ddim_sigma', ddim_sigma)
    model.diffuser.ddim_eta = ddim_eta
    model.diffuser.ddim_nsteps = ddim_nsteps

    # Verify buffers are registered
    ddim_buffers = ['t', 'ddim_alpha', 'ddim_alpha_sqrt', 'ddim_alpha_prev',
                    'ddim_sqrt_one_minus_alpha', 'ddim_sigma']
    all_present = True
    for buffer_name in ddim_buffers:
        if hasattr(model.diffuser, buffer_name):
            print(f"  ✓ {buffer_name} registered")
        else:
            print(f"  ✗ {buffer_name} NOT found")
            all_present = False

    if not all_present:
        raise RuntimeError("Some DDIM buffers failed to register!")

    # Save patched checkpoint
    output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_ddim.ckpt"
    print(f"\nSaving patched checkpoint to: {output_path.name}")

    # Save using trainer to preserve all metadata
    from lightning import Trainer
    trainer = Trainer(default_root_dir=str(checkpoint_path.parent))
    trainer.strategy.connect(model)
    trainer.save_checkpoint(output_path)

    print(f"✓ Checkpoint patched successfully!")
    print(f"  Original: {checkpoint_path.name}")
    print(f"  Patched:  {output_path.name}")

    return output_path


def main():
    """Patch all finetuned checkpoints in the NEWS_TRADES directory."""

    print("=" * 80)
    print("DDIM Checkpoint Patcher")
    print("=" * 80)
    print("\nThis script will add DDIM buffers to your finetuned checkpoints")
    print("so they can be used with DDIM sampling.\n")

    # Find all finetuned checkpoints
    checkpoint_dir = Path(cst.DIR_SAVED_MODEL) / "NEWS_TRADES"

    if not checkpoint_dir.exists():
        print(f"Error: Directory not found: {checkpoint_dir}")
        print("Please ensure your finetuned checkpoints are in data/checkpoints/NEWS_TRADES/")
        return

    checkpoints = list(checkpoint_dir.glob("*_finetuned_model.ckpt"))

    if not checkpoints:
        print(f"No finetuned checkpoints found in {checkpoint_dir}")
        print("Looking for files matching: *_finetuned_model.ckpt")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) to patch:")
    for ckpt in checkpoints:
        print(f"  - {ckpt.name}")

    print("\n" + "=" * 80)

    # Patch each checkpoint
    patched_checkpoints = []
    for checkpoint_path in checkpoints:
        try:
            output_path = patch_checkpoint(checkpoint_path)
            patched_checkpoints.append(output_path)
        except Exception as e:
            print(f"\n✗ Error patching {checkpoint_path.name}:")
            print(f"  {type(e).__name__}: {e}")
            continue

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nSuccessfully patched {len(patched_checkpoints)} / {len(checkpoints)} checkpoint(s)")

    if patched_checkpoints:
        print("\nPatched checkpoints:")
        for path in patched_checkpoints:
            print(f"  ✓ {path.name}")

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Rename the patched checkpoints to replace the originals:")
        for orig, patched in zip(checkpoints, patched_checkpoints):
            print(f"   mv {patched.name} {orig.name}")

        print("\n2. Or update your simulation to use the _patched.ckpt files")
        print("\n3. You can now use DDIM sampling with these checkpoints!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
