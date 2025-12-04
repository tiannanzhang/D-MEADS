# MPS Backend Compatibility

This document describes the MPS (Metal Performance Shaders) backend compatibility implementation for D-MEADS.

## Overview

MPS is Apple's GPU acceleration framework for M1/M2/M3 Macs. The codebase now automatically detects and uses MPS when available, providing significant performance improvements on Apple Silicon.

## Device Selection Priority

The device is automatically selected in the following priority order:

1. **CUDA** - NVIDIA GPU (if available)
2. **MPS** - Apple Silicon GPU (if available)
3. **CPU** - Fallback option

## Changes Made

### Core Files Modified

#### 1. `constants.py`
- Updated device selection logic to detect and prioritize MPS
- Added automatic fallback chain: CUDA → MPS → CPU

```python
# Device selection: prioritize CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
```

#### 2. `main.py`
- Updated `set_torch()` to only apply CUDA-specific optimizations when CUDA is available
- Updated accelerator selection to recognize MPS as a GPU backend
- Made TF32 settings conditional on CUDA availability

#### 3. `finetune_with_news.py`
- Updated accelerator selection for PyTorch Lightning Trainer
- Made pin_memory conditional (only beneficial for CUDA)
- Now supports both CUDA and MPS for GPU acceleration

#### 4. `preprocessing/SentimentAnalyzer.py`
- Updated auto-detection logic to support MPS
- Maintains same priority: CUDA → MPS → CPU

#### 5. `evaluation/quantitative_eval/predictive_lstm.py`
- Updated device detection for evaluation tasks
- Supports all three backends: CUDA, MPS, and CPU

#### 6. `run.py`
- Enhanced device information printing
- Shows MPS availability when running on Apple Silicon
- Displays CUDA device name when using NVIDIA GPUs

## Testing

A comprehensive test suite has been created to verify MPS compatibility:

```bash
source ~/.zshrc && conda activate dmeads && python3 test_mps_compatibility.py
```

The test suite validates:
- ✓ Device detection and selection
- ✓ Basic tensor operations (addition, matrix multiplication)
- ✓ Neural network forward and backward passes
- ✓ Embedding layer operations
- ✓ Non-blocking transfers

## Usage

No changes are needed to your existing code! The implementation automatically detects and uses the best available backend.

### Checking Your Device

When you run the main script, you'll see output like:

```
Device: mps
MPS (Metal Performance Shaders) is available
```

Or for CUDA:
```
Device: cuda
CUDA version: 12.1
CUDA device: NVIDIA RTX 4090
```

### Verifying MPS is Working

Run the test script to verify everything is working correctly:

```bash
python3 test_mps_compatibility.py
```

All tests should pass with "✓ PASS" status.

## Performance Notes

### MPS Benefits
- Significant speedup over CPU on Apple Silicon (M1/M2/M3)
- Typically 3-5x faster for neural network operations
- Lower power consumption compared to external GPUs

### Limitations
- MPS may be slower than high-end NVIDIA GPUs (RTX 3090/4090)
- Some advanced CUDA features are not available on MPS
- Pin memory optimization is CUDA-specific (disabled for MPS)

## Known Issues

### Non-blocking Transfers
The codebase uses `non_blocking=True` for some tensor transfers. This is primarily a CUDA optimization and may not provide benefits on MPS, but it doesn't cause issues - PyTorch handles this gracefully.

### TF32 Precision
TF32 (TensorFloat-32) is a CUDA-specific feature and is not available on MPS. The code now conditionally enables TF32 only when CUDA is available.

## Troubleshooting

### MPS Not Detected
If MPS is not being used despite having Apple Silicon:

1. Check PyTorch version (requires PyTorch 1.12+):
   ```bash
   python3 -c "import torch; print(torch.__version__)"
   ```

2. Verify MPS availability:
   ```bash
   python3 -c "import torch; print(torch.backends.mps.is_available())"
   ```

3. Update PyTorch if needed:
   ```bash
   pip3 install --upgrade torch
   ```

### Out of Memory Errors
MPS has separate memory from system RAM. If you encounter OOM errors:

1. Reduce batch size in your configuration
2. Close other GPU-intensive applications
3. Monitor memory usage with Activity Monitor

### Slower Than Expected
If MPS performance is slower than expected:

1. Ensure you're not running other GPU-intensive tasks
2. Check that the model is actually on MPS (not CPU)
3. Verify thermal throttling isn't occurring (check Activity Monitor)

## Backward Compatibility

All changes are backward compatible:
- Works on systems with CUDA GPUs
- Works on systems without any GPU (CPU fallback)
- No changes needed to existing configurations or scripts

## Testing on Different Systems

The implementation has been designed to work across:
- ✓ Apple Silicon Macs (M1/M2/M3) with MPS
- ✓ Linux/Windows with NVIDIA GPUs (CUDA)
- ✓ Any system with CPU fallback

## Additional Information

For more details about MPS:
- [Apple MPS Documentation](https://developer.apple.com/metal/pytorch/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

For issues or questions, please refer to the main repository documentation or open an issue on GitHub.
