#!/bin/bash
# Installation script for ViViT training requirements
# For RTX 5090 (compute capability 12.0), use nightly PyTorch

echo "=========================================="
echo "Installing ViViT Training Requirements"
echo "=========================================="

# Check if RTX 5090 or newer GPU (compute capability > 9.0)
USE_NIGHTLY=false
if command -v python3 &> /dev/null; then
    GPU_CAP=$(python3 -c "import torch; print(torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")
    if [ "$GPU_CAP" -gt 9 ]; then
        echo "Detected GPU with compute capability > 9.0"
        echo "Using PyTorch nightly for RTX 5090 support"
        USE_NIGHTLY=true
    fi
fi

# Install PyTorch first
if [ "$USE_NIGHTLY" = true ]; then
    echo "Installing PyTorch nightly (CUDA 12.6)..."
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
else
    echo "Installing PyTorch stable (CUDA 12.6)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
fi

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements.txt

echo "=========================================="
echo "Installation completed!"
echo "=========================================="


