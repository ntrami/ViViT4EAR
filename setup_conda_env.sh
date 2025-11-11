#!/bin/bash
# Script to create conda environment for ViViT training

ENV_NAME="vivit"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "=========================================="
echo "Creating conda environment: $ENV_NAME"
echo "=========================================="

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists."
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Using existing environment. Activate it with: conda activate $ENV_NAME"
        exit 0
    fi
fi

# Create conda environment with Python 3.9 (compatible with PyTorch and PyTorchVideo)
echo "Creating conda environment with Python 3.9..."
conda create -n $ENV_NAME python=3.9 -y

# Activate environment
echo "Activating environment..."
conda activate $ENV_NAME

# Install PyTorch with CUDA support (adjust CUDA version based on your system)
# For CUDA 11.8:
echo "Installing PyTorch with CUDA 11.8 support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import torch; import pytorchvideo; import transformers; print(\"All packages installed successfully!\")'"
echo ""

