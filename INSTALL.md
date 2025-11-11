# Installation Guide for ViViT Training

## Quick Install (Recommended)

For RTX 5090 or newer GPUs:
```bash
# Install PyTorch nightly with CUDA 12.6 (supports compute capability 12.0)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126

# Install other requirements
pip install -r requirements.txt
```

For other GPUs (up to compute capability 9.0):
```bash
# Install PyTorch stable with CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install other requirements
pip install -r requirements.txt
```

## Automated Installation

Use the install script (automatically detects GPU and uses appropriate PyTorch version):
```bash
bash install_requirements.sh
```

## Manual Installation

If you prefer to install manually:

1. **Install PyTorch** (choose one based on your GPU):
   ```bash
   # For RTX 5090 (sm_120)
   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
   
   # For other GPUs
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

2. **Install other dependencies**:
   ```bash
   pip install transformers>=4.30.0
   pip install pytorchvideo>=0.1.5
   pip install pandas>=1.5.0
   pip install numpy>=1.21.0
   pip install tqdm>=4.64.0
   pip install scikit-learn>=1.0.0
   pip install av>=10.0.0
   pip install Pillow>=9.0.0
   ```

## Verify Installation

Check if PyTorch can detect your GPU:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Troubleshooting

### RTX 5090 "no kernel image" error
- Make sure you installed PyTorch **nightly** build, not stable
- Stable PyTorch only supports up to compute capability 9.0 (sm_90)
- RTX 5090 requires compute capability 12.0 (sm_120) which is only in nightly builds

### CUDA version mismatch
- Ensure your CUDA version matches: `nvidia-smi` shows CUDA 13.0, but PyTorch is built for CUDA 12.6
- This is usually fine as CUDA has backward compatibility
- If issues persist, try matching CUDA versions exactly

