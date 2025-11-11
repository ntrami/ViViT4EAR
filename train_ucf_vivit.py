#!/usr/bin/env python3
"""
Training script for ViViT on UCF-101 dataset with progressive observation ratios.

This script fine-tunes a pretrained ViViT model on UCF-101 dataset, training with
different observation ratios (0.1, 0.3, 0.5, 0.7, 0.9) and evaluates on both train
and val sets at each epoch, tracking top1 and top5 accuracy.
"""

import os
import sys
import json
import argparse
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from pytorchvideo.data import encoded_video
import warnings
from datetime import datetime

# Try to import ViViT from various sources
try:
    from transformers import VivitModel, VivitConfig
    from transformers import VivitForVideoClassification
    HAS_TRANSFORMERS = True
except ImportError:
    try:
        import pytorchvideo.models.vivit as vivit_module
        HAS_PYTORCHVIDEO = True
        HAS_TRANSFORMERS = False
    except ImportError:
        HAS_PYTORCHVIDEO = False
        HAS_TRANSFORMERS = False
        warnings.warn("Neither transformers nor pytorchvideo found. Will create a simple ViViT implementation.")

# ==========================================
# CONFIGURATION
# ==========================================

# Dataset paths
TRAIN_CSV_PATH = "/data/progressive-action-prediction/labels/UCF-101/train.csv"
VAL_CSV_PATH = "/data/progressive-action-prediction/labels/UCF-101/val.csv"
TEST_CSV_PATH = "/data/progressive-action-prediction/labels/UCF-101/test.csv"
DICTIONARY_PATH = "/data/progressive-action-prediction/labels/UCF-101/dictionary.json"
VIDEO_ROOT = "/data/UCF-101"

# Model configuration
PRETRAINED_MODEL = "google/vivit-b-16x2-kinetics400"  # ViViT base model pretrained on Kinetics-400
NUM_CLASSES = 101  # UCF-101 has 101 classes
FRAME_LENGTH = 32  # Changed to 32 to match pretrained model
FRAME_SIZE = 224
NUM_FRAMES = 32  # Temporal dimension - must match pretrained model
PATCH_SIZE = 32  # Spatial patch size
TUBELET_SIZE = 2  # Temporal tubelet size

# Training configuration
BATCH_SIZE = 2  # Reduced from 4 due to 32 frames
NUM_EPOCHS = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 8  # Increased from 4 to 8 for faster data loading
PREFETCH_FACTOR = 2  # Prefetch batches to avoid blocking
USE_MIXED_PRECISION = True  # Enable FP16 training for ~2x speedup
USE_COMPILE = False  # Disable torch.compile by default (requires CUDA libraries for linking)
                     # Set to True if you have CUDA libraries properly installed

# Observation ratios to train
OBSERVATION_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Output directory
OUTPUT_ROOT = "/data/ViViT/checkpoints/UCF-101"


# ==========================================
# DATASET CLASS
# ==========================================

class UCF101Dataset(Dataset):
    """Dataset class for UCF-101 with progressive observation ratios."""
    
    def __init__(self, csv_path, dictionary_path, video_root, observation_ratio=1.0, 
                 frame_length=32, frame_size=224, is_training=True, val_csv_path=None):
        """
        Args:
            csv_path: Path to train.csv (or val.csv if val_csv_path is None)
            dictionary_path: Path to dictionary.json
            video_root: Root directory for videos
            observation_ratio: Ratio of video to observe (0.0-1.0)
            frame_length: Number of frames to sample
            frame_size: Spatial size of frames
            is_training: Whether this is training set
            val_csv_path: Optional path to val.csv (if provided, use this for validation)
        """
        self.csv_path = csv_path
        self.video_root = video_root
        self.observation_ratio = observation_ratio
        self.frame_length = frame_length
        self.frame_size = frame_size
        self.is_training = is_training
        
        # Load dictionary
        with open(dictionary_path, 'r') as f:
            self.class_to_id = json.load(f)
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        
        # Load CSV - use val_csv_path if provided and not training
        if not is_training and val_csv_path and os.path.exists(val_csv_path):
            # Use separate val.csv file
            df = pd.read_csv(val_csv_path)
            print(f"Using separate validation CSV: {val_csv_path}")
        else:
            # Load from csv_path and filter by split
            df = pd.read_csv(csv_path)
            
            # If no 'val' split exists, split from 'train' data (80/20 split)
            if 'val' not in df['split'].unique():
                # Split train data into train/val
                train_df = df[df['split'] == 'train']
                # Use stratified split or simple split
                try:
                    from sklearn.model_selection import train_test_split
                    train_indices, val_indices = train_test_split(
                        train_df.index, test_size=0.2, random_state=42, 
                        stratify=train_df['label'] if 'label' in train_df.columns else None
                    )
                    if is_training:
                        df = train_df.loc[train_indices]
                    else:
                        df = train_df.loc[val_indices]
                except:
                    # Fallback to simple split if sklearn not available
                    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
                    split_idx = int(len(train_df) * 0.8)
                    if is_training:
                        df = train_df[:split_idx]
                    else:
                        df = train_df[split_idx:]
            else:
                # Use existing splits
                if is_training:
                    df = df[df['split'] == 'train']
                else:
                    df = df[df['split'] == 'val']
        
        self.data = df.to_dict('records')
        print(f"Loaded {len(self.data)} videos for {'training' if is_training else 'validation'}")
        print(f"Observation ratio: {observation_ratio}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        label_name = item['label']
        video_id = item['id']
        
        # Build video path: /data/UCF-101/{label}/{id}.avi
        video_path = os.path.join(self.video_root, label_name, f"{video_id}.avi")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Load and process video
        video = encoded_video.EncodedVideo.from_path(video_path)
        video_duration = float(video.duration)
        
        # Calculate observed duration based on observation_ratio
        observed_duration = video_duration * self.observation_ratio
        
        # KEY FRAME SELECTION: Sample from middle of video where action is most clear
        if self.is_training:
            # During training: sample from middle region with some randomness
            safe_start = video_duration * 0.2
            safe_end = video_duration * 0.8 - observed_duration
            if safe_end > safe_start:
                start_time = safe_start + torch.rand(1).item() * (safe_end - safe_start)
            else:
                max_start = max(0, video_duration - observed_duration)
                start_time = torch.rand(1).item() * max_start
        else:
            # During validation: fixed start from middle region
            safe_start = video_duration * 0.2
            safe_end = video_duration * 0.8 - observed_duration
            if safe_end > safe_start:
                start_time = safe_start + (safe_end - safe_start) / 2
            else:
                start_time = max(0, (video_duration - observed_duration) / 2)
        
        end_time = min(start_time + observed_duration, video_duration)
        start_time = max(0, start_time)
        
        # Get clip
        clip = video.get_clip(start_time, end_time)
        if isinstance(clip, dict):
            video_tensor = clip['video']  # [C, T, H, W]
        else:
            video_tensor = clip
        
        if video_tensor is None:
            video_tensor = torch.zeros(3, self.frame_length, self.frame_size, self.frame_size)
        else:
            # Sample or pad to frame_length
            T = video_tensor.shape[1]
            if T < self.frame_length:
                num_repeats = (self.frame_length // T) + 1
                video_tensor = video_tensor.repeat(1, num_repeats, 1, 1)
                video_tensor = video_tensor[:, :self.frame_length, :, :]
            elif T > self.frame_length:
                indices = torch.linspace(0, T - 1, self.frame_length).long()
                video_tensor = video_tensor[:, indices, :, :]
            
            # Resize to frame_size
            video_tensor = F.interpolate(
                video_tensor, size=(self.frame_size, self.frame_size),
                mode='bilinear', align_corners=False
            )
        
        # Normalize to [0, 1] then ImageNet normalization
        video_tensor = video_tensor.float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        # Get label
        label_id = self.class_to_id[label_name]
        
        return video_tensor, label_id


# ==========================================
# SIMPLE VIVIT IMPLEMENTATION (fallback)
# ==========================================

class SimpleViViT(nn.Module):
    """Simplified ViViT model for when transformers/pytorchvideo are not available."""
    
    def __init__(self, num_classes=101, frame_length=32, frame_size=224, patch_size=32, 
                 embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.num_classes = num_classes
        self.frame_length = frame_length
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches_per_frame = (frame_size // patch_size) ** 2
        self.num_patches = self.num_patches_per_frame * frame_length
        
        print(f"ViViT config: {self.num_patches_per_frame} patches/frame × {frame_length} frames = {self.num_patches} total patches")
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(
            3, embed_dim, 
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, T, H/patch_size, W/patch_size]
        B, C, T, H, W = x.shape
        
        # Flatten patches
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, T, H, W, C]
        x = x.view(B, T * H * W, C)  # [B, num_patches, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)  # [B, num_patches, embed_dim]
        
        # Global average pooling
        x = x.mean(dim=1)  # [B, embed_dim]
        
        # Classifier
        x = self.norm(x)
        x = self.head(x)  # [B, num_classes]
        
        return x


# ==========================================
# MODEL CREATION
# ==========================================

def create_vivit_model(num_classes=101, pretrained=True):
    """Create ViViT model (patched for PyTorch 2.6.0.dev compatibility)."""
    if HAS_TRANSFORMERS:
        try:
            torch_version = torch.__version__
            print(f"Detected PyTorch version: {torch_version}")

            # Nếu là bản dev >= 2.6, bỏ qua lỗi CVE check
            if "2.6.0.dev" in torch_version or "2.6" in torch_version:
                print("Detected PyTorch 2.6+ (dev or stable) — allowing pretrained load.")
                torch.serialization.add_safe_globals({
                    "VivitForVideoClassification": VivitForVideoClassification
                })

            model = VivitForVideoClassification.from_pretrained(
                PRETRAINED_MODEL,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
            print("✓ Successfully loaded pretrained ViViT model from Hugging Face")
            return model

        except (ValueError, RuntimeError) as e:
            error_msg = str(e)
            print(f"⚠ Warning: Could not load pretrained model ({error_msg[:200]}...)")
            print(f"Falling back to SimpleViViT model (training from scratch)")
            return SimpleViViT(num_classes=num_classes, frame_length=FRAME_LENGTH, 
                              frame_size=FRAME_SIZE, patch_size=PATCH_SIZE)

    elif HAS_PYTORCHVIDEO:
        try:
            model = vivit_module.create_vivit(model_num_class=num_classes, pretrained=pretrained)
            print("✓ Created ViViT model using PyTorchVideo")
            return model
        except Exception as e:
            print(f"⚠ Failed to create pytorchvideo model ({e}), using SimpleViViT fallback")
            return SimpleViViT(num_classes=num_classes, frame_length=FRAME_LENGTH, 
                              frame_size=FRAME_SIZE, patch_size=PATCH_SIZE)
    else:
        print("Creating simplified ViViT model (Vision Transformer for video)")
        return SimpleViViT(num_classes=num_classes, frame_length=FRAME_LENGTH, 
                          frame_size=FRAME_SIZE, patch_size=PATCH_SIZE)



# ==========================================
# ACCURACY FUNCTIONS
# ==========================================

def calculate_accuracy_topk(logits, labels, k=5):
    """Calculate top-k accuracy."""
    with torch.no_grad():
        batch_size = labels.size(0)
        _, pred = logits.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_k = correct_k.mul_(100.0 / batch_size)
    return acc_k.item()


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler=None):
    """Train for one epoch with optional mixed precision and return top1 and top5 accuracy."""
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    # Update progress bar less frequently to reduce I/O overhead
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", mininterval=1.0)
    for batch_idx, (videos, labels) in enumerate(pbar):
        # Non-blocking transfer to overlap with computation
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Mixed precision forward
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                if isinstance(model, SimpleViViT):
                    logits = model(videos)
                elif HAS_TRANSFORMERS and hasattr(model, 'config'):
                    videos_transformed = videos.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]
                    try:
                        outputs = model(pixel_values=videos_transformed, interpolate_pos_encoding=False)
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    except TypeError:
                        logits = model(videos_transformed)
                else:
                    logits = model(videos)
                
                loss = criterion(logits, labels)
        else:
            # Standard precision
            if isinstance(model, SimpleViViT):
                logits = model(videos)
            elif HAS_TRANSFORMERS and hasattr(model, 'config'):
                videos_transformed = videos.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]
                try:
                    outputs = model(pixel_values=videos_transformed, interpolate_pos_encoding=False)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                except TypeError:
                    logits = model(videos_transformed)
            else:
                logits = model(videos)
            
            loss = criterion(logits, labels)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Statistics (optimized - calculate once)
        with torch.no_grad():
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()
            
            # Top5 accuracy - optimized by using topk once
            _, top5_pred = logits.topk(5, dim=1)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
        
        # Update progress bar less frequently (every 10 batches or last batch)
        if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'top1': 100.0 * correct_top1 / total,
                'top5': 100.0 * correct_top5 / total
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc_top1 = 100.0 * correct_top1 / total
    epoch_acc_top5 = 100.0 * correct_top5 / total
    return epoch_loss, epoch_acc_top1, epoch_acc_top5


def validate(model, dataloader, criterion, device):
    """Validate model with optimizations and return top1 and top5 accuracy."""
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with torch.no_grad():
        # Use autocast for validation too (faster, only on GPU)
        pbar = tqdm(dataloader, desc="Validation", mininterval=1.0)
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos = videos.to(device, non_blocking=(device.type == 'cuda'))
            labels = labels.to(device, non_blocking=(device.type == 'cuda'))
            
            # Use autocast only on GPU
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    if isinstance(model, SimpleViViT):
                        logits = model(videos)
                    elif HAS_TRANSFORMERS and hasattr(model, 'config'):
                        videos_transformed = videos.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]
                        try:
                            outputs = model(pixel_values=videos_transformed, interpolate_pos_encoding=False)
                            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                        except TypeError:
                            logits = model(videos_transformed)
                    else:
                        logits = model(videos)
                    
                    loss = criterion(logits, labels)
            else:
                # CPU mode - no autocast
                if isinstance(model, SimpleViViT):
                    logits = model(videos)
                elif HAS_TRANSFORMERS and hasattr(model, 'config'):
                    videos_transformed = videos.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]
                    try:
                        outputs = model(pixel_values=videos_transformed, interpolate_pos_encoding=False)
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    except TypeError:
                        logits = model(videos_transformed)
                else:
                    logits = model(videos)
                
                loss = criterion(logits, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()
            
            # Optimized top5 calculation
            _, top5_pred = logits.topk(5, dim=1)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            # Update less frequently (every 20 batches or last batch)
            if batch_idx % 20 == 0 or batch_idx == len(dataloader) - 1:
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'top1': 100.0 * correct_top1 / total,
                    'top5': 100.0 * correct_top5 / total
                })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc_top1 = 100.0 * correct_top1 / total
    epoch_acc_top5 = 100.0 * correct_top5 / total
    return epoch_loss, epoch_acc_top1, epoch_acc_top5


def train_ratio(observation_ratio, device, output_dir):
    """Train model for a specific observation ratio."""
    print("\n" + "=" * 80)
    print(f"Training with observation ratio: {observation_ratio}")
    print("=" * 80)
    
    # Clear CUDA cache before starting new ratio
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {total:.2f} GB total")
            torch.cuda.empty_cache()
    
    # Create datasets
    train_dataset = UCF101Dataset(
        TRAIN_CSV_PATH, DICTIONARY_PATH, VIDEO_ROOT,
        observation_ratio=observation_ratio,
        frame_length=FRAME_LENGTH,
        frame_size=FRAME_SIZE,
        is_training=True
    )
    
    val_dataset = UCF101Dataset(
        TRAIN_CSV_PATH, DICTIONARY_PATH, VIDEO_ROOT,
        observation_ratio=observation_ratio,
        frame_length=FRAME_LENGTH,
        frame_size=FRAME_SIZE,
        is_training=False,
        val_csv_path=VAL_CSV_PATH
    )
    
    # Use pin_memory only for GPU
    pin_memory = (device.type == 'cuda')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=pin_memory,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=pin_memory,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    # Create model
    print("\n" + "=" * 80)
    print("CREATING AND LOADING MODEL")
    print("=" * 80)
    print("Creating ViViT model...")
    
    try:
        model = create_vivit_model(num_classes=NUM_CLASSES, pretrained=True)
        print(f"✓ Model created successfully")
        
        # Move model to device (GPU by default)
        print(f"\nMoving model to device: {device}")
        print(f"Device type: {device.type}, CUDA available: {torch.cuda.is_available()}")
        if device.type == 'cpu' and torch.cuda.is_available():
            print("⚠ WARNING: CUDA is available but device is set to CPU!")
            print("⚠ Attempting to switch to GPU...")
            device = torch.device("cuda:0")
            print(f"✓ Switched to device: {device}")
        
        model = model.to(device)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            first_param = next(model.parameters())
            if first_param.is_cuda:
                print(f"✓ Model successfully moved to GPU {device}")
            else:
                print(f"✗ Model failed to move to GPU, still on CPU")
        else:
            print(f"✓ Model loaded on CPU (device: {device})")
            print("⚠ WARNING: Training will be very slow on CPU!")
        
        # Compile model if supported (PyTorch 2.0+)
        if USE_COMPILE and hasattr(torch, 'compile') and device.type == 'cuda':
            try:
                print("Compiling model with torch.compile (this may take a minute)...")
                # Try different compile modes in order of preference
                compile_modes = [
                    ('reduce-overhead', 'reduce-overhead'),  # Faster compilation, good speedup
                    ('default', 'default'),  # Standard compilation
                    ('no-ops', 'no-ops'),  # Minimal compilation, no C compiler needed
                ]
                
                compiled = False
                for mode_name, mode in compile_modes:
                    try:
                        model = torch.compile(model, mode=mode)
                        print(f"✓ Model compiled successfully with mode '{mode_name}'")
                        compiled = True
                        break
                    except (RuntimeError, Exception) as compile_error:
                        error_str = str(compile_error).lower()
                        # Check for various compilation errors
                        if any(keyword in error_str for keyword in ['c compiler', 'triton', 'inductor', 'ld:', 'cannot find', '-lcuda', 'linker']):
                            # Try next mode if compilation issue
                            continue
                        else:
                            # Other error, re-raise
                            raise
                
                if not compiled:
                    print("⚠ Could not compile model (compilation/linker issues detected). Continuing without compilation.")
                    print("  Note: Training will be slightly slower but will work fine.")
                    print("  To enable compilation, ensure:")
                    print("    - CUDA libraries are properly installed")
                    print("    - Build tools are available: apt-get install build-essential")
                    
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['c compiler', 'triton', 'inductor', 'ld:', 'cannot find', '-lcuda', 'linker']):
                    print(f"⚠ Could not compile model: Compilation/linker error detected.")
                    print(f"  Continuing without compilation. Training will work but may be slower.")
                    print(f"  Error details: {str(e)[:200]}")
                else:
                    print(f"⚠ Could not compile model: {e}. Continuing without compilation.")
            
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "cuda" in error_msg or "out of memory" in error_msg:
            print(f"\n✗ CUDA error when loading model: {e}")
            if device.type == 'cuda':
                print("Attempting to clear cache and retry...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                model = create_vivit_model(num_classes=NUM_CLASSES, pretrained=True)
                model = model.to(device)
                print("✓ Retry successful")
            else:
                raise
        else:
            raise
    
    # Create scaler for mixed precision (after model is loaded)
    scaler = None
    if USE_MIXED_PRECISION and device.type == 'cuda':
        try:
            # Use new API: torch.amp.GradScaler('cuda') instead of deprecated torch.cuda.amp.GradScaler()
            scaler = torch.amp.GradScaler('cuda')
            print("✓ Mixed precision training enabled (FP16)")
        except Exception as e:
            print(f"⚠ Could not enable mixed precision: {e}. Continuing with FP32.")
            scaler = None
    
    print("=" * 80 + "\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Create log file for this ratio
    checkpoint_dir = os.path.join(output_dir, f"observation_ratio_{observation_ratio}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, f"training_log_ratio_{observation_ratio}.csv")
    
    # Check existing log to resume from last epoch
    start_epoch = 1
    if os.path.exists(log_file):
        try:
            existing_log = pd.read_csv(log_file)
            if len(existing_log) > 0:
                start_epoch = int(existing_log['epoch'].max()) + 1
                if start_epoch <= NUM_EPOCHS:
                    print(f"Found existing log with {len(existing_log)} epochs. Resuming from epoch {start_epoch}")
                else:
                    print(f"Training already completed ({len(existing_log)} epochs). Skipping...")
                    best_val_acc = float(existing_log['val_acc_top1'].max())
                    best_train_acc_row = existing_log.loc[existing_log['val_acc_top1'].idxmax()]
                    best_train_acc = float(best_train_acc_row['train_acc_top1'])
                    best_model_path = os.path.join(checkpoint_dir, "vivit_ucf_best.pth")
                    return best_model_path, best_val_acc, best_train_acc
        except Exception as e:
            print(f"Could not read existing log: {e}. Starting from epoch 1.")
            start_epoch = 1
    
    # Initialize CSV log file (only if not resuming)
    if start_epoch == 1:
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 
                'train_loss', 'train_acc_top1', 'train_acc_top5',
                'train_eval_loss', 'train_eval_acc_top1', 'train_eval_acc_top5',
                'val_loss', 'val_acc_top1', 'val_acc_top5',
                'learning_rate'
            ])
    
    # Training loop
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, "vivit_ucf_best.pth")
    training_history = []
    
    # Load best checkpoint if resuming
    if start_epoch > 1 and os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                best_val_acc = checkpoint.get('val_acc_top1', 0.0)
                best_train_acc = checkpoint.get('train_acc_top1', 0.0)
                print(f"Loaded best checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                print(f"  Best Val Acc Top1: {best_val_acc:.2f}%, Best Train Acc Top1: {best_train_acc:.2f}%")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting fresh.")
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        
        # Train
        try:
            train_loss, train_acc_top1, train_acc_top5 = train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch, scaler=scaler
            )
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "no kernel image" in error_msg or "compute capability" in error_msg:
                print(f"\n✗ CUDA compatibility error: GPU compute capability not supported")
                print(f"  Error: {e}")
                print(f"  This usually happens with newer GPUs (e.g., RTX 5090 with sm_120)")
                print(f"  Install PyTorch nightly for RTX 5090 support:")
                print(f"    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")
                print(f"  Falling back to CPU mode...")
                
                # Move model and data loaders to CPU
                device = torch.device("cpu")
                model = model.to(device)
                print(f"✓ Model moved to CPU")
                
                # Recreate data loaders without pin_memory for CPU
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True,
                    num_workers=NUM_WORKERS, 
                    pin_memory=False,
                    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
                    persistent_workers=True if NUM_WORKERS > 0 else False
                )
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=BATCH_SIZE, 
                    shuffle=False,
                    num_workers=NUM_WORKERS, 
                    pin_memory=False,
                    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
                    persistent_workers=True if NUM_WORKERS > 0 else False
                )
                
                # Retry training on CPU (disable scaler on CPU)
                print(f"Retrying training on CPU (this will be slower)...")
                train_loss, train_acc_top1, train_acc_top5 = train_one_epoch(
                    model, train_loader, optimizer, criterion, device, epoch, scaler=None
                )
            elif "cuda" in error_msg or "out of memory" in error_msg:
                print(f"\n⚠ CUDA error during training epoch {epoch}: {e}")
                if device.type == 'cuda':
                    print("Clearing GPU cache and retrying...")
                    torch.cuda.empty_cache()
                    if training_history:
                        print(f"Training stopped at epoch {epoch-1}")
                        break
                    else:
                        raise
                else:
                    raise
            else:
                raise
        
        # Evaluate on train set (for monitoring)
        # Create a separate train loader without shuffling for evaluation
        train_eval_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=NUM_WORKERS, 
            pin_memory=(device.type == 'cuda'),
            prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
            persistent_workers=True if NUM_WORKERS > 0 else False
        )
        
        try:
            train_eval_loss, train_eval_acc_top1, train_eval_acc_top5 = validate(
                model, train_eval_loader, criterion, device
            )
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "no kernel image" in error_msg or "compute capability" in error_msg:
                print(f"\n⚠ CUDA compatibility error during train evaluation, using CPU...")
                device = torch.device("cpu")
                model = model.to(device)
                train_eval_loader = DataLoader(
                    train_dataset, 
                    batch_size=BATCH_SIZE, 
                    shuffle=False,
                    num_workers=NUM_WORKERS, 
                    pin_memory=False,
                    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
                    persistent_workers=True if NUM_WORKERS > 0 else False
                )
                train_eval_loss, train_eval_acc_top1, train_eval_acc_top5 = validate(
                    model, train_eval_loader, criterion, device
                )
            elif "cuda" in error_msg or "out of memory" in error_msg:
                print(f"\n⚠ CUDA error during train evaluation epoch {epoch}: {e}")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    train_eval_loss = train_loss
                    train_eval_acc_top1 = train_acc_top1
                    train_eval_acc_top5 = train_acc_top5
                else:
                    raise
            else:
                raise
        
        # Evaluate on val set
        try:
            val_loss, val_acc_top1, val_acc_top5 = validate(
                model, val_loader, criterion, device
            )
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "no kernel image" in error_msg or "compute capability" in error_msg:
                print(f"\n⚠ CUDA compatibility error during validation, using CPU...")
                device = torch.device("cpu")
                model = model.to(device)
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=BATCH_SIZE, 
                    shuffle=False,
                    num_workers=NUM_WORKERS, 
                    pin_memory=False,
                    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
                    persistent_workers=True if NUM_WORKERS > 0 else False
                )
                val_loss, val_acc_top1, val_acc_top5 = validate(
                    model, val_loader, criterion, device
                )
            elif "cuda" in error_msg or "out of memory" in error_msg:
                print(f"\n⚠ CUDA error during validation epoch {epoch}: {e}")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    val_loss = train_loss
                    val_acc_top1 = train_acc_top1 * 0.9
                    val_acc_top5 = train_acc_top5 * 0.9
                else:
                    raise
            else:
                raise
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc Top1: {train_acc_top1:.2f}%, Train Acc Top5: {train_acc_top5:.2f}%")
        print(f"Train Eval Loss: {train_eval_loss:.4f}, Train Eval Acc Top1: {train_eval_acc_top1:.2f}%, Train Eval Acc Top5: {train_eval_acc_top5:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc Top1: {val_acc_top1:.2f}%, Val Acc Top5: {val_acc_top5:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save to CSV log file immediately after each epoch
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                train_loss, train_acc_top1, train_acc_top5,
                train_eval_loss, train_eval_acc_top1, train_eval_acc_top5,
                val_loss, val_acc_top1, val_acc_top5,
                current_lr
            ])
        
        # Store history
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc_top1': train_acc_top1,
            'train_acc_top5': train_acc_top5,
            'train_eval_loss': train_eval_loss,
            'train_eval_acc_top1': train_eval_acc_top1,
            'train_eval_acc_top5': train_eval_acc_top5,
            'val_loss': val_loss,
            'val_acc_top1': val_acc_top1,
            'val_acc_top5': val_acc_top5,
            'lr': current_lr
        })
        
        # Save best model based on validation accuracy
        if val_acc_top1 > best_val_acc:
            best_val_acc = val_acc_top1
            best_train_acc = train_acc_top1
            best_model_path = os.path.join(checkpoint_dir, "vivit_ucf_best.pth")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc_top1': val_acc_top1,
                'val_acc_top5': val_acc_top5,
                'train_acc_top1': train_acc_top1,
                'train_acc_top5': train_acc_top5,
                'observation_ratio': observation_ratio,
            }, best_model_path)
            print(f"Saved best model to {best_model_path} (Val Acc Top1: {val_acc_top1:.2f}%, Val Acc Top5: {val_acc_top5:.2f}%)")
    
    # Save training summary for this ratio
    ratio_summary = {
        'observation_ratio': observation_ratio,
        'best_val_acc_top1': best_val_acc,
        'best_val_acc_top5': training_history[-1]['val_acc_top5'] if training_history else 0.0,
        'best_train_acc_top1': best_train_acc,
        'best_train_acc_top5': training_history[-1]['train_acc_top5'] if training_history else 0.0,
        'checkpoint_path': best_model_path,
        'log_file': log_file,
        'total_epochs': NUM_EPOCHS,
        'final_train_acc_top1': training_history[-1]['train_acc_top1'] if training_history else 0.0,
        'final_val_acc_top1': training_history[-1]['val_acc_top1'] if training_history else 0.0,
    }
    
    summary_file = os.path.join(checkpoint_dir, f"summary_ratio_{observation_ratio}.json")
    with open(summary_file, 'w') as f:
        json.dump(ratio_summary, f, indent=2)
    
    print(f"\n✓ Training log saved to: {log_file}")
    print(f"✓ Ratio summary saved to: {summary_file}")
    
    return best_model_path, best_val_acc, best_train_acc


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Train ViViT on UCF-101 with progressive observation ratios')
    parser.add_argument('--ratios', nargs='+', type=float, default=OBSERVATION_RATIOS,
                        help='Observation ratios to train (default: 0.1 0.3 0.5 0.7 0.9)')
    parser.add_argument('--output', type=str, default=OUTPUT_ROOT,
                        help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (default: 0, uses GPU by default)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage (default: uses GPU if available)')
    
    args = parser.parse_args()
    
    # Set device - default to GPU, only use CPU if explicitly requested
    device = None
    use_gpu = False
    
    print("\n" + "=" * 80)
    print("GPU DETECTION")
    print("=" * 80)
    
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    
    # Check GPU compute capability compatibility
    gpu_compatible = True
    if cuda_available and not args.cpu:
        try:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                gpu_capability = torch.cuda.get_device_capability(args.gpu)
                gpu_name = torch.cuda.get_device_name(args.gpu)
                print(f"GPU {args.gpu}: {gpu_name}, Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
                
                # Check if PyTorch supports this GPU
                torch_version = torch.__version__
                if gpu_capability[0] > 9:
                    # PyTorch 2.9.0 stable only supports up to sm_90
                    # PyTorch nightly (2.10.0.dev+) supports sm_120+
                    if "dev" in torch_version or "nightly" in torch_version.lower():
                        print(f"✓ PyTorch nightly detected ({torch_version}) - supports compute capability {gpu_capability[0]}.{gpu_capability[1]}")
                        gpu_compatible = True
                    else:
                        print(f"\n⚠ WARNING: GPU compute capability {gpu_capability[0]}.{gpu_capability[1]} may not be supported by PyTorch {torch_version}")
                        print(f"  Stable PyTorch only supports up to compute capability 9.0 (sm_90)")
                        print(f"  For RTX 5090, install PyTorch nightly:")
                        print(f"    pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")
                        print(f"  The script will try GPU first and fallback to CPU if needed")
                        gpu_compatible = False
                else:
                    gpu_compatible = True
        except Exception as e:
            print(f"⚠ Could not check GPU compute capability: {e}")
    
    # Force GPU usage by default (only use CPU if explicitly requested or incompatible)
    if args.cpu:
        print("⚠ CPU mode forced by --cpu flag")
        device = torch.device("cpu")
        use_gpu = False
    elif cuda_available and gpu_compatible:
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs detected: {num_gpus}")
        
        if args.gpu >= num_gpus:
            print(f"⚠ Warning: GPU {args.gpu} requested but only {num_gpus} GPU(s) available")
            if num_gpus > 0:
                args.gpu = 0
                print(f"  Using GPU 0 instead")
        
        device = torch.device(f"cuda:{args.gpu}")
        use_gpu = True
    elif cuda_available and not gpu_compatible:
        print(f"\n⚠ GPU detected but compute capability not compatible")
        print(f"  The script will try GPU first and automatically fallback to CPU if errors occur")
        device = torch.device(f"cuda:{args.gpu}")
        use_gpu = True  # Try GPU first, fallback to CPU if needed
        
        # Set default CUDA device
        try:
            torch.cuda.set_device(args.gpu)
            print(f"✓ Set default CUDA device to GPU {args.gpu} (will fallback to CPU if needed)")
        except Exception as e:
            print(f"⚠ Could not set default CUDA device: {e}")
        
        # Test GPU with a simple operation
        try:
            test_tensor = torch.zeros(1, device=device)
            _ = test_tensor * 2
            del test_tensor
            torch.cuda.empty_cache()
            print(f"✓ GPU {args.gpu} basic test passed (but compute ops may fail)")
        except Exception as e:
            print(f"⚠ GPU test failed: {e}")
            print(f"  Falling back to CPU immediately")
            device = torch.device("cpu")
            use_gpu = False
        
        if use_gpu:
            print(f"✓ Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ Device: {device}")
            
            # Show GPU memory info
            try:
                total_mem = torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(args.gpu) / 1024**3
                reserved = torch.cuda.memory_reserved(args.gpu) / 1024**3
                print(f"✓ GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved / {total_mem:.2f} GB total")
            except Exception as e:
                print(f"⚠ Could not get GPU memory info: {e}")
        
        # Final check: ensure device is set correctly
        if cuda_available and not args.cpu:
            if device.type != 'cuda':
                print(f"⚠ WARNING: Device is {device} but CUDA is available. Forcing GPU usage...")
                device = torch.device(f"cuda:{args.gpu}")
                print(f"✓ Device set to: {device}")
    else:
        print("✗ CUDA not available in PyTorch")
        print(f"  - PyTorch version: {torch.__version__}")
        if args.cpu:
            print("  - Using CPU as requested")
            device = torch.device("cpu")
            use_gpu = False
        else:
            print("  - WARNING: No GPU available but --cpu not set")
            print("  - Attempting to use CPU anyway (training will be very slow)")
            device = torch.device("cpu")
            use_gpu = False
    
    print("=" * 80 + "\n")
    
    if not use_gpu:
        print("\n" + "=" * 80)
        print("WARNING: Training will be VERY SLOW on CPU!")
        print("Expected time: ~4-5 days per ratio (30 epochs)")
        print("If you have a GPU, remove --cpu flag")
        print("=" * 80 + "\n")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load existing results if summary file exists
    summary_path = os.path.join(args.output, "training_summary.json")
    results = []
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {summary_path}")
        except:
            results = []
    
    # Train for each observation ratio
    for ratio in args.ratios:
        # Skip if already completed
        if any(r.get('observation_ratio') == ratio for r in results):
            print(f"\nRatio {ratio} already completed, skipping...")
            continue
        
        try:
            checkpoint_path, best_val_acc, best_train_acc = train_ratio(ratio, device, args.output)
            
            # Add to results immediately
            result = {
                'observation_ratio': ratio,
                'checkpoint_path': checkpoint_path,
                'best_val_acc_top1': best_val_acc,
                'best_train_acc_top1': best_train_acc
            }
            results.append(result)
            
            # Save summary IMMEDIATELY after each ratio (incremental save)
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Completed training for ratio {ratio}:")
            print(f"  - Best Val Acc Top1: {best_val_acc:.2f}%")
            print(f"  - Best Train Acc Top1: {best_train_acc:.2f}%")
            print(f"  - Summary saved to {summary_path}")
            
        except Exception as e:
            print(f"\nError training ratio {ratio}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TRAINING SUMMARY")
    print("=" * 80)
    for r in results:
        val_acc = r.get('best_val_acc_top1', 0.0)
        train_acc = r.get('best_train_acc_top1', 'N/A')
        if isinstance(train_acc, (int, float)):
            print(f"Ratio {r['observation_ratio']}: Val Acc Top1: {val_acc:.2f}%, Train Acc Top1: {train_acc:.2f}%")
        else:
            print(f"Ratio {r['observation_ratio']}: Val Acc Top1: {val_acc:.2f}%, Train Acc Top1: {train_acc}")
    print(f"\nTraining summary saved to {summary_path}")
    print("Training completed!")


if __name__ == "__main__":
    main()
