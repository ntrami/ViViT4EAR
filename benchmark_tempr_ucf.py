#!/usr/bin/env python3
"""
Benchmark script for evaluating TEMPr models on UCF-101 test set.

This script evaluates TEMPr models trained with different observation ratios (0.1, 0.3, 0.5, 0.7, 0.9)
on the UCF-101 test set, measuring both accuracy and average inference time per video.
Results are saved to a CSV file for later visualization.
"""

import os
import sys
import json
import time
import warnings
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pytorchvideo
from pytorchvideo.data import encoded_video

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset
from network.symbol_builder import Combined
from network.config import get_config

# ==========================================
# CONFIGURATION
# ==========================================

# Model checkpoints for different observation ratios
CHECKPOINTS = {
    "0.1": "/data/progressive-action-prediction/results/UCF-101/latents_256_heads_8/samplers_3/observation_ratio_0.1/Tempr_h_movinet_ada/Tempr_h_movinet_ada_best.pth",
    "0.3": "/data/progressive-action-prediction/results/UCF-101/latents_256_heads_8/samplers_3/observation_ratio_0.3/Tempr_h_movinet_ada/Tempr_h_movinet_ada_best.pth",
    "0.5": "/data/progressive-action-prediction/results/UCF-101/latents_256_heads_8/samplers_3/observation_ratio_0.5/Tempr_h_movinet_ada/Tempr_h_movinet_ada_best.pth",
    "0.7": "/data/progressive-action-prediction/results/UCF-101/latents_256_heads_8/samplers_3/observation_ratio_0.7/Tempr_h_movinet_ada/Tempr_h_movinet_ada_best.pth",
    "0.9": "/data/progressive-action-prediction/results/UCF-101/latents_256_heads_8/samplers_3/observation_ratio_0.9/Tempr_h_movinet_ada/Tempr_h_movinet_ada_best.pth"
}

# Dataset paths
DICTIONARY_PATH = "/data/progressive-action-prediction/labels/UCF-101/dictionary.json"
TEST_CSV_PATH = "/data/progressive-action-prediction/labels/UCF-101/test.csv"
VIDEO_ROOT = "/data/UCF-101"
OUTPUT_CSV = "/data/progressive-action-prediction/logs/tempr_ucf_benchmark.csv"

# Model configuration (same for all ratios)
DATASET = "UCF-101"
BACKBONE = "movinet"
HEAD = "Tempr_h"
POOL = "max"  # Changed from "ada" to "max" to avoid adaPool dependency
NUM_SAMPLERS = 3
FRAME_LENGTH = 16
FRAME_SIZE = 224
GPU_ID = 0

# Evaluation configuration
NUM_TEST_SAMPLES = 1000  # Number of random samples from test set (set to None to use all)
RANDOM_SEED = 42  # Random seed for reproducibility

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_class_mapping(dictionary_path):
    """
    Load class mapping from dictionary.json.
    
    Args:
        dictionary_path: Path to dictionary.json file
        
    Returns:
        class_name_to_id: Dictionary mapping class name to class ID
        id_to_class_name: Dictionary mapping class ID to class name
    """
    with open(dictionary_path, 'r') as f:
        name_to_id = json.load(f)
    
    # Create reverse mapping (id -> name)
    id_to_name = {v: k for k, v in name_to_id.items()}
    
    # Note: The dictionary maps class_name -> class_id (not necessarily sequential)
    return name_to_id, id_to_name


def load_test_data(test_csv_path, num_samples=1000, random_seed=42):
    """
    Load test data from CSV file and randomly sample a subset.
    
    Args:
        test_csv_path: Path to test.csv file
        num_samples: Number of samples to randomly select (default: 1000, None to use all)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        DataFrame with randomly sampled test videos
    """
    df = pd.read_csv(test_csv_path)
    
    # Randomly sample records if num_samples is specified
    if num_samples is None:
        df_sampled = df
        print(f"Using all {len(df)} records from test set")
    elif len(df) > num_samples:
        df_sampled = df.sample(n=num_samples, random_state=random_seed).reset_index(drop=True)
        print(f"Randomly sampled {num_samples} records from {len(df)} total records")
    else:
        df_sampled = df
        print(f"Using all {len(df)} records (less than requested {num_samples})")
    
    return df_sampled


def process_video(video_path, observation_ratio, num_samplers=3, frame_length=16, frame_size=224, fps=30.0):
    """
    Process video into model input format with progressive sampling.
    
    Args:
        video_path: Path to video file
        observation_ratio: Ratio of video to observe (0.1, 0.3, 0.5, 0.7, 0.9)
        num_samplers: Number of progressive samplers
        frame_length: Number of frames per sampler
        frame_size: Frame size (height/width)
        fps: Frames per second (assumed)
        
    Returns:
        input_tensor: Tensor of shape [1, num_samplers, C, T, H, W]
    """
    # Load video
    video = encoded_video.EncodedVideo.from_path(video_path)
    video_duration = float(video.duration)
    
    # Calculate total frames and observed frames
    total_frames = int(video_duration * fps)
    num_observed_frames = int(total_frames * observation_ratio)
    
    # Sample frames for each sampler (progressive sampling)
    sampled_clips = []
    for s in range(1, num_samplers + 1):
        # Calculate range for this sampler
        range_max = int(num_observed_frames * (s / num_samplers))
        
        # Sequential sampling - create indices evenly spaced
        clip_start = 0
        clip_end = max(range_max - 1, 0)
        
        if clip_end <= 0:
            # If no frames available, use first frame
            indices = [0] * frame_length
        else:
            # Create evenly spaced indices
            if clip_end < frame_length:
                # If we have fewer frames than needed, repeat the last index
                indices = list(range(clip_start, clip_end + 1))
                while len(indices) < frame_length:
                    indices.append(indices[-1] if indices else 0)
            else:
                # Evenly sample frame_length frames from [0, clip_end]
                step = max(1, clip_end // frame_length)
                indices = list(range(clip_start, min(clip_end + 1, total_frames), step))
                
                # Ensure we have exactly frame_length frames
                while len(indices) < frame_length:
                    indices.append(indices[-1] if indices else 0)
        
        indices = indices[:frame_length]
        
        # Get clip from video
        time_start = indices[0] / fps if len(indices) > 0 else 0
        time_end = indices[-1] / fps if len(indices) > 0 else video_duration
        
        clip = video.get_clip(time_start, time_end)
        
        # Get frames - pytorchvideo returns dict with 'video' key
        if isinstance(clip, dict):
            video_tensor = clip['video']  # Shape: [C, T, H, W]
        else:
            video_tensor = clip
        
        # Extract specific frames if needed
        if video_tensor.shape[1] < frame_length:
            # Pad by repeating last frame
            padding = frame_length - video_tensor.shape[1]
            last_frame = video_tensor[:, -1:, :, :]
            video_tensor = torch.cat([video_tensor, last_frame.repeat(1, padding, 1, 1)], dim=1)
        elif video_tensor.shape[1] > frame_length:
            # Take evenly spaced frames
            step = video_tensor.shape[1] // frame_length
            selected_frames = [i * step for i in range(frame_length)]
            video_tensor = video_tensor[:, selected_frames, :, :]
        
        # Ensure exactly frame_length frames
        video_tensor = video_tensor[:, :frame_length, :, :]
        
        # Apply transformations
        # Normalize to [0, 1]
        video_tensor = video_tensor.float() / 255.0
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        video_tensor = (video_tensor - mean) / std
        
        # Resize to frame_size x frame_size
        video_tensor = F.interpolate(
            video_tensor, size=(frame_size, frame_size),
            mode='bilinear', align_corners=False
        )
        
        sampled_clips.append(video_tensor)
    
    # Stack all samplers: [num_samplers, C, T, H, W]
    input_tensor = torch.stack(sampled_clips, dim=0)
    
    # Add batch dimension: [1, num_samplers, C, T, H, W]
    input_tensor = input_tensor.unsqueeze(0)
    
    return input_tensor


def load_model(checkpoint_path, observation_ratio, device):
    """
    Load TEMPr model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        observation_ratio: Observation ratio (for logging)
        device: torch device
        
    Returns:
        model: Loaded model in eval mode
    """
    print(f"\nLoading model for observation ratio {observation_ratio}...")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Get dataset config
    dataset_cfg = dataset.get_config(name=DATASET)
    
    # Get input config
    input_conf = get_config(name=f"{HEAD} w/ {BACKBONE}")
    
    # Create model
    kwargs = {
        'backbone': BACKBONE,
        'head': HEAD,
        'pool': POOL,
        'num_samplers': NUM_SAMPLERS,
        'num_classes': dataset_cfg['num_classes'],
        'input_conf': input_conf,
    }
    
    net = Combined(**kwargs)
    net = torch.nn.DataParallel(net).to(device)
    net.eval()
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['state_dict'], strict=False)
    
    print("Model loaded successfully!")
    return net


def evaluate_ratio(model_path, observation_ratio, test_df, class_name_to_id, id_to_class_name, device):
    """
    Evaluate a single model checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        observation_ratio: Observation ratio (as string, e.g., "0.1")
        test_df: DataFrame with test videos
        class_name_to_id: Dictionary mapping class name to ID
        id_to_class_name: Dictionary mapping ID to class name
        device: torch device
        
    Returns:
        accuracy: Accuracy on test set
        avg_inference_time: Average inference time per video (seconds)
        num_processed: Number of videos successfully processed
    """
    # Load model
    obs_ratio_float = float(observation_ratio)
    model = load_model(model_path, observation_ratio, device)
    
    # Statistics
    correct = 0
    total = 0
    total_inference_time = 0.0
    num_processed = 0
    missing_videos = 0
    
    # Process each video
    print(f"\nEvaluating {len(test_df)} videos...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Ratio {observation_ratio}"):
        video_id = row['id']
        label_name = row['label']
        
        # Build video path - UCF-101 has videos organized in subdirectories by label
        video_path = os.path.join(VIDEO_ROOT, label_name, f"{video_id}.avi")
        
        # Check if video exists
        if not os.path.exists(video_path):
            missing_videos += 1
            if missing_videos <= 5:  # Only warn for first 5 missing videos
                warnings.warn(f"Video not found: {video_path}")
            continue
        
        try:
            # Process video
            video_tensor = process_video(
                video_path, 
                observation_ratio=obs_ratio_float,
                num_samplers=NUM_SAMPLERS,
                frame_length=FRAME_LENGTH,
                frame_size=FRAME_SIZE
            )
            
            # Move to device
            video_tensor = video_tensor.to(device)
            
            # Inference
            with torch.no_grad():
                # Synchronize GPU
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                # Forward pass
                output = model(video_tensor)
                
                # Get final prediction (model may return tuple)
                if isinstance(output, tuple):
                    final_pred = output[0]
                else:
                    final_pred = output
                
                # Synchronize GPU
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                inference_time = end_time - start_time
                total_inference_time += inference_time
            
            # Get predicted class
            pred_class_id = torch.argmax(final_pred, dim=1).item()
            
            # Get ground truth class ID
            true_class_id = class_name_to_id.get(label_name)
            
            if true_class_id is None:
                warnings.warn(f"Unknown label: {label_name}")
                continue
            
            # Check if correct
            if pred_class_id == true_class_id:
                correct += 1
            
            total += 1
            num_processed += 1
            
        except Exception as e:
            warnings.warn(f"Error processing video {video_id}: {str(e)}")
            continue
    
    # Calculate metrics
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    avg_inference_time = (total_inference_time / num_processed) if num_processed > 0 else 0.0
    
    if missing_videos > 5:
        print(f"Warning: {missing_videos} videos were missing (only first 5 were shown)")
    
    return accuracy, avg_inference_time, num_processed


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():
    """Main function to run benchmark."""
    print("=" * 80)
    print("TEMPr UCF-101 Benchmark")
    print("=" * 80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load class mapping
    print(f"\nLoading class mapping from: {DICTIONARY_PATH}")
    class_name_to_id, id_to_class_name = load_class_mapping(DICTIONARY_PATH)
    print(f"Loaded {len(class_name_to_id)} classes")
    
    # Load test data (randomly sample records if specified)
    print(f"\nLoading test data from: {TEST_CSV_PATH}")
    test_df = load_test_data(TEST_CSV_PATH, num_samples=NUM_TEST_SAMPLES, random_seed=RANDOM_SEED)
    print(f"Loaded {len(test_df)} test videos for evaluation")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_CSV)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Results storage
    results = []
    
    # Evaluate each observation ratio
    for ratio, checkpoint_path in CHECKPOINTS.items():
        print("\n" + "=" * 80)
        print(f"Evaluating observation ratio: {ratio}")
        print("=" * 80)
        
        try:
            accuracy, avg_inference_time, num_processed = evaluate_ratio(
                checkpoint_path,
                ratio,
                test_df,
                class_name_to_id,
                id_to_class_name,
                device
            )
            
            # Store results
            results.append({
                'observation_ratio': ratio,
                'accuracy': accuracy,
                'avg_inference_time_seconds': avg_inference_time,
                'num_processed_videos': num_processed,
                'checkpoint_path': checkpoint_path
            })
            
            print(f"\nResults for ratio {ratio}:")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Average inference time: {avg_inference_time*1000:.2f} ms per video")
            print(f"  Processed videos: {num_processed}/{len(test_df)}")
            
        except Exception as e:
            print(f"\nError evaluating ratio {ratio}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results to CSV
    print("\n" + "=" * 80)
    print("Saving results to CSV...")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 80)
    print("Benchmark completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

