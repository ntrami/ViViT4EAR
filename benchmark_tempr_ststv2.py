#!/usr/bin/env python3
"""
Benchmark script for evaluating TEMPr models on Something-Something v2 test set.

This script evaluates TEMPr models trained with different observation ratios (0.1, 0.3, 0.5, 0.7, 0.9)
on the Something-Something v2 test set, measuring both accuracy and average inference time per video.
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
    "0.1": "/data/progressive-action-prediction/st-ckpt/Tempr_h_movinet_ada_obs_01.pth",
    "0.3": "/data/progressive-action-prediction/st-ckpt/Tempr_h_movinet_ada_obs_03.pth",
    "0.5": "/data/progressive-action-prediction/st-ckpt/Tempr_h_movinet_ada_obs_05.pth",
    "0.7": "/data/progressive-action-prediction/st-ckpt/Tempr_h_movinet_ada_obs_07.pth",
    "0.9": "/data/progressive-action-prediction/st-ckpt/Tempr_h_movinet_ada_obs_09.pth"
}

# Dataset paths
LABELS_JSON_PATH = "/data/progressive-action-prediction/labels/smthng-smthng/v2/labels.json"
TEST_CSV_PATH = "/data/progressive-action-prediction/labels/smthng-smthng/v2/test-answers.csv"
VIDEO_ROOT = "/data/stst_test"  # Updated according to prompt
OUTPUT_CSV = "/data/progressive-action-prediction/logs/tempr_STST_benchmark.csv"  # Updated according to prompt

# Model configuration (same for all ratios)
DATASET = "smthng-smthng_v2"
BACKBONE = "movinet"
HEAD = "Tempr_h"
# Try to use ADA pooling (as models were trained), fallback to MAX if not available
# Check if adaPool is available
try:
    import adapool_cuda
    from adaPool import AdaPool1d
    POOL = "ada"  # Use ADA pooling as models were trained with
    ADAPOOL_AVAILABLE = True
except ImportError:
    POOL = "max"  # Fallback to MAX pooling if adaPool not available
    ADAPOOL_AVAILABLE = False
    print("=" * 80)
    print("WARNING: adaPool is NOT available! Using MAX pooling instead.")
    print("This may cause SIGNIFICANT accuracy degradation!")
    print("Models were trained with ADA pooling, results may not be accurate.")
    print("=" * 80)
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

def load_class_mapping(labels_json_path):
    """
    Load class mapping from labels.json.
    
    Args:
        labels_json_path: Path to labels.json file
        
    Returns:
        label_to_id: Dictionary mapping label name to class ID
        id_to_label: Dictionary mapping class ID to label name
    """
    with open(labels_json_path, 'r') as f:
        label_to_id = json.load(f)
    
    # Convert string IDs to integers
    label_to_id = {k: int(v) for k, v in label_to_id.items()}
    
    # Create reverse mapping (id -> label)
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    return label_to_id, id_to_label


def validate_video(video_path):
    """
    Quick validation to check if video file exists and can be decoded.
    
    Args:
        video_path: Path to video file
        
    Returns:
        bool: True if video is valid, False otherwise
    """
    if not os.path.exists(video_path):
        return False
    
    try:
        # Quick check: try to load video and get at least one frame
        video = encoded_video.EncodedVideo.from_path(video_path)
        if video.duration <= 0:
            return False
        
        # Try to get first frame
        clip = video.get_clip(0.0, min(0.1, video.duration))
        if clip is None:
            return False
        
        if isinstance(clip, dict):
            video_tensor = clip.get('video')
        else:
            video_tensor = clip
        
        if video_tensor is None or video_tensor.shape[1] == 0:
            return False
        
        return True
    except Exception:
        return False


def load_test_data(test_csv_path, num_samples=1000, random_seed=42, video_root=None):
    """
    Load test data from CSV file and filter to get only valid videos.
    Continues until we have num_samples valid videos.
    
    Args:
        test_csv_path: Path to test-answers.csv file
        num_samples: Number of valid samples to collect (default: 1000, None to use all valid)
        random_seed: Random seed for reproducibility (default: 42)
        video_root: Root directory for video files (for validation)
        
    Returns:
        DataFrame with valid test videos (columns: id, label)
    """
    # Read CSV with semicolon separator
    df = pd.read_csv(test_csv_path, sep=';', header=None, names=['id', 'label'])
    print(f"Loaded {len(df)} total records from CSV")
    
    if num_samples is None:
        # Use all records, but still validate them
        if video_root:
            print("Validating all videos...")
            valid_videos = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating videos"):
                video_id = str(row['id']).strip()
                if video_id.endswith('.webm'):
                    video_id = video_id[:-5]
                video_path = os.path.join(video_root, f"{video_id}.webm")
                if validate_video(video_path):
                    valid_videos.append(idx)
            df_valid = df.iloc[valid_videos].reset_index(drop=True)
            print(f"Found {len(df_valid)} valid videos out of {len(df)} total")
            return df_valid
        else:
            return df
    
    # Shuffle for randomness
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Collect valid videos until we have enough
    valid_videos = []
    print(f"Collecting {num_samples} valid videos...")
    
    for idx, row in tqdm(df_shuffled.iterrows(), total=len(df_shuffled), desc="Finding valid videos"):
        if len(valid_videos) >= num_samples:
            break
        
        video_id = str(row['id']).strip()
        if video_id.endswith('.webm'):
            video_id = video_id[:-5]
        
        video_path = os.path.join(video_root, f"{video_id}.webm")
        
        if validate_video(video_path):
            valid_videos.append(idx)
    
    if len(valid_videos) < num_samples:
        print(f"Warning: Only found {len(valid_videos)} valid videos out of requested {num_samples}")
        print(f"Will use {len(valid_videos)} valid videos for evaluation")
    
    df_valid = df_shuffled.iloc[valid_videos].reset_index(drop=True)
    print(f"Selected {len(df_valid)} valid videos for evaluation")
    
    return df_valid


def process_video(video_path, observation_ratio, num_samplers=3, frame_length=16, frame_size=224, fps=30.0):
    """
    Process video into model input format with progressive sampling.
    Handles short videos by getting actual frame count and using the entire video if needed.
    
    Args:
        video_path: Path to video file
        observation_ratio: Ratio of video to observe (0.1, 0.3, 0.5, 0.7, 0.9)
        num_samplers: Number of progressive samplers
        frame_length: Number of frames per sampler
        frame_size: Frame size (height/width)
        fps: Frames per second (assumed, will be adjusted if video is too short)
        
    Returns:
        input_tensor: Tensor of shape [1, num_samplers, C, T, H, W]
    """
    # Load video
    video = encoded_video.EncodedVideo.from_path(video_path)
    video_duration = float(video.duration)
    
    # Get actual video clip to determine real frame count
    # Use the entire video to count actual frames
    full_clip = video.get_clip(0.0, video_duration)
    if isinstance(full_clip, dict):
        full_video_tensor = full_clip['video']  # Shape: [C, T, H, W]
    else:
        full_video_tensor = full_clip
    
    # Handle None case (video decode failure)
    if full_video_tensor is None:
        # Create dummy tensor with zeros for all samplers
        dummy_tensor = torch.zeros(3, frame_length, frame_size, frame_size)
        sampled_clips = [dummy_tensor] * num_samplers
        input_tensor = torch.stack(sampled_clips, dim=0).unsqueeze(0)
        return input_tensor
    
    # Get actual frame count from video
    actual_frame_count = full_video_tensor.shape[1] if len(full_video_tensor.shape) > 1 else 1
    
    # If video is very short, use minimum frames needed
    min_frames_needed = frame_length * num_samplers
    if actual_frame_count < min_frames_needed:
        # For very short videos, we'll use the entire video and duplicate frames
        # Adjust observation ratio to use full video if it's too short
        actual_fps = max(actual_frame_count / video_duration, 1.0) if video_duration > 0 else fps
    else:
        actual_fps = fps
    
    # Calculate total frames and observed frames
    # Use actual frame count instead of estimated
    total_frames = actual_frame_count
    num_observed_frames = max(1, int(total_frames * observation_ratio))
    
    # Sample frames for each sampler (progressive sampling)
    sampled_clips = []
    for s in range(1, num_samplers + 1):
        # Calculate range for this sampler
        range_max = max(1, int(num_observed_frames * (s / num_samplers)))
        
        # Sequential sampling - create indices evenly spaced
        clip_start = 0
        clip_end = min(range_max - 1, total_frames - 1)
        clip_end = max(0, clip_end)
        
        # Create indices for frames
        if clip_end < frame_length:
            # Very short video - use all available frames and repeat
            if total_frames > 0:
                indices = list(range(0, min(total_frames, frame_length)))
                # Repeat indices to fill frame_length
                while len(indices) < frame_length:
                    if len(indices) > 0:
                        indices.extend(indices[:min(len(indices), frame_length - len(indices))])
                    else:
                        indices = [0] * frame_length
            else:
                indices = [0] * frame_length
        else:
            # Evenly sample frame_length frames from [0, clip_end]
            step = max(1, clip_end // frame_length) if clip_end > 0 else 1
            indices = list(range(0, min(clip_end + 1, total_frames), step))
            
            # Ensure we have exactly frame_length frames
            while len(indices) < frame_length:
                if len(indices) > 0:
                    # Repeat last few indices
                    last_idx = indices[-1]
                    indices.append(min(last_idx + 1, total_frames - 1) if total_frames > 0 else last_idx)
                else:
                    indices = [0] * frame_length
        
        indices = indices[:frame_length]
        
        # Get clip from video - use actual frame indices
        # For short videos, use the entire video
        if total_frames < frame_length:
            time_start = 0.0
            time_end = video_duration
        else:
            # Convert frame indices to time
            time_start = max(0.0, (indices[0] / total_frames) * video_duration) if total_frames > 0 else 0.0
            time_end = min(video_duration, (indices[-1] / total_frames) * video_duration) if total_frames > 0 else video_duration
        
        # Ensure time_end > time_start
        if time_end <= time_start:
            time_end = min(time_start + 0.1, video_duration)
        
        clip = video.get_clip(time_start, time_end)
        
        # Get frames - pytorchvideo returns dict with 'video' key
        if isinstance(clip, dict):
            video_tensor = clip['video']  # Shape: [C, T, H, W]
        else:
            video_tensor = clip
        
        # Handle None case (video decode failure)
        if video_tensor is None:
            # Use first frame from full video if available
            if full_video_tensor is not None and full_video_tensor.shape[1] > 0:
                first_frame = full_video_tensor[:, 0:1, :, :]
                video_tensor = first_frame.repeat(1, frame_length, 1, 1)
            else:
                video_tensor = torch.zeros(3, frame_length, frame_size, frame_size)
        
        # Ensure we have at least one frame
        if video_tensor.shape[1] == 0:
            if full_video_tensor is not None and full_video_tensor.shape[1] > 0:
                first_frame = full_video_tensor[:, 0:1, :, :]
                video_tensor = first_frame.repeat(1, frame_length, 1, 1)
            else:
                video_tensor = torch.zeros(3, frame_length, frame_size, frame_size)
        
        # Extract specific frames if needed
        if video_tensor.shape[1] < frame_length:
            # Pad by repeating frames
            num_repeats = (frame_length // video_tensor.shape[1]) + 1
            repeated_frames = video_tensor.repeat(1, num_repeats, 1, 1)
            video_tensor = repeated_frames[:, :frame_length, :, :]
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
    print(f"Using pooling method: {POOL}")
    if not ADAPOOL_AVAILABLE and POOL == "max":
        print(f"⚠️  WARNING: Models were likely trained with 'ada' pooling!")
        print(f"   Using 'max' pooling instead may cause:")
        print(f"   - Significant accuracy degradation (explaining poor results)")
        print(f"   - Incorrect model behavior")
        print(f"   - Results not comparable to original performance")
    
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


def evaluate_ratio(model_path, observation_ratio, test_df, label_to_id, id_to_label, device):
    """
    Evaluate a single model checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        observation_ratio: Observation ratio (as string, e.g., "0.1")
        test_df: DataFrame with test videos
        label_to_id: Dictionary mapping label name to ID
        id_to_label: Dictionary mapping ID to label name
        device: torch device
        
    Returns:
        accuracy: Accuracy on test set (as percentage 0-100)
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
    label_not_found_count = 0
    
    # Debug: Track prediction distribution
    prediction_stats = {}
    
    # Process each video
    print(f"\nEvaluating {len(test_df)} videos...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Ratio {observation_ratio}"):
        video_id = str(row['id']).strip()
        label_name = str(row['label']).strip()
        
        # Remove .webm extension if present (to handle cases where id already has extension)
        if video_id.endswith('.webm'):
            video_id = video_id[:-5]  # Remove .webm extension
        
        # Build video path - videos are named as {id}.webm
        video_path = os.path.join(VIDEO_ROOT, f"{video_id}.webm")
        
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
            
            # Debug: Track prediction distribution
            if pred_class_id not in prediction_stats:
                prediction_stats[pred_class_id] = 0
            prediction_stats[pred_class_id] += 1
            
            # Get ground truth class ID
            true_class_id = label_to_id.get(label_name)
            
            if true_class_id is None:
                label_not_found_count += 1
                if label_not_found_count <= 5:
                    warnings.warn(f"Unknown label: {label_name} for video {video_id}")
                continue
            
            # Check if correct
            if pred_class_id == true_class_id:
                correct += 1
            
            total += 1
            num_processed += 1
            
        except Exception as e:
            warnings.warn(f"Error processing video {video_id}: {str(e)}")
            continue
    
    # Calculate metrics with validation
    if total == 0:
        print(f"WARNING: No videos were successfully processed!")
        accuracy = 0.0
    else:
        accuracy = (correct / total) * 100.0
        
        # Validate: correct should never exceed total
        if correct > total:
            print(f"ERROR: correct ({correct}) > total ({total})! This should never happen.")
            accuracy = 100.0  # Cap at 100%
        elif accuracy > 100.0:
            print(f"ERROR: accuracy ({accuracy}) > 100%! Capping to 100%.")
            accuracy = 100.0
    
    avg_inference_time = (total_inference_time / num_processed) if num_processed > 0 else 0.0
    
    # Print debug information
    print(f"\nEvaluation statistics for ratio {observation_ratio}:")
    print(f"  Total processed: {num_processed}")
    print(f"  Total with valid labels: {total}")
    print(f"  Correct predictions: {correct}")
    print(f"  Accuracy: {accuracy:.4f}%")
    if label_not_found_count > 5:
        print(f"  Warning: {label_not_found_count} videos had unknown labels (only first 5 were shown)")
    if missing_videos > 5:
        print(f"  Warning: {missing_videos} videos were missing (only first 5 were shown)")
    
    # Debug: Show top predicted classes (if model is predicting same class)
    if prediction_stats:
        sorted_stats = sorted(prediction_stats.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_stats[:5]
        print(f"  Top 5 predicted classes: {top_5}")
        if len(prediction_stats) == 1:
            print(f"  WARNING: Model predicted only one class for all videos! This suggests a model issue.")
    
    return accuracy, avg_inference_time, num_processed


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():
    """Main function to run benchmark."""
    print("=" * 80)
    print("TEMPr Something-Something v2 Benchmark")
    print("=" * 80)
    
    # Show pooling status
    if ADAPOOL_AVAILABLE:
        print(f"\n✓ adaPool is available - using {POOL} pooling (correct for trained models)")
    else:
        print("\n" + "!" * 80)
        print("⚠️  CRITICAL WARNING:")
        print(f"  adaPool is NOT available - using {POOL} pooling instead")
        print(f"  Models were likely trained with 'ada' pooling")
        print(f"  This mismatch may cause:")
        print(f"    - Significant accuracy degradation (explaining poor/inconsistent results)")
        print(f"    - Results not comparable to original model performance")
        print(f"    - Models may not work correctly with different pooling method")
        print(f"\n  To fix: Install adaPool by compiling from source in:")
        print(f"    /data/progressive-action-prediction/adaPool/pytorch/")
        print("!" * 80 + "\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load class mapping
    print(f"\nLoading class mapping from: {LABELS_JSON_PATH}")
    label_to_id, id_to_label = load_class_mapping(LABELS_JSON_PATH)
    print(f"Loaded {len(label_to_id)} classes")
    
    # Load test data (filter to get only valid videos)
    print(f"\nLoading test data from: {TEST_CSV_PATH}")
    test_df = load_test_data(
        TEST_CSV_PATH, 
        num_samples=NUM_TEST_SAMPLES, 
        random_seed=RANDOM_SEED,
        video_root=VIDEO_ROOT
    )
    print(f"Loaded {len(test_df)} valid test videos for evaluation")
    
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
                label_to_id,
                id_to_label,
                device
            )
            
            # Store results (accuracy is in percentage 0-100)
            # Validate accuracy is reasonable before storing
            if accuracy < 0 or accuracy > 100:
                print(f"WARNING: Stored accuracy {accuracy} is outside valid range [0, 100]!")
                print(f"This indicates a bug in accuracy calculation.")
            
            results.append({
                'observation_ratio': float(ratio),  # Ensure it's a float
                'accuracy': float(accuracy),  # Ensure it's a float, stored as percentage (0-100)
                'avg_inference_time_seconds': float(avg_inference_time),
                'num_processed_videos': int(num_processed),
                'checkpoint_path': checkpoint_path
            })
            
            print(f"\nResults for ratio {ratio}:")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Average inference time: {avg_inference_time*1000:.2f} ms per video")
            print(f"  Processed videos: {num_processed}/{len(test_df)}")
            
            # Save results after each ratio (incremental save)
            results_df = pd.DataFrame(results)
            # Ensure accuracy column is float and validate values
            if 'accuracy' in results_df.columns:
                results_df['accuracy'] = results_df['accuracy'].astype(float)
                invalid_mask = (results_df['accuracy'] < 0) | (results_df['accuracy'] > 100)
                if invalid_mask.any():
                    print(f"  WARNING: Found invalid accuracy values in CSV!")
                    print(f"  Invalid rows: {results_df[invalid_mask]}")
            results_df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
            print(f"  Results saved to: {OUTPUT_CSV}")
            
        except Exception as e:
            print(f"\nError evaluating ratio {ratio}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final save results to CSV
    print("\n" + "=" * 80)
    print("Saving final results to CSV...")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    # Ensure accuracy column is float and validate values
    if 'accuracy' in results_df.columns:
        results_df['accuracy'] = results_df['accuracy'].astype(float)
        invalid_mask = (results_df['accuracy'] < 0) | (results_df['accuracy'] > 100)
        if invalid_mask.any():
            print(f"\nWARNING: Found invalid accuracy values in final CSV!")
            print(f"Invalid rows:\n{results_df[invalid_mask]}")
            print(f"\nThis indicates a bug - accuracy should be between 0 and 100!")
    results_df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
    print(f"\nResults saved to: {OUTPUT_CSV}")
    
    # Print validation summary
    if 'accuracy' in results_df.columns:
        print(f"\nAccuracy validation summary:")
        print(f"  Min accuracy: {results_df['accuracy'].min():.6f}%")
        print(f"  Max accuracy: {results_df['accuracy'].max():.6f}%")
        print(f"  All values in valid range [0, 100]: {((results_df['accuracy'] >= 0) & (results_df['accuracy'] <= 100)).all()}")
    
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

