"""
CLIP LOADER FOR VGG VIA VIDEO FORMAT
Helper functions to load training clips into SIPEC's dataloader
"""
import json
import os
import numpy as np
from glob import glob
from SwissKnife.utils import loadVideo


def parse_via_video_json_for_clip(json_data, clip_filename, framerate):
    """
    Parse VGG Via VIDEO JSON and extract frame-level labels for a specific clip.
    
    Args:
        json_data: Loaded VGG Via JSON dict
        clip_filename: Name of the clip file (e.g., "104-137_clip_0001.mp4")
        framerate: Frame rate of the clip video
    
    Returns:
        list of labels, one per frame
    """
    # Find the file ID for this clip
    clip_fid = None
    for fid, file_data in json_data['file'].items():
        if file_data['fname'] == clip_filename:
            clip_fid = fid
            clip_duration = file_data.get('duration_seconds', None)
            break
    
    if clip_fid is None:
        raise ValueError(f"Clip {clip_filename} not found in JSON")
    
    # Calculate number of frames
    if clip_duration:
        num_frames = int(clip_duration * framerate)
    else:
        # Fallback: will be set based on actual video length
        num_frames = None
    
    # Initialize labels array
    if num_frames:
        labels = ['none'] * num_frames
    else:
        labels = []
    
    # Find all metadata entries for this clip
    for meta_id, meta in json_data['metadata'].items():
        if str(meta['vid']) == str(clip_fid):
            start_sec, end_sec = meta['z']
            behavior = meta['av']['1']
            
            start_frame = int(start_sec * framerate)
            end_frame = int(end_sec * framerate)
            
            # Extend labels array if needed (when duration wasn't in JSON)
            if not num_frames and end_frame > len(labels):
                labels.extend(['none'] * (end_frame - len(labels)))
            
            # Fill in behavior labels for the current metadata range
            if num_frames:
                end_frame = min(end_frame, num_frames)
            
            labels[start_frame:end_frame] = [behavior] * (end_frame - start_frame)
    
    # No backfilling - unlabelled frames will remain as 'none'
    # They will be filtered out during clip loading
    
    return labels


def load_clips_from_directory(clips_dir, annotations_json, framerate=1, greyscale=False):
    """
    Load all video clips from a directory and their corresponding labels.
    Drops frames with 'none' labels (unlabelled frames).
    
    Args:
        clips_dir: Path to directory containing clip videos (e.g., "output/train/")
        annotations_json: Path to VGG Via JSON file
        framerate: Frame rate of the clip videos (default 1 fps)
        greyscale: Whether to load videos as greyscale
    
    Returns:
        tuple: (videos, labels)
            - videos: numpy array of shape (total_frames, height, width, channels)
            - labels: numpy array of shape (total_frames,) with behavior labels
    """
    # Load JSON
    with open(annotations_json, 'r') as f:
        json_data = json.load(f)
    
    # Find all video files in directory
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(glob(os.path.join(clips_dir, ext)))
    
    video_files.sort()  # Ensure consistent ordering
    
    if not video_files:
        raise ValueError(f"No video files found in {clips_dir}")
    
    print(f"Loading {len(video_files)} clips from {clips_dir}")
    
    all_videos = []
    all_labels = []
    
    for i, video_path in enumerate(video_files):
        clip_filename = os.path.basename(video_path)
        
        # Load video
        print(f"  ({i+1}/{len(video_files)}) Loading {clip_filename}...", end=' ')
        video = loadVideo(video_path, greyscale=greyscale, num_frames=None)
        
        min_frames = 30 * framerate   # 30s
        if len(video) < min_frames:
            print(f"SKIPPED (too short: {len(video)} frames, required: {min_frames})")
            continue
        
        # Get labels for this clip
        try:
            labels = parse_via_video_json_for_clip(json_data, clip_filename, framerate)
            
            # Ensure labels match video length
            if len(labels) != len(video):
                print(f"WARNING: Label length ({len(labels)}) != video length ({len(video)}), truncating/padding")
                if len(labels) < len(video):
                    labels.extend(['none'] * (len(video) - len(labels)))
                else:
                    labels = labels[:len(video)]
            
            # Filter out unlabelled frames
            labels_array = np.array(labels)
            valid_mask = labels_array != 'none'
            
            if not np.any(valid_mask):
                print(f"SKIPPED (no labelled frames)")
                continue
            
            filtered_video = video[valid_mask]
            filtered_labels = labels_array[valid_mask].tolist()
            
            dropped_frames = len(video) - len(filtered_video)
            if dropped_frames > 0:
                print(f"{len(filtered_video)} frames (dropped {dropped_frames} unlabelled)")
            else:
                print(f"{len(filtered_video)} frames")
            
            all_videos.append(filtered_video)
            all_labels.append(filtered_labels)
            
        except ValueError as e:
            print(f"ERROR: {e}")
            continue
    
    # Concatenate all clips
    print(f"\nConcatenating {len(all_videos)} clips...")
    x = np.vstack(all_videos)
    y = np.hstack(all_labels)
    
    print(f"Total frames: {len(x)}")
    print(f"Total labels: {len(y)}")
    
    return x, y


def load_training_data(output_dir, annotations_filename='annotations.json', 
                       framerate=1, greyscale=False,
                       train_splits=None, val_splits=None, test_splits=None):
    """
    Load train, val, and test data from clip extraction output directory.
    
    Args:
        output_dir: Base output directory containing train/, val/, test/ folders
        annotations_filename: Name of the JSON annotations file
        framerate: Frame rate of clips (default 1 fps)
        greyscale: Whether to load as greyscale
        train_splits: List of splits to combine for training (e.g., ['train', 'val'])
        val_splits: List of splits to combine for validation (e.g., ['test'])
        test_splits: List of splits to combine for testing (e.g., None or [])
        
        If all are None, defaults to standard split: train/val/test separately
    
    Returns:
        dict with keys: 'train', 'val', 'test'
        Each value is a tuple of (x, y) arrays
    """
    # Default to standard split if not specified
    if train_splits is None and val_splits is None and test_splits is None:
        train_splits = ['train']
        val_splits = ['val']
        test_splits = ['test']
    
    # Convert None to empty list
    if train_splits is None:
        train_splits = []
    if val_splits is None:
        val_splits = []
    if test_splits is None:
        test_splits = []
        
    # Convert strings to lists if needed (config parser might return strings)
    if isinstance(train_splits, str):
        train_splits = [s.strip() for s in train_splits.split(',') if s.strip()]
    if isinstance(val_splits, str):
        val_splits = [s.strip() for s in val_splits.split(',') if s.strip()]
    if isinstance(test_splits, str):
        test_splits = [s.strip() for s in test_splits.split(',') if s.strip()]
    
    annotations_path = os.path.join(output_dir, annotations_filename)
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    
    # Load all original splits
    all_splits = {}
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split_name)
        if os.path.exists(split_dir) and os.listdir(split_dir):
            print("\n" + "="*80)
            print(f"LOADING {split_name.upper()} DATA")
            print("="*80)
            x, y = load_clips_from_directory(
                split_dir, annotations_path, framerate, greyscale
            )
            all_splits[split_name] = (x, y)
        else:
            all_splits[split_name] = (np.array([]), np.array([]))
    
    # Combine splits according to specification
    def combine_splits(split_list):
        """Combine multiple original splits into one"""
        if not split_list or split_list == []:
            return np.array([]), np.array([])
        
        x_combined = []
        y_combined = []
        
        for split_name in split_list:
            if split_name in all_splits:
                x, y = all_splits[split_name]
                if len(x) > 0:
                    x_combined.append(x)
                    y_combined.append(y)
        
        if x_combined:
            return np.vstack(x_combined), np.hstack(y_combined)
        else:
            return np.array([]), np.array([])
    
    # Create final splits
    print("\n" + "="*80)
    print("COMBINING SPLITS")
    print("="*80)
    
    x_train, y_train = combine_splits(train_splits)
    x_val, y_val = combine_splits(val_splits)
    x_test, y_test = combine_splits(test_splits)
    
    print(f"Training set: {train_splits} → {len(x_train)} frames")
    print(f"Validation set: {val_splits} → {len(x_val)} frames")
    print(f"Test set: {test_splits} → {len(x_test)} frames")
    
    data = {
        'train': (x_train, y_train),
        'val': (x_val, y_val),
        'test': (x_test, y_test)
    }
    
    return data


def load_training_metadata(clips_output_dir, framerate=1, 
                          train_splits=None, val_splits=None, test_splits=None):
    """
    Load video file paths and labels without loading actual frames.
    Supports .mp4, .avi video files.
    
    Args:
        clips_output_dir: Base directory with train/val/test subdirs
        framerate: Frame rate (not used but kept for compatibility)
        train_splits: List of splits to combine for training (e.g., ['train', 'val'])
        val_splits: List of splits to combine for validation (e.g., ['test'])
        test_splits: List of splits to combine for testing (e.g., None or [])
        
        If all are None, defaults to standard split: train/val/test separately
    
    Returns:
        dict with keys 'train', 'val', 'test'
        Each value is (list_of_paths, list_of_labels)
    """
    import os
    import re
    from glob import glob
    
    # Default to standard split if not specified
    if train_splits is None and val_splits is None and test_splits is None:
        train_splits = ['train']
        val_splits = ['val']
        test_splits = ['test']
    
    # Convert None to empty list
    if train_splits is None:
        train_splits = []
    if val_splits is None:
        val_splits = []
    if test_splits is None:
        test_splits = []
        
    # Convert strings to lists if needed (config parser might return strings)
    if isinstance(train_splits, str):
        train_splits = [s.strip() for s in train_splits.split(',') if s.strip()]
    if isinstance(val_splits, str):
        val_splits = [s.strip() for s in val_splits.split(',') if s.strip()]
    if isinstance(test_splits, str):
        test_splits = [s.strip() for s in test_splits.split(',') if s.strip()]
    
    print(f"\n{'='*70}")
    print(f"LOADING METADATA FROM: {clips_output_dir}")
    print(f"Train splits: {train_splits}")
    print(f"Val splits: {val_splits}")
    print(f"Test splits: {test_splits}")
    print(f"{'='*70}")
    
    # Check base directory
    if not os.path.exists(clips_output_dir):
        print(f"❌ ERROR: Base directory does not exist!")
        print(f"   Path: {clips_output_dir}")
        return {'train': ([], []), 'val': ([], []), 'test': ([], [])}
    
    print(f"✓ Base directory exists")
    
    # Helper function to load a split
    def load_split(split_name, split_dir):
        print(f"\n{'─'*70}")
        print(f"Processing: {split_name}")
        print(f"{'─'*70}")
        print(f"Looking in: {split_dir}")
        
        if not os.path.exists(split_dir):
            print(f"  ❌ Directory does not exist!")
            return [], []
        
        print(f"  ✓ Directory exists")
        
        # Find video files
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']:
            pattern = os.path.join(split_dir, ext)
            found = sorted(glob(pattern))
            if found:
                video_files.extend(found)
        
        video_files = sorted(list(set(video_files)))
        
        if not video_files:
            print(f"  ❌ No video files found!")
            return [], []
        
        print(f"  ✓ Found {len(video_files)} video files")
        print(f"  Sample filenames:")
        for i, path in enumerate(video_files[:3]):
            print(f"    {i+1}. {os.path.basename(path)}")
        
        # Extract labels from filenames
        labels = []
        for path in video_files:
            filename = os.path.basename(path)
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split('_')

            try:
                # Look for behavior keywords
                behavior_keywords = ['lying-asleep', 'lying-awake', 'upright', 'obstructed']
                label = 'unknown'

                # Pattern 1: sideview format - behavior_chunk_##
                # Pattern 2: topview format - behavior_topview_chunk_##

                # Find 'chunk' position
                chunk_idx = None
                for i, part in enumerate(parts):
                    if part == 'chunk':
                        chunk_idx = i
                        break

                if chunk_idx is not None:
                    # Check if topview/sideview is present
                    if chunk_idx >= 2 and parts[chunk_idx - 1] in ['topview', 'sideview']:
                        # Format: ...behavior_topview_chunk_##
                        candidate = parts[chunk_idx - 2]
                    elif chunk_idx >= 1:
                        # Format: ...behavior_chunk_##
                        candidate = parts[chunk_idx - 1]
                    else:
                        candidate = None

                    if candidate:
                        candidate_lower = candidate.lower()
                        # Direct match or contains behavior keyword
                        if candidate_lower in behavior_keywords:
                            label = candidate_lower.replace('-', ' ')
                        elif any(kw.replace('-', '') in candidate_lower for kw in behavior_keywords):
                            label = candidate_lower.replace('-', ' ')

                # Fallback: search all parts for behavior keywords
                if label == 'unknown':
                    for part in parts:
                        part_lower = part.lower()
                        if part_lower in behavior_keywords:
                            label = part_lower.replace('-', ' ')
                            break

                labels.append(label)
            except Exception as e:
                print(f"  ⚠ Error extracting label from {filename}: {e}")
                labels.append('unknown')
        
        # Show label statistics
        from collections import Counter
        unique_labels = set(labels)
        print(f"  ✓ Unique labels found: {unique_labels}")
        
        label_counts = Counter(labels)
        print(f"  Label distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"    {label}: {count}")
        
        return video_files, labels
    
    # Load all original splits
    all_splits = {}
    for split_name in ['train', 'val', 'test']:
        split_dir = os.path.join(clips_output_dir, split_name)
        if os.path.exists(split_dir):
            paths, labels = load_split(split_name, split_dir)
            all_splits[split_name] = (paths, labels)
        else:
            all_splits[split_name] = ([], [])
    
    # Combine splits according to specification
    def combine_splits(split_list):
        """Combine multiple original splits into one"""
        if not split_list or split_list == []:
            return [], []
        
        combined_paths = []
        combined_labels = []
        
        for split_name in split_list:
            if split_name in all_splits:
                paths, labels = all_splits[split_name]
                combined_paths.extend(paths)
                combined_labels.extend(labels)
        
        return combined_paths, combined_labels
    
    # Create final splits
    train_paths, train_labels = combine_splits(train_splits)
    print(f"\nDEBUG AFTER COMBINE:")
    print(f"  train_splits input: {train_splits}")
    print(f"  train_paths result: {len(train_paths)}")
    print(f"  all_splits keys: {all_splits.keys()}")
    print(f"  all_splits['train'] length: {len(all_splits.get('train', [[]])[0])}")
    val_paths, val_labels = combine_splits(val_splits)
    test_paths, test_labels = combine_splits(test_splits)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"FINAL SPLIT CONFIGURATION")
    print(f"{'='*70}")
    print(f"Training set: {train_splits} → {len(train_paths)} clips")
    print(f"Validation set: {val_splits} → {len(val_paths)} clips")
    print(f"Test set: {test_splits} → {len(test_paths)} clips")
    
    # Print label distributions for final splits
    from collections import Counter
    if train_labels:
        train_dist = Counter(train_labels)
        print(f"\nTraining label distribution:")
        for label, count in sorted(train_dist.items()):
            print(f"  {label}: {count}")
    
    if val_labels:
        val_dist = Counter(val_labels)
        print(f"\nValidation label distribution:")
        for label, count in sorted(val_dist.items()):
            print(f"  {label}: {count}")
    
    if test_labels:
        test_dist = Counter(test_labels)
        print(f"\nTest label distribution:")
        for label, count in sorted(test_dist.items()):
            print(f"  {label}: {count}")
    
    print(f"{'='*70}\n")
    
    data = {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }
    
    print(f"\n{'='*70}")
    print(f"METADATA LOADING COMPLETE")
    print(f"{'='*70}\n")
    
    return data