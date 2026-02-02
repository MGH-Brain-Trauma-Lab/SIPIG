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
                       framerate=1, greyscale=False, combine_val_test=False):
    """
    Load train, val, and test data from clip extraction output directory.
    
    Args:
        output_dir: Base output directory containing train/, val/, test/ folders
        annotations_filename: Name of the JSON annotations file
        framerate: Frame rate of clips (default 1 fps)
        greyscale: Whether to load as greyscale
        combine_val_test: If True, combine val and test into a single validation set
    
    Returns:
        dict with keys: 'train', 'val', 'test'
        Each value is a tuple of (x, y) arrays
    """
    annotations_path = os.path.join(output_dir, annotations_filename)
    
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    
    data = {}
    
    # Load train
    train_dir = os.path.join(output_dir, 'train')
    if os.path.exists(train_dir):
        print("\n" + "="*80)
        print("LOADING TRAINING DATA")
        print("="*80)
        x_train, y_train = load_clips_from_directory(
            train_dir, annotations_path, framerate, greyscale
        )
        data['train'] = (x_train, y_train)
    
    if combine_val_test:
        # Combine val and test into single validation set
        print("\n" + "="*80)
        print("COMBINING VAL + TEST INTO SINGLE VALIDATION SET")
        print("="*80)
        
        val_dir = os.path.join(output_dir, 'val')
        test_dir = os.path.join(output_dir, 'test')
        
        x_combined = []
        y_combined = []
        
        if os.path.exists(val_dir) and os.listdir(val_dir):
            print("Loading val data...")
            x_val, y_val = load_clips_from_directory(
                val_dir, annotations_path, framerate, greyscale
            )
            x_combined.append(x_val)
            y_combined.append(y_val)
        
        if os.path.exists(test_dir):
            print("Loading test data...")
            x_test, y_test = load_clips_from_directory(
                test_dir, annotations_path, framerate, greyscale
            )
            x_combined.append(x_test)
            y_combined.append(y_test)
        
        if x_combined:
            x_combined = np.vstack(x_combined)
            y_combined = np.hstack(y_combined)
            
            print(f"\nCombined validation set: {len(x_combined)} frames")
            data['val'] = (x_combined, y_combined)
            data['test'] = (x_combined, y_combined)  # Same as val
        else:
            print("WARNING: No val or test data found!")
            data['val'] = (np.array([]), np.array([]))
            data['test'] = (np.array([]), np.array([]))
    
    else:
        # Load val and test separately (original behavior)
        val_dir = os.path.join(output_dir, 'val')
        if os.path.exists(val_dir) and os.listdir(val_dir):
            print("\n" + "="*80)
            print("LOADING VALIDATION DATA")
            print("="*80)
            x_val, y_val = load_clips_from_directory(
                val_dir, annotations_path, framerate, greyscale
            )
            data['val'] = (x_val, y_val)
        
        test_dir = os.path.join(output_dir, 'test')
        if os.path.exists(test_dir):
            print("\n" + "="*80)
            print("LOADING TEST DATA")
            print("="*80)
            x_test, y_test = load_clips_from_directory(
                test_dir, annotations_path, framerate, greyscale
            )
            data['test'] = (x_test, y_test)
    
    return data


def load_training_metadata(clips_output_dir, framerate=1, combine_val_test=False):
    """
    Load video file paths and labels without loading actual frames.
    Supports .mp4, .avi video files.
    
    Args:
        clips_output_dir: Base directory with train/val/test subdirs
        framerate: Frame rate (not used but kept for compatibility)
        combine_val_test: If True, combine val and test into single validation set
    
    Returns:
        dict with keys 'train', 'val', 'test'
        Each value is (list_of_paths, list_of_labels)
    """
    import os
    import re
    from glob import glob
    
    print(f"\n{'='*70}")
    print(f"LOADING METADATA FROM: {clips_output_dir}")
    if combine_val_test:
        print(f"MODE: COMBINING VAL + TEST")
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
                if len(parts) >= 2 and parts[-2] == 'chunk':
                    label = parts[-3]
                elif len(parts) >= 1:
                    behavior_keywords = ['lying-asleep', 'lying-awake', 'upright', 'obstructed', 
                                        'lying_asleep', 'lying_awake', 'lying', 'asleep', 'awake']
                    
                    for part in reversed(parts):
                        if part.lower() in behavior_keywords or any(kw in part.lower() for kw in behavior_keywords):
                            label = part
                            break
                    else:
                        label = parts[-2] if len(parts) > 1 else parts[0]
                else:
                    label = 'unknown'
                
                label = label.replace('-', ' ')
                label = label.strip().lower()
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
    
    # Load train
    train_dir = os.path.join(clips_output_dir, 'train')
    train_paths, train_labels = load_split('train', train_dir)
    
    if combine_val_test:
        # Combine val and test
        print(f"\n{'='*70}")
        print("COMBINING VAL + TEST")
        print(f"{'='*70}")
        
        val_dir = os.path.join(clips_output_dir, 'val')
        test_dir = os.path.join(clips_output_dir, 'test')
        
        val_paths, val_labels = load_split('val', val_dir)
        test_paths, test_labels = load_split('test', test_dir)
        
        # Combine
        combined_paths = val_paths + test_paths
        combined_labels = val_labels + test_labels
        
        print(f"\n{'─'*70}")
        print(f"COMBINED VALIDATION SET")
        print(f"  Val clips: {len(val_paths)}")
        print(f"  Test clips: {len(test_paths)}")
        print(f"  Total: {len(combined_paths)}")
        print(f"{'─'*70}")
        
        from collections import Counter
        if combined_labels:
            combined_dist = Counter(combined_labels)
            print(f"  Combined label distribution:")
            for label, count in sorted(combined_dist.items()):
                print(f"    {label}: {count}")
        
        data = {
            'train': (train_paths, train_labels),
            'val': (combined_paths, combined_labels),
            'test': (combined_paths, combined_labels)  # Same as val
        }
    else:
        # Load separately
        val_dir = os.path.join(clips_output_dir, 'val')
        test_dir = os.path.join(clips_output_dir, 'test')
        
        val_paths, val_labels = load_split('val', val_dir)
        test_paths, test_labels = load_split('test', test_dir)
        
        data = {
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels),
            'test': (test_paths, test_labels)
        }
    
    print(f"\n{'='*70}")
    print(f"METADATA LOADING COMPLETE")
    print(f"{'='*70}\n")
    
    return data