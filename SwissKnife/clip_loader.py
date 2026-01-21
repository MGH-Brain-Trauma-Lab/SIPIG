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
    
    for video_number, video_path in enumerate(video_files):
        clip_filename = os.path.basename(video_path)
        
        # Load video
        print(f" ({video_number + 1}/{len(video_files)}) Loading {clip_filename}...", end=' ')
        video = loadVideo(video_path, greyscale=greyscale, num_frames=None)
        # ADD THIS: Expand dims for greyscale to add channel dimension
        if greyscale and len(video.shape) == 3:  # (frames, height, width)
            video = np.expand_dims(video, axis=-1)  # (frames, height, width, 1)
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
                       framerate=1, greyscale=False):
    """
    Load train, val, and test data from clip extraction output directory.
    
    Args:
        output_dir: Base output directory containing train/, val/, test/ folders
        annotations_filename: Name of the JSON annotations file
        framerate: Frame rate of clips (default 1 fps)
        greyscale: Whether to load as greyscale
    
    Returns:
        dict with keys: 'train', 'val', 'test' (if they exist)
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
    
    # Load val
    val_dir = os.path.join(output_dir, 'val')
    if os.path.exists(val_dir) and os.listdir(val_dir):
        print("\n" + "="*80)
        print("LOADING VALIDATION DATA")
        print("="*80)
        x_val, y_val = load_clips_from_directory(
            val_dir, annotations_path, framerate, greyscale
        )
        data['val'] = (x_val, y_val)
    
    # Load test
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
