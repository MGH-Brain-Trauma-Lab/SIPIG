"""
Inference script for pig behavior classification on extracted frames.
Streams frames through model, saves logits (not classifications) to metadata CSV.

Usage:
    python inference_pig_behavior_frames.py --checkpoint path/to/model.h5
    python inference_pig_behavior_frames.py --checkpoint checkpoints/model.h5 --batch-size 64
    python inference_pig_behavior_frames.py --checkpoint model.h5 --workers 4 --debug
"""
import os
import sys
import csv
import logging
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================================
# Configuration
# ============================================================================

# Paths
EXTRACTION_ROOT = "/media/tbiinterns/P01-X/pig_behavior_extraction"
METADATA_CSV = f"{EXTRACTION_ROOT}/metadata/extraction_master.csv"

# Model input size (must match training config)
# **USER: VERIFY THIS MATCHES YOUR MODEL'S TRAINING CONFIG**
MODEL_INPUT_SIZE = (75, 75)  # (height, width) - CHANGE IF YOUR MODEL DIFFERS

# Preprocessing settings (must match training)
NORMALIZE_0_1 = True  # Divide by 255 to get [0, 1]
NORMALIZE_IMAGENET = False  # Use ImageNet mean/std normalization
GREYSCALE = False  # Convert to grayscale

# Inference batch size
BATCH_SIZE = 32

# Parallel frame loading
NUM_WORKERS = 4  # Threads for loading images


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    log_dir = Path(EXTRACTION_ROOT) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized (debug={'ON' if debug else 'OFF'})")
    logger.info(f"Log file: {log_file}")
    
    return logger


# ============================================================================
# Frame Loading
# ============================================================================

def load_and_preprocess_frame(
    frame_path: str,
    target_size: Tuple[int, int],
    normalize_0_1: bool = NORMALIZE_0_1,
    normalize_imagenet: bool = NORMALIZE_IMAGENET,
    greyscale: bool = GREYSCALE
) -> np.ndarray:
    """
    Load and preprocess a single frame.
    
    Args:
        frame_path: Path to frame image
        target_size: (height, width) to resize to
        normalize_0_1: Normalize to [0, 1] by dividing by 255
        normalize_imagenet: Apply ImageNet normalization (mean/std)
        greyscale: Convert to greyscale
        
    Returns:
        Preprocessed frame as float32 array
    """
    # Read image
    img = cv2.imread(frame_path)
    
    if img is None:
        raise ValueError(f"Failed to load image: {frame_path}")
    
    # Convert to greyscale if needed
    if greyscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
    else:
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize (using same method as training - squashed resize, no aspect ratio preservation)
    img = cv2.resize(img, (target_size[1], target_size[0]))
    
    # Convert to float32
    img = img.astype(np.float32)
    
    # Normalization
    if normalize_0_1:
        # Standard [0, 1] normalization
        img = img / 255.0
    
    if normalize_imagenet:
        # ImageNet normalization: (x - mean) / std
        # Mean and std for RGB channels
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        if not normalize_0_1:
            img = img / 255.0  # Must be [0, 1] first
        
        img = (img - mean) / std
    
    return img


def load_frames_batch(
    frame_paths: List[str],
    target_size: Tuple[int, int],
    num_workers: int = 4,
    normalize_0_1: bool = NORMALIZE_0_1,
    normalize_imagenet: bool = NORMALIZE_IMAGENET,
    greyscale: bool = GREYSCALE
) -> np.ndarray:
    """
    Load and preprocess a batch of frames in parallel.
    
    Args:
        frame_paths: List of frame paths
        target_size: (height, width) to resize to
        num_workers: Number of parallel threads
        normalize_0_1: Normalize to [0, 1]
        normalize_imagenet: Apply ImageNet normalization
        greyscale: Convert to greyscale
        
    Returns:
        Batch of frames as numpy array [batch, height, width, channels]
    """
    frames = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                load_and_preprocess_frame,
                path,
                target_size,
                normalize_0_1,
                normalize_imagenet,
                greyscale
            ): i
            for i, path in enumerate(frame_paths)
        }
        
        # Collect results in order
        results = [None] * len(frame_paths)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                # Return None for failed frames (will be filtered later)
                results[idx] = None
                logging.warning(f"Failed to load frame {frame_paths[idx]}: {e}")
    
    # Filter out None values and stack
    valid_frames = [f for f in results if f is not None]
    
    if not valid_frames:
        raise ValueError("No valid frames loaded in batch")
    
    return np.array(valid_frames, dtype=np.float32)


# ============================================================================
# Metadata Handler
# ============================================================================

class MetadataHandler:
    """Handle reading and updating metadata CSV with logits."""
    
    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.logger = logging.getLogger(__name__)
        
        # Verify file exists
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    
    def load_metadata(self) -> List[Dict]:
        """
        Load metadata CSV.
        
        Returns:
            List of metadata dicts
        """
        metadata = []
        
        with open(self.metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            
            # Verify expected columns exist
            expected_cols = {'frame_path', 'pig_id', 'pen', 'date', 'week_number'}
            if not expected_cols.issubset(reader.fieldnames):
                missing = expected_cols - set(reader.fieldnames)
                raise ValueError(f"Missing required columns in CSV: {missing}")
            
            for row in reader:
                metadata.append(row)
        
        self.logger.info(f"Loaded {len(metadata)} frame records from metadata")
        
        return metadata
    
    def save_metadata_with_logits(
        self,
        metadata: List[Dict],
        output_path: str = None
    ):
        """
        Save metadata with logits to new CSV.
        
        Args:
            metadata: List of metadata dicts (must include logit_0, logit_1, logit_2)
            output_path: Where to save (default: overwrite original)
        """
        if output_path is None:
            output_path = self.metadata_path
        
        # Ensure logit columns exist
        if metadata and 'logit_0' not in metadata[0]:
            raise ValueError("Metadata missing logit columns")
        
        # Get all fieldnames (preserve order, add logits at end)
        fieldnames = list(metadata[0].keys())
        
        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata)
        
        self.logger.info(f"Saved metadata with logits: {output_path}")


# ============================================================================
# Model Inference
# ============================================================================

class BehaviorInference:
    """Handle model loading and inference."""
    
    def __init__(
        self,
        checkpoint_path: str,
        input_size: Tuple[int, int] = MODEL_INPUT_SIZE,
        batch_size: int = BATCH_SIZE
    ):
        self.checkpoint_path = checkpoint_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = self._load_model()
        
        # Get class names from model if available
        self.class_names = self._get_class_names()
    
    def _load_model(self) -> tf.keras.Model:
        """Load trained model from checkpoint."""
        self.logger.info(f"Loading model from: {self.checkpoint_path}")
        
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Load model
        model = tf.keras.models.load_model(self.checkpoint_path)
        
        self.logger.info(f"Model loaded successfully")
        self.logger.info(f"  Input shape: {model.input_shape}")
        self.logger.info(f"  Output shape: {model.output_shape}")
        
        # Verify output is logits (should be 3 classes for pig behavior)
        num_classes = model.output_shape[-1]
        if num_classes != 3:
            self.logger.warning(
                f"Expected 3 output classes, got {num_classes}. "
                "This may not be a pig behavior model."
            )
        
        return model
    
    def _get_class_names(self) -> List[str]:
        """Try to extract class names from model metadata."""
        # Common pig behavior classes (adjust if your model differs)
        default_classes = ['lying', 'sitting', 'standing']
        
        # Try to get from model config (may not be saved)
        try:
            if hasattr(self.model, 'class_names'):
                return self.model.class_names
        except:
            pass
        
        return default_classes
    
    def predict_logits(self, frames: np.ndarray) -> np.ndarray:
        """
        Run inference and return raw logits.
        
        Args:
            frames: Batch of preprocessed frames [batch, H, W, 3]
            
        Returns:
            Logits array [batch, num_classes]
        """
        # Run prediction
        logits = self.model.predict(frames, batch_size=self.batch_size, verbose=0)
        
        return logits
    
    def process_frame_batch(
        self,
        frame_paths: List[str],
        num_workers: int = NUM_WORKERS
    ) -> np.ndarray:
        """
        Load frames and run inference.
        
        Args:
            frame_paths: List of frame paths
            num_workers: Parallel loading threads
            
        Returns:
            Logits for each frame [batch, num_classes]
        """
        # Load and preprocess frames
        frames = load_frames_batch(
            frame_paths,
            self.input_size,
            num_workers,
            normalize_0_1=NORMALIZE_0_1,
            normalize_imagenet=NORMALIZE_IMAGENET,
            greyscale=GREYSCALE
        )
        
        # Run inference
        logits = self.predict_logits(frames)
        
        return logits


# ============================================================================
# Main Inference Pipeline
# ============================================================================

def run_inference(
    checkpoint_path: str,
    metadata_path: str = METADATA_CSV,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    debug: bool = False
) -> str:
    """
    Run inference on all extracted frames and save logits.
    
    Args:
        checkpoint_path: Path to model checkpoint (.h5)
        metadata_path: Path to extraction metadata CSV
        batch_size: Inference batch size
        num_workers: Parallel frame loading threads
        debug: Enable debug logging
        
    Returns:
        Path to output CSV with logits
    """
    logger = setup_logging(debug=debug)
    
    logger.info("="*80)
    logger.info("PIG BEHAVIOR INFERENCE")
    logger.info("="*80)
    logger.info(f"Model: {checkpoint_path}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Workers: {num_workers}")
    logger.info("="*80)
    
    # Initialize components
    logger.info("\nInitializing...")
    metadata_handler = MetadataHandler(metadata_path)
    inference_engine = BehaviorInference(
        checkpoint_path=checkpoint_path,
        batch_size=batch_size
    )
    
    # Load metadata
    logger.info("\nLoading metadata...")
    metadata = metadata_handler.load_metadata()
    total_frames = len(metadata)
    
    logger.info(f"  Total frames to process: {total_frames}")
    logger.info(f"  Estimated batches: {(total_frames + batch_size - 1) // batch_size}")
    
    # Process frames in batches
    logger.info("\nRunning inference...")
    
    processed = 0
    failed = 0
    
    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_metadata = metadata[batch_start:batch_end]
            
            # Get frame paths
            frame_paths = [row['frame_path'] for row in batch_metadata]
            
            try:
                # Run inference
                logits = inference_engine.process_frame_batch(
                    frame_paths,
                    num_workers=num_workers
                )
                
                # Add logits to metadata
                for i, row in enumerate(batch_metadata):
                    if i < len(logits):  # In case some frames failed to load
                        row['logit_0'] = f"{logits[i, 0]:.6f}"
                        row['logit_1'] = f"{logits[i, 1]:.6f}"
                        row['logit_2'] = f"{logits[i, 2]:.6f}"
                        processed += 1
                    else:
                        # Frame failed to load
                        row['logit_0'] = "NaN"
                        row['logit_1'] = "NaN"
                        row['logit_2'] = "NaN"
                        failed += 1
                
            except Exception as e:
                logger.error(f"Batch {batch_start}-{batch_end} failed: {e}")
                
                # Mark all as failed
                for row in batch_metadata:
                    row['logit_0'] = "NaN"
                    row['logit_1'] = "NaN"
                    row['logit_2'] = "NaN"
                    failed += len(batch_metadata)
            
            pbar.update(len(batch_metadata))
    
    # Save results
    logger.info("\nSaving results...")
    output_path = metadata_path.replace('.csv', '_with_logits.csv')
    metadata_handler.save_metadata_with_logits(metadata, output_path)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("INFERENCE COMPLETE")
    logger.info("="*80)
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"  Successful: {processed}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"Output: {output_path}")
    logger.info("="*80)
    
    return output_path


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run inference on extracted pig behavior frames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_pig_behavior_frames.py --checkpoint model.h5
  python inference_pig_behavior_frames.py --checkpoint checkpoints/model.h5 --batch-size 64
  python inference_pig_behavior_frames.py --checkpoint model.h5 --workers 8 --debug
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.h5)'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        default=METADATA_CSV,
        help=f'Path to extraction metadata CSV (default: {METADATA_CSV})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Inference batch size (default: {BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=NUM_WORKERS,
        help=f'Number of parallel frame loading threads (default: {NUM_WORKERS})'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Run inference
    try:
        output_path = run_inference(
            checkpoint_path=args.checkpoint,
            metadata_path=args.metadata,
            batch_size=args.batch_size,
            num_workers=args.workers,
            debug=args.debug
        )
        
        print(f"\n✓ Success! Logits saved to: {output_path}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()