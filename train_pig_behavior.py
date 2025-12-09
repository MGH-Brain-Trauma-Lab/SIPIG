"""
Training script for pig behavior classification using extracted clips
"""
from SwissKnife.clip_loader import load_training_data
from SwissKnife.dataloader import Dataloader
from SwissKnife.behavior import train_behavior
from SwissKnife.utils import load_config
from datetime import datetime

# Configuration
CLIPS_OUTPUT_DIR = '/home/tbiinterns/Desktop/semiology_ml/distilled_clips_all/'  # Where you extracted clips
CONFIG_NAME = 'default'  # Use SIPEC's default behavior config

# Load data
print("Loading clips...")
data = load_training_data(
    CLIPS_OUTPUT_DIR,
    framerate=1,  # Your downsampled fps
    greyscale=False  # Set to True if you want grayscale
)

x_train, y_train = data['train']
x_test, y_test = data['test']

# If you have validation data
if 'val' in data:
    x_val, y_val = data['val']
    # You could combine val into test, or use it separately
    # For now, let's use test as-is

print(f"\nData loaded:")
print(f"  Train: {x_train.shape}, labels: {len(y_train)}")
print(f"  Test: {x_test.shape}, labels: {len(y_test)}")

# Check label distribution
from collections import Counter
print(f"\nTrain label distribution: {Counter(y_train)}")
print(f"Test label distribution: {Counter(y_test)}")

# Load SIPEC config
config = load_config(f"configs/behavior/{CONFIG_NAME}")

# Image dimensions - IMPORTANT! Match your clip size
config['image_x'] = 200  # Use 75 to avoid memory issues
config['image_y'] = 200

# Core parameters
config['num_classes'] = 4  # lying asleep, lying awake, upright, obstructed
config['normalize_data'] = True
config['encode_labels'] = True
config['use_class_weights'] = True  # Recommended for imbalanced data
config['look_back'] = 10  # Required for recurrent models
config['use_generator'] = True  # Enable generator to save memory
config['do_flow'] = False  # Disable optical flow
config['undersample_data'] = True  # Keep all data

# Model type selection
config['train_sequential_model'] = False  # Use regular CNN, not RNN

# Recognition model parameters (for non-sequential training)
config['recognition_model_name'] = 'xception'  # or 'mobilenet', 'resnet50', etc.
config['recognition_model_optimizer'] = 'adam'
config['recognition_model_lr'] = 0.0001  # Learning rate
config['recognition_model_scheduler_lr'] = True  # Use learning rate scheduler
config['recognition_model_scheduler_lr_patience'] = 5  # LR scheduler patience
config['recognition_model_scheduler_factor'] = 1.1  # LR reduction factor (divides by factor)
config['recognition_model_early_stopping'] = True  # Use early stopping
config['recognition_model_early_stopping_patience'] = 10  # Early stopping patience
config['recognition_model_epochs'] = 10  # Max epochs
config['recognition_model_batch_size'] = 16  # Smaller batch size for memory
config['recognition_model_fix'] = False  # Don't fix layers
config['recognition_model_remove_classification'] = False  # Keep classification layers

# Sequential model parameters (needed even if not used)
config['sequential_backbone'] = 'lstm'
config['sequential_model_optimizer'] = 'adam'
config['sequential_model_lr'] = 0.0001
config['sequential_model_use_scheduler'] = False
config['sequential_model_scheduler_lr'] = False
config['sequential_model_scheduler_factor'] = 1

# Create SIPEC dataloader
print("\nCreating dataloader...")
dataloader = Dataloader(x_train, y_train, x_test, y_test, config=config)

# Prepare data
print("Preparing data...")
dataloader.prepare_data(
    downscale=(75, 75),
    remove_behaviors=['none'],  # Keep all behaviors
    flatten=False,
    recurrent=config.get('train_sequential_model', False)
)

# Train model
print("\n" + "="*80)
print("TRAINING MODEL")
print("="*80)

model, results, report = train_behavior(
    dataloader,
    config,
    num_classes=config['num_classes']
)

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print("\nClassification Report:")
print(report)
print(f"\nResults: {results}")

# Save model
now = datetime.now().strftime("%m-%d-%Y_%HH-%MM-%SS")
name = f'pig_behavior_model_{now}.h5'
model.recognition_model.save(name)
print(f"\nModel saved to: {name}")