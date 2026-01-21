"""
Training script for pig behavior classification using extracted clips
"""
from SwissKnife.clip_loader import load_training_data
from SwissKnife.dataloader import Dataloader
from SwissKnife.behavior import train_behavior
from SwissKnife.utils import load_config
from SwissKnife.augmentations import mouse_identification
from sklearn.utils import class_weight
from datetime import datetime 
import numpy as np
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import metrics
import tensorflow as tf
import os
import random

# =========== SET SEED ==========
SEED = None  # Change this to whatever you want, or set to None for random
# SEED = None  # Uncomment this line for random seed

if SEED is None:
    SEED = random.randint(0, 999999)
    print(f"\n{'='*80}")
    print(f"USING RANDOM SEED: {SEED}")
    print(f"{'='*80}\n")
else:
    print(f"\n{'='*80}")
    print(f"USING FIXED SEED: {SEED}")
    print(f"{'='*80}\n")

# Set all random seeds
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# For additional reproducibility with GPU operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# ==================================================

# =========== ALLOW GPU GROWTH (allows for more gpu use) ==========
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Also try limiting total memory
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=20480)]  # 20GB limit
            )
    except RuntimeError as e:
        print(e)
# ==================================================

# =============== LOWER PRECISION needed for larger images ===================== 
# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
# ==================================================

# Configuration
#CLIPS_OUTPUT_DIR = '/home/tbiinterns/Desktop/semiology_ml/training_data/temporal_extraction/temporal_split_5min_1fps_petite/'
#CLIPS_OUTPUT_DIR = '/home/tbiinterns/Desktop/semiology_ml/training_data/stride_temporal_split_5min_1fps_topview_200/'
#CLIPS_OUTPUT_DIR = '/home/tbiinterns/Desktop/semiology_ml/training_data/stride_temporal_split_5min_1fps_jan6_max154_lying/'
CLIPS_OUTPUT_DIR = '/home/tbiinterns/Desktop/semiology_ml/training_data/combined_data_01-14-25_200/'
CONFIG_NAME = 'default'
USE_STREAMING = True  # ← TOGGLE THIS to switch modes

# =========== LOAD TRAINING/VAL/TEST DATA ==========
if USE_STREAMING:
    print("Loading clip metadata (streaming mode)...")
    from SwissKnife.clip_loader import load_training_metadata
    
    metadata = load_training_metadata(CLIPS_OUTPUT_DIR, framerate=1)
    x_train, y_train = metadata['train']
    x_val, y_val = metadata['val']
    x_test, y_test = metadata['test']
    
    print(f"\nMetadata loaded:")
    print(f"  Train: {len(x_train)} clips")
    print(f"  Val: {len(x_val)} clips")
    print(f"  Test: {len(x_test)} clips")
else:
    print("Loading all clips into memory (traditional mode)...")
    data = load_training_data(CLIPS_OUTPUT_DIR, framerate=1, greyscale=False)
    
    x_train, y_train = data['train']
    x_val, y_val = data['val']
    x_test, y_test = data['test']
    
    print(f"\nData loaded:")
    print(f"  Train: {x_train.shape}, labels: {len(y_train)}")
    print(f"  Val: {x_val.shape}, labels: {len(y_val)}")
    print(f"  Test: {x_test.shape}, labels: {len(y_test)}")

# Check label distribution (works for both modes)
from collections import Counter
train_dist = Counter(y_train)
print(f"\nTrain label distribution: {train_dist}")

# +++++++++ COMBINE LYING INTO ONE CATEGORY +++++++++
# y_train = ['lying' if 'lying' in label else label for label in y_train]
# y_test = ['lying' if 'lying' in label else label for label in y_test]
# if y_val is not None:
#     y_val = ['lying' if 'lying' in label else label for label in y_val]
# NOTE: Also need to change num_classes to 3
# ++++++++++++++++++++++++++++++++++++++++++++++++++++

# Check label distribution
from collections import Counter
train_dist = Counter(y_train)
print(f"\nTrain label distribution: {train_dist}")
val_dist = Counter(y_val)
print(f"Val label distribution: {val_dist}")
test_dist = Counter(y_test)
print(f"Test label distribution: {test_dist}")
# ========================================================

# ================== CONFIGS ==========================
# Load SIPEC config
config = load_config(f"configs/behavior/{CONFIG_NAME}")

# Major data parameters (will crash with wrong values)
config['use_streaming'] = USE_STREAMING  # Pass to config
config['use_generator'] = True  # Must be True for streaming
IMG_DIM = 200
config['image_x'] = IMG_DIM
config['image_y'] = IMG_DIM
config['num_classes'] = 4

# Recognition Model Parameters
config['train_recognition_model'] = True  # Force boolean
config['recognition_model_lr'] = 3e-5
config['recognition_model_epochs'] = 50
config['recognition_model_batch_size'] = 16
config['backbone'] = 'mobilenet'
# config['backbone'] = 'xception'
config['recognition_model_optimizer'] = 'adam'
config['recognition_model_loss'] = 'focal_loss'

# Pretrained Recognition Model Parameters
# config['pretrained_weights_path'] = '../simclr/training/simclr_checkpoints_minimal_aug/encoder_epoch_90.h5'
config['freeze_pretrained'] = False  # Freeze backbone, only train classification head

# Sequential Model Parameters
config['train_sequential_model'] = False  # Force boolean
config['recognition_model_fix'] = True
config['recognition_model_remove_classification'] = True

config['sequential_model_lr'] = 0.0001
config['sequential_model_epochs'] = 10
config['sequential_model_batch_size'] = 8
config['sequential_backbone'] = 'lstm'
config['sequential_model_optimizer'] = 'adam'
config['look_back'] = 5
config["temporal_causal"] = False # personally created parameter --> used in SwissKnife/behavior.py
config['sequential_model_loss'] = 'focal_loss'

# Data Parameters
config['undersample_data'] = False
config['recognition_model_augmentation'] = 3 # 0-3 levels
config['use_class_weights'] = True  # Force boolean
config['normalize_data'] = True
config['encode_labels'] = True
config['do_flow'] = False

# Scheduler (Not really used but required)
config['recognition_model_use_scheduler'] = False  # Force boolean
config['sequential_model_use_scheduler'] = False
config['recognition_model_scheduler_lr'] = config['recognition_model_lr']#0.0001  # Initial LR for scheduler
config['recognition_model_scheduler_factor'] = 1 #1.1  # Division factor per epoch
config['recognition_model_scheduler_lower_lr'] = 0.0000001  # Minimum LR


# ========================================================

# ================= WANDB INITIAL LOGGING ========================
wandb.init(
    project="SIPIG-initial",
    name=f"train_{datetime.now().strftime('%m%d_%H%M')}",
    config={
        **config,  # Unpack all config items
        # Add computed statistics
        "seed": SEED, 
        "dataset": CLIPS_OUTPUT_DIR.split('/')[-2],
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "test_samples": len(y_test),
        "train_distribution": dict(train_dist),
        "val_distribution": dict(val_dist),
        "test_distribution": dict(test_dist),
    }
)

# Log distribution bar charts
import pandas as pd
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (name, dist) in enumerate([('Train', train_dist), ('Val', val_dist if val_dist else {}), ('Test', test_dist)]):
    if dist:
        axes[idx].bar(dist.keys(), dist.values())
        axes[idx].set_title(f'{name} Distribution')
        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel('Count')
        axes[idx].tick_params(axis='x', rotation=45)
plt.tight_layout()
wandb.log({"data_distributions": wandb.Image(fig)})
plt.close(fig)

print("\nConfig being used:")
for key in ['train_recognition_model', 'use_class_weights', 'recognition_model_use_scheduler']:
    print(f"  {key}: {config[key]} (type: {type(config[key])})")
# =======================================================

# ================= OVERSAMPLING ========================
# from imblearn.over_sampling import RandomOverSampler

# print("\nOversampling minority classes...")
# ros = RandomOverSampler(random_state=42)
# x_train_flat = x_train.reshape(len(x_train), -1)
# x_train_resampled, y_train_resampled = ros.fit_resample(x_train_flat, y_train)
# x_train = x_train_resampled.reshape(-1, *x_train.shape[1:])
# y_train = y_train_resampled

# print(f"After oversampling: {Counter(y_train)}")
# ==========================================================

print("\n=== DEBUG INFO ===")
print(f"Type of x_train: {type(x_train)}")
print(f"Length of x_train: {len(x_train)}")
if len(x_train) > 0:
    print(f"Type of first element: {type(x_train[0])}")
    print(f"First element value: {x_train[0][:100] if isinstance(x_train[0], str) else 'NOT A STRING'}")
print("==================\n")

# ================= DATA LOADER =================
print("\nCreating dataloader...")
dataloader = Dataloader(x_train, y_train, x_val, y_val, config=config)

# Prepare data
print("Preparing data...")
if USE_STREAMING:
    # Streaming mode - only encode labels
    dataloader.prepare_streaming_data(target_size=(config['image_x'], config['image_y']))
else:
    # Traditional mode - load and process everything
    dataloader.prepare_data(
        downscale=(config['image_x'], config['image_y']),
        remove_behaviors=[],
        flatten=False,
        recurrent=False
    )
# =================================================

# ================= CLASS WEIGHTS =================
# CRITICAL: Calculate class weights BEFORE training
class_weights = None
if config['use_class_weights']:
    print("\nCalculating class weights...")
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train_encoded),
        y=y_train_encoded
    )
    
    print(f"Class weights: {class_weights}")
    print(f"Classes: {label_encoder.classes_}")
    
# ++++ SET MANUAL CLASS WEIGHTS ++++
#     MANUAL WEIGHTS - Set these to whatever you want
#     manual_weights = {
#         'lying asleep': 0.0,
#         'lying awake': 1000.0,
#         'obstructed': 0.0,
#         'upright': 0.0,
#     }
    
#     # Convert to array in the order that label_encoder expects
#     class_weights = np.array([manual_weights[cls] for cls in label_encoder.classes_])
    
#     print(f"Manual class weights: {class_weights}")
#     print(f"Classes: {label_encoder.classes_}")
# +++++++++++++++++++++++++++++++++++
    
    print(f"Weight distribution:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {cls}: {class_weights[i]:.4f}")

    # Log class weights to wandb
    wandb.config.update({
        "class_weights": {label_encoder.classes_[i]: float(class_weights[i]) 
                         for i in range(len(class_weights))},
        "classes": list(label_encoder.classes_)
    })
    
    if class_weights is not None:
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"\nClass weights as dict: {class_weights_dict}")
        class_weights = class_weights_dict  # Pass dict instead of array
# ==========================================================
        
# =================== MODEL TRAINING =======================

print("\n" + "="*80)
print("TRAINING MODEL")
print("="*80)

model, results, report = train_behavior(
    dataloader,
    config,
    num_classes=config['num_classes'],
    encode_labels=False,
    class_weights=class_weights
)

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print("\nValidation Results:")
print("Classification Report:")
print(report)
print(f"\nMetrics (acc, f1, corr): {results}")

# =================== WANDB RESULTS LOGGING =======================

# Log validation results to wandb
wandb.log({
    "val_balanced_accuracy": results[0],
    "val_macro_f1": results[1],
    "val_pearson_corr": results[2],
})

# Log classification report as text
wandb.log({"val_classification_report": wandb.Html(f"<pre>{report}</pre>")})

# Log validation results to wandb
wandb.log({
    "val_balanced_accuracy": results[0],
    "val_macro_f1": results[1],
    "val_pearson_corr": results[2],
})

# Get predictions to calculate per-class metrics
val_predictions = model.recognition_model.predict(dataloader.x_test)
val_pred_labels = np.argmax(val_predictions, axis=-1)
val_true_labels = np.argmax(dataloader.y_test, axis=-1)

# Parse classification report for per-class metrics
report_dict = metrics.classification_report(
    val_true_labels,
    val_pred_labels,
    target_names=dataloader.label_encoder.classes_,
    output_dict=True
)

# Log per-class metrics with class names
for class_name in dataloader.label_encoder.classes_:
    wandb.log({
        f"val_{class_name}_precision": report_dict[class_name]['precision'],
        f"val_{class_name}_recall": report_dict[class_name]['recall'],
        f"val_{class_name}_f1": report_dict[class_name]['f1-score'],
    })

# ======================================================================

# ================== MAKE AND LOG CONFUSION MATRIX =====================
print("\n" + "="*80)
print("MAKING VAL CONFUSION MATRIX")
print("="*80)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(val_true_labels, val_pred_labels)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['lying_asleep', 'lying_awake', 'upright', 'obstructed'],
            yticklabels=['lying_asleep', 'lying_awake', 'upright', 'obstructed'],
            ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Validation Set Confusion Matrix')

wandb.log({"val_confusion_matrix": wandb.Image(fig)})
plt.close(fig)

# Save model
now = datetime.now().strftime("%m-%d-%Y_%HH-%MM-%SS")
name = f'pig_behavior_model_{now}_seed{SEED}.h5'
model.recognition_model.save(name)
print(f"\nModel saved to: {name}")
print(f"Seed used: {SEED}")

# Save model as wandb artifact
artifact = wandb.Artifact(
    name=f'pig-behavior-model-{now}',
    type='model',
    description='Pig behavior classification model (xception)',
    metadata={
        'architecture': 'xception',
        'num_classes': 4,
        'input_size': (75, 75, 3),
        'val_accuracy': results[0],
        'val_f1': results[1],
    }
)
artifact.add_file(name)
wandb.log_artifact(artifact)
# ===========================================================

# =========== SAVE MODEL HISTORY + WEIGHTS TO WANDB =========
# Save training history if available
if hasattr(model, 'recognition_model_history') and model.recognition_model_history:
    import pickle
    history_name = f'pig_behavior_history_{now}.pkl'
    with open(history_name, 'wb') as f:
        pickle.dump(model.recognition_model_history.history, f)
    print(f"Training history saved to: {history_name}")
    
    # Log training curves
    history = model.recognition_model_history.history
    
    # Create training plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Categorical Accuracy
    axes[0, 1].plot(history['categorical_accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history['val_categorical_accuracy'], label='Val Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning Rate (if available)
    if 'lr' in history:
        axes[1, 0].plot(history['lr'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Hide empty subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    wandb.log({"training_curves": wandb.Image(fig)})
    plt.close(fig)

# Finish wandb run
wandb.finish()
print("\nWandB run completed!")
