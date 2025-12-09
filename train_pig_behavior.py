"""
Training script for pig behavior classification using extracted clips
"""
from SwissKnife.clip_loader import load_training_data
from SwissKnife.dataloader import Dataloader
from SwissKnife.behavior import train_behavior
from SwissKnife.utils import load_config
from sklearn.utils import class_weight
from datetime import datetime
import numpy as np

# Configuration
CLIPS_OUTPUT_DIR = '/home/tbiinterns/Desktop/semiology_ml/distilled_clips/'
CONFIG_NAME = 'default'

# Load data
print("Loading clips...")
data = load_training_data(
    CLIPS_OUTPUT_DIR,
    framerate=1,
    greyscale=False
)

x_train, y_train = data['train']

# Handle validation data
if 'val' in data:
    x_val, y_val = data['val']
    print(f"Using separate validation set")
else:
    x_val, y_val = None, None
    print(f"No validation set found, will use test as validation")

# Handle test data
if 'test' in data:
    x_test, y_test = data['test']
else:
    # If no test set, use val as test (shouldn't happen with your setup)
    x_test, y_test = x_val, y_val
    x_val, y_val = None, None

print(f"\nData loaded:")
print(f"  Train: {x_train.shape}, labels: {len(y_train)}")
if x_val is not None:
    print(f"  Val: {x_val.shape}, labels: {len(y_val)}")
print(f"  Test: {x_test.shape}, labels: {len(y_test)}")

# Check label distribution
from collections import Counter
print(f"\nTrain label distribution: {Counter(y_train)}")
if y_val is not None:
    print(f"Val label distribution: {Counter(y_val)}")
print(f"Test label distribution: {Counter(y_test)}")

# Load SIPEC config
config = load_config(f"configs/behavior/{CONFIG_NAME}")

# CRITICAL: Fix string booleans from config file
config['train_recognition_model'] = True  # Force boolean
config['train_sequential_model'] = False  # Force boolean
config['recognition_model_use_scheduler'] = True  # Force boolean
config['use_class_weights'] = True  # Force boolean

# Image dimensions
config['image_x'] = 200
config['image_y'] = 200

# Core parameters
config['num_classes'] = 4
config['normalize_data'] = True
config['encode_labels'] = True
config['look_back'] = 10
config['use_generator'] = True
config['do_flow'] = False
config['undersample_data'] = False

# Model architecture
config['backbone'] = 'xception'

# Recognition model parameters
config['recognition_model_optimizer'] = 'adam'
config['recognition_model_lr'] = 0.0001
config['recognition_model_epochs'] = 10
config['recognition_model_batch_size'] = 16
config['recognition_model_fix'] = False
config['recognition_model_remove_classification'] = False

# Scheduler parameters (these need to match Model class attributes)
config['recognition_model_scheduler_lr'] = 0.0001  # Initial LR for scheduler
config['recognition_model_scheduler_factor'] = 1.1  # Division factor per epoch
config['recognition_model_scheduler_lower_lr'] = 0.0000001  # Minimum LR

# Sequential model parameters (needed even if not training sequential)
config['sequential_backbone'] = 'lstm'
config['sequential_model_optimizer'] = 'adam'
config['sequential_model_lr'] = 0.0001
config['sequential_model_use_scheduler'] = False
config['sequential_model_epochs'] = 50
config['sequential_model_batch_size'] = 16

print("\nConfig being used:")
for key in ['train_recognition_model', 'use_class_weights', 'recognition_model_use_scheduler']:
    print(f"  {key}: {config[key]} (type: {type(config[key])})")

# Create SIPEC dataloader with appropriate validation data
print("\nCreating dataloader...")
if x_val is not None:
    # Use val set for validation during training
    dataloader = Dataloader(x_train, y_train, x_val, y_val, config=config)
    print("Using validation set for training validation")
else:
    # Use test set for validation during training
    dataloader = Dataloader(x_train, y_train, x_test, y_test, config=config)
    print("Using test set for training validation")

# Prepare data
print("Preparing data...")
dataloader.prepare_data(
    downscale=(75, 75),  # Don't downscale - already at target size
    remove_behaviors=[],  # Don't remove any behaviors
    flatten=False,
    recurrent=False  # Not using sequential model
)

# CRITICAL: Calculate class weights BEFORE training
class_weights = None
if config['use_class_weights']:
    print("\nCalculating class weights...")
    # Need to encode labels first to get numeric values
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

# Train model
print("\n" + "="*80)
print("TRAINING MODEL")
print("="*80)

model, results, report = train_behavior(
    dataloader,
    config,
    num_classes=config['num_classes'],
    encode_labels=False,  # Already encoded in prepare_data
    class_weights=class_weights  # CRITICAL: Pass class weights!
)

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print("\nValidation Results:")
print("Classification Report:")
print(report)
print(f"\nMetrics (acc, f1, corr): {results}")

# If we have a separate test set, evaluate on it
if x_val is not None and x_test is not None:
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    # Create a separate dataloader for test evaluation
    test_dataloader = Dataloader(x_train, y_train, x_test, y_test, config=config)
    test_dataloader.prepare_data(
        downscale=None,
        remove_behaviors=[],
        flatten=False,
        recurrent=False
    )
    
    # Evaluate on test set
    from sklearn import metrics
    from scipy.stats import pearsonr
    
    print(f"Test set size: {len(test_dataloader.x_test)}")
    
    # Get predictions
    test_predictions = model.recognition_model.predict(test_dataloader.x_test)
    test_pred_labels = np.argmax(test_predictions, axis=-1)
    test_true_labels = np.argmax(test_dataloader.y_test, axis=-1)
    
    # Calculate metrics
    test_acc = metrics.balanced_accuracy_score(test_true_labels, test_pred_labels)
    test_f1 = metrics.f1_score(test_true_labels, test_pred_labels, average='macro')
    test_corr = pearsonr(test_pred_labels, test_true_labels)[0]
    
    test_report = metrics.classification_report(
        test_true_labels,
        test_pred_labels,
        target_names=['lying_asleep', 'lying_awake', 'upright', 'obstructed']
    )
    
    print("\nTest Set Classification Report:")
    print(test_report)
    print(f"\nTest Set Metrics:")
    print(f"  Balanced Accuracy: {test_acc:.4f}")
    print(f"  Macro F1: {test_f1:.4f}")
    print(f"  Pearson Correlation: {test_corr:.4f}")

# Save model
now = datetime.now().strftime("%m-%d-%Y_%HH-%MM-%SS")
name = f'pig_behavior_model_{now}.h5'
model.recognition_model.save(name)
print(f"\nModel saved to: {name}")

# Save training history if available
if hasattr(model, 'recognition_model_history') and model.recognition_model_history:
    import pickle
    history_name = f'pig_behavior_history_{now}.pkl'
    with open(history_name, 'wb') as f:
        pickle.dump(model.recognition_model_history.history, f)
    print(f"Training history saved to: {history_name}")