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
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import metrics

# Configuration
CLIPS_OUTPUT_DIR = '/home/tbiinterns/Desktop/semiology_ml/training_data/distilled_clips_gpu_test/'
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
train_dist = Counter(y_train)
print(f"\nTrain label distribution: {train_dist}")
if y_val is not None:
    val_dist = Counter(y_val)
    print(f"Val label distribution: {val_dist}")
test_dist = Counter(y_test)
print(f"Test label distribution: {test_dist}")

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
config['recognition_model_epochs'] = 15
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

val_dist = None
if y_val is not None:
    val_dist = Counter(y_val)
    print(f"Val label distribution: {val_dist}")

wandb.init(
    project="SIPIG-initial",
    name=f"train_{datetime.now().strftime('%m%d_%H%M')}",
    config={
        # Model config
        "architecture": config['backbone'],
        "dataset": "distilled_clips",
        "epochs": config['recognition_model_epochs'],
        "batch_size": config['recognition_model_batch_size'],
        "learning_rate": config['recognition_model_lr'],
        "optimizer": config['recognition_model_optimizer'],
        "image_size": (75, 75),
        "num_classes": config['num_classes'],
        "use_class_weights": config['use_class_weights'],
        "use_scheduler": config['recognition_model_use_scheduler'],
        "scheduler_lr": config['recognition_model_scheduler_lr'],
        "scheduler_factor": config['recognition_model_scheduler_factor'],
        "normalize_data": config['normalize_data'],
        "use_generator": config['use_generator'],
        # Data statistics
        "train_samples": len(y_train),
        "val_samples": len(y_val) if y_val is not None else 0,
        "test_samples": len(y_test),
        "train_distribution": dict(train_dist),
        "val_distribution": dict(val_dist) if val_dist is not None else None,
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

# =================oversampling========================
from imblearn.over_sampling import RandomOverSampler

print("\nOversampling minority classes...")
ros = RandomOverSampler(random_state=42)
x_train_flat = x_train.reshape(len(x_train), -1)
x_train_resampled, y_train_resampled = ros.fit_resample(x_train_flat, y_train)
x_train = x_train_resampled.reshape(-1, *x_train.shape[1:])
y_train = y_train_resampled

print(f"After oversampling: {Counter(y_train)}")
# =========================================

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
    downscale=(75, 75),
    remove_behaviors=[],
    flatten=False,
    recurrent=False
)

# ================= class weights =================
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

# ============================================
        
# Train model
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

# =================== wandb logging =======================

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

# Parse classification report for per-class metrics
report_dict = metrics.classification_report(
    np.argmax(dataloader.y_test, axis=-1),
    res,
    target_names=label_encoder.classes_,
    output_dict=True
)

# Log per-class metrics with class names
for class_name in label_encoder.classes_:
    wandb.log({
        f"val_{class_name}_precision": report_dict[class_name]['precision'],
        f"val_{class_name}_recall": report_dict[class_name]['recall'],
        f"val_{class_name}_f1": report_dict[class_name]['f1-score'],
    })

# =================== wandb logging =======================
    
# If we have a separate test set, evaluate on it
if x_val is not None and x_test is not None:
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    # Create a separate dataloader for test evaluation
    test_dataloader = Dataloader(x_train, y_train, x_test, y_test, config=config)
    test_dataloader.prepare_data(
        downscale=(75, 75),
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

    test_pred_labels = np.argmax(test_predictions, axis=-1)
    test_pred_behaviors = label_encoder.inverse_transform(test_pred_labels)
    print(f"\nPrediction distribution: {Counter(test_pred_behaviors)}")
    
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
    
    # Log test results to wandb
    wandb.log({
        "test_balanced_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "test_pearson_corr": test_corr,
    })
    
    # Log test classification report
    wandb.log({"test_classification_report": wandb.Html(f"<pre>{test_report}</pre>")})
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(test_true_labels, test_pred_labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['lying_asleep', 'lying_awake', 'upright', 'obstructed'],
                yticklabels=['lying_asleep', 'lying_awake', 'upright', 'obstructed'],
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Test Set Confusion Matrix')
    
    wandb.log({"test_confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

# Save model
now = datetime.now().strftime("%m-%d-%Y_%HH-%MM-%SS")
name = f'pig_behavior_model_{now}.h5'
model.recognition_model.save(name)
print(f"\nModel saved to: {name}")

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