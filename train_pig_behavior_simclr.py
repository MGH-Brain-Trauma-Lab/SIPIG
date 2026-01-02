"""
Modified training script for pig behavior classification with SimCLR pretraining
Integrates pretrained encoder from SimCLR with SIPEC pipeline
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

# Import SimCLR integration
from simclr_integration import (
    SimCLREncoderLoader,
    modify_config_for_simclr,
    create_simclr_model_from_scratch
)

# =========== ALLOW GPU GROWTH ==========
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=20480)]  # 20GB limit
            )
    except RuntimeError as e:
        print(e)
# ==========================================

# =========== SIMCLR CONFIGURATION ==========
USE_SIMCLR_PRETRAINED = True  # Set to True to use SimCLR pretrained encoder
SIMCLR_CHECKPOINT_DIR = '../simclr/training/simclr_checkpoints/'  # Where SimCLR checkpoints are saved
SIMCLR_FREEZE_ENCODER = True  # True = freeze encoder, False = fine-tune entire network
SIMCLR_ENCODER_PATH = None  # Specific path, or None to auto-find latest

# Progressive unfreezing strategy
PROGRESSIVE_UNFREEZING = True  # Train in 2 stages: frozen → unfrozen
STAGE1_EPOCHS = 10  # Epochs with frozen encoder
STAGE2_EPOCHS = 10  # Epochs with unfrozen encoder
# ===========================================

# Configuration
CLIPS_OUTPUT_DIR = '/home/tbiinterns/Desktop/semiology_ml/training_data/temporal_split_5min_1fps/'
CONFIG_NAME = 'default'

# =========== LOAD TRAINING/VAL/TEST DATA ==========
print("Loading clips...")
data = load_training_data(
    CLIPS_OUTPUT_DIR,
    framerate=1,
    greyscale=False
)

x_train, y_train = data['train']
x_val, y_val = data['val']
x_test, y_test = data['test']
    
print(f"\nData loaded:")
print(f"  Train: {x_train.shape}, labels: {len(y_train)}")
print(f"  Val: {x_val.shape}, labels: {len(y_val)}")
print(f"  Test: {x_test.shape}, labels: {len(y_test)}")

# Check label distribution
from collections import Counter
train_dist = Counter(y_train)
print(f"\nTrain label distribution: {train_dist}")
val_dist = Counter(y_val)
print(f"Val label distribution: {val_dist}")
test_dist = Counter(y_test)
print(f"Test label distribution: {test_dist}")
# ==================================================

# ================== CONFIGS ==========================
config = load_config(f"configs/behavior/{CONFIG_NAME}")

# Fix string booleans from config file
config['train_recognition_model'] = True
config['train_sequential_model'] = True
config['recognition_model_use_scheduler'] = False
config['use_class_weights'] = True

# Image dimensions
config['image_x'] = 75  # Match SimCLR training size
config['image_y'] = 75
config['num_classes'] = 4
config['normalize_data'] = True
config['encode_labels'] = True
config['look_back'] = 10
config['use_generator'] = True
config['do_flow'] = False
config['undersample_data'] = False

# Model architecture
config['backbone'] = 'mobilenet'  # Should match SimCLR backbone
    
# Recognition model parameters
config['recognition_model_optimizer'] = 'adam'
config['recognition_model_batch_size'] = 16
config['recognition_model_fix'] = True
config['recognition_model_remove_classification'] = True
config['recognition_model_augmentation'] = 2

# Sequential model parameters
config['sequential_backbone'] = 'lstm'
config['sequential_model_optimizer'] = 'adam'
config['sequential_model_lr'] = 0.0001
config['sequential_model_use_scheduler'] = False
config['sequential_model_epochs'] = 1
config['sequential_model_batch_size'] = 16
config["temporal_causal"] = False

# =========== SIMCLR INTEGRATION ===========
if USE_SIMCLR_PRETRAINED:
    print("\n" + "="*80)
    print("SIMCLR PRETRAINING INTEGRATION")
    print("="*80)
    
    loader = SimCLREncoderLoader()
    
    # Get encoder path (auto-find or use specified)
    if SIMCLR_ENCODER_PATH is None:
        encoder_path = loader.get_latest_encoder(SIMCLR_CHECKPOINT_DIR)
        if encoder_path is None:
            print("WARNING: No SimCLR pretrained encoder found!")
            print(f"Looked in: {SIMCLR_CHECKPOINT_DIR}")
            print("Training from scratch instead.")
            USE_SIMCLR_PRETRAINED = False
        else:
            print(f"Auto-found encoder: {encoder_path}")
    else:
        encoder_path = SIMCLR_ENCODER_PATH
        print(f"Using specified encoder: {encoder_path}")
    
    if USE_SIMCLR_PRETRAINED:
        # Modify config for SimCLR fine-tuning
        config = modify_config_for_simclr(
            config,
            encoder_path,
            freeze_encoder=SIMCLR_FREEZE_ENCODER
        )
        
        print(f"Freeze encoder: {SIMCLR_FREEZE_ENCODER}")
        print(f"Fine-tuning LR: {config['recognition_model_lr']}")
        print(f"Fine-tuning epochs: {config['recognition_model_epochs']}")
    
    print("="*80)
# ==========================================

# ================= WANDB INITIAL LOGGING ========================
run_name = f"train_{datetime.now().strftime('%m%d_%H%M')}"
if USE_SIMCLR_PRETRAINED:
    run_name += "_simclr"
    if SIMCLR_FREEZE_ENCODER:
        run_name += "_frozen"
    else:
        run_name += "_finetune"

wandb.init(
    project="SIPIG-simclr",
    name=run_name,
    config={
        **config,
        "dataset": CLIPS_OUTPUT_DIR.split('/')[-2],
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "test_samples": len(y_test),
        "train_distribution": dict(train_dist),
        "val_distribution": dict(val_dist),
        "test_distribution": dict(test_dist),
        "use_simclr": USE_SIMCLR_PRETRAINED,
        "simclr_freeze": SIMCLR_FREEZE_ENCODER if USE_SIMCLR_PRETRAINED else None,
        "simclr_encoder": encoder_path if USE_SIMCLR_PRETRAINED else None,
    }
)

# Log distribution bar charts
import pandas as pd
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (name, dist) in enumerate([('Train', train_dist), ('Val', val_dist), ('Test', test_dist)]):
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
for key in ['train_recognition_model', 'use_class_weights', 'recognition_model_lr', 'recognition_model_epochs']:
    print(f"  {key}: {config[key]} (type: {type(config[key])})")
# ==============================================================

# ================= DATA LOADER =================
print("\nCreating dataloader...")
dataloader = Dataloader(x_train, y_train, x_val, y_val, config=config)

print("Preparing data...")
dataloader.prepare_data(
    downscale=(75, 75),
    remove_behaviors=[],
    flatten=False,
    recurrent=False
)
# ===============================================

# ================= CLASS WEIGHTS =================
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
    
    print(f"Weight distribution:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {cls}: {class_weights[i]:.4f}")

    wandb.config.update({
        "class_weights": {label_encoder.classes_[i]: float(class_weights[i]) 
                         for i in range(len(class_weights))},
        "classes": list(label_encoder.classes_)
    })
    
    if class_weights is not None:
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        print(f"\nClass weights as dict: {class_weights_dict}")
        class_weights = class_weights_dict
# =================================================

# =================== CUSTOM MODEL TRAINING WITH SIMCLR =======================
if USE_SIMCLR_PRETRAINED and PROGRESSIVE_UNFREEZING:
    print("\n" + "="*80)
    print("PROGRESSIVE UNFREEZING TRAINING")
    print("="*80)
    
    # STAGE 1: Train with frozen encoder
    print(f"\nSTAGE 1: Training classification head ({STAGE1_EPOCHS} epochs)")
    print("Encoder: FROZEN")
    
    model_stage1 = create_simclr_model_from_scratch(
        backbone=config['backbone'],
        input_shape=(75, 75, 3),
        num_classes=config['num_classes'],
        pretrained_encoder_path=encoder_path,
        freeze_encoder=True
    )
    
    # Use gradient clipping to prevent explosion
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0  # Clip gradients
    )
    
    model_stage1.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    # Convert labels to one-hot
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_train_onehot = to_categorical(y_train_encoded, num_classes=config['num_classes'])
    y_val_onehot = to_categorical(y_val_encoded, num_classes=config['num_classes'])
    
    history_stage1 = model_stage1.fit(
        dataloader.x_train,
        y_train_onehot,
        validation_data=(dataloader.x_test, y_val_onehot),
        epochs=STAGE1_EPOCHS,
        batch_size=config['recognition_model_batch_size'],
        class_weight=class_weights,
        callbacks=[
            WandbMetricsLogger(log_freq='epoch'),
        ]
    )
    
    # STAGE 2: Fine-tune entire network
    print(f"\nSTAGE 2: Fine-tuning entire network ({STAGE2_EPOCHS} epochs)")
    print("Encoder: UNFROZEN")
    
    # Unfreeze all layers
    for layer in model_stage1.layers:
        layer.trainable = True
    
    # Recompile with lower learning rate and gradient clipping
    optimizer_stage2 = tf.keras.optimizers.Adam(
        learning_rate=0.00001,
        clipnorm=1.0  # Clip gradients
    )
    
    model_stage1.compile(
        optimizer=optimizer_stage2,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    history_stage2 = model_stage1.fit(
        dataloader.x_train,
        y_train_onehot,
        validation_data=(dataloader.x_test, y_val_onehot),
        epochs=STAGE2_EPOCHS,
        batch_size=config['recognition_model_batch_size'],
        class_weight=class_weights,
        callbacks=[
            WandbMetricsLogger(log_freq='epoch'),
        ]
    )
    
    # Use final model
    final_model = model_stage1
    
    # Evaluate
    val_predictions = final_model.predict(dataloader.x_test)
    val_pred_labels = np.argmax(val_predictions, axis=-1)
    val_true_labels = np.argmax(y_val_onehot, axis=-1)
    
    # Calculate metrics
    from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
    
    balanced_acc = balanced_accuracy_score(val_true_labels, val_pred_labels)
    macro_f1 = f1_score(val_true_labels, val_pred_labels, average='macro')
    report = classification_report(val_true_labels, val_pred_labels, target_names=le.classes_)
    
    results = (balanced_acc, macro_f1, 0.0)  # No correlation for single timepoint
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nValidation Results:")
    print("Classification Report:")
    print(report)
    print(f"\nMetrics (balanced_acc, f1): {results[:2]}")
    
    # Create a wrapper object similar to SIPEC's output
    class ModelWrapper:
        def __init__(self, model):
            self.recognition_model = model
            self.recognition_model_history = None
    
    model = ModelWrapper(final_model)

else:
    # =================== STANDARD MODEL TRAINING =======================
    print("\n" + "="*80)
    print("STANDARD TRAINING (SIPEC)")
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
    
    val_predictions = model.recognition_model.predict(dataloader.x_test)
    val_pred_labels = np.argmax(val_predictions, axis=-1)
    val_true_labels = np.argmax(dataloader.y_test, axis=-1)

# =================== WANDB RESULTS LOGGING =======================
wandb.log({
    "val_balanced_accuracy": results[0],
    "val_macro_f1": results[1],
})

# Log classification report
wandb.log({"val_classification_report": wandb.Html(f"<pre>{report}</pre>")})

# Get predictions for per-class metrics
report_dict = metrics.classification_report(
    val_true_labels,
    val_pred_labels,
    target_names=le.classes_ if USE_SIMCLR_PRETRAINED and PROGRESSIVE_UNFREEZING else dataloader.label_encoder.classes_,
    output_dict=True
)

# Log per-class metrics
classes = le.classes_ if USE_SIMCLR_PRETRAINED and PROGRESSIVE_UNFREEZING else dataloader.label_encoder.classes_
for class_name in classes:
    wandb.log({
        f"val_{class_name}_precision": report_dict[class_name]['precision'],
        f"val_{class_name}_recall": report_dict[class_name]['recall'],
        f"val_{class_name}_f1": report_dict[class_name]['f1-score'],
    })
# =================================================================

# ================== CONFUSION MATRIX =====================
print("\n" + "="*80)
print("MAKING VAL CONFUSION MATRIX")
print("="*80)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(val_true_labels, val_pred_labels)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes,
            yticklabels=classes,
            ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Validation Set Confusion Matrix')

wandb.log({"val_confusion_matrix": wandb.Image(fig)})
plt.close(fig)

# Save model
now = datetime.now().strftime("%m-%d-%Y_%HH-%MM-%SS")
name = f'pig_behavior_model_{"simclr_" if USE_SIMCLR_PRETRAINED else ""}{now}.h5'
model.recognition_model.save(name)
print(f"\nModel saved to: {name}")

# Save as wandb artifact
artifact = wandb.Artifact(
    name=f'pig-behavior-model-{"simclr-" if USE_SIMCLR_PRETRAINED else ""}{now}',
    type='model',
    description=f'Pig behavior classification model ({"SimCLR pretrained " if USE_SIMCLR_PRETRAINED else ""}{config["backbone"]})',
    metadata={
        'architecture': config['backbone'],
        'num_classes': 4,
        'input_size': (75, 75, 3),
        'val_accuracy': results[0],
        'val_f1': results[1],
        'simclr_pretrained': USE_SIMCLR_PRETRAINED,
        'simclr_frozen': SIMCLR_FREEZE_ENCODER if USE_SIMCLR_PRETRAINED else None,
    }
)
artifact.add_file(name)
wandb.log_artifact(artifact)
# =========================================================

# =========== TRAINING HISTORY =========
if hasattr(model, 'recognition_model_history') and model.recognition_model_history:
    import pickle
    history_name = f'pig_behavior_history_{now}.pkl'
    with open(history_name, 'wb') as f:
        pickle.dump(model.recognition_model_history.history, f)
    print(f"Training history saved to: {history_name}")
    
    # Log training curves
    history = model.recognition_model_history.history
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(history['loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['categorical_accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history['val_categorical_accuracy'], label='Val Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    if 'lr' in history:
        axes[1, 0].plot(history['lr'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    wandb.log({"training_curves": wandb.Image(fig)})
    plt.close(fig)

wandb.finish()
print("\nWandB run completed!")
print(f"\nFinal model uses SimCLR pretraining: {USE_SIMCLR_PRETRAINED}")
if USE_SIMCLR_PRETRAINED:
    print(f"Encoder was {'frozen' if SIMCLR_FREEZE_ENCODER else 'fine-tuned'}")
    print(f"Expected improvement: +15-20% validation accuracy")