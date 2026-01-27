"""
Training script for pig behavior classification using extracted clips
All hyperparameters loaded from SIPEC config files.

Usage:
    python train_pig_behavior.py --config configs/behavior/default
    python train_pig_behavior.py --config configs/behavior/p01_side_75
"""
import os
import sys
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics, preprocessing
from sklearn.utils import class_weight
import wandb

from SwissKnife.clip_loader import load_training_data, load_training_metadata
from SwissKnife.dataloader import Dataloader
from SwissKnife.behavior import train_behavior
from SwissKnife.utils import load_config


def setup_seed(seed=None):
    """Set all random seeds for reproducibility."""
    if seed is None:
        seed = random.randint(0, 999999)
        print(f"\n{'='*80}")
        print(f"USING RANDOM SEED: {seed}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"USING FIXED SEED: {seed}")
        print(f"{'='*80}\n")
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    return seed


def setup_gpu(config):
    """Configure GPU settings from config."""
    # Force GPU growth
    if config.get('gpu_allow_growth', True):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Deterministic operations
    if config.get('gpu_deterministic_ops', True):
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Configure GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory limit if specified
                memory_limit = config.get('gpu_memory_limit_mb')
                if memory_limit is not None:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Mixed precision
    if config.get('use_mixed_precision', False):
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Using mixed precision (float16)")


def load_data(config):
    """Load training/validation/test data based on config."""
    use_streaming = config['use_streaming']
    clips_dir = config['clips_output_dir']
    framerate = config.get('framerate', 1)
    greyscale = config.get('greyscale', False)
    
    if use_streaming:
        print("Loading clip metadata (streaming mode)...")
        metadata = load_training_metadata(clips_dir, framerate=framerate)
        x_train, y_train = metadata['train']
        x_val, y_val = metadata['val']
        x_test, y_test = metadata['test']
        
        print(f"\nMetadata loaded:")
        print(f"  Train: {len(x_train)} clips")
        print(f"  Val: {len(x_val)} clips")
        print(f"  Test: {len(x_test)} clips")
    else:
        print("Loading all clips into memory (traditional mode)...")
        data = load_training_data(clips_dir, framerate=framerate, greyscale=greyscale)
        
        x_train, y_train = data['train']
        x_val, y_val = data['val']
        x_test, y_test = data['test']
        
        print(f"\nData loaded:")
        print(f"  Train: {x_train.shape}, labels: {len(y_train)}")
        print(f"  Val: {x_val.shape}, labels: {len(y_val)}")
        print(f"  Test: {x_test.shape}, labels: {len(y_test)}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def merge_lying_classes(y_train, y_val, y_test, config):
    """Merge label categories if specified in config."""
    if config.get('merge_lying_labels', False):
        print("\nMerging lying labels...")
        y_train = ['lying' if 'lying' in label else label for label in y_train]
        y_test = ['lying' if 'lying' in label else label for label in y_test]
        if y_val is not None:
            y_val = ['lying' if 'lying' in label else label for label in y_val]
    
    return y_train, y_val, y_test


def print_label_distributions(y_train, y_val, y_test):
    """Print label distribution statistics."""
    train_dist = Counter(y_train)
    val_dist = Counter(y_val) if y_val is not None else {}
    test_dist = Counter(y_test)
    
    print(f"\nLabel distributions:")
    print(f"  Train: {train_dist}")
    print(f"  Val: {val_dist}")
    print(f"  Test: {test_dist}")
    
    return train_dist, val_dist, test_dist


def calculate_class_weights(y_train, config):
    """Calculate class weights for training."""
    if not config.get('use_class_weights', False):
        return None
    
    print("\nCalculating class weights...")
    label_encoder = preprocessing.LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train_encoded),
        y=y_train_encoded
    )
    
    print(f"Class weights:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {cls}: {class_weights[i]:.4f}")
    
    # Convert to dict for Keras
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    return class_weights_dict


def initialize_wandb(config, seed, distributions, class_weights_dict=None):
    """Initialize Weights & Biases logging."""
    if not config.get('wandb_enabled', True):
        return
    
    train_dist, val_dist, test_dist = distributions
    
    # Prepare config for wandb
    wandb_config = {
        'seed': seed,
        'dataset': Path(config['clips_output_dir']).name,
        'train_samples': sum(train_dist.values()),
        'val_samples': sum(val_dist.values()) if val_dist else 0,
        'test_samples': sum(test_dist.values()),
        'train_distribution': dict(train_dist),
        'val_distribution': dict(val_dist) if val_dist else {},
        'test_distribution': dict(test_dist),
    }
    
    # Add all config items
    wandb_config.update(config)
    
    if class_weights_dict is not None:
        wandb_config['class_weights'] = {str(k): float(v) for k, v in class_weights_dict.items()}
    
    # Initialize wandb
    exp_name = config.get('experiment_name', 'pig_behavior')
    run_name = f"{exp_name}_{datetime.now().strftime('%m%d_%H%M')}"
    wandb.init(
        project=config.get('wandb_project', 'SIPIG-initial'),
        name=run_name,
        config=wandb_config,
        notes=config.get('experiment_description', '')
    )
    
    # Log distribution plots if enabled
    if config.get('log_distributions', True):
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


def log_validation_results(config, model, dataloader, results, report, seed):
    """Log validation results to wandb."""
    if not config.get('wandb_enabled', True):
        return
    
    # Log basic metrics
    wandb.log({
        "val_balanced_accuracy": results[0],
        "val_macro_f1": results[1],
        "val_pearson_corr": results[2],
    })
    
    # Log classification report
    wandb.log({"val_classification_report": wandb.Html(f"<pre>{report}</pre>")})
    
    # Get predictions for detailed metrics
    use_streaming = config['use_streaming']
    
    if use_streaming:
        print("Loading validation data for final evaluation...")
        val_batches = min(10, len(dataloader.validation_generator))
        val_x, val_y = [], []
        for i in range(val_batches):
            batch_x, batch_y = dataloader.validation_generator[i]
            val_x.append(batch_x)
            val_y.append(batch_y)
        val_x = np.concatenate(val_x, axis=0)
        val_y = np.concatenate(val_y, axis=0)
    else:
        val_x = dataloader.x_test
        val_y = dataloader.y_test
    
    val_predictions = model.recognition_model.predict(val_x)
    val_pred_labels = np.argmax(val_predictions, axis=-1)
    val_true_labels = np.argmax(val_y, axis=-1)
    
    # Log per-class metrics
    report_dict = metrics.classification_report(
        val_true_labels,
        val_pred_labels,
        target_names=dataloader.label_encoder.classes_,
        output_dict=True
    )
    
    for class_name in dataloader.label_encoder.classes_:
        wandb.log({
            f"val_{class_name}_precision": report_dict[class_name]['precision'],
            f"val_{class_name}_recall": report_dict[class_name]['recall'],
            f"val_{class_name}_f1": report_dict[class_name]['f1-score'],
        })
    
    # Log confusion matrix
    if config.get('log_confusion_matrix', True):
        cm = metrics.confusion_matrix(val_true_labels, val_pred_labels)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=dataloader.label_encoder.classes_,
                    yticklabels=dataloader.label_encoder.classes_,
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Validation Set Confusion Matrix')
        wandb.log({"val_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
    
    # Log training curves
    if config.get('log_training_curves', True) and hasattr(model, 'recognition_model_history'):
        if model.recognition_model_history:
            history = model.recognition_model_history.history
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss
            axes[0, 0].plot(history['loss'], label='Train Loss')
            axes[0, 0].plot(history['val_loss'], label='Val Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy
            axes[0, 1].plot(history['categorical_accuracy'], label='Train Accuracy')
            axes[0, 1].plot(history['val_categorical_accuracy'], label='Val Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Learning rate
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


def save_model(config, model, results, seed, epoch):
    """Save model and artifacts."""
    if not config.get('save_model', True):
        return None
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate model name
    now = datetime.now().strftime("%m-%d-%Y_%HH-%MM-%SS")
    model_prefix = config.get('model_name_prefix', 'pig_behavior_model')
    model_name = f"{model_prefix}_{now}_seed{seed}_epoch{epoch}.h5"
    model_path = checkpoint_dir / model_name
    
    # Save model
    model.recognition_model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save training history
    if config.get('save_history', True) and hasattr(model, 'recognition_model_history'):
        if model.recognition_model_history:
            import pickle
            history_name = f"{model_prefix}_history_{now}.pkl"
            history_path = checkpoint_dir / history_name
            with open(history_path, 'wb') as f:
                pickle.dump(model.recognition_model_history.history, f)
            print(f"Training history saved to: {history_path}")
    
    # Save to wandb
    if config.get('save_to_wandb', True) and config.get('wandb_enabled', True):
        if config.get('log_model_artifact', True):
            artifact = wandb.Artifact(
                name=f'pig-behavior-model-{now}',
                type='model',
                description=config.get('experiment_description', ''),
                metadata={
                    'architecture': config['backbone'],
                    'num_classes': config['num_classes'],
                    'input_size': (config['image_x'], config['image_y'], 3),
                    'val_accuracy': results[0],
                    'val_f1': results[1],
                    'seed': seed,
                }
            )
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact)
            print(f"Model artifact logged to wandb")
    
    return model_path


def main(config_path):
    """Main training pipeline."""
    print(f"\n{'='*80}")
    print(f"Loading config from: {config_path}")
    print(f"{'='*80}\n")
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup seed
    seed = setup_seed(config.get('seed'))
    
    # Setup GPU
    setup_gpu(config)
    
    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(config)
    
    # Merge labels if specified
    y_train, y_val, y_test = merge_lying_classes(y_train, y_val, y_test, config)
    
    # Print distributions
    distributions = print_label_distributions(y_train, y_val, y_test)
    
    # Calculate class weights
    class_weights_dict = calculate_class_weights(y_train, config)
    
    # Initialize wandb
    initialize_wandb(config, seed, distributions, class_weights_dict)
    
    # Create dataloader
    print("\nCreating dataloader...")
    dataloader = Dataloader(x_train, y_train, x_val, y_val, config=config)
    
    # Prepare data
    print("Preparing data...")
    if config['use_streaming']:
        dataloader.prepare_streaming_data(
            target_size=(config['image_x'], config['image_y'])
        )
    else:
        dataloader.prepare_data(
            downscale=(config['image_x'], config['image_y']),
            remove_behaviors=[],
            flatten=False,
            recurrent=False
        )
    
    # Train model
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    model, results, report = train_behavior(
        dataloader,
        config,
        num_classes=config['num_classes'],
        encode_labels=False,
        class_weights=class_weights_dict
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nValidation Results:")
    print(report)
    print(f"\nMetrics (acc, f1, corr): {results}")
    
    # Log results
    log_validation_results(config, model, dataloader, results, report, seed)
    
    # Save model
    model_path = save_model(config, model, results, seed, config['recognition_model_epochs'])
    
    # Finish wandb
    if config.get('wandb_enabled', True):
        wandb.finish()
        print("\nWandB run completed!")
    
    print(f"\n{'='*80}")
    print(f"Training completed successfully!")
    print(f"Seed: {seed}")
    if model_path:
        print(f"Model: {model_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train pig behavior classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_pig_behavior.py --config configs/behavior/default
  python train_pig_behavior.py --config configs/behavior/p01_side_75
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/behavior/p01_side_75',
        help='Path to SIPEC config file (without extension)'
    )
    
    args = parser.parse_args()
    main(args.config)