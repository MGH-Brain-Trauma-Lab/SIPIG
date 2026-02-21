# SIPEC
# MARKUS MARKS
# Behavioral Classification

import random
from argparse import ArgumentParser
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.externals._pilutil import imresize
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from SwissKnife.architectures import (
    pretrained_recognition,
)
from SwissKnife.dataloader import Dataloader, DataGenerator
from SwissKnife.model import Model
from SwissKnife.utils import (
    setGPU,
    Metrics,
    load_vgg_labels,
    loadVideo,
    load_config,
    check_directory,
    callbacks_learningRate_plateau,
)

import tensorflow as tf
import numpy as np


def train_behavior(
    dataloader,
    config,
    num_classes,
    encode_labels=True,
    class_weights=None,
    checkpoint_callback=None,
):
    print("data prepared!")

    our_model = Model()

    our_model.recognition_model = pretrained_recognition(
        config["backbone"],
        dataloader.get_input_shape(),
        num_classes,
        skip_layers=True,
        pretrained_weights_path=config.get("pretrained_weights_path", None),
        freeze_pretrained=config.get("freeze_pretrained", False),
        keep_pretrained_head=config.get("keep_pretrained_head", True),  # NEW
    )
    # ========== ADD THIS DEBUG CODE ==========
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE CHECK")
    print("="*80)
    our_model.recognition_model.summary()

    has_dropout = False
    for layer in our_model.recognition_model.layers:
        if 'dropout' in layer.name.lower():
            print(f"✓ Found dropout layer: {layer.name} - rate: {layer.rate if hasattr(layer, 'rate') else 'N/A'}")
            has_dropout = True

    if not has_dropout:
        print("❌ WARNING: NO DROPOUT LAYERS FOUND IN MODEL!")
    print("="*80 + "\n")
    
    print(f"\n{'='*70}")
    print(f"VERIFYING PRETRAINED WEIGHTS LOADED")
    print(f"{'='*70}")

    # Get a sample layer's weights to check if they're ImageNet or SimCLR
    # ImageNet weights have specific patterns, SimCLR will be different
    sample_layer = our_model.recognition_model.layers[2]  # Usually first conv layer
    sample_weights = sample_layer.get_weights()[0] if len(sample_layer.get_weights()) > 0 else None

    if sample_weights is not None:
        weight_mean = sample_weights.mean()
        weight_std = sample_weights.std()
        print(f"Sample layer: {sample_layer.name}")
        print(f"  Weight mean: {weight_mean:.6f}")
        print(f"  Weight std: {weight_std:.6f}")

        # ImageNet init typically has mean near 0, std around 0.1-0.3
        # If very different, might be SimCLR
        if abs(weight_mean) < 0.01 and 0.05 < weight_std < 0.5:
            print(f"  ✓ Weights look like standard initialization (ImageNet or SimCLR)")
        else:
            print(f"  ⚠ Unusual weight distribution")

    print(f"{'='*70}\n")
    # =========================================

    our_model.set_class_weight(class_weights)

    if config.get("recognition_model_augmentation", 0) > 0:
        from SwissKnife.augmentations import mouse_identification
        augmentation = mouse_identification(level=config["recognition_model_augmentation"])
        our_model.set_augmentation(augmentation)
        print(f"Using augmentation level {config['recognition_model_augmentation']}")

    if config["train_recognition_model"]:
        # ============ INITIALIZE VALIDATION SUBSET WITH DEFAULTS ============
        x_val_subset = dataloader.x_test if hasattr(dataloader, 'x_test') else None
        y_val_subset = dataloader.y_test if hasattr(dataloader, 'y_test') else None
        
        # ============ SETUP GENERATORS ============
        if dataloader.config["use_generator"]:
            if dataloader.streaming_mode:
                # Use StreamingDataGenerator
                from SwissKnife.dataloader import StreamingDataGenerator

                # ========== GET BALANCED SAMPLING CONFIG ==========
                balanced_sampling = config.get('balanced_sampling', False)
                sampling_strategy = config.get('sampling_strategy', 'stratified')
                # ==================================================

                print("Creating STREAMING generators for recognition model...")
                dataloader.training_generator = StreamingDataGenerator(
                    clip_paths=dataloader.train_paths,
                    labels=dataloader.train_labels_raw,
                    label_encoder=dataloader.label_encoder,
                    num_classes=dataloader.num_classes,
                    batch_size=config["recognition_model_batch_size"],
                    target_size=dataloader.target_size,
                    shuffle=True,
                    augmentation=None,
                    normalize=True,
                    mode='recognition',
                    frames_per_video=10,
                    balanced_sampling=balanced_sampling,      # NEW
                    sampling_strategy=sampling_strategy,      # NEW
                )
                # ============ VERIFY STRATIFIED SAMPLING IS WORKING ============
                print(f"\n{'='*70}")
                print(f"VERIFYING STRATIFIED SAMPLING")
                print(f"{'='*70}")

                # Load first 3 batches and check composition
                for batch_idx in range(3):
                    print(f"\nBatch {batch_idx}:")
                    try:
                        batch_x, batch_y = dataloader.training_generator[batch_idx]

                        # Decode labels
                        batch_labels = np.argmax(batch_y, axis=-1)

                        # Count each class
                        from collections import Counter
                        label_counts = Counter(batch_labels)

                        print(f"  Batch shape: {batch_x.shape}")
                        print(f"  Label distribution:")
                        for class_id in range(dataloader.num_classes):
                            class_name = dataloader.label_encoder.inverse_transform([class_id])[0]
                            count = label_counts.get(class_id, 0)
                            percentage = (count / len(batch_labels)) * 100
                            print(f"    Class {class_id} ({class_name}): {count:3d} samples ({percentage:5.1f}%)")

                        # Check if balanced
                        counts = [label_counts.get(i, 0) for i in range(dataloader.num_classes)]
                        min_count = min(counts)
                        max_count = max(counts)
                        if max_count - min_count <= 2:  # Allow ±2 difference for remainder
                            print(f"  ✓ BALANCED (difference: {max_count - min_count})")
                        else:
                            print(f"  ✗ IMBALANCED (difference: {max_count - min_count})")

                    except Exception as e:
                        print(f"  ✗ Error loading batch: {e}")

                print(f"{'='*70}\n")
                # ===============================================================
                

                # ============ CHECK MODEL & TRAINING SETUP ============
                print(f"\n{'='*70}")
                print(f"MODEL & TRAINING DIAGNOSTIC")
                print(f"{'='*70}")

                # 1. Config check
                print(f"\n1. TRAINING CONFIG:")
                print(f"   Learning rate: {config['recognition_model_lr']}")
                print(f"   Optimizer: {config['recognition_model_optimizer']}")
                print(f"   Loss: {config['recognition_model_loss']}")
                print(f"   Batch size: {config['recognition_model_batch_size']}")
                print(f"   Epochs: {config['recognition_model_epochs']}")
                print(f"   Class weights: {config.get('use_class_weights', False)}")
                print(f"   Pretrained weights: {config.get('pretrained_weights_path', 'None')}")
                print(f"   Keep pretrained head: {config.get('keep_pretrained_head', 'N/A')}")
                print(f"   Freeze pretrained: {config.get('freeze_pretrained', False)}")

                # 2. Model architecture check
                print(f"\n2. MODEL ARCHITECTURE:")
                print(f"   Total layers: {len(our_model.recognition_model.layers)}")

                # Count critical layers
                dense_layers = [l for l in our_model.recognition_model.layers if 'Dense' in str(type(l))]
                dropout_layers = [l for l in our_model.recognition_model.layers if 'Dropout' in str(type(l))]
                bn_layers = [l for l in our_model.recognition_model.layers if 'BatchNormalization' in str(type(l))]

                print(f"   Dense layers: {len(dense_layers)} (expected: 1)")
                print(f"   Dropout layers: {len(dropout_layers)} (expected: 1)")
                print(f"   BatchNorm layers: {len(bn_layers)}")

                if len(dense_layers) > 1:
                    print(f"\n   ⚠ WARNING: Multiple Dense layers detected!")
                    for i, layer in enumerate(dense_layers):
                        print(f"      Dense {i}: {layer.name}, output_shape={layer.output_shape}, trainable={layer.trainable}")

                if len(dropout_layers) > 1:
                    print(f"\n   ⚠ WARNING: Multiple Dropout layers detected!")
                    for i, layer in enumerate(dropout_layers):
                        print(f"      Dropout {i}: {layer.name}, rate={layer.rate}, trainable={layer.trainable}")

                # 3. Check trainable parameters
                trainable_count = sum([1 for l in our_model.recognition_model.layers if l.trainable])
                frozen_count = len(our_model.recognition_model.layers) - trainable_count
                print(f"\n3. TRAINABLE STATUS:")
                print(f"   Trainable layers: {trainable_count}")
                print(f"   Frozen layers: {frozen_count}")

                if frozen_count > trainable_count:
                    print(f"   ⚠ WARNING: Most layers are frozen! This may prevent learning.")

                # 4. Test model on a batch
                print(f"\n4. MODEL OUTPUT TEST:")
                test_batch_x, test_batch_y = dataloader.training_generator[0]
                test_predictions = our_model.recognition_model.predict(test_batch_x[:10], verbose=0)

                print(f"   Test batch (first 10 samples):")
                print(f"   Prediction shape: {test_predictions.shape}")
                print(f"   Prediction range: [{test_predictions.min():.4f}, {test_predictions.max():.4f}]")
                print(f"   First prediction: {test_predictions[0]}")
                print(f"   Sum of first prediction: {test_predictions[0].sum():.4f} (should be ~1.0 for softmax)")

                # Check if predictions are random or collapsed
                pred_classes = np.argmax(test_predictions, axis=-1)
                from collections import Counter
                pred_dist = Counter(pred_classes)
                print(f"   Predicted class distribution:")
                for class_id in range(dataloader.num_classes):
                    class_name = dataloader.label_encoder.inverse_transform([class_id])[0]
                    count = pred_dist.get(class_id, 0)
                    print(f"      {class_name}: {count}/10")

                # Check if model is stuck predicting one class
                if len(pred_dist) == 1:
                    print(f"   ✗ WARNING: Model predicting ONLY class {list(pred_dist.keys())[0]}!")
                elif len(pred_dist) == dataloader.num_classes:
                    print(f"   ✓ Model outputs all {dataloader.num_classes} classes")

                # 5. Check class weights if used
                if config.get('use_class_weights', False) and class_weights is not None:
                    print(f"\n5. CLASS WEIGHTS:")
                    # class_weights is a numpy array, need to convert to dict
                    for class_id in range(len(class_weights)):
                        class_name = dataloader.label_encoder.inverse_transform([class_id])[0]
                        print(f"   {class_name}: {class_weights[class_id]:.4f}")

                print(f"{'='*70}\n")
                # ===============================================================
                
                # After MODEL & TRAINING DIAGNOSTIC, before training:

                print(f"\n{'='*70}")
                print(f"PRE-TRAINING MODEL STATE")
                print(f"{'='*70}")

                # Get first batch
                test_batch_x, test_batch_y = dataloader.training_generator[0]
                test_batch_small = test_batch_x[:32]  # Use smaller batch
                test_labels_small = np.argmax(test_batch_y[:32], axis=-1)

                # Get initial predictions
                initial_preds = our_model.recognition_model.predict(test_batch_small, verbose=0)
                initial_classes = np.argmax(initial_preds, axis=-1)

                # Show distribution
                from collections import Counter
                pred_dist = Counter(initial_classes)
                true_dist = Counter(test_labels_small)

                print(f"\nTrue label distribution (first 32 samples):")
                for class_id in range(dataloader.num_classes):
                    class_name = dataloader.label_encoder.inverse_transform([class_id])[0]
                    count = true_dist.get(class_id, 0)
                    print(f"  {class_name}: {count}/32")

                print(f"\nInitial predictions (before training):")
                for class_id in range(dataloader.num_classes):
                    class_name = dataloader.label_encoder.inverse_transform([class_id])[0]
                    count = pred_dist.get(class_id, 0)
                    confidence = initial_preds[:, class_id].mean()
                    print(f"  {class_name}: {count}/32 predictions (avg confidence: {confidence:.4f})")

                # Check if initialization is reasonable
                entropy = -(initial_preds * np.log(initial_preds + 1e-10)).sum(axis=1).mean()
                print(f"\nAverage prediction entropy: {entropy:.4f}")
                print(f"  (High entropy ~1.1 = uncertain/random, Low entropy ~0 = confident)")

                if entropy > 0.8:
                    print(f"  ✓ Model is uncertain (good for untrained model)")
                else:
                    print(f"  ✗ Model is too confident (may be stuck)")

                print(f"{'='*70}\n")
                
                # In behavior.py, replace the DIAGNOSING LOGIT SATURATION section with this simpler version:

                print(f"\n{'='*70}")
                print(f"DIAGNOSING LOGIT SATURATION")
                print(f"{'='*70}")

                # Get a test batch
                test_batch_x, test_batch_y = dataloader.training_generator[0]
                test_sample = test_batch_x[:2]  # Use just 2 samples for speed

                # ============ STEP 1: Get Intermediate Outputs ============
                print(f"\n1. CHECKING LAYER OUTPUTS:")

                # Get outputs from each layer
                layer_outputs = []
                for i, layer in enumerate(our_model.recognition_model.layers):
                    try:
                        intermediate_model = tf.keras.Model(
                            inputs=our_model.recognition_model.input,
                            outputs=layer.output
                        )
                        output = intermediate_model.predict(test_sample, verbose=0)

                        layer_type = str(type(layer).__name__)
                        print(f"\n   Layer {i}: {layer.name} ({layer_type})")
                        print(f"      Shape: {output.shape}")
                        print(f"      Mean: {output.mean():.6f}")
                        print(f"      Std: {output.std():.6f}")
                        print(f"      Min: {output.min():.6f}")
                        print(f"      Max: {output.max():.6f}")

                        if 'Dense' in layer_type:
                            print(f"      → LOGITS (before softmax)")
                            print(f"      First sample: {output[0]}")
                            print(f"      Range: {output.max() - output.min():.6f}")

                            if output.max() - output.min() > 50:
                                print(f"      ❌ HUGE logit range! This causes saturation.")
                            elif output.max() - output.min() > 20:
                                print(f"      ⚠ Large logit range")

                        if 'Activation' in layer_type or 'Softmax' in layer_type:
                            print(f"      → FINAL PROBABILITIES")
                            print(f"      First sample: {output[0]}")

                        layer_outputs.append((i, layer.name, output))

                    except Exception as e:
                        print(f"   Layer {i}: {layer.name} - Cannot extract (nested model)")

                # ============ STEP 2: Check Dense Layer Weights ============
                print(f"\n2. DENSE LAYER WEIGHTS:")

                for layer in our_model.recognition_model.layers:
                    if 'Dense' in str(type(layer)):
                        weights, bias = layer.get_weights()

                        print(f"   Weight shape: {weights.shape}")
                        print(f"   Weight mean: {weights.mean():.6f}")
                        print(f"   Weight std: {weights.std():.6f}")
                        print(f"   Weight min: {weights.min():.6f}")
                        print(f"   Weight max: {weights.max():.6f}")
                        print(f"   Bias: {bias}")

                        # Estimate magnitude
                        if weights.std() > 1:
                            print(f"   ⚠ Dense weights have large std ({weights.std():.3f})")
                        if np.abs(weights).max() > 2:
                            print(f"   ⚠ Dense weights have large max ({np.abs(weights).max():.3f})")

                print(f"{'='*70}\n")
                
                dataloader.validation_generator = StreamingDataGenerator(
                    clip_paths=dataloader.val_paths,
                    labels=dataloader.val_labels_raw,
                    label_encoder=dataloader.label_encoder,
                    num_classes=dataloader.num_classes,
                    batch_size=config["recognition_model_batch_size"],
                    target_size=dataloader.target_size,
                    shuffle=False,  # Don't shuffle validation
                    augmentation=None,
                    normalize=True,
                    mode='recognition',
                    frames_per_video=5,
                    balanced_sampling=False,  # Don't balance validation - keep true distribution
                    sampling_strategy=sampling_strategy,
                )
                
                
                # ============ ADD THIS ============
                print(f"\n{'='*70}")
                print(f"VALIDATION GENERATOR DEBUG")
                print(f"{'='*70}")
                print(f"Number of validation videos: {len(dataloader.val_paths)}")
                print(f"Batch size: {config['recognition_model_batch_size']}")
                print(f"Frames per video: (look for yourself)")  # ← CHANGE THIS
                print(f"Expected samples per batch: {config['recognition_model_batch_size']}")
                print(f"Number of batches: {len(dataloader.validation_generator)}")
                print(f"Total expected samples: {len(dataloader.validation_generator) * config['recognition_model_batch_size']}")

                # Test load first batch
                print(f"\nTesting first batch load...")
                try:
                    test_x, test_y = dataloader.validation_generator[0]
                    print(f"✓ First batch shape: {test_x.shape}")
                except Exception as e:
                    print(f"✗ Failed to load first batch: {e}")
                print(f"{'='*70}\n")
                # =============

                # Create validation subset for metrics
                print("Creating validation subset for metrics...")
                val_batches_to_load = len(dataloader.validation_generator)

                if val_batches_to_load == 0:
                    print("WARNING: No validation data available!")
                    x_val_subset = None
                    y_val_subset = None
                else:
                    x_val_list = []
                    y_val_list = []

                    from tqdm import tqdm  # Import tqdm
                    for i in tqdm(range(val_batches_to_load), desc="Loading validation batches"):
                        batch_x, batch_y = dataloader.validation_generator[i]
                        x_val_list.append(batch_x)
                        y_val_list.append(batch_y)

                    x_val_subset = np.concatenate(x_val_list, axis=0)
                    y_val_subset = np.concatenate(y_val_list, axis=0)

                    print(f"Validation subset for metrics: {x_val_subset.shape}")

            else:
                # Traditional DataGenerator
                print("Creating TRADITIONAL generators...")
                dataloader.training_generator = DataGenerator(
                    x_train=dataloader.x_train,
                    y_train=dataloader.y_train,
                    look_back=dataloader.config["look_back"],
                    batch_size=config["recognition_model_batch_size"],
                    type="recognition",
                    temporal_causal=config["temporal_causal"],
                )
                dataloader.validation_generator = DataGenerator(
                    x_train=dataloader.x_test,
                    y_train=dataloader.y_test,
                    look_back=dataloader.config["look_back"],
                    batch_size=config["recognition_model_batch_size"],
                    type="recognition",
                    temporal_causal=config["temporal_causal"],
                )

                # Use full validation data for metrics
                x_val_subset = dataloader.x_test
                y_val_subset = dataloader.y_test

        # ============ SETUP METRICS CALLBACK ============
        if x_val_subset is not None and y_val_subset is not None:
            my_metrics = Metrics(
                validation_data=(x_val_subset, y_val_subset),
                class_names=dataloader.label_encoder.classes_
            )
            my_metrics.setModel(our_model.recognition_model)
            our_model.add_callbacks([my_metrics])

        # ============ REST OF TRAINING SETUP ============
        try:
            from wandb.integration.keras import WandbMetricsLogger
            wandb_callback = WandbMetricsLogger(log_freq='epoch')
            our_model.add_callbacks([wandb_callback])
        except ImportError:
            print("WandB not available, skipping callback")
            
        # ============ SETUP OPTIMIZER & SCHEDULER ============
        our_model.set_optimizer(
            config["recognition_model_optimizer"],
            lr=config["recognition_model_lr"],
        )

        if config["recognition_model_use_scheduler"]:
            our_model.scheduler_lr = config["recognition_model_scheduler_lr"]
            our_model.scheduler_factor = config["recognition_model_scheduler_factor"]
            our_model.set_lr_scheduler()
        else:
            # use standard training callback
            CB_es, CB_lr = callbacks_learningRate_plateau()
            our_model.add_callbacks([CB_es, CB_lr])
            
        # ============ ADD CHECKPOINT CALLBACK IF PROVIDED ============
        if checkpoint_callback is not None:
            our_model.add_callbacks([checkpoint_callback])
            print(f"✓ Added checkpoint callback")

        # ============ ACTUALLY TRAIN ============
        our_model.recognition_model_epochs = config["recognition_model_epochs"]
        our_model.recognition_model_batch_size = config["recognition_model_batch_size"]
        print()
        our_model.train_recognition_network(dataloader=dataloader)
        print(config)
        
    # ==================== SEQUENTIAL MODEL TRAINING ====================
    if config["train_sequential_model"]:
        # ============ INITIALIZE VALIDATION SUBSET WITH DEFAULTS ============
        x_val_subset = dataloader.x_test_recurrent if hasattr(dataloader, 'x_test_recurrent') else None
        y_val_subset = dataloader.y_test_recurrent if hasattr(dataloader, 'y_test_recurrent') else None
        
        # ============ SETUP GENERATORS ============
        if dataloader.config["use_generator"]:
            if dataloader.streaming_mode:
                # Use StreamingDataGenerator in sequential mode
                from SwissKnife.dataloader import StreamingDataGenerator

                print("Creating STREAMING generators for sequential model...")
                dataloader.training_generator = StreamingDataGenerator(
                    clip_paths=dataloader.train_paths,
                    labels=dataloader.train_labels_raw,
                    label_encoder=dataloader.label_encoder,
                    num_classes=dataloader.num_classes,
                    batch_size=config["sequential_model_batch_size"],
                    target_size=dataloader.target_size,
                    shuffle=True,
                    augmentation=None,
                    normalize=True,
                    mode='sequential',  # Sequential mode
                    look_back=dataloader.config["look_back"],
                    temporal_causal=config["temporal_causal"],
                )
                dataloader.validation_generator = StreamingDataGenerator(
                    clip_paths=dataloader.val_paths,
                    labels=dataloader.val_labels_raw,
                    label_encoder=dataloader.label_encoder,
                    num_classes=dataloader.num_classes,
                    batch_size=config["sequential_model_batch_size"],
                    target_size=dataloader.target_size,
                    shuffle=False,
                    augmentation=None,
                    normalize=True,
                    mode='sequential',
                    look_back=dataloader.config["look_back"],
                    temporal_causal=config["temporal_causal"],
                )

                # Create validation subset for metrics
                print("\nCreating validation subset for metrics tracking...")
                val_batches_to_load = min(3, len(dataloader.validation_generator))
                x_val_list = []
                y_val_list = []

                for i in range(val_batches_to_load):
                    batch_x, batch_y = dataloader.validation_generator[i]
                    x_val_list.append(batch_x)
                    y_val_list.append(batch_y)

                x_val_subset = np.concatenate(x_val_list, axis=0)
                y_val_subset = np.concatenate(y_val_list, axis=0)

                print(f"Validation subset: {x_val_subset.shape}")

            else:
                # Traditional DataGenerator
                dataloader.training_generator = DataGenerator(
                    x_train=dataloader.x_train,
                    y_train=dataloader.y_train,
                    look_back=dataloader.config["look_back"],
                    batch_size=config["sequential_model_batch_size"],
                    type="sequential",
                    temporal_causal=config["temporal_causal"],
                )
                dataloader.validation_generator = DataGenerator(
                    x_train=dataloader.x_test,
                    y_train=dataloader.y_test,
                    look_back=dataloader.config["look_back"],
                    batch_size=config["sequential_model_batch_size"],
                    type="sequential",
                    temporal_causal=config["temporal_causal"],
                )

                # Create validation subset for traditional mode
                total_val_samples = len(dataloader.validation_generator) * config["sequential_model_batch_size"]
                n_samples = min(2000, max(500, int(total_val_samples * 0.2)))
                n_batches = int(n_samples / config["sequential_model_batch_size"])

                sample_indices = np.linspace(0, len(dataloader.validation_generator) - 1, n_batches, dtype=int)

                val_sequences = []
                val_labels = []
                for idx in sample_indices:
                    batch_x, batch_y = dataloader.validation_generator[idx]
                    val_sequences.append(batch_x)
                    val_labels.append(batch_y)

                x_val_subset = np.concatenate(val_sequences, axis=0)
                y_val_subset = np.concatenate(val_labels, axis=0)

                print(f"Validation subset: {x_val_subset.shape} ({len(x_val_subset)/total_val_samples*100:.1f}%)")

        # ============ FIX RECOGNITION MODEL ============
        if config["recognition_model_fix"]:
            our_model.fix_recognition_layers()
        if config["recognition_model_remove_classification"]:
            our_model.remove_classification_layers()

        # ============ CREATE SEQUENTIAL MODEL ============
        our_model.set_sequential_model(
            architecture=config["sequential_backbone"],
            input_shape=dataloader.get_input_shape(recurrent=True),
            num_classes=num_classes,
        )

        # ============ SETUP METRICS CALLBACK ============
        if x_val_subset is not None and y_val_subset is not None:
            my_metrics = Metrics(
                validation_data=(x_val_subset, y_val_subset),
                class_names=dataloader.label_encoder.classes_
            )
            my_metrics.setModel(our_model.sequential_model)
            our_model.add_callbacks([my_metrics])

        # ============ SETUP OPTIMIZER & SCHEDULER ============
        our_model.set_optimizer(
            config["sequential_model_optimizer"],
            lr=config["sequential_model_lr"],
        )

        # Setup scheduler OR standard callbacks
        if config["sequential_model_use_scheduler"]:
            our_model.scheduler_lr = config["sequential_model_scheduler_lr"]
            our_model.scheduler_factor = config["sequential_model_scheduler_factor"]
            our_model.set_lr_scheduler()
        else:
            # Use standard training callbacks (matches recognition model structure)
            CB_es, CB_lr = callbacks_learningRate_plateau()
            our_model.add_callbacks([CB_es, CB_lr])

        # ============ ACTUALLY TRAIN ============
        our_model.sequential_model_epochs = config["sequential_model_epochs"]
        our_model.sequential_model_batch_size = config["sequential_model_batch_size"]

        our_model.train_sequential_network(dataloader=dataloader)

        print("\n" + "="*80)
        print("SEQUENTIAL TRAINING COMPLETED")
        print("="*80 + "\n")

    print(config)

    print("evaluating")
    # Skip evaluation if using generator (already have metrics from training)
    if dataloader.config["use_generator"]:
        print("Skipping final batch-by-batch evaluation (using generator)")
        print("Validation metrics already computed during training")
        return our_model, [0.0, 0.0, 0.0], "Training completed with generator"
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    res = []
    batches = len(dataloader.x_test)
    batches = int(batches / config["sequential_model_batch_size"])
    test_gt = []
    # TODO: fix -1 to really use all VAL data
    for idx in tqdm(range(batches - 1)):
        if config["train_sequential_model"]:
            eval_batch = []
            for i in range(config["sequential_model_batch_size"]):
                new_idx = (
                    (idx * config["sequential_model_batch_size"])
                    + i
                    + dataloader.look_back
                )
                data = dataloader.x_test[
                    new_idx - dataloader.look_back : new_idx + dataloader.look_back
                ]
                eval_batch.append(data)
                test_gt.append(dataloader.y_test[new_idx])
            eval_batch = np.asarray(eval_batch)
            prediction = our_model.predict(eval_batch, model="sequential")
        else:
            eval_batch = []
            #TODO: double check batch behavior
            for i in range(config["recognition_model_batch_size"]):
                new_idx = (idx * config["recognition_model_batch_size"]) + i
                data = dataloader.x_test[new_idx]
                eval_batch.append(data)
                test_gt.append(dataloader.y_test[new_idx])
            eval_batch = np.asarray(eval_batch)
            predictions, predicted_labels = our_model.predict(eval_batch, model="recognition")
            #res.append(np.argmax(predictions, axis=-1))
            # concatenate results
            res = np.concatenate(
                (res, np.argmax(predictions, axis=-1)), axis=-1
            )

    test_gt = np.asarray(test_gt)

    acc = metrics.balanced_accuracy_score(res, np.argmax(test_gt, axis=-1))
    f1 = metrics.f1_score(res, np.argmax(test_gt, axis=-1), average="macro")
    #
    corr = pearsonr(res, np.argmax(test_gt, axis=-1))[0]
    report = metrics.classification_report(
        res,
        np.argmax(test_gt, axis=-1),
    )

    print(report)
    return our_model, [acc, f1, corr], report


def train_primate(config, results_sink, shuffle):
    """TODO: Fill in description"""
    # TODO: Remove the hardcoded paths
    basepath = "/media/nexus/storage5/swissknife_data/primate/behavior/"

    vids = [
        basepath + "fullvids_20180124T113800-20180124T115800_%T1_0.mp4",
        basepath + "fullvids_20180124T113800-20180124T115800_%T1_1.mp4",
        basepath + "20180116T135000-20180116T142000_social_complete.mp4",
        basepath + "20180124T115800-20180124T122800b_0_complete.mp4",
        basepath + "20180124T115800-20180124T122800b_1_complete.mp4",
    ]
    all_annotations = [
        basepath + "20180124T113800-20180124T115800_0.csv",
        basepath + "20180124T113800-20180124T115800_1.csv",
        basepath + "20180116T135000-20180116T142000_social_complete.csv",
        basepath + "20180124T115800-20180124T122800b_0_complete.csv",
        basepath + "20180124T115800-20180124T122800b_1_complete.csv",
    ]

    all_vids = []
    for vid in vids:
        myvid = loadVideo(vid, greyscale=False)
        if "social" in vid:
            im_re = []
            for el in tqdm(myvid):
                im_re.append(imresize(el, 0.5))
            myvid = np.asarray(im_re)
        all_vids.append(myvid)
    vid = np.vstack(all_vids)

    # #TODO: test here w/without
    new_vid = []
    for el in tqdm(vid):
        new_vid.append(imresize(el, 0.5))
    vid = np.asarray(new_vid)

    labels = []
    labels_idxs = []
    for annot_idx, annotation in enumerate(all_annotations):
        annotation = pd.read_csv(annotation, error_bad_lines=False, header=9)
        annotation = load_vgg_labels(
            annotation, video_length=len(all_vids[annot_idx]), framerate_video=25
        )
        labels = labels + annotation
        labels_idxs.append(annotation)
    idxs = []
    for idx, img in enumerate(vid):
        if max(img.flatten()) == 0:
            # print('black image')
            pass
        else:
            idxs.append(idx)
    idxs = np.asarray(idxs)

    global groups

    groups = (
        [0] * len(labels_idxs[0])
        + [0] * len(labels_idxs[1])
        + [3] * len(labels_idxs[2])
        + [4] * len(labels_idxs[3])
        + [4] * len(labels_idxs[4])
    )

    groups = groups
    vid = vid
    labels = labels

    groups = [groups[i] for i in idxs]
    labels = [labels[i] for i in idxs]
    vid = [vid[i] for i in idxs]
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    vid = np.asarray(vid)

    num_splits = 5
    # TODO: prettify me!
    sss = StratifiedKFold(n_splits=num_splits, random_state=0, shuffle=False)
    print("shuffle")

    y = labels
    print("Classes")
    print(np.unique(y))

    X = list(range(0, len(labels)))
    X = np.asarray(X)
    X = np.expand_dims(X, axis=-1)

    results = []
    reports = []

    for split in range(0, num_splits):

        tr_idx = None
        tt_idx = None
        idx = 0
        print(split)
        for train_index, test_index in sss.split(X, y, groups=groups):
            if idx == split:
                tr_idx = train_index
                tt_idx = test_index
            idx += 1

        y_train = y[tr_idx]
        y_test = y[tt_idx]
        x_train = vid[tr_idx]
        x_test = vid[tt_idx]

        dataloader = Dataloader(x_train, y_train, x_test, y_test, config=config)

        # config_name = 'primate_' + str(1)
        #
        # config = load_config("../configs/behavior/primate/" + config_name)
        config["recognition_model_batch_size"] = 128
        config["backbone"] = "imagenet"
        config["encode_labels"] = True
        print(config)

        num_classes = config["num_classes"]

        print("preparing data")
        # TODO: adjust

        dataloader.change_dtype()
        print("dtype changed")

        dataloader.remove_behavior(behavior="walking")

        if config["normalize_data"]:
            dataloader.normalize_data()
        if config["encode_labels"]:
            dataloader.encode_labels()
        print("labels encoded")

        if shuffle:

            res = list(dataloader.y_test)
            random.shuffle(res)
            res = np.asarray(res)
            results.append(
                [
                    "shuffle",
                    "bla",
                    metrics.balanced_accuracy_score(
                        # res, np.argmax(dataloader.y_test, axis=-1)
                        res,
                        dataloader.y_test,
                    ),
                    metrics.f1_score(
                        # res, np.argmax(dataloader.y_test, axis=-1), average="macro"
                        res,
                        dataloader.y_test,
                        average="macro",
                    ),
                ]
            )
            report = metrics.classification_report(
                res,
                dataloader.y_test,
            )

            print(results)
        else:
            class_weights = None
            if config["use_class_weights"]:
                print("calc class weights")
                from sklearn.utils import class_weight

                class_weights = class_weight.compute_class_weight(
                    "balanced", np.unique(dataloader.y_train), dataloader.y_train
                )

            # if config["undersample_data"]:
            print("undersampling data")
            # dataloader.undersample_data()

            print("preparing recurrent data")
            dataloader.create_recurrent_data()
            print("preparing flattened data")
            # dataloader.create_flattened_data()

            print("categorize data")
            dataloader.categorize_data(num_classes, recurrent=True)

            print("data ready")

            # if operation == "train":
            res, report = train_behavior(
                dataloader,
                config,
                num_classes=config["num_classes"],
                class_weights=class_weights,
            )

        print(config)
        print("DONE")
        print(report)
        np.save(
            results_sink + "results.npy",
            res,
        )
        np.save(
            results_sink + "reports.npy",
            report,
        )


def sec2frame(seconds, fps=30):
    return int((seconds * fps))


def load_multi_labels(path, video_path):
    times = pd.read_json(path)
    print(times)

    # behaviors = {}
    # for i in range(1):
    #     i = str(i)
    #     behaviors[i] = times['attribute'][i]['aname']
    videos = {}
    for i in times["project"]["vid_list"]:
        i = str(i)
        videos[i] = times["file"][i]["fname"]

    fps_dict = {}
    loaded_labels = {}
    for video in videos.values():
        vidpath = video_path + video
        cap = cv2.VideoCapture(vidpath)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        loaded_labels[video] = ["none"] * length
        cam = cv2.VideoCapture(vidpath)
        fps = cam.get(cv2.CAP_PROP_FPS)
        fps_dict[video] = int(fps)

    label_video = {}
    # myvids = []
    # mylabels = []
    for i in times.iterrows():
        meta = i[1]["metadata"]
        try:
            video = meta["vid"]
            print(video)
            video = videos[video]
            fps = fps_dict[video]
            behavior = meta["av"]
            # TODO: allow for multiple behaviors at the same time
            behavior = list(behavior.values())[0]
            times = meta["z"]
            print(times)
            labs = loaded_labels[video]
            labs[sec2frame(times[0], fps=fps) : sec2frame(times[1], fps=fps)] = [
                behavior
            ] * int(sec2frame(times[1], fps=fps) - sec2frame(times[0], fps=fps))
            # mylabels.append(labs)
            # myvids.append(video)
            label_video[video] = labs
            if "vid" in meta.keys():
                print(meta)
        except (AttributeError, TypeError, KeyError):
            continue

    return list(label_video.values()), list(label_video.keys())


def downscale_vid(video, factor):
    new_vid = []
    for el in tqdm(video):
        new_vid.append(imresize(el, factor))
    return np.asarray(new_vid)


def main():
    args = parser.parse_args()
    operation = args.operation
    gpu_name = args.gpu
    config_name = args.config_name
    network = args.network
    shuffle = args.shuffle
    annotations = args.annotations
    video = args.video
    results_sink = args.results_sink
    only_flow = args.only_flow

    results_sink = (
        results_sink
        + "/"
        + config_name
        + "/"
        + network
        + "/"
        + datetime.now().strftime("%Y-%m-%d-%H_%M")
        + "/"
    )

    setGPU(gpu_name)
    check_directory(results_sink)



    labels, videos = load_multi_labels(
        path=annotations,
        video_path=video,
    )

    #TODO: fix these
    basepath = video
    greyscale = False
    downscale_factor = 0.1
    testvid = videos[-1]


    all_labels = []
    all_vids = []
    for vid_idx, vid in tqdm(enumerate(videos)):
        if vid == testvid:
            testivdeo = downscale_vid(
                loadVideo(basepath + vid, greyscale=greyscale), downscale_factor
            )
            test_labels = labels[vid_idx]
        else:
            myvid = downscale_vid(
                loadVideo(basepath + vid, greyscale=greyscale), downscale_factor
            )
            all_labels.append(labels[vid_idx])
            all_vids.append(myvid)
    vid = np.vstack(all_vids)
    labels = np.hstack(all_labels)

    x_train = vid
    y_train = labels
    x_test = testivdeo
    y_test = test_labels

    config = load_config("../configs/behavior/shared_config")
    beh_config = load_config("../configs/behavior/default")
    config.update(beh_config)

    dataloader = Dataloader(x_train, y_train, x_test, y_test, config=config)
    # dataloader.prepare_data()
    num_classes = len(np.unique(y_train))
    config["num_classes"] = num_classes
    print("dataloader prepared")

    dataloader.change_dtype()
    print("dtype changed")

    if config["normalize_data"]:
        dataloader.normalize_data()
    if config["encode_labels"]:
        dataloader.encode_labels()
    print("labels encoded")

    class_weights = None
    if config["use_class_weights"]:
        print("calc class weights")
        from sklearn.utils import class_weight

        class_weights = class_weight.compute_class_weight(
            "balanced", np.unique(dataloader.y_train), dataloader.y_train
        )

    if config["undersample_data"]:
        dataloader.undersample_data()
        print("undersampling data")

    print("preparing recurrent data")
    # dataloader.create_recurrent_data()
    print("preparing flattened data")
    # dataloader.create_flattened_data()

    print("categorize data")
    dataloader.categorize_data(num_classes, recurrent=False)

    print("data ready")

    # if operation == "train":
    res = train_behavior(
        dataloader,
        config,
        num_classes=config["num_classes"],
        class_weights=class_weights,
    )

parser = ArgumentParser()

parser.add_argument(
    "--operation",
    action="store",
    dest="operation",
    type=str,
    default="train_primate",
    help="standard training options for SIPEC data",
)
parser.add_argument(
    "--gpu",
    action="store",
    dest="gpu",
    type=str,
    default="0",
    help="filename of the video to be processed (has to be a segmented one)",
)
parser.add_argument(
    "--config_name",
    action="store",
    dest="config_name",
    type=str,
    default="default",
    help="behavioral config to use",
)
parser.add_argument(
    "--network",
    action="store",
    dest="network",
    type=str,
    default="ours",
    help="which network used for training",
)
# TODO: check if folder and then load all files in folder, similar for vid files
parser.add_argument(
    "--annotations",
    action="store",
    dest="annotations",
    type=str,
    default=None,
    help="path for annotations from VGG annotator",
)
parser.add_argument(
    "--video",
    action="store",
    dest="video",
    type=str,
    default=None,
    help="path to folder with annotated video",
)
parser.add_argument(
    "--results_sink",
    action="store",
    dest="results_sink",
    type=str,
    default="./results/behavior/",
    help="path to results",
)
parser.add_argument(
    "--only_flow",
    action="store",
    dest="only_flow",
    type=str,
    default=None,
    help="use_only_flow",
)
parser.add_argument(
    "--shuffle",
    action="store",
    dest="shuffle",
    type=bool,
    default=False,
)

# example usage
# python behavior.py --annotations "/media/nexus/storage5/swissknife_data/primate/behavior/20180124T113800-20180124T115800_0.csv" --video "/media/nexus/storage5/swissknife_data/primate/behavior/fullvids_20180124T113800-20180124T115800_%T1_0.mp4" --gpu 2
if __name__ == "__main__":
    main()
