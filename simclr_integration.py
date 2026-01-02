"""
SimCLR Integration Utilities
Load pretrained encoder and integrate with SIPEC behavior classification
"""
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json


class SimCLREncoderLoader:
    """Load and integrate SimCLR pretrained encoder"""
    
    @staticmethod
    def load_pretrained_encoder(encoder_path, freeze_layers=True, freeze_until_layer=None):
        """
        Load a pretrained SimCLR encoder
        
        Args:
            encoder_path: Path to saved encoder .h5 file
            freeze_layers: Whether to freeze encoder weights initially
            freeze_until_layer: Freeze all layers up to this layer name (None = all layers)
        
        Returns:
            Loaded encoder model
        """
        print(f"Loading pretrained encoder from: {encoder_path}")
        
        encoder = keras.models.load_model(encoder_path, compile=False)
        
        if freeze_layers:
            if freeze_until_layer is None:
                # Freeze all layers
                for layer in encoder.layers:
                    layer.trainable = False
                print(f"Froze all {len(encoder.layers)} encoder layers")
            else:
                # Freeze up to specific layer
                freeze_count = 0
                for layer in encoder.layers:
                    if layer.name == freeze_until_layer:
                        break
                    layer.trainable = False
                    freeze_count += 1
                print(f"Froze {freeze_count} encoder layers up to '{freeze_until_layer}'")
        
        return encoder
    
    @staticmethod
    def get_latest_encoder(checkpoint_dir='../simclr/training/simclr_checkpoints/'):
        """
        Get the path to the latest/best encoder from checkpoint directory
        
        Args:
            checkpoint_dir: Directory containing SimCLR checkpoints
        
        Returns:
            Path to best encoder, or None if not found
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # Try to load results.json which has best encoder path
        results_path = checkpoint_dir / 'training_results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
                best_path = results.get('best_encoder_path')
                if best_path and Path(best_path).exists():
                    return best_path
        
        # Otherwise find latest encoder file
        encoder_files = list(checkpoint_dir.glob('encoder_epoch_*.h5'))
        if encoder_files:
            # Sort by epoch number
            encoder_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            return str(encoder_files[-1])
        
        return None
    
    @staticmethod
    def create_classifier_from_encoder(encoder, num_classes, input_shape=(75, 75, 3)):
        """
        Create a behavior classifier using pretrained encoder
        
        Args:
            encoder: Pretrained encoder model
            num_classes: Number of behavior classes
            input_shape: Input image shape
        
        Returns:
            Full classification model
        """
        inputs = keras.Input(shape=input_shape)
        
        # Get features from encoder
        features = encoder(inputs, training=False)  # Set training=False for frozen encoder
        
        # Add batch normalization to stabilize features
        x = keras.layers.BatchNormalization()(features)
        
        # Add classification head
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='simclr_classifier')
        
        return model
    
    @staticmethod
    def replace_backbone_with_pretrained(sipec_model, encoder, freeze_encoder=True):
        """
        Replace SIPEC model's backbone with pretrained encoder
        
        This modifies the recognition_model in place
        
        Args:
            sipec_model: Trained SIPEC model object
            encoder: Pretrained SimCLR encoder
            freeze_encoder: Whether to freeze encoder weights
        
        Returns:
            Modified sipec_model
        """
        # Get the input and output of current model
        old_model = sipec_model.recognition_model
        
        # Create new model with pretrained encoder
        inputs = old_model.input
        
        # Apply encoder
        features = encoder(inputs, training=not freeze_encoder)
        
        # Keep the same classification head
        # Find the last layer before the final Dense layer
        for layer in old_model.layers[::-1]:
            if isinstance(layer, keras.layers.Dense) and layer.units == sipec_model.num_classes:
                # This is the classification layer
                outputs = layer(features)
                break
        
        # Create new model
        new_model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Copy over the optimizer and compile settings
        new_model.compile(
            optimizer=old_model.optimizer,
            loss=old_model.loss,
            metrics=old_model.metrics
        )
        
        # Replace in sipec_model
        sipec_model.recognition_model = new_model
        
        print("Replaced backbone with pretrained encoder")
        return sipec_model


def modify_config_for_simclr(config, encoder_path, freeze_encoder=True):
    """
    Modify SIPEC config to use pretrained SimCLR encoder
    
    Args:
        config: SIPEC config dictionary
        encoder_path: Path to pretrained encoder
        freeze_encoder: Whether to freeze encoder during training
    
    Returns:
        Modified config with SimCLR parameters
    """
    config['use_simclr_pretrained'] = True
    config['simclr_encoder_path'] = encoder_path
    config['simclr_freeze_encoder'] = freeze_encoder
    
    # Adjust training parameters for fine-tuning
    if freeze_encoder:
        # Only training classification head - can use higher LR
        config['recognition_model_lr'] = 0.001
        config['recognition_model_epochs'] = 10  # Faster convergence
    else:
        # Fine-tuning entire network - use lower LR
        config['recognition_model_lr'] = 0.00001
        config['recognition_model_epochs'] = 20
    
    return config


def create_simclr_model_from_scratch(backbone='mobilenet', 
                                     input_shape=(75, 75, 3),
                                     num_classes=4,
                                     pretrained_encoder_path=None,
                                     freeze_encoder=True):
    """
    Create a complete model with optional SimCLR pretraining
    
    This is an alternative to modifying SIPEC's train_behavior function
    
    Args:
        backbone: 'mobilenet' or 'xception'
        input_shape: Input image shape
        num_classes: Number of behavior classes
        pretrained_encoder_path: Path to SimCLR pretrained encoder (None = train from scratch)
        freeze_encoder: Whether to freeze encoder if using pretrained
    
    Returns:
        Compiled Keras model
    """
    if pretrained_encoder_path:
        # Load pretrained encoder
        loader = SimCLREncoderLoader()
        encoder = loader.load_pretrained_encoder(
            pretrained_encoder_path,
            freeze_layers=freeze_encoder
        )
        
        # Create classifier
        model = loader.create_classifier_from_encoder(
            encoder, num_classes, input_shape
        )
    else:
        # Create model from scratch
        if backbone == 'mobilenet':
            base_model = keras.applications.MobileNetV3Small(
                input_shape=input_shape,
                include_top=False,
                weights=None,
                pooling='avg'
            )
        elif backbone == 'xception':
            base_model = keras.applications.Xception(
                input_shape=input_shape,
                include_top=False,
                weights=None,
                pooling='avg'
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        inputs = keras.Input(shape=input_shape)
        features = base_model(inputs)
        x = keras.layers.Dropout(0.5)(features)
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


# Example usage patterns
def example_integration_patterns():
    """
    Examples of how to integrate SimCLR with your training script
    """
    
    # Pattern 1: Load and use in custom training
    print("="*80)
    print("PATTERN 1: Direct model creation")
    print("="*80)
    
    model = create_simclr_model_from_scratch(
        backbone='mobilenet',
        input_shape=(75, 75, 3),
        num_classes=4,
        pretrained_encoder_path='./simclr_checkpoints/encoder_epoch_100.h5',
        freeze_encoder=True
    )
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model ready for training!")
    
    # Pattern 2: Get latest encoder automatically
    print("\n" + "="*80)
    print("PATTERN 2: Auto-find latest encoder")
    print("="*80)
    
    loader = SimCLREncoderLoader()
    encoder_path = loader.get_latest_encoder('./simclr_checkpoints/')
    
    if encoder_path:
        print(f"Found encoder: {encoder_path}")
        model = create_simclr_model_from_scratch(
            pretrained_encoder_path=encoder_path,
            freeze_encoder=True
        )
    else:
        print("No pretrained encoder found, training from scratch")
        model = create_simclr_model_from_scratch()
    
    # Pattern 3: Progressive unfreezing for fine-tuning
    print("\n" + "="*80)
    print("PATTERN 3: Progressive unfreezing")
    print("="*80)
    
    # Stage 1: Train only classification head (10 epochs)
    model = create_simclr_model_from_scratch(
        pretrained_encoder_path=encoder_path,
        freeze_encoder=True  # Frozen
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='categorical_crossentropy')
    # model.fit(x_train, y_train, epochs=10)
    
    # Stage 2: Unfreeze and fine-tune (10 more epochs)
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='categorical_crossentropy')  # Lower LR!
    # model.fit(x_train, y_train, epochs=10)
    
    print("\nPatterns demonstrated!")


if __name__ == '__main__':
    example_integration_patterns()