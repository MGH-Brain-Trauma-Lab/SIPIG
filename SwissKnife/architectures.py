"""
SIPEC
MARKUS MARKS
MODEL ARCHITECTURES
"""
from tensorflow.keras import Model, regularizers
from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB7,
    InceptionResNetV2,
    InceptionV3,
    ResNet50,
    Xception,
    MobileNetV3Small,
)
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    GaussianNoise,
    Input,
    LeakyReLU,
    MaxPooling2D,
    TimeDistributed,
    ZeroPadding2D,
    concatenate,
)
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.applications.efficientnet import EfficientNetB1


def posenet(
    input_shape,
    num_classes,
    backbone="efficientnetb5",
    fix_backbone=False,
    gaussian_noise=0.05,
    features=256,
    bias=False,
):
    """Model that implements SIPEC:PoseNet architecture.

    This model uses an EfficientNet backbone and deconvolves generated features into landmarks in imagespace.
    It operates on single images and can be used in conjuntion with SIPEC:SegNet to perform top-down pose estimation.

    Parameters
    ----------
    input_shape : keras compatible input shape (W,H,Channels)
        keras compatible input shape (features,)
    num_classes : int
        Number of joints/landmarks to detect.
    backbone : str
        Backbone/feature detector to use, default is EfficientNet5. Choose smaller/bigger backbone depending on GPU memory.
    gaussian_noise : float
        Kernel size of gaussian noise layers to use.
    features : int
        Number of feature maps to generate at each level.
    bias : bool
        Use bias for deconvolutional layers.

    Returns
    -------
    keras.model
        SIPEC:PoseNet
    """
    if backbone == "efficientnetb5":
        recognition_model = EfficientNetB5(
            include_top=False,
            input_shape=input_shape,
            pooling=None,
            weights="imagenet",
        )
    elif backbone == "efficientnetb7":
        recognition_model = EfficientNetB7(
            include_top=False,
            input_shape=input_shape,
            pooling=None,
            weights="imagenet",
        )
    elif backbone == "efficientnetb1":
        recognition_model = EfficientNetB1(
            include_top=False,
            input_shape=input_shape,
            pooling=None,
            weights="imagenet",
        )
    else:
        raise NotImplementedError

    new_input = Input(
        batch_shape=(None, input_shape[0], input_shape[1], input_shape[2])
    )

    if fix_backbone:
        for layer in recognition_model.layers:
            layer.trainable = False

    x = Conv2D(3, kernel_size=(1, 1), strides=(1, 1))(new_input)
    x = recognition_model(x)

    for _ in range(4):
        x = Conv2DTranspose(
            features, kernel_size=(2, 2), strides=(2, 2), padding="valid", use_bias=bias
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = GaussianNoise(gaussian_noise)(x)

    x = Conv2DTranspose(
        features, kernel_size=(2, 2), strides=(2, 2), padding="valid", use_bias=bias
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(
        num_classes, kernel_size=(1, 1), strides=(1, 1), padding="valid"
    )(x)

    x = Activation("sigmoid")(x)

    model = Model(inputs=new_input, outputs=x)
    model.summary()

    return model


def classification_scratch(input_shape):
    """
    Args:
        input_shape:
    """
    dropout = 0.3
    # conv model

    model = Sequential()

    model.add(
        Conv2D(
            64,
            kernel_size=(4, 4),
            strides=(4, 4),
            padding="valid",
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(512, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    #     model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation("softmax"))

    return model


def classification_large(input_shape, num_classes):
    """
    Args:
        input_shape:
    """
    dropout = 0.1
    # conv model

    model = Sequential()

    model.add(
        Conv2D(
            64,
            kernel_size=(4, 4),
            strides=(4, 4),
            padding="valid",
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    # model.add(Conv2D(512, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    # model.add(Dropout(dropout))
    #
    # model.add(Conv2D(1024, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    # model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    #     model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    return model


def classification_small(input_shape, num_classes):
    """
    Args:
        input_shape:
        num_classes:
    """
    dropout = 0.33
    # conv model

    model = Sequential()

    model.add(
        Conv2D(
            64,
            kernel_size=(4, 4),
            strides=(4, 4),
            padding="valid",
            input_shape=input_shape,
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(256, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Conv2D(512, kernel_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(32, kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    #     model.add(Dropout(dropout))

    #     model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    return model


def dlc_model_sturman(input_shape, num_classes):
    """Model that implements behavioral classification based on
    Deeplabcut generated features as in Sturman et al.

    Reimplementation of the model used in the publication Sturman et al.
    that performs action recognition on top of pose estimation

    Parameters
    ----------
    input_shape : keras compatible input shape (W,H,Channels)
        keras compatible input shape (features,)
    num_classes : int
        Number of behaviors to classify.

    Returns
    -------
    keras.model
        Sturman et al. model
    """
    model = Sequential()

    model.add(
        Dense(
            256, input_shape=(input_shape[-1],), kernel_regularizer=regularizers.l2(0.0)
        )
    )
    model.add(Activation("relu"))
    model.add(Dropout(0.4))

    model.add(Dense(128, kernel_regularizer=regularizers.l2(0.0)))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    # TODO: parametrize # behaviors
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    return model

# TODO: remove unused code
def dlc_model(input_shape, num_classes):
    """Model for classification on top of pose estimation.

    Classification model for behavior, operating on pose estimation.
    This model has more free parameters than Sturman et al.

    Parameters
    ----------
    input_shape : keras compatible input shape (W,H,Channels)
        keras compatible input shape (features,)
    num_classes : int
        Number of behaviors to classify.

    Returns
    -------
    keras.model
        behavior (from pose estimates) model
    """
    dropout = 0.3

    model = Sequential()

    model.add(Dense(256, input_shape=(input_shape[-1],)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout))

    # TODO: parametrize # behaviors
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    return model


# TODO: remove unused code
def recurrent_model_old(
    recognition_model, recurrent_input_shape, classes=4, recurrent_dropout=None
):
    """
    Args:
        recognition_model:
        recurrent_input_shape:
        classes:
        recurrent_dropout:
    """
    input_sequences = Input(shape=recurrent_input_shape)
    sequential_model_helper = TimeDistributed(recognition_model)(input_sequences)

    if recurrent_dropout:
        # TODO: adjust bidirectional
        k = LSTM(units=128, return_sequences=True, recurrent_dropout=recurrent_dropout)(
            sequential_model_helper
        )
        k = LSTM(units=64, return_sequences=True, recurrent_dropout=recurrent_dropout)(
            k
        )
        k = LSTM(units=32, return_sequences=False, recurrent_dropout=recurrent_dropout)(
            k
        )

    else:
        # As of TF 2, one can just use LSTM and there is no CuDNNLSTM
        k = Bidirectional(LSTM(units=128, return_sequences=True))(
            sequential_model_helper
        )
        k = Bidirectional(LSTM(units=64, return_sequences=True))(k)
        k = Bidirectional(LSTM(units=32, return_sequences=False))(k)

    dout = 0.3
    k = Dense(256)(k)
    k = Activation("relu")(k)
    k = Dropout(dout)(k)
    k = Dense(128)(k)
    k = Activation("relu")(k)
    k = Dropout(dout)(k)
    k = Dense(64)(k)
    k = Dropout(dout)(k)
    k = Activation("relu")(k)
    k = Dense(32)(k)
    k = Activation("relu")(k)
    # TODO: modelfy me!
    k = Dense(classes)(k)
    k = Activation("softmax")(k)

    sequential_model = Model(inputs=input_sequences, outputs=k)

    return sequential_model


def recurrent_model_tcn(
    recognition_model,
    recurrent_input_shape,
    classes=4,
):
    """Recurrent architecture for classification of temporal sequences
    of images based on temporal convolution architecture (TCN).
    This architecture is used for BehaviorNet in SIPEC.

    Parameters
    ----------
    recognition_model : keras.model
        Pretrained recognition model that extracts features for individual frames.
    recurrent_input_shape : np.ndarray - (Time, Width, Height, Channels)
        Shape of the images over time.
    classes : int
        Number of behaviors to recognise.

    Returns
    -------
    keras.model
        BehaviorNet
    """
    input_sequences = Input(shape=recurrent_input_shape)
    sequential_model_helper = TimeDistributed(recognition_model)(input_sequences)
    k = BatchNormalization()(sequential_model_helper)

    # TODO: config me!
    filters = 64
    kernel_size = 2
    # dout = 0.1
    alpha = 0.3
    act_fcn = "relu"
    k = Conv1D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=1,
        kernel_initializer="he_normal",
    )(k)
    # k_1 = Flatten(k)
    k = BatchNormalization()(k)
    # k = Activation(LeakyReLU(alpha=0.3))(k)
    # k = Activation(Activation('relu'))(k)
    # k = wave_net_activation(k)
    k = Activation(act_fcn)(k)
    # k = SpatialDropout1D(rate=dout)(k)

    k = Conv1D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=2,
        kernel_initializer="he_normal",
    )(k)
    k = BatchNormalization()(k)
    # k = Activation(LeakyReLU(alpha=0.3))(k)
    # k = Activation(Activation('relu'))(k)
    # k = wave_net_activation(k)
    k = Activation(act_fcn)(k)
    # k = SpatialDropout1D(rate=dout)(k)

    k = Conv1D(
        filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=4,
        kernel_initializer="he_normal",
    )(k)
    k = BatchNormalization()(k)
    # k = Activation(LeakyReLU(alpha=0.3))(k)
    # k = Activation(Activation('relu'))(k)
    # k = wave_net_activation(k)
    k = Activation(act_fcn)(k)
    # k = SpatialDropout1D(rate=dout)(k)

    k = Conv1D(
        1,
        kernel_size=1,
        padding="same",
        dilation_rate=1,
        kernel_initializer="he_normal",
    )(k)
    k = Activation(Activation("relu"))(k)
    k = Flatten()(k)

    k = Dense(64)(k)
    k = Activation(LeakyReLU(alpha=alpha))(k)
    # k = Dropout(dout)(k)
    k = Dense(32)(k)
    k = Activation(LeakyReLU(alpha=alpha))(k)
    # k = Dropout(dout)(k)
    k = Dense(16)(k)
    k = Activation(LeakyReLU(alpha=alpha))(k)

    k = Dense(classes)(k)
    k = Activation("softmax")(k)

    sequential_model = Model(inputs=input_sequences, outputs=k)

    return sequential_model


# def recurrent_model_lstm(
#     recognition_model, recurrent_input_shape, classes=4, recurrent_dropout=None
# ):
#     """Recurrent architecture for classification of temporal sequences of
#     images based on LSTMs or GRUs.
#     This architecture is used for IdNet in SIPEC.

#     Parameters
#     ----------
#     recognition_model : keras.model
#         Pretrained recognition model that extracts features for individual frames.
#     recurrent_input_shape : np.ndarray - (Time, Width, Height, Channels)
#         Shape of the images over time.
#     classes : int
#         Number of behaviors to recognise.
#     recurrent_dropout : float
#         Recurrent dropout factor to use.

#     Returns
#     -------
#     keras.model
#         IdNet
#     """
#     input_sequences = Input(shape=recurrent_input_shape)
#     sequential_model_helper = TimeDistributed(recognition_model)(input_sequences)
#     k = BatchNormalization()(sequential_model_helper)

#     dout = 0.2

#     if recurrent_dropout:
#         # TODO: adjust bidirectional
#         k = LSTM(units=128, return_sequences=True, recurrent_dropout=recurrent_dropout)(
#             k
#         )
#         k = LSTM(units=64, return_sequences=True, recurrent_dropout=recurrent_dropout)(
#             k
#         )
#         k = LSTM(units=32, return_sequences=False, recurrent_dropout=recurrent_dropout)(
#             k
#         )

#     else:
#         # As of TF 2, one can just use LSTM and there is no CuDNNGRU
#         k = Bidirectional(GRU(units=128, return_sequences=True))(k)
#         k = Activation(LeakyReLU(alpha=0.3))(k)
#         k = Bidirectional(GRU(units=64, return_sequences=True))(k)
#         k = Activation(LeakyReLU(alpha=0.3))(k)
#         k = Bidirectional(GRU(units=32, return_sequences=False))(k)
#         k = Activation(LeakyReLU(alpha=0.3))(k)

#     # k = Dense(256)(k)
#     # k = Activation('relu')(k)
#     # k = Dropout(dout)(k)

#     # k = Dense(128)(k)
#     # k = Activation("relu")(k)
#     # k = Dropout(dout)(k)
#     # k = Dense(64)(k)

#     k = Dropout(dout)(k)
#     k = Dense(classes)(k)
#     k = Activation("softmax")(k)

#     sequential_model = Model(inputs=input_sequences, outputs=k)

#     return sequential_model

def recurrent_model_lstm(recognition_model, input_shape, classes):
    """
    Build LSTM model that processes sequences using recognition model features.
    Uses modern tf.function approach instead of TimeDistributed to avoid TFOpLambda issues.
    """
    from tensorflow.keras.layers import Input, LSTM, Dense, Lambda
    from tensorflow.keras.models import Model
    import tensorflow as tf
    
    input_sequences = Input(shape=input_shape)
    
    # Extract features from each timestep using recognition model
    def extract_features(sequences):
        """Apply recognition model to each frame in the sequence"""
        # Get shapes
        batch_size = tf.shape(sequences)[0]
        n_timesteps = sequences.shape[1]
        
        # Reshape: (batch, timesteps, h, w, c) -> (batch*timesteps, h, w, c)
        frames = tf.reshape(sequences, [-1] + list(sequences.shape[2:]))
        
        # Apply recognition model to all frames at once
        features = recognition_model(frames, training=False)
        
        # Reshape back: (batch*timesteps, features) -> (batch, timesteps, features)
        n_features = features.shape[-1]
        sequences_features = tf.reshape(features, [batch_size, n_timesteps, n_features])
        
        return sequences_features
    
    # Apply feature extraction
    sequential_model_helper = Lambda(extract_features)(input_sequences)

    # ++++++++++ dropout ++++++++++
    sequential_model_helper = LSTM(
        128, 
        return_sequences=False,
        dropout=0.3,              # ← Add input dropout
        recurrent_dropout=0.2     # ← Add recurrent dropout
    )(sequential_model_helper)
    
    # Add dropout before output
    sequential_model_helper = Dropout(0.3)(sequential_model_helper)  # ← Add this
    # +++++++++++++++++++++++++++++
    # -----------------------------
#     # LSTM layer
#     sequential_model_helper = LSTM(128, return_sequences=False)(sequential_model_helper)
    # -----------------------------
    
    # Output layer
    sequential_model_prediction = Dense(classes, activation="softmax")(sequential_model_helper)
    
    # Build model
    sequential_model = Model(inputs=input_sequences, outputs=sequential_model_prediction)
    
    return sequential_model


# TODO: adaptiv size
# def pretrained_recognition(
#     model_name, input_shape, num_classes, skip_layers=False, 
#     pretrained_weights_path=None, freeze_pretrained=False,
#     keep_pretrained_head=True  # Default to True for backward compatibility
# ):
#     """
#     Build recognition model with optional pretrained weights.

#     Parameters
#     ----------
#     ...
#     keep_pretrained_head : bool
#         If True: Keep existing classification head from checkpoint (fine-tuning)
#         If False: Remove head and add fresh classification head (transfer learning)
#         Only applies when pretrained_weights_path is provided
#     """

#     # === LOAD CUSTOM PRETRAINED WEIGHTS IF PROVIDED ===
#     if pretrained_weights_path is not None:
#         import os
#         import tensorflow as tf

#         if os.path.exists(pretrained_weights_path):
#             print(f"\n{'='*80}")
#             print(f"Loading from: {pretrained_weights_path}")
#             print(f"Keep pretrained head: {keep_pretrained_head}")
#             print(f"Freeze backbone: {freeze_pretrained}")
#             print(f"{'='*80}\n")

#             try:
#                 # Load the model
#                 loaded_model = tf.keras.models.load_model(pretrained_weights_path, compile=False)

#                 # ============ OPTION 1: KEEP PRETRAINED HEAD ============
#                 if keep_pretrained_head:
#                     output_classes = loaded_model.output_shape[-1]

#                     # Verify output matches expected
#                     if output_classes == num_classes:
#                         print(f"✓ Model has {num_classes} output classes (matches config)")
#                         print(f"✓ Using complete model as-is")

#                         # Optionally freeze layers
#                         if freeze_pretrained:
#                             # Freeze everything except final classification layer
#                             for layer in loaded_model.layers[:-2]:  # Keep Dense + Activation trainable
#                                 layer.trainable = False
#                             frozen_count = len([l for l in loaded_model.layers if not l.trainable])
#                             print(f"✓ Froze {frozen_count} layers (keeping final head trainable)")

#                         # Validate no duplicates
#                         dense_count = len([l for l in loaded_model.layers if 'Dense' in str(type(l))])
#                         if dense_count > 1:
#                             print(f"\n⚠ WARNING: Found {dense_count} Dense layers (expected 1)")
#                             print(f"⚠ This checkpoint may have been corrupted by training bug")
#                             raise ValueError(
#                                 "Loaded model has duplicate classification layers. "
#                                 "Use a clean checkpoint or set keep_pretrained_head=False"
#                             )

#                         loaded_model.summary()
#                         return loaded_model

#                     else:
#                         print(f"✗ Model has {output_classes} output classes, config needs {num_classes}")
#                         raise ValueError(
#                             f"Mismatch: loaded model has {output_classes} classes, "
#                             f"but config specifies {num_classes}. "
#                             f"Either:\n"
#                             f"  1. Change num_classes in config to {output_classes}, OR\n"
#                             f"  2. Set keep_pretrained_head=False to replace the head"
#                         )

#                 # ============ OPTION 2: REPLACE HEAD WITH FRESH ONE ============
#                 else:
#                     print(f"✓ Removing pretrained classification head")
#                     print(f"✓ Building fresh classification head for {num_classes} classes")

#                     # Extract backbone by removing last layers
#                     # Typically: backbone → BatchNorm → Dropout → Dense → Activation
#                     # We want to keep up to and including the backbone

#                     # Find where classification head starts
#                     # (after the main backbone, before Dense layer)
#                     dense_layer_idx = None
#                     for i, layer in enumerate(loaded_model.layers):
#                         if 'Dense' in str(type(layer)):
#                             dense_layer_idx = i
#                             break

#                     if dense_layer_idx is None:
#                         raise ValueError("Could not find Dense layer in loaded model")

#                     # Extract backbone (everything before Dense)
#                     # Typically Dense is at -2 (Dense → Activation)
#                     # But might have BatchNorm/Dropout before it

#                     # Safe approach: go back from Dense to find last backbone layer
#                     # Backbone is usually a Functional model (MobileNet, ResNet, etc.)
#                     backbone_output_idx = None
#                     for i in range(dense_layer_idx - 1, -1, -1):
#                         layer = loaded_model.layers[i]
#                         # MobileNet/ResNet appear as Functional models
#                         if 'Functional' in str(type(layer)) or 'Model' in str(type(layer)):
#                             backbone_output_idx = i
#                             break
#                         # Or look for the actual MobileNet/ResNet layer
#                         if any(name in layer.name.lower() for name in ['mobilenet', 'resnet', 'densenet', 'efficientnet']):
#                             backbone_output_idx = i
#                             break

#                     if backbone_output_idx is None:
#                         # Fallback: use layer before first BatchNorm after input
#                         print("⚠ Could not auto-detect backbone, using conservative cutoff")
#                         backbone_output_idx = dense_layer_idx - 3  # Before BatchNorm → Dropout → Dense

#                     print(f"  Extracting backbone: layers 0-{backbone_output_idx}")

#                     # Build new model with fresh head
#                     backbone = tf.keras.Model(
#                         inputs=loaded_model.input,
#                         outputs=loaded_model.layers[backbone_output_idx].output
#                     )

#                     # Freeze backbone if requested
#                     if freeze_pretrained:
#                         for layer in backbone.layers:
#                             layer.trainable = False
#                         print(f"✓ Froze backbone ({len(backbone.layers)} layers)")

#                     # Add fresh classification head
#                     x = backbone.output
#                     x = BatchNormalization()(x)

#                     # Use same dropout as original architecture
#                     if model_name == "mobilenet":
#                         dropout_rate = 0.7
#                     elif model_name == "densenet":
#                         dropout_rate = 0.5
#                     else:
#                         dropout_rate = 0.25

#                     x = Dropout(dropout_rate)(x)
#                     x = Dense(num_classes)(x)
#                     x = Activation("softmax")(x)

#                     new_model = tf.keras.Model(inputs=backbone.input, outputs=x)

#                     print(f"✓ Fresh classification head added ({num_classes} classes)")
#                     new_model.summary()

#                     return new_model

#             except Exception as e:
#                 print(f"ERROR loading pretrained model: {e}")
#                 print("Falling back to ImageNet weights...\n")
#                 pretrained_weights_path = None  # Fall through to standard loading
#         else:
#             print(f"WARNING: Pretrained weights not found at {pretrained_weights_path}")
#             print("Falling back to ImageNet weights...\n")
#             pretrained_weights_path = None

#     # ============ STANDARD IMAGENET LOADING (unchanged) ============
#     if model_name == "mobilenet":
#         recognition_model = MobileNetV3Small(
#             include_top=False,
#             input_shape=(input_shape[0], input_shape[1], 3),
#             pooling="avg",
#             weights="imagenet",
#         )
#     elif model_name == "resnet":
#         recognition_model = ResNet50(
#             include_top=False,
#             input_shape=(input_shape[0], input_shape[1], 3),
#             pooling="avg",
#             weights="imagenet",
#         )
#     elif model_name == "densenet":
#         recognition_model = DenseNet121(
#             include_top=False,
#             input_shape=(input_shape[0], input_shape[1], 3),
#             pooling="avg",
#             weights="imagenet",
#         )
#     # ... other backbones ...
#     else:
#         raise NotImplementedError(f"Backbone {model_name} not supported")

#     # Build full model with head
#     new_input = Input(
#         batch_shape=(None, input_shape[0], input_shape[1], input_shape[2])
#     )

#     x = Conv2D(3, kernel_size=(1, 1), strides=(1, 1))(new_input)
#     x = recognition_model(x)
#     x = BatchNormalization()(x)

#     if model_name == "mobilenet":
#         dout = 0.9
#         x = Dropout(dout)(x)
#     elif model_name == "densenet":
#         dout = 0.5
#         x = Dropout(dout)(x)
#     else:
#         dout = 0.25
#         x = Dropout(dout)(x)

#     x = Dense(num_classes)(x)
#     x = Activation("softmax")(x)

#     recognition_model = Model(inputs=new_input, outputs=x)
#     recognition_model.summary()

#     return recognition_model
def pretrained_recognition(
        model_name, input_shape, num_classes, skip_layers=False, 
        pretrained_weights_path=None, freeze_pretrained=False,
        keep_pretrained_head=True
    ):
        """
        Build recognition model with optional pretrained weights.
        """

        # === LOAD CUSTOM PRETRAINED WEIGHTS IF PROVIDED ===
        if pretrained_weights_path is not None:
            import os
            if os.path.exists(pretrained_weights_path):
                print(f"\n{'='*80}")
                print(f"Loading custom pretrained weights from: {pretrained_weights_path}")
                print(f"Freeze backbone: {freeze_pretrained}")
                print(f"Keep pretrained head: {keep_pretrained_head}")
                print(f"{'='*80}\n")

                import tensorflow as tf
                from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, Dense, Activation
                from tensorflow.keras.models import Model

                try:
                    # Load the pretrained model
                    loaded_model = tf.keras.models.load_model(pretrained_weights_path, compile=False)

                    print(f"✓ Model loaded successfully")
                    print(f"  Input shape: {loaded_model.input_shape}")
                    print(f"  Output shape: {loaded_model.output_shape}")

                    output_size = loaded_model.output_shape[-1]

                    # ============ OPTION 1: KEEP PRETRAINED HEAD ============
                    if keep_pretrained_head:
                        if output_size == num_classes:
                            print(f"✓ Output matches num_classes ({num_classes})")
                            print(f"✓ Using loaded model as-is\n")

                            if freeze_pretrained:
                                for layer in loaded_model.layers[:-2]:
                                    layer.trainable = False
                                print(f"✓ Froze {sum(1 for l in loaded_model.layers if not l.trainable)} layers")

                            loaded_model.summary()
                            return loaded_model
                        else:
                            raise ValueError(
                                f"Output mismatch: model has {output_size} outputs, need {num_classes}. "
                                f"Set keep_pretrained_head=False to add new classification head."
                            )

                    # ============ OPTION 2: USE AS BACKBONE, ADD NEW HEAD ============
                    else:
                        print(f"✓ Using loaded model as backbone (output: {output_size} features)")
                        print(f"✓ Adding fresh classification head ({num_classes} classes)\n")

                        # Use loaded model as backbone
                        backbone = loaded_model

                        # Freeze backbone if requested
                        if freeze_pretrained:
                            for layer in backbone.layers:
                                layer.trainable = False
                            print(f"✓ Froze backbone ({len(backbone.layers)} layers)")

                        # Build new model with classification head
                        new_input = Input(
                            batch_shape=(None, input_shape[0], input_shape[1], input_shape[2])
                        )

                        # Get features from backbone
                        x = backbone(new_input)

                        # Add classification head
#                         x = BatchNormalization()(x)
                        x = tf.keras.layers.LayerNormalization()(x)  # ← TO THIS


                        # Use dropout based on backbone type
                        if model_name == "mobilenet":
                            dropout_rate = 0.7
                        elif model_name == "densenet":
                            dropout_rate = 0.5
                        else:
                            dropout_rate = 0.25

                        x = Dropout(dropout_rate)(x)
                        x = Dense(num_classes)(x)
                        x = Activation("softmax")(x)

                        full_model = Model(inputs=new_input, outputs=x)

                        print(f"✓ Model with new head created")
                        full_model.summary()

                        return full_model

                except Exception as e:
                    print(f"\n{'='*70}")
                    print(f"❌ ERROR LOADING PRETRAINED WEIGHTS")
                    print(f"{'='*70}")
                    print(f"Exception: {type(e).__name__}: {e}")
                    print(f"\nFull traceback:")
                    import traceback
                    traceback.print_exc()
                    print(f"\n⚠ FALLING BACK TO IMAGENET WEIGHTS")
                    print(f"{'='*70}\n")

                    pretrained_weights_path = None
            else:
                print(f"\n❌ Pretrained weights file not found: {pretrained_weights_path}")
                print(f"⚠ FALLING BACK TO IMAGENET WEIGHTS\n")
                pretrained_weights_path = None

        # ============ STANDARD IMAGENET LOADING (fallback) ============
        print(f"\nBuilding model with ImageNet initialization...")

        if model_name == "mobilenet":
            recognition_model = MobileNetV3Small(
                include_top=False,
                input_shape=(input_shape[0], input_shape[1], 3),
                pooling="avg",
                weights="imagenet",
            )
        elif model_name == "densenet":
            recognition_model = DenseNet121(
                include_top=False,
                input_shape=(input_shape[0], input_shape[1], 3),
                pooling="avg",
                weights="imagenet",
            )
        elif model_name == "resnet":
            recognition_model = ResNet50(
                include_top=False,
                input_shape=(input_shape[0], input_shape[1], 3),
                pooling="avg",
                weights="imagenet",
            )
        # ... add other backbones ...
        else:
            raise NotImplementedError(f"Backbone {model_name} not supported")

        # Build full model with classification head
        new_input = Input(
            batch_shape=(None, input_shape[0], input_shape[1], input_shape[2])
        )

        x = Conv2D(3, kernel_size=(1, 1), strides=(1, 1))(new_input)
        x = recognition_model(x)
        x = BatchNormalization()(x)

        if model_name == "mobilenet":
            dout = 0.7
        elif model_name == "densenet":
            dout = 0.5
        else:
            dout = 0.25

        x = Dropout(dout)(x)
        x = Dense(num_classes)(x)
        x = Activation("softmax")(x)

        recognition_model = Model(inputs=new_input, outputs=x)

        print(f"\n✓ Model with ImageNet backbone created")
        recognition_model.summary()

        return recognition_model


# Model with hyperparameters from idtracker.ai


def idtracker_ai(input_shape, classes):
    """Implementation of the idtracker.ai identification module as described in the supplementary of Romero-Ferrero et al.

    Parameters
    ----------
    input_shape : keras compatible input shape (W,H,Channels)
        keras compatible input shape (features,)
    num_classes : int
        Number of behaviors to classify..

    Returns
    -------
    keras.model
        idtracker.ai identification module
    """

    # dropout = 0.2
    # conv model

    model = Sequential()

    model.add(
        Conv2D(
            16,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",
            input_shape=input_shape,
        )
    )
    model.add(Activation("relu"))

    model.add(
        MaxPooling2D(
            strides=(2, 2),
        )
    )

    model.add(
        Conv2D(
            64,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",
            input_shape=input_shape,
        )
    )
    model.add(Activation("relu"))

    model.add(
        MaxPooling2D(
            strides=(2, 2),
        )
    )

    model.add(
        Conv2D(
            100,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",
            input_shape=input_shape,
        )
    )
    model.add(Activation("relu"))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Activation("relu"))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model


# TODO: remove unused code
def SkipConNet(x_train, dropout):
    """
    Args:
        x_train:
        dropout:
    """
    inputs = Input(shape=(x_train.shape[1], 1))

    dout = dropout

    features = 128

    # a layer instance is callable on a tensor, and returns a tensor
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(
        inputs
    )
    x = GaussianNoise(0.5)(x)
    x = BatchNormalization()(x)
    x_1 = Activation("relu")(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x_2 = Activation("relu")(x)
    # x = Dropout(dout)(x_2)

    x = concatenate([x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x_3 = Activation("relu")(x)
    # x = Dropout(dout)(x_3)

    x = concatenate([x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x_4 = Activation("relu")(x)
    # x = Dropout(dout)(x_4)

    x = concatenate([x_4, x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x_5 = Activation("relu")(x)
    # x = Dropout(dout)(x_5)

    x = concatenate([x_5, x_4, x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    # x_6 = Activation('relu')(x)
    # x = Dropout(dout)(x_6)
    x_6 = Activation("relu")(x)

    x = concatenate([x_6, x_5, x_4, x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    # x_6 = Activation('relu')(x)
    # x = Dropout(dout)(x_6)
    x_7 = Activation("relu")(x)

    x = concatenate([x_7, x_6, x_5, x_4, x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    # x_6 = Activation('relu')(x)
    # x = Dropout(dout)(x_6)
    x_8 = Activation("relu")(x)

    x = concatenate([x_8, x_7, x_6, x_5, x_4, x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    # x_6 = Activation('relu')(x)
    # x = Dropout(dout)(x_6)
    x_9 = Activation("relu")(x)

    x = concatenate([x_9, x_7, x_6, x_5, x_4, x_3, x_2, x_1], axis=1)
    x = Dropout(dout)(x)

    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dout)(x)
    x = Conv1D(features, kernel_size=6, strides=2, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    # x_6 = Activation('relu')(x)
    # x = Dropout(dout)(x_6)
    x = Activation("relu")(x)

    x = Flatten()(x)
    x = Dropout(dout)(x)

    x = Dense(1024, kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # x = Dropout(dout)(x)
    x = Dense(1024, kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # x = Dropout(dout)(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    predictions = Dense(3, activation="softmax")(x)
    model_mlp = Model(inputs=inputs, outputs=predictions)

    return model_mlp
