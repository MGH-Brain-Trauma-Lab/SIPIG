# SIPEC
# MARKUS MARKS
# Dataloader
import multiprocessing
import pickle

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from skimage.registration import optical_flow_tvl1
from sklearn import preprocessing
from sklearn.externals._pilutil import imresize
from sklearn.utils import class_weight
from tensorflow import keras
from tqdm import tqdm

import tensorflow as tf


# adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        x_train,
        y_train,
        look_back,
        type="recognition",
        batch_size=32,
        shuffle=True,
        temporal_causal=False,
    ):
        # self.dim = dim
        self.batch_size = batch_size
        self.look_back = look_back
        self.list_IDs = np.array(range(self.look_back, len(x_train) - self.look_back))
        # self.n_channels = n_channels
        self.shuffle = shuffle
        # self.augmentation = augmentation
        self.type = type
        self.x_train = x_train
        self.y_train = y_train
        self.dlc_train_flat = None
        self.temporal_causal = temporal_causal

        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []

        # Generate data
        for _, ID in enumerate(list_IDs_temp):
            if self.type == "recognition":
                X.append(self.x_train[ID])
            else:
                if self.temporal_causal:
                    X.append(self.x_train[ID - (2 * self.look_back) : ID])
                else:
                    X.append(self.x_train[ID - self.look_back : ID + self.look_back])

            y.append(self.y_train[ID])
            # _y = self.y_train[ID - self.look_back: ID + self.look_back]
            # y.append(self.label_encoder.transform(_y))

        return np.asarray(X).astype("float32"), np.asarray(y).astype("int")


def create_dataset(dataset, look_back=5, oneD=False):
    # """Create a recurrent dataset from array.
    # Args:
    #     dataset: Numpy/List of dataset.
    #     look_back: Number of future/past timepoints to add to current timepoint.
    #     oneD: Boolean flag whether data is one dimensional or not.
    # """
    """Summary line.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        dataset
    """
    dataX = []
    print("creating recurrency")
    for i in tqdm(range(look_back, len(dataset) - look_back)):
        if oneD:
            a = dataset[i - look_back : i + look_back]
        else:
            a = dataset[i - look_back : i + look_back, :]
        dataX.append(a)
    return np.array(dataX)


class Dataloader:
    def __init__(self, x_train, y_train, x_test, y_test, config=None):
        """
        Args:
            x_train: Either numpy array of images OR list of file paths (for streaming)
            y_train: Labels for training data
            x_test: Either numpy array of images OR list of file paths (for streaming)
            y_test: Labels for test/validation data
            config: Configuration dictionary
        """
        # Detect if we're in streaming mode (paths vs data)
        self.streaming_mode = False
        if isinstance(x_train, (list, np.ndarray)) and len(x_train) > 0:
            if isinstance(x_train[0], str):
                # Streaming mode - we have file paths
                self.streaming_mode = True
                self.train_paths = list(x_train)
                self.val_paths = list(x_test)
                self.train_labels_raw = list(y_train)
                self.val_labels_raw = list(y_test)
                self.x_train = None  # Don't load into memory
                self.x_test = None
                print("Dataloader initialized in STREAMING mode (paths only)")
            else:
                # Traditional mode - we have actual data
                self.x_train = x_train
                self.x_test = x_test
                self.streaming_mode = False
                print("Dataloader initialized in TRADITIONAL mode (data in memory)")
        else:
            self.x_train = x_train
            self.x_test = x_test
            self.streaming_mode = False
        
        self.y_train = y_train
        self.y_test = y_test
        
        # Rest of the original __init__ code stays the same
        self.config = config if config else {}
        self.look_back = self.config.get("look_back", 1)
        
        # Initialize other attributes
        self.training_generator = None
        self.validation_generator = None
        self.label_encoder = None
        self.num_classes = None

        self.label_encoder = None

        self.x_train_recurrent = None
        self.x_test_recurrent = None
        self.y_train_recurrent = None
        self.y_test_recurrent = None

        self.use_generator = False

    def encode_label(self, label):
        return self.label_encoder.transform(label)

    def encode_labels(self):
        self.label_encoder = preprocessing.LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(self.y_train)
        self.y_test = self.label_encoder.transform(self.y_test)

    def decode_labels(self, labels):
        """
        Args:
            labels:
        """
        decoded = self.label_encoder.inverse_transform(labels)
        return decoded

    def categorize_data(self, num_classes, recurrent=False):
        """
        Args:
            num_classes:
            recurrent:
        """
        self.y_train = self.y_train.astype(int)
        # TODO: parametrize num behaviors
        self.y_train = keras.utils.to_categorical(
            self.y_train, num_classes=num_classes, dtype="int"
        )

        self.y_test = self.y_test.astype(int)
        self.y_test = keras.utils.to_categorical(
            self.y_test, num_classes=num_classes, dtype="int"
        )

        if recurrent:
            self.y_train_recurrent = self.y_train_recurrent.astype(int)
            # TODO: parametrize num behaviors
            self.y_train_recurrent = keras.utils.to_categorical(
                self.y_train_recurrent, num_classes=num_classes, dtype="int"
            )

            self.y_test_recurrent = self.y_test_recurrent.astype(int)
            self.y_test_recurrent = keras.utils.to_categorical(
                self.y_test_recurrent, num_classes=num_classes, dtype="int"
            )

    def normalize_data(self, mode="default"):
        if mode == "default":
            # TODO: double check this here
            # self.mean = self.x_train[1000:-1000].mean(axis=0)
            # self.std = np.std(self.x_train[1000:-1000], axis=0)
            self.mean = self.x_train.mean(axis=0)
            self.std = np.std(self.x_train, axis=0)
            self.x_train = self.x_train - self.mean
            self.x_train /= self.std
            self.x_test = self.x_test - self.mean
            self.x_test /= self.std

            if not self.dlc_train is None:
                self.mean_dlc = self.dlc_train.mean(axis=0)
                self.std_dlc = self.dlc_train.std(axis=0)
                self.dlc_train -= self.mean_dlc
                self.dlc_test -= self.mean
                self.dlc_train /= self.std_dlc
                self.dlc_test /= self.std_dlc
        elif mode == "xception":
            self.x_train /= 127.5
            self.x_train -= 1.0
            self.x_test /= 127.5
            self.x_test -= 1.0

            if not self.dlc_train is None:
                self.dlc_train /= 127.5
                self.dlc_train -= 1.0
                self.dlc_test /= 127.5
                self.dlc_test -= 1.0

        else:
            self.x_train /= 255.0
            self.x_test /= 255.0
            if not self.dlc_train is None:
                self.dlc_train /= 255.0
                self.dlc_test /= 255.0


    def create_dataset(dataset, oneD, look_back=5):
        """
        Args:
            oneD:
            look_back:
        """
        dataX = []
        for i in range(look_back, len(dataset) - look_back):
            if oneD:
                a = dataset[i - look_back : i + look_back]
            else:
                a = dataset[i - look_back : i + look_back, :]
            dataX.append(a)
        return np.array(dataX)

    def create_recurrent_labels(self, only_test=False):

        if only_test:
            self.y_test_recurrent = self.y_test[self.look_back : -self.look_back]
            self.y_test = self.y_test[self.look_back : -self.look_back]
        else:
            self.y_train_recurrent = self.y_train[self.look_back : -self.look_back]
            self.y_test_recurrent = self.y_test[self.look_back : -self.look_back]

            self.y_train = self.y_train[self.look_back : -self.look_back]
            self.y_test = self.y_test[self.look_back : -self.look_back]

    def create_recurrent_data(self, oneD=False, recurrent_labels=True, only_test=False):
        """
        Args:
            oneD:
        """
        if only_test:
            self.x_test_recurrent = create_dataset(
                self.x_test, self.look_back, oneD=oneD
            )
        else:
            self.x_train_recurrent = create_dataset(
                self.x_train, self.look_back, oneD=oneD
            )
            self.x_test_recurrent = create_dataset(
                self.x_test, self.look_back, oneD=oneD
            )

        # also shorten normal data so all same length
        if only_test:
            self.x_test = self.x_test[self.look_back : -self.look_back]
        else:
            self.x_test = self.x_test[self.look_back : -self.look_back]
            self.x_train = self.x_train[self.look_back : -self.look_back]

        if recurrent_labels:
            self.create_recurrent_labels(only_test=only_test)

    def create_recurrent_data_dlc(self, recurrent_labels=True):

        self.dlc_train_recurrent = create_dataset(self.dlc_train, self.look_back)
        self.dlc_test_recurrent = create_dataset(self.dlc_test, self.look_back)

        # also shorten normal data so all same length
        self.dlc_train = self.dlc_train[self.look_back : -self.look_back]
        self.dlc_test = self.dlc_test[self.look_back : -self.look_back]

        if recurrent_labels:
            self.create_recurrent_labels()

    # TODO: redo all like this, i.e. gettters instead of changing data
    def expand_dims(self):
        self.x_train = np.expand_dims(self.x_train, axis=-1)
        self.x_test = np.expand_dims(self.x_test, axis=-1)
        if self.x_test_recurrent is not None:
            self.x_train_recurrent = np.expand_dims(self.x_train_recurrent, axis=-1)
            self.x_test_recurrent = np.expand_dims(self.x_test_recurrent, axis=-1)

    def create_flattened_data(self):
        if self.with_dlc:
            _shape = self.dlc_train.shape
            self.dlc_train_flat = self.dlc_train.reshape(
                (_shape[0], _shape[1] * _shape[2])
            )
            _shape = self.dlc_test.shape
            self.dlc_test_flat = self.dlc_test.reshape(
                (_shape[0], _shape[1] * _shape[2])
            )

            _shape = self.dlc_train_recurrent.shape
            self.dlc_train_recurrent_flat = self.dlc_train_recurrent.reshape(
                (_shape[0], _shape[1] * _shape[2] * _shape[3])
            )
            _shape = self.dlc_test_recurrent.shape
            self.dlc_test_recurrent_flat = self.dlc_test_recurrent.reshape(
                (_shape[0], _shape[1] * _shape[2] * _shape[3])
            )

    def decimate_labels(self, percentage, balanced=False):
        """decimate labels to a given percentate percentage in [0,1] :return:

        Args:
            percentage:
            balanced:
        """
        if balanced:
            # TODO: do w class weights and probability in choice fcn
            raise NotImplementedError
        if self.x_train_recurrent is not None:
            num_labels = int(len(self.x_train_recurrent) * percentage)
            indices = np.arange(0, len(self.x_train_recurrent) - 1)
            random_idxs = np.random.choice(indices, size=num_labels, replace=False)
            self.x_train = self.x_train[random_idxs]
            self.y_train = self.y_train[random_idxs]
            self.x_train_recurrent = self.x_train_recurrent[random_idxs]
            self.y_train_recurrent = self.y_train_recurrent[random_idxs]
        if self.config["train_ours"] or self.config["train_ours_plus_dlc"]:
            num_labels = int(len(self.x_train) * percentage)
            indices = np.arange(0, len(self.x_train))
            random_idxs = np.random.choice(indices, size=num_labels, replace=False)
            self.x_train = self.x_train[random_idxs]
            self.y_train = self.y_train[random_idxs]
        if self.dlc_train is not None:
            num_labels = int(len(self.dlc_train) * percentage)
            indices = np.arange(0, len(self.dlc_train))
            random_idxs = np.random.choice(indices, size=num_labels, replace=False)
            self.dlc_train = self.dlc_train[random_idxs]
            self.dlc_train_flat = self.dlc_train_flat[random_idxs]
            # self.y_train = self.y_train[random_idxs]
        if hasattr(self, "dlc_train_recurrent"):
            self.dlc_train_recurrent = self.dlc_train_recurrent[random_idxs]
            self.dlc_train_recurrent_flat = self.dlc_train_recurrent_flat[random_idxs]
            # self.y_train_recurrent = self.y_train_recurrent[random_idxs]

    # old
    def reduce_labels(self, behavior, num_labels):

        """
        Args:
            behavior:
            num_labels:
        """
        idx_behavior = self.y_train == behavior
        idx_behavior = np.asarray(idx_behavior)
        idx_behavior_true = np.where(idx_behavior == 1)[0]

        # select only a subset of labels for the behavior
        selected_labels = np.random.choice(idx_behavior_true, num_labels, replace=False)

        self.y_train = ["none"] * len(idx_behavior)
        self.y_train = np.asarray(self.y_train)
        self.y_train[selected_labels] = behavior

        idx_behavior = self.y_test == behavior
        idx_behavior = np.asarray(idx_behavior)
        idx_behavior_true = np.where(idx_behavior == 1)[0]
        self.y_test = ["none"] * len(idx_behavior)
        self.y_test = np.asarray(self.y_test)
        self.y_test[idx_behavior_true] = behavior

    def remove_behavior(self, behavior):
        """
        Args:
            behavior:
        """
        idx_behavior = self.y_test == behavior
        idx_behavior = np.asarray(idx_behavior)
        self.y_test[idx_behavior] = "none"

        idx_behavior = self.y_train == behavior
        idx_behavior = np.asarray(idx_behavior)
        self.y_train[idx_behavior] = "none"

    def undersample_data(self):
        random_under_sampler = RandomUnderSampler(
            0.2, sampling_strategy="majority", random_state=42
        )

        shape = self.x_train.shape
        if len(shape) == 2:
            self.x_train, self.y_train = random_under_sampler.fit_sample(
                self.x_train, self.y_train
            )

        if len(shape) == 3:
            self.x_train = self.x_train.reshape((shape[0], shape[1] * shape[2]))
            self.x_train, self.y_train = random_under_sampler.fit_sample(
                self.x_train, self.y_train
            )
            self.x_train = self.x_train.reshape(
                (self.x_train.shape[0], shape[1], shape[2])
            )
            self.x_train = np.expand_dims(self.x_train, axis=-1)
        if len(shape) == 4:
            self.x_train = self.x_train.reshape(
                (shape[0], shape[1] * shape[2] * shape[3])
            )
            self.x_train, self.y_train = random_under_sampler.fit_resample(
                self.x_train, self.y_train
            )
            self.x_train = self.x_train.reshape(
                (self.x_train.shape[0], shape[1], shape[2], shape[3])
            )
        else:
            raise NotImplementedError

        # TODO: undersample recurrent

    def change_dtype(self, dtype="uint8"):
        self.x_train = np.asarray(self.x_train, dtype=dtype)
        self.x_test = np.asarray(self.x_test, dtype=dtype)

    def get_input_shape(self, recurrent=False):
        """
        Get input shape for model architecture.
        Works in both streaming and traditional modes.
        """
        if self.streaming_mode:
            # In streaming mode, use target_size from config
            height = self.target_size[0] if hasattr(self, 'target_size') else 200
            width = self.target_size[1] if hasattr(self, 'target_size') else 200
            channels = 3
            if recurrent:
                return (self.look_back * 2, height, width, channels)
            return (height, width, channels)
        else:
            # Traditional mode - use actual data shape
            if recurrent:
                if hasattr(self, 'x_train_recurrent') and self.x_train_recurrent is not None:
                    return self.x_train_recurrent.shape[1:]
            if self.x_train is not None:
                return self.x_train.shape[1:]
            return (200, 200, 3)  # Default fallback

    # TODO: parallelize
    def downscale_frames(self, factor=0.5):
        im_re = []
        for el in tqdm(self.x_train):
            im_re.append(imresize(el, factor))
        self.x_train = np.asarray(im_re)
        im_re = []
        for el in tqdm(self.x_test):
            im_re.append(imresize(el, factor))
        self.x_test = np.asarray(im_re)

    def prepare_data(
        self, downscale=0.5, remove_behaviors=[], flatten=False, recurrent=False, normalization_mode='default'
    ):
        print("preparing data")
        print("changing dtype")
        self.change_dtype()

        print("removing behaviors")
        for behavior in remove_behaviors:
            self.remove_behavior(behavior=behavior)
        print("downscaling")
        if downscale:
            self.downscale_frames(factor=downscale)
        print("normalizing data")
        if self.config["normalize_data"]:
            self.normalize_data()
        print("doing flow")
        if self.config["do_flow"]:
            self.create_flow_data()
        print("encoding labels")
        if self.config["encode_labels"]:
            print("test")
            self.encode_labels()
        print("labels encoded")
        print("using class weights")
        if self.config["use_class_weights"]:
            print("calc class weights")
            self.class_weights = class_weight.compute_class_weight(
                "balanced", np.unique(self.y_train), self.y_train
            )
        if self.config["undersample_data"]:
            print("undersampling data")
            self.undersample_data()
        print("using generator")
        if self.config["use_generator"]:
            self.categorize_data(self.num_classes, recurrent=recurrent)
        else:
            print("preparing recurrent data")
            self.create_recurrent_data()
            print("preparing flattened data")
            if flatten:
                self.create_flattened_data()
            print("categorize data")
            self.categorize_data(self.num_classes, recurrent=recurrent)

        print("data ready")

    def flow_single(self, fr1, fr2):
        if len(fr1.shape) > 2:
            fr2 = fr2[:, :, 0]
            fr1 = fr1[:, :, 0]
        v, u = optical_flow_tvl1(fr1, fr2)
        v = np.expand_dims(v, axis=-1)
        u = np.expand_dims(u, axis=-1)
        s = np.stack([u, v], axis=2)[:, :, :, 0]
        return v

    def do_flow(self, videodata, num_cores=(multiprocessing.cpu_count())):

        flow_data = Parallel(n_jobs=num_cores)(
            delayed(self.flow_single)(videodata[i], videodata[i - 1])
            for i in tqdm(range(1, len(videodata)))
        )
        flow_data = np.array(flow_data)
        flow_data = np.vstack([np.expand_dims(flow_data[0], axis=0), flow_data])

        return flow_data

    def create_flow_data(self):

        flow_data_train = self.do_flow(self.x_train)
        # self.x_train = np.stack([flow_data_train, self.x_train])
        a = np.swapaxes(flow_data_train, 0, -1)
        b = np.swapaxes(self.x_train, 0, -1)
        self.x_train = np.swapaxes(np.vstack([a, b]), 0, -1)
        # self.x_train = self.x_train[:, :, :, 0, :]

        flow_data_test = self.do_flow(self.x_test)
        # self.x_test = np.stack([flow_data_test, self.x_test], axis=-1)
        a = np.swapaxes(flow_data_test, 0, -1)
        b = np.swapaxes(self.x_test, 0, -1)
        self.x_test = np.swapaxes(np.vstack([a, b]), 0, -1)
        # self.x_test = self.x_test[:, :, :, 0, :]

        pass

    def expand_dims(self, axis=-1):

        self.x_train = np.expand_dims(self.x_train, axis=axis)
        self.x_test = np.expand_dims(self.x_test, axis=axis)

    def save(self, path):
        self.x_train = None
        self.x_train_recurrent = None
        self.x_test = None
        self.x_test_recurrent = None
        self.y_train = None
        self.y_train_recurrent = None
        self.y_test = None
        self.y_test_recurrent = None
        pickle.dump(self, open(path, "wb"))

    def get_num_classes(self):
        return self.num_classes
    
    def prepare_streaming_data(self, target_size=(200, 200)):
        """
        Prepare data for streaming mode - only encodes labels.
        Does NOT load images into memory.
        
        Args:
            target_size: Target image size (used by generator, not here)
        """
        if not self.streaming_mode:
            raise ValueError("prepare_streaming_data() only works in streaming mode")
        
        print("Preparing streaming data (encoding labels only)...")
        
        # Store target size for generators
        self.target_size = target_size
        
        # Encode labels
        from sklearn import preprocessing
        self.label_encoder = preprocessing.LabelEncoder()
        all_labels = self.train_labels_raw + self.val_labels_raw
        self.label_encoder.fit(all_labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Number of classes: {self.num_classes}")
        
        # Store for compatibility
        self.y_train = self.label_encoder.transform(self.train_labels_raw)
        self.y_test = self.label_encoder.transform(self.val_labels_raw)

    
# =====================================================================
# STREAMING DATA GENERATOR (for memory-efficient training)
# =====================================================================

import cv2

class StreamingDataGenerator(tf.keras.utils.Sequence):
    """
    Generator that loads clips from disk on-demand.
    Memory efficient - only loads one batch at a time.
    Supports both single-frame (recognition) and sequential (LSTM) modes.
    """
    
    def __init__(
        self, 
        clip_paths, 
        labels, 
        label_encoder,
        num_classes,
        batch_size=16, 
        target_size=(200, 200),
        shuffle=True,
        augmentation=None,
        normalize=True,
        mode='recognition',  # NEW: 'recognition' or 'sequential'
        look_back=5,  # NEW: for sequential mode
        temporal_causal=False,  # NEW: for sequential mode
        frames_per_video=1,
    ):
        """
        Args:
            clip_paths: List of file paths to video/image files
            labels: List of string labels corresponding to clips
            label_encoder: sklearn LabelEncoder for encoding labels
            num_classes: Number of classes
            batch_size: Batch size
            target_size: (height, width) to resize frames to
            shuffle: Whether to shuffle at end of epoch
            augmentation: Optional augmentation function
            normalize: Whether to normalize to [0, 1]
            mode: 'recognition' (single frame) or 'sequential' (frame sequences)
            look_back: Number of frames before/after center frame for sequential mode
            temporal_causal: If True, only use past frames in sequential mode
            frames_per_video: Number of frames to sample from each video when streaming
        """
        self.clip_paths = np.array(clip_paths)
        self.labels = labels
        self.label_encoder = label_encoder
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.normalize = normalize
        self.mode = mode
        self.look_back = look_back
        self.temporal_causal = temporal_causal
        self.frames_per_video = frames_per_video
        
        if frames_per_video > 1 and mode == 'recognition':
            print(f"Expanding dataset: {frames_per_video} frames per video")
            expanded_paths = []
            expanded_labels = []
            for path, label in zip(clip_paths, labels):
                for _ in range(frames_per_video):
                    expanded_paths.append(path)
                    expanded_labels.append(label)

            self.clip_paths = np.array(expanded_paths)
            self.labels = expanded_labels
            self.label_encoder = label_encoder
            self.encoded_labels = label_encoder.transform(expanded_labels)

            print(f"Dataset expanded: {len(clip_paths)} videos → {len(self.clip_paths)} samples")
        else:
            # Original behavior for sequential mode or frames_per_video=1
            self.clip_paths = np.array(clip_paths)
            self.labels = labels
            self.label_encoder = label_encoder
            self.encoded_labels = label_encoder.transform(labels)

        # Create indices for shuffling
        self.indices = np.arange(len(self.clip_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.clip_paths) / self.batch_size))
    
    def _load_frame_from_path(self, path, frame_idx=None):
        """
        Load a single frame from various file types.
        
        Args:
            path: File path
            frame_idx: For videos, which frame to extract (None = random/middle)
        
        Returns:
            frame as numpy array (H, W, C)
        """
        if path.endswith('.npy'):
            # Load numpy array (single frame)
            frame = np.load(path)
        
        elif path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Load image file
            frame = cv2.imread(path)
            if frame is None:
                raise ValueError(f"Could not load image: {path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        elif path.endswith(('.mp4', '.avi', '.mov')):
            # Load frame from video
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {path}")
            
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                raise ValueError(f"Video has no frames: {path}")
            
            # Determine which frame to read
            if frame_idx is not None:
                # Use specified frame index
                use_frame_idx = min(frame_idx, total_frames - 1)
            elif self.shuffle and self.mode == 'recognition':
                # Random frame for training
                use_frame_idx = np.random.randint(0, total_frames)
            else:
                # Middle frame for validation or as default
                use_frame_idx = total_frames // 2
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, use_frame_idx)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError(f"Could not read frame {use_frame_idx} from video: {path}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        
        return frame
    
    def _load_sequence_from_video(self, path):
        """
        Load a sequence of frames from a video for sequential models.
        
        Args:
            path: Video file path
        
        Returns:
            sequence of frames: (timesteps, H, W, C)
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has no frames: {path}")
        
        # Determine sequence parameters
        if self.temporal_causal:
            # Only past frames: need look_back*2 frames total
            required_frames = self.look_back * 2
        else:
            # Past and future: need look_back*2 frames (look_back before + look_back after)
            required_frames = self.look_back * 2
        
        # Pick a center frame
        if self.shuffle and self.mode == 'sequential':
            # Random center frame (but ensure we have enough frames around it)
            min_center = self.look_back if not self.temporal_causal else required_frames
            max_center = total_frames - self.look_back
            if max_center <= min_center:
                center_frame = total_frames // 2
            else:
                center_frame = np.random.randint(min_center, max_center)
        else:
            # Use middle frame
            center_frame = total_frames // 2
        
        # Determine frame indices to extract
        if self.temporal_causal:
            # Only past frames
            start_frame = max(0, center_frame - required_frames)
            end_frame = center_frame
        else:
            # Past and future frames
            start_frame = max(0, center_frame - self.look_back)
            end_frame = min(total_frames, center_frame + self.look_back)
        
        # Extract frames
        sequence = []
        for frame_idx in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sequence.append(frame)
        
        cap.release()
        
        # Pad sequence if needed (in case video is too short)
        while len(sequence) < required_frames:
            if len(sequence) > 0:
                sequence.append(sequence[-1])  # Repeat last frame
            else:
                # Create black frame if video was completely empty
                sequence.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Truncate if too long
        sequence = sequence[:required_frames]
        
        return np.array(sequence)
    
    def __getitem__(self, idx):
        """
        Generate one batch of data.
        Loads frames from video files on-demand.
        """
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get paths and labels for this batch
        batch_paths = self.clip_paths[batch_indices]
        batch_labels = self.encoded_labels[batch_indices]
        
        # Load data based on mode
        if self.mode == 'recognition':
            # Load single frames
            batch_frames = []
            for path in batch_paths:
                frame = self._load_frame_from_path(path)
                
                # Resize
                if frame.shape[:2] != self.target_size:
                    frame = cv2.resize(frame, self.target_size)
                
                # Apply augmentation if provided
                if self.augmentation is not None:
                    frame = self.augmentation(image=frame)
                
                # Normalize
                if self.normalize:
                    frame = frame.astype('float32') / 255.0
                
                batch_frames.append(frame)
            
            X = np.array(batch_frames, dtype='float32')
        
        elif self.mode == 'sequential':
            # Load sequences of frames
            batch_sequences = []
            for path in batch_paths:
                sequence = self._load_sequence_from_video(path)
                
                # Resize each frame in sequence
                resized_sequence = []
                for frame in sequence:
                    if frame.shape[:2] != self.target_size:
                        frame = cv2.resize(frame, self.target_size)
                    
                    # Apply augmentation if provided
                    if self.augmentation is not None:
                        frame = self.augmentation(image=frame)
                    
                    # Normalize
                    if self.normalize:
                        frame = frame.astype('float32') / 255.0
                    
                    resized_sequence.append(frame)
                
                batch_sequences.append(resized_sequence)
            
            X = np.array(batch_sequences, dtype='float32')
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # One-hot encode labels
        y = tf.keras.utils.to_categorical(batch_labels, num_classes=self.num_classes)
        
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indices at end of each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
            