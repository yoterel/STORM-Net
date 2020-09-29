import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
import cv2
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance
from keras_unet.models import vanilla_unet
from keras_unet.models import custom_unet
import matplotlib.pyplot as plt


def split_data(x, y, with_test_set=True):
    if with_test_set:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
        return x_train, x_val, y_train, y_val, x_test, y_test
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
        return x_train, x_val, y_train, y_val


def pairwise(iterable):
    """
    turns an iterable into a pairwise iterable
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    :param iterable:
    :return:
    """
    a = iter(iterable)
    return zip(a, a)


def create_semantic_seg_model(input_shape, learning_rate):
    model = custom_unet(
        input_shape,
        use_batch_norm=True,
        upsample_mode='deconv',
        num_classes=1,
        filters=64,
        dropout=0.2,
        output_activation='sigmoid'
    )
    model.compile(
        # optimizer=Adam(),
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss='binary_crossentropy',
        # loss=jaccard_distance,
        metrics=[iou, iou_thresholded, 'binary_accuracy']
    )
    model.summary()
    return model


def create_fc_model(input_shape, learning_rate):
    model = Sequential()
    model.add(Conv1D(256, 9, activation='relu', input_shape=input_shape))
    model.add(Conv1D(256, 9, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Conv1D(256, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5))
    opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()
    return model


def create_fc2_model(input_shape, output_shape, learning_rate):
    model = Sequential()
    model.add(Conv1D(64, 2, activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, 2, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_shape))
    opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()
    return model


def create_lstm_model(input_shape, learning_rate):
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dense(5))
    opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()
    return model


def shuffle_timeseries(x):
    """
    Shuffles the frames in-place pair-wise for augmentation.
    Each consecutive frames pair is either shuffled, or not, randomly.
    """
    for ndx in np.ndindex(x.shape[0]):
        b = np.reshape(x[ndx], (x.shape[1] // 2, 2, x.shape[2]))
        for ndxx in np.ndindex(b.shape[0]):
            np.random.shuffle(b[ndxx])


def mask_data(x):
    """
    masks 20% of the frames in-place for augmentation.
    """
    percent = 20
    shp = x.shape
    for ndx in np.ndindex(shp[0]):
        a = np.random.choice(shp[1], percent * shp[1] // 100, replace=False)
        x[ndx, a] = np.zeros((shp[2],))
    return


def shuffle_data(x):
    """
    Shuffles the stickers in-place to create orderless data.
    Each one-dimensional slice is shuffled independently.
    """
    b = x
    shp = b.shape[:-1]
    shuf_shp = b.shape[-1]
    for ndx in np.ndindex(shp):
        c = np.reshape(b[ndx], (shuf_shp // 2, 2))  # shuffles in groups of 2 since each sticker has 2 coordinates
        np.random.shuffle(c[3:])  # shuffles only non-facial stickers
    return


def center_data(x):
    """
    centers the stickers in place to create centered data
    """
    b = x
    zero_indices = np.copy(b == 0)
    for ndx in range(b.shape[0]):
        xvec_cent = np.true_divide(b[ndx, :, ::2].sum(1), (b[ndx, :, ::2] != 0).sum(1))
        xvec_cent = np.nan_to_num(xvec_cent)
        yvec_cent = np.true_divide(b[ndx, :, 1::2].sum(1), (b[ndx, :, 1::2] != 0).sum(1))
        yvec_cent = np.nan_to_num(yvec_cent)
        b[ndx, :, ::2] += np.expand_dims(0.5 - xvec_cent, axis=1)
        b[ndx, :, 1::2] += np.expand_dims(0.5 - yvec_cent, axis=1)
    b[zero_indices] = 0
    return


def perturb_data(x):
    """
    NOTE: doesn't work - results aren't normalized, scaled, or clamped to [0-1]
    perturbs the stickers in-place for augmentation.
    Each sticker is perturbed separately
    """
    mag = 1.
    my_perturbation = np.random.normal(size=x.shape)
    x += my_perturbation * mag
    return
# def scale_data(X):
    # timesteps_per_sample = 40
    # number_of_features = 27
    # X = X
    # x_reshaped = np.reshape(X, (len(X)*timesteps_per_sample, number_of_features))
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = scaler.fit(x_reshaped)
    # x_scaled = scaler.transform(x_reshaped)
    # x_scaled_reshaped = np.reshape(x_scaled, (len(x_scaled) // timesteps_per_sample, timesteps_per_sample, number_of_features))
    # # invert transform
    # x_inverted = scaler.inverse_transform(x_scaled)
    # x_inverted_reshaped = np.reshape(x_inverted, (len(x_inverted)//timesteps_per_sample, timesteps_per_sample, number_of_features))
    # return x_scaled_reshaped, x_inverted_reshaped, scaler


class DataGenerator(keras.utils.Sequence):
    """Generates data for rotation estimation task"""
    def __init__(self,
                 X,
                 Y,
                 batch_size=16,
                 dim=(40, 18),
                 shuffle_batches=True,
                 shuffle_timestamps=True,
                 shuffle_stickers=True,
                 mask_stickers=True,
                 center_stickers=True,
                 perturb_stickers=False):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.X = X
        self.Y = Y
        self.shuffle_batches = shuffle_batches
        self.shuffle_timestamps = shuffle_timestamps
        self.shuffle_stickers = shuffle_stickers
        self.perturb_stickers = perturb_stickers
        self.center_stickers = center_stickers
        self.mask_stickers = mask_stickers
        self.on_epoch_end()
        self.indexes = np.arange(len(self.X))
        if self.shuffle_batches:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.X))
        if self.shuffle_batches:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indices):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, np.shape(self.Y)[1]))

        # Generate data
        for i, index in enumerate(indices):
            # Store sample
            X[i] = self.X[index]

            # Store class
            y[i] = self.Y[index]
        if self.shuffle_timestamps:
            shuffle_timeseries(X)
        if self.shuffle_stickers:
            shuffle_data(X)
        if self.perturb_stickers:
            perturb_data(X)
        if self.mask_stickers:
            mask_data(X)
        if self.center_stickers:
            center_data(X)
        return X, y


def preprocess_semantic_labels(orig_label_folder, train_label_dir):
    train_label_dir.mkdir(parents=True, exist_ok=True)
    for file in sorted(orig_label_folder.glob("*")):
        img = cv2.imread(str(file), 0)
        img[img == 1] = 0
        img[img == 3] = 0
        img[img == 2] = 1
        dst_file = Path.joinpath(train_label_dir, file.name)
        cv2.imwrite(str(dst_file), img)
        # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        # num_of_classes = np.count_nonzero(hist)
        # print(file, num_of_classes)


def get_patches(img_arr, size=256, stride=256):
    """
    Takes single image or array of images and returns
    crops using sliding window method.
    If stride < size it will do overlapping.
    """
    # check size and stride
    if size % stride != 0:
        raise ValueError('size % stride must be equal 0')

    patches_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1

    if img_arr.ndim == 3 or img_arr.ndim == 2:
        i_max = img_arr.shape[0] // stride - overlapping
        j_max = img_arr.shape[1] // stride - overlapping
        for i in range(i_max):
            for j in range(j_max):
                # print(i*stride, i*stride+size)
                # print(j*stride, j*stride+size)
                patches_list.append(
                    img_arr[i * stride:i * stride + size,
                    j * stride:j * stride + size
                    ])

    elif img_arr.ndim == 4:
        i_max = img_arr.shape[0] // stride - overlapping
        j_max = img_arr.shape[1] // stride - overlapping
        for im in img_arr:
            for i in range(i_max):
                for j in range(j_max):
                    # print(i*stride, i*stride+size)
                    # print(j*stride, j*stride+size)
                    patches_list.append(
                        im[i * stride:i * stride + size,
                        j * stride:j * stride + size
                        ])

    else:
        raise ValueError('img_arr.ndim must be equal 2, 3 or 4')

    return np.stack(patches_list)


def get_local_range(x, local_env_size, frames_to_use):
    """
    gets non negative local envrionment of indices around x given size of environment and maximum frames
    :param x: the index about we find a local environment
    :param local_env_size: the size of the local environment
    :param frames_to_use: maximum index allowed
    :return:
    """
    diff_minus = max(x - local_env_size // 2, 0)
    diff_plus = min(x + local_env_size // 2, frames_to_use)
    if diff_plus - diff_minus != (local_env_size - 1):
        if diff_minus == 0:
            diff_plus -= x - local_env_size // 2
        else:
            assert (1 == 0)
    return diff_minus, diff_plus