import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


class DataGenerator(tf.keras.utils.Sequence):
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
        with np.errstate(all='ignore'):  # we replace nans with zero immediately after possible division by zero
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


def mask_facial_landmarks(x):
    """
    masks facial landmarks in-place in every frame of x if less than 3 of them are present in it (per frame)
    note: masking in this sense means to zeroify the data.
    :param x: the np tensor representing the data (shaped batch_size x 10 x 14)
    :return: None
    """
    for batch in range(x.shape[0]):
        b = np.reshape(x[batch], (x.shape[1], x.shape[2] // 2, 2))  # reshape to 10 x 7 x 2
        c = np.all(b == 0, axis=2)  # find stickers that are zeros
        for frame in range(c.shape[0]):  # iterate over frames
            if c[frame, 0] or c[frame, 1] or c[frame, 2]:  # if one of the first 3 stickers is 0, set all of them to 0
                b[frame, :3] = 0


def get_augmented(X_train,
                  Y_train,
                  X_val=None,
                  Y_val=None,
                  batch_size=32,
                  seed=0,
                  data_gen_args=None):
    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    if data_gen_args is None:
        data_gen_args = dict(rotation_range=10.,
                             # width_shift_range=0.02,
                             height_shift_range=0.02,
                             shear_range=5,
                             # zoom_range=0.3,
                             horizontal_flip=True,
                             vertical_flip=False,
                             fill_mode='constant'
                             )
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)

    train_generator = zip(X_train_augmented, Y_train_augmented)

    if not (X_val is None) and not (Y_val is None):
        # Validation data, no data augmentation, but we create a generator anyway
        X_datagen_val = ImageDataGenerator(**data_gen_args)
        Y_datagen_val = ImageDataGenerator(**data_gen_args)
        X_datagen_val.fit(X_val, augment=True, seed=seed)
        Y_datagen_val.fit(Y_val, augment=True, seed=seed)
        X_val_augmented = X_datagen_val.flow(X_val, batch_size=batch_size, shuffle=True, seed=seed)
        Y_val_augmented = Y_datagen_val.flow(Y_val, batch_size=batch_size, shuffle=True, seed=seed)

        # combine generators into one which yields image and masks
        val_generator = zip(X_val_augmented, Y_val_augmented)

        return train_generator, val_generator
    else:
        return train_generator