import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import data_augmentations


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
        if self.perturb_stickers:
            perturb_data(X)
        if self.shuffle_timestamps:
            shuffle_timeseries(X)
        if self.shuffle_stickers:
            shuffle_data(X)
        if self.mask_stickers:
            mask_data(X)
        if self.center_stickers:
            data_augmentations.center_data(X)
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





def perturb_data(x):
    """
    NOTE: doesn't work - results aren't normalized, scaled, or clamped to [0-1]
    perturbs the stickers in-place for augmentation.
    Each sticker is perturbed separately
    """
    b_shape = x.shape[0]
    t_shape = x.shape[1]
    b = np.copy(x)
    another_view = np.reshape(b, (b_shape, t_shape, b.shape[-1] // 2, 2))
    zero_locs = np.where(another_view == np.array([0, 0]))
    noise_mag = 5
    noise_shift = noise_mag / 2
    noise = (np.random.random_sample(b.shape) * noise_mag) - noise_shift
    b += noise
    another_view[zero_locs] = 0
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