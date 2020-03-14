import json
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "5";
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


def load_semantic_seg_model(weights_loc):
    old_model = keras.models.load_model(weights_loc,
                                        custom_objects={'iou': iou, 'iou_thresholded': iou_thresholded})
    old_model.layers.pop(0)
    input_shape = (512, 1024, 3)  # replace input layer with this shape so unet forward will work
    new_input = keras.layers.Input(input_shape)
    new_outputs = old_model(new_input)
    new_model = keras.engine.Model(new_input, new_outputs)
    new_model.summary()
    return new_model


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


def load_model(model_dir, pretrained_model_name, learning_rate):
    pretrained_weight_location = Path.joinpath(model_dir, "{}_best_weights.h5".format(pretrained_model_name))
    model = keras.models.load_model(str(pretrained_weight_location))
    opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()
    return model


def load_db(root_dir):
    X = []
    Y = []
    for file in sorted(root_dir.glob("*.json")):
        x, y = extract_session_data(file, use_scale=True)
        if x is not None:
            X.append(x)
            Y.append(y)
    timesteps_per_sample = X[0].shape[0]
    number_of_features = X[0].shape[1]
    X = np.reshape(X, (len(X), timesteps_per_sample, number_of_features))
    Y = np.array(Y)
    return X, Y


def extract_session_data(file, use_scale=True):
    timesteps_per_sample = 0
    session = open(file, 'r')
    number_of_features = 0
    # cap_scale_max = 1.2
    # cap_scale_min = 0.8
    x_session = []
    sticker_count = 0
    for i, line in enumerate(session):
        timesteps_per_sample += 1
        my_dict = json.loads(line)
        sticker_3d_locs = my_dict["stickers_locs"]
        valid_stickers = my_dict["valid_stickers"]
        cap_rotation = my_dict["cap_rot"]
        rescalex = 1. / 960
        rescaley = 1. / 540
        sticker_2d_locs = [[member[0]['x'] * rescalex, member[0]['y'] * rescaley] for (i, member) in enumerate(zip(sticker_3d_locs, valid_stickers))]
        if i == 0:
            number_of_features = len(sticker_2d_locs) * 2
        for i in range(len(valid_stickers)):
            if not valid_stickers[i]:
                sticker_2d_locs[i] = [0, 0]
            else:
                sticker_count += 1
        if sticker_2d_locs[0] == [0, 0] or sticker_2d_locs[1] == [0, 0] or sticker_2d_locs[2] == [0, 0]:
            sticker_2d_locs[0] = [0, 0]
            sticker_2d_locs[1] = [0, 0]
            sticker_2d_locs[2] = [0, 0]
        if cap_rotation['x'] > 180:
            cap_rotation['x'] -= 360
        if cap_rotation['y'] > 180:
            cap_rotation['y'] -= 360
        if cap_rotation['z'] > 180:
            cap_rotation['z'] -= 360
        if use_scale:
            # cap_scalex = (cap_scalex - cap_scale_min) / (cap_scale_max - cap_scale_min)
            # cap_scalez = (cap_scalez - cap_scale_min) / (cap_scale_max - cap_scale_min)
            cap_scalex = my_dict["scalex"]
            cap_scaley = my_dict["scaley"]
            cap_scalez = my_dict["scalez"]
            cap_rots = (cap_rotation['x'], cap_rotation['y'], cap_rotation['z'], cap_scalex, cap_scaley, cap_scalez)
        else:
            cap_rots = (cap_rotation['x'], cap_rotation['y'], cap_rotation['z'])
        x_session.append(sticker_2d_locs)
    if sticker_count >= 20:  # remove datapoints with less than 20 sticker occurrences
        x_session = np.reshape(x_session, (timesteps_per_sample, number_of_features))
        y_session = np.array(cap_rots)
    else:
        x_session = None
        y_session = None
    return x_session, y_session


def split_data(x, y, with_test_set=True):
    if with_test_set:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
        return x_train, x_val, y_train, y_val, x_test, y_test
    else:
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)
        return x_train, x_val, y_train, y_val


def serialize_data(file_path, x_train, x_val, y_train, y_val, x_test=None, y_test=None):
    f = open(file_path, 'wb')
    if x_test is None or y_test is None:
        pickle.dump([x_train, x_val, y_train, y_val], f)
    else:
        pickle.dump([x_train, x_val, y_train, y_val, x_test, y_test], f)
    f.close()


def deserialize_data(file_path, with_test_set=True):
    f = open(file_path, 'rb')
    if with_test_set:
        x_train, x_val, y_train, y_val, x_test, y_test = pickle.load(f)
        f.close()
        return x_train, x_val, y_train, y_val, x_test, y_test
    else:
        x_train, x_val, y_train, y_val = pickle.load(f)
        f.close()
        return x_train, x_val, y_train, y_val


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


def plot_semanticseg_results(org_imgs,
                             mask_imgs,
                             pred_imgs=None,
                             nm_img_to_plot=10,
                             figsize=4,
                             alpha=0.5
                             ):
    """
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.
    """
    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

    fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize))
    axes[0, 0].set_title("original", fontsize=15)
    axes[0, 1].set_title("ground truth", fontsize=15)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=15)
        axes[0, 3].set_title("overlay", fontsize=15)
    else:
        axes[0, 2].set_title("overlay", fontsize=15)
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 3].imshow(mask_to_red(pred_imgs[im_id]),
                              cmap=get_cmap(pred_imgs), alpha=alpha)
            axes[m, 3].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(mask_to_red(zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size)),
                              cmap=get_cmap(mask_imgs), alpha=alpha)
            axes[m, 2].set_axis_off()
        im_id += 1
    fig.show()
    fig.savefig("results.png")


def zero_pad_mask(mask, desired_size):
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask


def reshape_arr(arr):
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])


def get_cmap(arr):
    if arr.ndim == 3:
        return 'gray'
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return 'jet'
        elif arr.shape[3] == 1:
            return 'gray'


def mask_to_red(mask):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    """
    c1 = mask
    c2 = np.zeros(mask.shape)
    c3 = np.zeros(mask.shape)
    c4 = mask.reshape(mask.shape)
    return np.stack((c1, c2, c3, c4), axis=-1)


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


def plot_patches(img_arr, org_img_size, stride=None, size=None):
    """
    Plots all the patches for the first image in 'img_arr' trying to reconstruct the original image
    """

    # check parameters
    if type(org_img_size) is not tuple:
        raise ValueError('org_image_size must be a tuple')

    if img_arr.ndim == 3:
        img_arr = np.expand_dims(img_arr, axis=0)

    if size is None:
        size = img_arr.shape[1]

    if stride is None:
        stride = size

    i_max = (org_img_size[0] // stride) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride) + 1 - (size // stride)

    fig, axes = plt.subplots(i_max, j_max)
    fig.subplots_adjust(hspace=0.01, wspace=0.05)
    jj = 0
    for i in range(i_max):
        for j in range(j_max):
            axes[i, j].imshow(img_arr[jj])
            axes[i, j].set_axis_off()
            jj += 1
    plt.savefig('test.png')
