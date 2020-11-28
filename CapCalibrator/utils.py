import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import hashlib
import cv2


def shape_to_np(shape, dtype="int"):
    """
    converts list of tuples into np array
    :param shape: the tuples
    :param dtype: type of return value
    :return: the np array
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def split_data(x, y, with_test_set=True):
    """
    splits x and y to train/val/test or train/val
    :param x: data
    :param y: labels
    :param with_test_set: whether to also create also test set
    :return: the split sections
    """
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


def md5_from_vid(path):
    block_size = 2 ** 14
    md5 = hashlib.md5()
    with open(str(path), 'rb') as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()
