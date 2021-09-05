import numpy as np


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
