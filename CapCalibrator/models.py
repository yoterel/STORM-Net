import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D,\
    Dropout, UpSampling2D, Input, concatenate, Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras import backend as K


def threshold_binarize(x, threshold=0.5):
    ge = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(ge, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y


def iou(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred = threshold_binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def upsample_conv(filters, kernel_size, strides, padding):
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)


def conv2d_block(
        inputs,
        use_batch_norm=True,
        dropout=0.3,
        filters=16,
        kernel_size=(3, 3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same'):
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(
        inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = Dropout(dropout)(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def custom_unet(
        input_shape,
        num_classes=1,
        use_batch_norm=True,
        upsample_mode='deconv',  # 'deconv' or 'simple'
        use_dropout_on_upsampling=False,
        dropout=0.3,
        dropout_change_per_layer=0.0,
        filters=16,
        num_layers=4,
        output_activation='sigmoid'):  # 'sigmoid' or 'softmax'

    if upsample_mode == 'deconv':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


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
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss='binary_crossentropy',
        # loss=jaccard_distance,
        metrics=[iou, iou_thresholded, 'binary_accuracy']
    )
    # model.summary()
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
    opt = tf.keras.optimizers.Adam(lr=learning_rate)
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
    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
    # model.summary()
    return model