import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # suppress more warnings & info from tf
import models
import logging


def load_semantic_seg_model(weights_loc):
    logging.info("Loading unet model from: " + str(weights_loc))
    g = tf.Graph()
    with g.as_default():
        old_model = tf.keras.models.load_model(weights_loc,
                                               custom_objects={'iou': models.iou, 'iou_thresholded': models.iou_thresholded})
        old_model.layers.pop(0)
        input_shape = (512, 1024, 3)  # replace input layer with this shape so unet forward will work
        new_input = tf.keras.layers.Input(input_shape)
        new_outputs = old_model(new_input)
        new_model = tf.keras.Model(new_input, new_outputs)
        # if verbosity:
        #     new_model.summary()
    return new_model, g


def load_keras_model(pretrained_model_path, output_shape, learning_rate):
    model = tf.keras.models.load_model(str(pretrained_model_path))
    if model.output_shape[-1] != output_shape:
        model.pop()  # to remove last layer
        new_model = tf.keras.Sequential([
            model,
            tf.keras.layers.Dense(output_shape)
        ])
        model = new_model
    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)
    # model.summary()
    return model


def load_clean_keras_model(path):
    logging.info("Loading STORM model from: " + str(path))
    g = tf.Graph()
    with g.as_default():
        model = tf.keras.models.load_model(str(path))
        # if v:
        #     model.summary()
    return model, g