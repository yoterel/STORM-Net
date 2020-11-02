import argparse
import sys
from pathlib import Path
from time import time
import utils
import file_io
import logging

def train(args):
    import keras
    from tensorflow.python.keras.callbacks import TensorBoard
    model_dir = Path("models")
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    data_dir = args.data_path
    model_name = args.model_name
    pretrained_model_name = args.pretrained_model_path
    if args.log:
        event_file_path = args.log
        if event_file_path.is_file():
            logging.info("Warnning: log file already exists. Overwriting.")
        stdout = sys.stdout
        # set hyper parameters
        sys.stdout = open(str(event_file_path), 'w+')
        # model_graph_path = event_file_path.parent
        # my_log_dir = Path.joinpath(logs_dir, "{}_{}".format(model_name, time()))
        # my_log_dir.mkdir(parents=True, exist_ok=True)
    best_weight_location = Path.joinpath(model_dir, "{}.h5".format(model_name))
    pickle_file_path = Path.joinpath(cache_dir, "serialized_synthetic_data.pickle")

    # hyper parameters #
    hp = {"batch_size": 16,
          "learning_rate": 1e-4,
          "early_stopping_patience": 10,
          "reduce_lr_patience": 3,
          "epochs": 1000,
          "verbosity": 1}
    logging.info(hp.items())
    logging.info("data path: " + str(data_dir))
    # load data
    if not pickle_file_path.is_file():
        logging.info("loading raw data")
        X, Y = file_io.load_raw_json_db(data_dir)
        logging.info("creating train/val split")
        x_train, x_val, y_train, y_val = utils.split_data(X, Y, with_test_set=False)
        # X_train = np.expand_dims(X_train, axis=0)
        # X_val = np.expand_dims(X_val, axis=0)
        # y_train = np.expand_dims(y_train, axis=0)
        # y_val = np.expand_dims(y_val, axis=0)
        logging.info("saving train/val split to cache folder: " + str(cache_dir))
        file_io.serialize_data(pickle_file_path, x_train, x_val, y_train, y_val)
    else:
        logging.info("loading train/val split from cache folder: " + str(cache_dir))
        x_train, x_val, y_train, y_val = file_io.deserialize_data(pickle_file_path, with_test_set=False)
    input_shape = (x_train.shape[1], x_train.shape[2])
    output_shape = y_train.shape[-1]
    if pretrained_model_name:
        model = file_io.load_keras_model(pretrained_model_name, hp["learning_rate"])
    else:
        model = utils.create_fc2_model(input_shape, output_shape, hp["learning_rate"])
    # keras.utils.plot_model(model, to_file=str(model_graph_path)+"_graph.png", show_shapes=True)

    # set model callbacks
    checkpoint = keras.callbacks.ModelCheckpoint(str(best_weight_location),
                                                 monitor='val_loss',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 mode='min',
                                                 period=1)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0.001,
                                               patience=hp["early_stopping_patience"],
                                               mode='min',
                                               verbose=1)
    # trains well even without rducing lr...
    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
    #                                               factor=0.2,
    #                                               patience=hp["reduce_lr_patience"],
    #                                               verbose=1,
    #                                               mode='min',
    #                                               min_delta=0.001,
    #                                               cooldown=0,
    #                                               min_lr=1e-6)
    callbacks = [checkpoint, early_stop]
    if args.tensorboard:
        tensor_board = TensorBoard(log_dir=Path(args.tensorboard),
                                   write_graph=True,
                                   write_images=True)
        callbacks.append(tensor_board)
    dim = x_train.shape[1:]
    training_generator = utils.DataGenerator(x_train, y_train, batch_size=hp["batch_size"], dim=dim)
    validation_generator = utils.DataGenerator(x_val, y_val, batch_size=hp["batch_size"], dim=dim, mask_stickers=False)

    # start training
    H = model.fit_generator(generator=training_generator,
                            epochs=hp["epochs"],
                            verbose=hp["verbosity"],
                            callbacks=callbacks,
                            validation_data=validation_generator)

    # set stdout back to normal
    if args.log:
        sys.stdout = stdout


def configure_environment(gpu_id):
    import os
    if gpu_id == -1:
        gpu_id = ""
    else:
        gpu_id = str(gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id  # set gpu visibility prior to importing tf and keras
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras


def parse_arguments():
    parser = argparse.ArgumentParser(description='This script fine-tunes STORM-Net')
    parser.add_argument("model_name", help="The name to give the newly trained model (without extension).")
    parser.add_argument("pretrained_model_path", help="The path to the pretrained model file")
    parser.add_argument("data_path", help="The path to the folder containing the synthetic data")
    parser.add_argument("--gpu_id", type=int, default=-1, help="Which GPU to use (or -1 for cpu)")
    parser.add_argument("--tensorboard", help="If present, writes training stats to this file path (readable with tensorboard)")
    parser.add_argument("--log", help="If present, stdout will be redirected to this log file path.")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    # cmd_line = 'telaviv_model_b17 models/telaviv_model_b16.h5 data/telaviv_model'.split()
    args = parser.parse_args()
    args.pretrained_model_path = Path(args.pretrained_model_path)
    if args.log:
        args.log = Path(args.log)
    args.data_path = Path(args.data_path)
    return args


if __name__ == "__main__":
    args = parse_arguments()
    configure_environment(args.gpu_id)
    train(args)
    logging.info("Done!")
