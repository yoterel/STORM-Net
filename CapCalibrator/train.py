import argparse
import sys
from pathlib import Path
import utils
import file_io
import logging
import keras.callbacks as cb


class CustomLogging(cb.Callback):
    """
    performs custom logging to a shared queue, or to a log, or both
    """
    def __init__(self, queue=None, stoprequest=None, log_results=None):
        super(CustomLogging, self).__init__()
        self.queue = queue
        self.stoprequest = stoprequest
        self.log_results = log_results

    def on_batch_end(self, batch, logs=None):
        if self.stoprequest:
            if self.stoprequest.isSet():
                self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        if self.stoprequest:
            if self.stoprequest.isSet():
                self.model.stop_training = True
        loss = logs.get("val_loss")
        lr = logs.get("lr")
        if self.log_results:
            logging.info("epoch: {}, validation_loss: {}, lr: {}".format(epoch, loss, lr))
        if self.queue:
            self.queue.put(["finetune_data", epoch, loss])

    def on_train_end(self, logs=None):
        if self.log_results:
            logging.info("Training finished.")
        if self.queue:
            self.queue.put(["finetune_done"])


def train(model_name, data_path, pretrained_model_path, tensorboard, verbosity, output_path, queue=None, event=None):
    import keras
    keras.backend.clear_session()
    from tensorflow.python.keras.callbacks import TensorBoard
    model_dir = output_path
    data_dir = data_path
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    model_name = model_name
    pretrained_model_name = pretrained_model_path
    best_weight_location = Path.joinpath(model_dir, "{}.h5".format(model_name))
    pickle_file_path = Path(cache_dir, "{}_serialized.pickle".format(data_dir.name))

    # hyper parameters #
    hp = {"batch_size": 16,
          "learning_rate": 1e-4,
          "early_stopping_patience": 20,
          "reduce_lr_patience": 100,
          "epochs": 2000,
          "verbosity": verbosity,
          "pre_trained_model": pretrained_model_name,
          "shuffle_frames_pairwise": True}
    logging.info(hp.items())
    # load data
    if not pickle_file_path.is_file():
        logging.info("loading raw data")
        X, Y = file_io.load_raw_json_db(data_dir)
        logging.info("creating train-validation split")
        x_train, x_val, y_train, y_val = utils.split_data(X, Y, with_test_set=False)
        # X_train = np.expand_dims(X_train, axis=0)
        # X_val = np.expand_dims(X_val, axis=0)
        # y_train = np.expand_dims(y_train, axis=0)
        # y_val = np.expand_dims(y_val, axis=0)
        logging.info("saving train-validation split to: " + str(pickle_file_path))
        file_io.serialize_data(pickle_file_path, x_train, x_val, y_train, y_val)
    else:
        logging.info("loading train-validation split from: " + str(pickle_file_path))
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
                                                 verbose=0,
                                                 save_best_only=True,
                                                 mode='min',
                                                 period=1)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0.001,
                                               patience=hp["early_stopping_patience"],
                                               mode='min',
                                               verbose=0)
    # trains well even without rducing lr...
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.5,
                                                  patience=hp["reduce_lr_patience"],
                                                  verbose=0,
                                                  mode='min',
                                                  min_delta=0.001,
                                                  cooldown=0,
                                                  min_lr=1e-6)
    callbacks = [checkpoint, reduce_lr]
    if queue and event:
        callbacks.append(CustomLogging(queue, event))
    else:
        callbacks.append(CustomLogging(queue, event, True))
    if tensorboard:
        tensor_board = TensorBoard(log_dir=Path(tensorboard),
                                   write_graph=True,
                                   write_images=True)
        callbacks.append(tensor_board)
    dim = x_train.shape[1:]
    training_generator = utils.DataGenerator(x_train, y_train,
                                             batch_size=hp["batch_size"],
                                             dim=dim,
                                             shuffle_timestamps=hp["shuffle_frames_pairwise"])
    validation_generator = utils.DataGenerator(x_val, y_val,
                                               batch_size=hp["batch_size"],
                                               dim=dim,
                                               mask_stickers=False,
                                               shuffle_timestamps=hp["shuffle_frames_pairwise"])

    # start training
    _ = model.fit_generator(generator=training_generator,
                            epochs=hp["epochs"],
                            verbose=hp["verbosity"],
                            callbacks=callbacks,
                            validation_data=validation_generator)


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
    parser.add_argument("data_path", help="The path to the folder containing the synthetic data")
    parser.add_argument("--output_path", default="models", help="The trained model will be saved to this folder")
    parser.add_argument("--pretrained_model_path", help="The path to the pretrained model file")
    parser.add_argument("--gpu_id", type=int, default=-1, help="Which GPU to use (or -1 for cpu)")
    parser.add_argument("--tensorboard",
                        help="If present, writes training stats to this path (readable with tensorboard)")
    parser.add_argument("--log",
                        help="If present, writes training log to this path")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info", help="Selects verbosity level")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    if args.pretrained_model_path:
        args.pretrained_model_path = Path(args.pretrained_model_path)
    else:
        args.pretrained_model_path = None
    args.output_path = Path(args.output_path)
    if args.log:
        args.log = Path(args.log)
    if args.tensorboard:
        args.tensorboard = Path(args.tensorboard)
    args.data_path = Path(args.data_path)
    return args


if __name__ == "__main__":
    args = parse_arguments()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    configure_environment(args.gpu_id)
    train(args.model_name, args.data_path, args.pretrained_model_path, args.tensorboard, args.verbosity, args.output_path)
    logging.info("Done!")
