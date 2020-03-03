import sys
import os
import utils
from pathlib import Path
from time import time
import numpy as np
import math
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import keras
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import mean_squared_error

# set paths
model_name = 'scene3_batch16_lr1e4_supershuffle_noise6'
pretrained_model_name = 'scene3_batch16_lr1e4_supershuffle_noise5'
root_dir = Path("/disk1/yotam/capnet")
data_dir = Path.joinpath(root_dir, "scene3_100k")
model_dir = Path.joinpath(root_dir, "models")
logs_dir = Path.joinpath(root_dir, "logs")
best_weight_location = Path.joinpath(model_dir, "{}_best_weights.h5".format(model_name))
my_log_dir = Path.joinpath(logs_dir, "{}_{}".format(model_name, time()))
my_log_dir.mkdir(parents=True, exist_ok=True)
model_graph_path = Path.joinpath(my_log_dir, model_name)
event_file_path = Path.joinpath(my_log_dir, model_name+"_events.txt")
pickle_file_path = Path.joinpath(data_dir, "data.pickle")
# set stdout to a file, instead of sending prints to terminal
stdout = sys.stdout
# set hyper parameters
redirect_sysout = True
batch_size = 16
learning_rate = 1e-4
early_stopping_patience = 20
reduce_lr_patience = 3
epochs = 2000
verbosity = 1  # set verbosity of fit

# ------------------------------------- START ---------------------------------------------- #
if redirect_sysout:
    sys.stdout = open(str(event_file_path), 'w+')

print("datapath:", data_dir)
# load data
if not pickle_file_path.is_file():
    X, Y = utils.load_db(data_dir)
    x_train, x_val, y_train, y_val, x_test, y_test = utils.split_data(X, Y)
    # X_train = np.expand_dims(X_train, axis=0)
    # X_val = np.expand_dims(X_val, axis=0)
    # y_train = np.expand_dims(y_train, axis=0)
    # y_val = np.expand_dims(y_val, axis=0)
    utils.serialize_data(pickle_file_path, x_train, x_val, y_train, y_val, x_test, y_test)
else:
    x_train, x_val, y_train, y_val, x_test, y_test = utils.deserialize_data(pickle_file_path)
# utils.disarrange(X_train)
# sanity check
# y_predict = np.random.uniform(-3, 3, y_val.shape)
# testScore = math.sqrt(mean_squared_error(y_val, y_predict))
input_shape = (x_train.shape[1], x_train.shape[2])
output_shape = y_train.shape[-1]
if pretrained_model_name:
    model = utils.load_model(model_dir, pretrained_model_name, learning_rate)
else:
    model = utils.create_fc2_model(input_shape, output_shape, learning_rate)
keras.utils.plot_model(model, to_file=str(model_graph_path)+"_graph.png", show_shapes=True)
# set model callbacks
checkpoint = keras.callbacks.ModelCheckpoint(str(best_weight_location),
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='min',
                                             period=1)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=0.001,
                                           patience=early_stopping_patience,
                                           mode='min',
                                           verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.2,
                                              patience=reduce_lr_patience,
                                              verbose=1,
                                              mode='min',
                                              min_delta=0.001,
                                              cooldown=0,
                                              min_lr=1e-6)
tensor_board = TensorBoard(log_dir=my_log_dir,
                           write_graph=True,
                           write_images=True)
dim = x_train.shape[1:]
training_generator = utils.DataGenerator(x_train, y_train, batch_size=batch_size, dim=dim)
validation_generator = utils.DataGenerator(x_val, y_val, batch_size=batch_size, dim=dim, mask_stickers=False)
callbacks = [checkpoint, tensor_board]
# start training
H = model.fit_generator(generator=training_generator,
                        epochs=epochs,
                        verbose=verbosity,
                        callbacks=callbacks,
                        validation_data=validation_generator)

# set stdout back to normal
sys.stdout = stdout