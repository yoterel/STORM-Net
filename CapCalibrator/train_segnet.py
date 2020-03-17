import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import utils
from pathlib import Path
import numpy as np
from PIL import Image
from time import time
from keras_unet.utils import get_augmented
import keras
from tensorflow.python.keras.callbacks import TensorBoard
stdout = sys.stdout

# hyper parameters
redirect_sysout = True
batch_size = 8
learning_rate = 1e-4
early_stopping_patience = 20
reduce_lr_patience = 3
epochs = 1000
verbosity = 1  # set verbosity of fit
model_name = 'unet_try_2'

# paths
root_dir = Path("/disk1/yotam/capnet")
model_dir = Path.joinpath(root_dir, "models")
logs_dir = Path.joinpath(root_dir, "logs")
best_weight_location = Path.joinpath(model_dir, "{}_best_weights.h5".format(model_name))
my_log_dir = Path.joinpath(logs_dir, "{}_{}".format(model_name, time()))
my_log_dir.mkdir(parents=True, exist_ok=True)
model_graph_path = Path.joinpath(my_log_dir, model_name)
event_file_path = Path.joinpath(my_log_dir, model_name+"_events.txt")
db_dir = Path.joinpath(root_dir, "db_segmentation")
pickle_file_path = Path.joinpath(db_dir, "data.pickle")
image_dir = Path.joinpath(db_dir, "Frames")
orig_label_dir = Path.joinpath(db_dir, "Labels")
label_dir = Path.joinpath(db_dir, "Labels_stickers")
if not label_dir.is_dir():
    utils.preprocess_labels(orig_label_dir, label_dir)
if redirect_sysout:
    sys.stdout = open(str(event_file_path), 'w+')
sys.stdout.flush()

if not pickle_file_path.is_file():
    imgs = sorted(image_dir.glob("*.jpg"))
    masks = sorted(label_dir.glob("*.png"))
    imgs_list = []
    masks_list = []
    for image, mask in zip(imgs, masks):
        img_data = np.array(Image.open(image))
        mask_data = np.array(Image.open(mask))
        img_patches = utils.get_patches(img_data, size=128, stride=128)
        mask_patches = utils.get_patches(mask_data, size=128, stride=128)
        # utils.plot_patches(img_arr=patches, org_img_size=(1080, 1920))
        imgs_list.append(img_patches)
        masks_list.append(mask_patches)
    imgs_np = np.vstack(imgs_list)
    masks_np = np.vstack(masks_list)
    complete_background_patch = np.all(masks_np == np.zeros((1, 128, 128)), axis=(1, 2))
    non_bg_patch = np.logical_not(complete_background_patch)
    non_bg_indices = np.where(non_bg_patch)
    imgs_np = imgs_np[non_bg_indices]
    masks_np = masks_np[non_bg_indices]
    # utils.plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=10, figsize=6)
    x = np.asarray(imgs_np, dtype=np.float32)/255
    y = np.expand_dims(masks_np, axis=-1)
    print(x.shape, y.shape)
    x_train, x_val, y_train, y_val = utils.split_data(x, y, with_test_set=False)
    utils.serialize_data(pickle_file_path, x_train, x_val, y_train, y_val, None, None)
else:
    x_train, x_val, y_train, y_val = utils.deserialize_data(pickle_file_path, with_test_set=False)

input_shape = (None, None, 3)
model = utils.create_semantic_seg_model(input_shape, learning_rate)
keras.utils.plot_model(model, to_file=str(model_graph_path)+"_graph.png", show_shapes=True)

train_gen = get_augmented(
    x_train, y_train, batch_size=batch_size,
    data_gen_args=dict(
        rotation_range=15.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=50,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant'
    ))
steps_per_epoch = np.math.ceil(x_train.shape[0] / batch_size)
# sample_batch = next(train_gen)
# xx, yy = sample_batch
# print(xx.shape, yy.shape)
# plot_imgs(org_imgs=xx, mask_imgs=yy, nm_img_to_plot=2, figsize=6)
checkpoint = keras.callbacks.ModelCheckpoint(str(best_weight_location),
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='min',
                                             period=1)
tensor_board = TensorBoard(log_dir=my_log_dir,
                           write_graph=True,
                           write_images=True)
callbacks = [checkpoint, tensor_board]

H = model.fit_generator(generator=train_gen,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        verbose=verbosity,
                        callbacks=callbacks,
                        validation_data=(x_val, y_val))
# set stdout back to normal
sys.stdout = stdout