import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from pathlib import Path
import keras
import numpy as np
from PIL import Image
import utils
import pickle
import cv2

root_dir = Path("/disk1/yotam/capnet")
db_dir = Path.joinpath(root_dir, "db_segmentation")
pickle_file_path = Path.joinpath(db_dir, "data.pickle")
pred_results_file = Path.joinpath(db_dir, "pred_results.pickle")
image_dir = Path.joinpath(db_dir, "Frames")
orig_label_dir = Path.joinpath(db_dir, "Labels")
label_dir = Path.joinpath(db_dir, "Labels_stickers")
model_name = 'unet_try_2'
model_dir = Path.joinpath(root_dir, "models")
best_weight_location = Path.joinpath(model_dir, "{}_best_weights.h5".format(model_name))
if not pred_results_file.is_file():
    imgs = sorted(image_dir.glob("*.jpg"))
    masks = sorted(label_dir.glob("*.png"))
    imgs_list = []
    masks_list = []
    for image, mask in zip(imgs, masks):
        img_data = np.array(Image.open(image).resize((1024, 512)))
        mask_data = np.array(Image.open(mask).resize((1024, 512)))
        imgs_list.append(img_data)
        masks_list.append(mask_data)
    imgs_np = np.array(imgs_list)
    masks_np = np.array(masks_list)
    x = np.asarray(imgs_np, dtype=np.float32)/255
    y = np.expand_dims(masks_np, axis=-1)
    input_shape = (None, None, 3)
    my_model = utils.load_semantic_seg_model(str(best_weight_location))
    y_pred_list = []
    for i in range(x.shape[0]):
        print("predicting on image:", i)
        to_predict = np.expand_dims(x[i], axis=0)
        y_pred = my_model.predict(to_predict)
        y_pred_list.append(y_pred)
    y_pred_np = np.array(y_pred_list)
    y_pred_np = np.squeeze(y_pred_np)
    threshold, upper, lower = 0.5, 1, 0
    y_pred_np = np.where(y_pred_np > threshold, upper, lower)
    f = open(pred_results_file, 'wb')
    pickle.dump([imgs_np, masks_np, y_pred_np], f, protocol=4)
    f.close()
else:
    f = open(pred_results_file, 'rb')
    imgs_np, masks_np, y_pred_np = pickle.load(f)
    f.close()

utils.plot_semanticseg_results(org_imgs=imgs_np, mask_imgs=masks_np, pred_imgs=y_pred_np, nm_img_to_plot=10, figsize=6)
print("bye!")
# for image in test_image_dir.glob("*"):
#     dst = Path.joinpath(output_folder, image.stem + ".png")
#     new_model.predict_segmentation(
#         inp=str(image),
#         out_fname=str(dst)
#     )
