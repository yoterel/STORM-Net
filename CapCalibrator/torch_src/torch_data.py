import logging
import file_io
import geometry
import torch_src.MNI_torch as MNI_torch
import torch
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import utils
import copy


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)
        self.raw_data_file = opt.data_path / "data.pickle"
        if not self.raw_data_file.is_file():
            logging.info("loading raw data")
            X, Y = file_io.load_raw_json_db(opt.data_path, False, False)
            logging.info("creating train-validation split")
            x_train, x_val, y_train, y_val, x_test, y_test = utils.split_data(X, Y, with_test_set=True)
            # X_train = np.expand_dims(X_train, axis=0)
            # X_val = np.expand_dims(X_val, axis=0)
            # y_train = np.expand_dims(y_train, axis=0)
            # y_val = np.expand_dims(y_val, axis=0)
            logging.info("saving train-validation split to: " + str(self.raw_data_file))
            file_io.serialize_data(self.raw_data_file, x_train, x_val, y_train, y_val, x_test, y_test)
        else:
            logging.info("loading train-validation split from: " + str(self.raw_data_file))
            x_train, x_val, y_train, y_val, x_test, y_test = file_io.deserialize_data(self.raw_data_file, with_test_set=True)
        if self.opt.is_train:
            self.data = x_train
            self.labels = y_train
            self.labels[:, 1:] *= -1
            selector = 10000
            self.data = self.data[:selector]
            self.labels = {"rot_and_scale": self.labels[:selector]}
            # self.labels = {"rot_and_scale": self.labels}
        else:
            self.data = x_val
            self.labels = y_val
            self.labels[:, 1:] *= -1
            selector = 500
            self.data = self.data[:selector]
            self.labels = {"rot_and_scale": self.labels[:selector]}
            # self.labels = {"rot_and_scale": self.labels}
        self.transform_labels_to_point_cloud(save_result=True, force_recreate=False, use_gpu=True)

    def __getitem__(self, idx):
        x = self.data[idx]
        # self.shuffle_timeseries(x)
        self.shuffle_data(x)
        self.center_data(x)
        # if self.opt.is_train:
        #     self.mask_data(x)
        y1 = self.labels["rot_and_scale"][idx]
        y1_torch = torch.from_numpy(y1).float().to(self.opt.device)
        if self.opt.loss == "l2+projection":
            y2 = self.labels["raw_projected_data"][idx]
            y2_torch = torch.from_numpy(y2).float().to(self.opt.device)
            y_to_return = {"rot_and_scale": y1_torch,
                           "raw_projected_data": y2_torch}
        else:
            y_to_return = {"rot_and_scale": y1_torch}
        x_torch = torch.from_numpy(x).float().to(self.opt.device)

        return x_torch, y_to_return

    def __len__(self):
        return len(self.data)

    def transform_labels_to_point_cloud(self, save_result=True, force_recreate=False, use_gpu=False):
        projected_data_file = Path.joinpath(self.raw_data_file.parent, self.raw_data_file.stem + "_" + str(self.opt.is_train) + "_projected.pickle")
        if not force_recreate:
            if projected_data_file.is_file():
                data = file_io.load_from_pickle(projected_data_file)
                self.labels = data
                return
        rot_and_scale_labels = self.labels["rot_and_scale"]
        rs = []
        sc = []
        for i in range(len(rot_and_scale_labels)):
            # logging.info("Network Euler angels:" + str([y_predict[i][0], -y_predict[i][1], -y_predict[i][2]]))
            rot = R.from_euler('xyz', [rot_and_scale_labels[i][0], rot_and_scale_labels[i][1], rot_and_scale_labels[i][2]], degrees=True)
            scale_mat = np.identity(3)
            if rot_and_scale_labels.shape[-1] > 3:
                scale_mat[0, 0] = rot_and_scale_labels[0][3]  # xscale
                scale_mat[1, 1] = rot_and_scale_labels[0][4]  # yscale
                scale_mat[2, 2] = rot_and_scale_labels[0][5]  # zscale
            rotation_mat = rot.as_matrix()
            rs.append(rotation_mat)
            sc.append(scale_mat)
        # rs = np.array(rs)
        # sc = np.array(sc)
        names, data, format, _ = file_io.read_template_file(self.opt.template)
        names = names[0]
        data = data[0]
        data = geometry.to_standard_coordinate_system(names, data)
        assert 0 in names
        data_origin = data[:names.index(0), :]  # non numbered optodes are not calibrated
        data_others = data[names.index(0):, :]  # selects optodes for applying calibration
        origin_names = np.array(names[:names.index(0)])

        projected_data = []
        if use_gpu:
            anchors_xyz, selected_indices = geometry.sort_anchors(origin_names, data_origin)
            anchors_xyz_torch = torch.from_numpy(anchors_xyz).float().to(self.opt.device)
            selected_indices_torch = torch.from_numpy(selected_indices).to(self.opt.device)
            for i, (rot_mat, scale_mat) in enumerate(zip(rs, sc)):
                logging.info("processing: {} / {}".format(i, len(rs)))
                transformed_data_others = (rot_mat @ (scale_mat @ data_others.T)).T
                transformed_others_xyz_torch = torch.from_numpy(transformed_data_others).float().to(self.opt.device)
                torch_mni, _, _ = MNI_torch.torch_project_non_differentiable(anchors_xyz_torch,
                                                                             transformed_others_xyz_torch.unsqueeze(0),
                                                                             selected_indices_torch)
                # transformed_data.append([names, np.vstack((data_origin, data_others))])
                projected_data.append(torch_mni.cpu().numpy())
            raw_projected_data = np.array(projected_data)
        else:
            for i, (rot_mat, scale_mat) in enumerate(zip(rs, sc)):
                transformed_data_sim = rot_mat @ (scale_mat @ data_others.T)
                data_others = transformed_data_sim.T
                projected_data.append([names, np.vstack((data_origin, data_others))])
            projected_data = geometry.project_sensors_to_MNI(projected_data)
            raw_projected_data = np.array([x[1] for x in projected_data])[:, names.index(0):, :]
        self.labels["raw_projected_data"] = raw_projected_data
        if save_result:
            file_io.dump_to_pickle(projected_data_file, self.labels)
        return


    def shuffle_timeseries(self, x):
        """
        Shuffles the frames in-place pair-wise for augmentation.
        Each consecutive frames pair is either shuffled, or not, randomly.
        """
        b = np.reshape(x, (x.shape[0] // 2, 2, x.shape[1]))
        for ndxx in np.ndindex(b.shape[0]):
            np.random.shuffle(b[ndxx])

    def mask_data(self, x):
        """
        masks 20% of the frames in-place for augmentation.
        """
        percent = 20
        shp = x.shape
        a = np.random.choice(shp[0], percent * shp[0] // 100, replace=False)
        x[a] = np.zeros((shp[1],))
        return

    def shuffle_data(self, x):
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

    def center_data(self, x):
        """
        centers the stickers in place to create centered data
        """
        b = x
        zero_indices = np.copy(b == 0)
        with np.errstate(all='ignore'):  # we replace nans with zero immediately after possible division by zero
            xvec_cent = np.true_divide(b[:, ::2].sum(1), (b[:, ::2] != 0).sum(1))
            xvec_cent = np.nan_to_num(xvec_cent)
            yvec_cent = np.true_divide(b[:, 1::2].sum(1), (b[:, 1::2] != 0).sum(1))
            yvec_cent = np.nan_to_num(yvec_cent)
        b[:, ::2] += np.expand_dims(0.5 - xvec_cent, axis=1)
        b[:, 1::2] += np.expand_dims(0.5 - yvec_cent, axis=1)
        b[zero_indices] = 0
        return


class MyDataLoader:
    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)
        self.dataset = MyDataSet(opt)
        if self.opt.is_train:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=int(opt.num_threads)
            )
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.num_threads)
            )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data