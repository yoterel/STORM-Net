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
import itertools
import torch.distributions as D


class HeatMap(torch.nn.Module):
    """
    Layer to create a heatmap from a given set of landmarks
    """

    def __init__(self, img_size, patch_size, dont_use_gmm, device):
        """

        Parameters
        ----------
        img_size : tuple
            the image size of the returned heatmap
        patch_size : int
            the patchsize to use
        dont_use_gmm: dont use a gmm to create the heat maps, instead use "offsets" like in DAN (deep alignment network)
        device: the device to operate over
        """

        super().__init__()

        self.img_shape = img_size
        self.dont_use_gmm = dont_use_gmm
        self.half_size = patch_size // 2
        self.device = device
        self.offsets = torch.tensor(
            list(
                itertools.product(
                    range(-self.half_size, self.half_size + 1),
                    range(-self.half_size, self.half_size + 1)
                )
            )
        ).float().to(self.device)

    def draw_gmm(self, landmarks):
        valid_landmarks = torch.where(torch.all(landmarks != 0, dim=1))[0]
        if len(valid_landmarks):
            mix = D.Categorical(torch.ones(len(valid_landmarks), ).to(landmarks.device))
            my_distri = D.Normal(landmarks[valid_landmarks], torch.ones((len(valid_landmarks), 2)).to(landmarks.device)*5)
            comp = D.Independent(my_distri, 1)
            gmm = D.MixtureSameFamily(mix, comp)

            grid = torch.meshgrid(torch.arange(256), torch.arange(256))
            stacked_grid = torch.dstack((grid[0], grid[1])).to(landmarks.device)
            log_pdf = torch.exp(gmm.log_prob(stacked_grid))
            heatmap = (log_pdf - torch.min(log_pdf)) / (torch.max(log_pdf) - torch.min(log_pdf))
        else:
            heatmap = torch.zeros((256, 256)).to(landmarks.device)
        return heatmap.unsqueeze(0)

    def draw_offsets(self, landmark):
        """
        Draws a single point only

        Parameters
        ----------
        landmark : :class:`torch.Tensor`
            the landmarkto draw (of shape 1x2)

        Returns
        -------
        :class:`torch.Tensor`
            the heatmap containing one landmark
            (of shape ``1 x self.img_shape[0] x self.img_shape[1]``)

        """

        img = torch.zeros(1, *self.img_shape, device=landmark.device)
        if not torch.all(landmark == self.half_size):
            int_lmk = landmark.to(torch.long)
            locations = self.offsets.to(torch.long) + int_lmk
            diffs = landmark - int_lmk.to(landmark.dtype)

            offsets_subpix = self.offsets - diffs
            vals = 1 / (1 + (offsets_subpix ** 2).sum(dim=1) + 1e-6).sqrt()

            img[0, locations[:, 0], locations[:, 1]] = vals.clone()

        return img


    def draw_landmarks(self, landmarks, dont_use_gmm):
        """
        Draws a group of landmarks

        Parameters
        ----------
        landmarks : :class:`torch.Tensor`
            the landmarks to draw (of shape Num_Landmarks x 2)

        Returns
        -------
        :class:`torch.Tensor`
            the heatmap containing all landmarks
            (of shape ``1 x self.img_shape[0] x self.img_shape[1]``)
        """
        if dont_use_gmm:
            landmarks = landmarks.view(-1, 2)
            # landmarks = landmarks.clone()

            for i in range(landmarks.size(-1)):
                landmarks[:, i] = torch.clamp(
                    landmarks[:, i].clone(),
                    self.half_size,
                    self.img_shape[1 - i] - 1 - self.half_size)
            heatmap = torch.max(torch.cat([self.draw_offsets(lmk.unsqueeze(0)) for lmk in landmarks], dim=0),
                                dim=0,
                                keepdim=True)[0]
        else:
            heatmap = self.draw_gmm(landmarks)
        return heatmap


    def forward(self, landmark_batch):
        """
        Draws all landmarks from one batch element in one heatmap

        Parameters
        ----------
        landmark_batch : :class:`torch.Tensor`
            the landmarks to draw
            (of shape ``N x Num_landmarks x 2``))

        Returns
        -------
        :class:`torch.Tensor`
            a batch of heatmaps
            (of shape ``N x 1 x self.img_shape[0] x self.img_shape[1]``)

        """
        x = torch.cat([self.draw_landmarks(landmarks, self.dont_use_gmm)
                   for landmarks in landmark_batch], dim=0)
        return x


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)
        self.raw_data_file = opt.data_path / "data_split.pickle"
        if self.opt.is_train:
            if not self.raw_data_file.is_file() or self.opt.force_load_raw_data:
                logging.info("loading raw data. this might take a while.")
                X, Y = file_io.load_raw_json_db(opt.data_path, opt.scale_faces, False)
                logging.info("creating train-validation split")
                x_train, x_val, y_train, y_val = utils.split_data(X, Y, with_test_set=False)
                # X_train = np.expand_dims(X_train, axis=0)
                # X_val = np.expand_dims(X_val, axis=0)
                # y_train = np.expand_dims(y_train, axis=0)
                # y_val = np.expand_dims(y_val, axis=0)
                logging.info("saving train-validation split to: " + str(self.raw_data_file))
                file_io.serialize_data(self.raw_data_file, x_train, x_val, y_train, y_val)
            else:
                logging.info("(train data) loading train-validation split from: " + str(self.raw_data_file))
                x_train, x_val, y_train, y_val = file_io.deserialize_data(self.raw_data_file, with_test_set=False)
        else:
            logging.info("(val data) loading train-validation split from: " + str(self.raw_data_file))
            x_train, x_val, y_train, y_val = file_io.deserialize_data(self.raw_data_file, with_test_set=False)

        if self.opt.is_train:
            self.data = x_train
            self.labels = y_train
            self.labels[:, 1:3] *= -1
            selector = 10000
            self.data = self.data[:selector]
            self.labels = {"rot_and_scale": self.labels[:selector]}
            # self.labels = {"rot_and_scale": self.labels}
        else:
            self.data = x_val
            self.labels = y_val
            self.labels[:, 1:3] *= -1
            selector = 500
            self.data = self.data[:selector]
            self.labels = {"rot_and_scale": self.labels[:selector]}
            # self.labels = {"rot_and_scale": self.labels}
        if self.opt.loss == "l2+projection":
            self.transform_labels_to_point_cloud(save_result=True, force_recreate=False, use_gpu=True)
        if self.opt.architecture == "2dconv":
            self.heat_mapper = HeatMap((256, 256), 16, self.opt.dont_use_gmm, self.opt.device)


    def __getitem__(self, idx):
        x = self.data[idx]
        # self.shuffle_timeseries(x)
        self.center_data(x)
        if not self.opt.architecture == "2dconv":
            self.shuffle_data(x)
        x_torch = torch.from_numpy(x).float().to(self.opt.device)
        if self.opt.architecture == "2dconv":
            x_torch[:, 0::2] *= 256
            x_torch[:, 1::2] *= 256
            x_torch = self.heat_mapper(x_torch.reshape(10, -1, 2))
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


        return x_torch, y_to_return

    def __len__(self):
        return len(self.data)

    # def landmarks_to_heatmaps(self):
    #     from scipy.stats import multivariate_normal
    #     for instance in self.data:
    #
    #
    #     new_data = np.empty((len(self.data), 10, 256, 256), dtype=np.float32)
    #     grid = np.dstack(np.mgrid[:256, :256])
    #     for instance in self.data:
    #         spots = multivariate_normal(mea)
    #         heat_map =


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
        data_origin = data[:names.index(0), :]  # non numbered optodes are not coregistered
        data_others = data[names.index(0):, :]  # selects optodes for applying coregistration
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
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=self.opt.is_train,
            num_workers=int(opt.num_threads)
        )
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data