import torch
import torch.nn as nn
import argparse
from pathlib import Path
import logging
import file_io
import utils
import geometry
import numpy as np
from scipy.spatial.transform import Rotation as R
import MNI_torch


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.raw_data_file = opt.data_path / "serialized.pickle"
        if not self.raw_data_file.is_file():
            logging.info("loading raw data")
            X, Y = file_io.load_raw_json_db(opt.data_path, False, False)
            logging.info("creating train-validation split")
            x_train, x_val, y_train, y_val = utils.split_data(X, Y, with_test_set=False)
            # X_train = np.expand_dims(X_train, axis=0)
            # X_val = np.expand_dims(X_val, axis=0)
            # y_train = np.expand_dims(y_train, axis=0)
            # y_val = np.expand_dims(y_val, axis=0)
            logging.info("saving train-validation split to: " + str(self.raw_data_file))
            file_io.serialize_data(self.raw_data_file, x_train, x_val, y_train, y_val)
        else:
            logging.info("loading train-validation split from: " + str(self.raw_data_file))
            x_train, x_val, y_train, y_val = file_io.deserialize_data(self.raw_data_file, with_test_set=False)
        if self.opt.is_train:
            self.data = x_train
            self.labels = y_train
        else:
            self.data = x_val
            self.labels = y_val
        self.data = self.data[:10]
        self.labels = self.labels[:10]
        self.labels = self.transform_labels_to_point_cloud(True, False)

    def __getitem__(self, idx):
        x = self.data[idx]
        self.shuffle_timeseries(x)
        self.shuffle_data(x)
        self.mask_data(x)
        self.center_data(x)
        y = self.labels[idx]

        x_torch = torch.from_numpy(x).float()
        y_torch = torch.from_numpy(y).float()
        return x_torch, y_torch

    def __len__(self):
        return len(self.data)

    def transform_labels_to_point_cloud(self, save_result=True, force_recreate=False):
        projected_data_file = Path.joinpath(self.raw_data_file.parent, self.raw_data_file.stem + "_" + str(self.opt.is_train) + "_projected.pickle")
        if not force_recreate:
            if projected_data_file.is_file():
                data = file_io.load_from_pickle(projected_data_file)
                return data
        labels = self.labels
        rs = []
        sc = []
        for i in range(len(labels)):
            # logging.info("Network Euler angels:" + str([y_predict[i][0], -y_predict[i][1], -y_predict[i][2]]))
            rot = R.from_euler('xyz', [labels[i][0], -labels[i][1], -labels[i][2]], degrees=True)
            scale_mat = np.identity(3)
            if labels.shape[-1] > 3:
                scale_mat[0, 0] = labels[0][3]  # xscale
                scale_mat[1, 1] = labels[0][4]  # yscale
                scale_mat[2, 2] = labels[0][5]  # zscale
            rotation_mat = rot.as_matrix()
            rs.append(rotation_mat)
            sc.append(scale_mat)
        # rs = np.array(rs)
        # sc = np.array(sc)
        names, data, format, _ = file_io.read_template_file(opt.template)
        names = names[0]
        data = data[0]
        data = geometry.to_standard_coordinate_system(names, data)
        assert 0 in names
        data_origin = data[:names.index(0), :]  # non numbered optodes are not calibrated
        data_others = data[names.index(0):, :]  # selects optodes for applying calibration
        transformed_data = []
        for i, (rot_mat, scale_mat) in enumerate(zip(rs, sc)):
            transformed_data_sim = rot_mat @ (scale_mat @ data_others.T)
            data_others = transformed_data_sim.T
            transformed_data.append([names, np.vstack((data_origin, data_others))])
        projected_data = geometry.project_sensors_to_MNI(transformed_data)
        raw_projected_data = np.array([x[1] for x in projected_data])[:, names.index(0):, :]
        if save_result:
            file_io.dump_to_pickle(projected_data_file, raw_projected_data)
        return raw_projected_data


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
        self.opt = opt
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


class MyNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(MyNetwork, self).__init__()
        self.opt = opt
        self.naive_net = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(140, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, opt.network_output_size),
        )
        self.net = torch.nn.Sequential(
            nn.Conv1d(in_channels=opt.network_input_size, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, opt.network_output_size),
        ).to(self.opt.device)
        self.anchors_xyz, self.sensors_xyz, self.selected_indices = self.load_static_model()

    def forward(self, x):
        out = self.naive_net(x)
        mat_out = self.euler_to_matrix(out)

        rots = np.empty((x.shape[0], 3, 3), dtype=float)
        # for i in range(x.shape[0]):
        #      rot = R.from_euler('xyz', list(out[i].cpu().detach().numpy()), degrees=True)
        #      rots[i] = rot.as_matrix()
        # if not torch.all(torch.isclose(mat_out, torch.from_numpy(rots).float())):
        #     logging.warning("matrix from euler different than scipy matrix!")
        transformed_sensors = torch.transpose(torch.bmm(mat_out, self.sensors_xyz.T.repeat(x.shape[0], 1, 1)), 1, 2)
        projected_out = MNI_torch.torch_project(self.anchors_xyz, transformed_sensors, self.selected_indices)
        return projected_out

    def load_static_model(self):
        names, data, format, _ = file_io.read_template_file(opt.template)
        names = names[0]
        data = data[0]
        data = geometry.to_standard_coordinate_system(names, data)
        assert 0 in names
        data_anchors = data[:names.index(0), :]  # non numbered optodes are not calibrated
        origin_names = np.array(names[:names.index(0)])
        data_sensors = data[names.index(0):, :]  # selects optodes for applying calibration
        anchors_xyz, selected_indices = geometry.sort_anchors(origin_names, data_anchors)
        return torch.from_numpy(anchors_xyz).float(), torch.from_numpy(data_sensors).float(), selected_indices

    def euler_to_matrix(self, x):
        """
        converts euler angels to matrix (x is batched, b x 3)
        :param x:
        :return:
        """
        Rx = torch.zeros(x.shape[0], 3, 3)
        Ry = torch.zeros(x.shape[0], 3, 3)
        Rz = torch.zeros(x.shape[0], 3, 3)
        cos = torch.cos(x * np.pi / 180)
        sin = torch.sin(x * np.pi / 180)
        Rx[:, 0, 0] = 1
        Rx[:, 1, 1] = cos[:, 0]
        Rx[:, 1, 2] = -sin[:, 0]
        Rx[:, 2, 1] = sin[:, 0]
        Rx[:, 2, 2] = cos[:, 0]

        Ry[:, 1, 1] = 1
        Ry[:, 0, 0] = cos[:, 1]
        Ry[:, 0, 2] = sin[:, 1]
        Ry[:, 2, 0] = -sin[:, 1]
        Ry[:, 2, 2] = cos[:, 1]

        Rz[:, 2, 2] = 1
        Rz[:, 0, 0] = cos[:, 2]
        Rz[:, 0, 1] = -sin[:, 2]
        Rz[:, 1, 0] = sin[:, 2]
        Rz[:, 1, 1] = cos[:, 2]

        return torch.bmm(torch.bmm(Rz, Ry), Rx)


class MyModel():
    def __init__(self, opt):
        self.opt = opt
        self.network = MyNetwork(opt)
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=opt.lr,
                                          betas=(opt.beta1, 0.999),
                                          weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')


def train_loop(opt):
    opt.is_train = True
    train_dataset = MyDataLoader(opt)
    opt.is_train = False
    val_dataset = MyDataLoader(opt)
    model = MyModel(opt)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(opt.number_of_epochs):
        for batch_index, (input, target) in enumerate(train_dataset):
            model.optimizer.zero_grad()
            output = model.network(input)
            train_loss = loss_fn(output, target)
            logging.info("train: epoch: {}, batch {} / {}, loss: {}".format(epoch,
                                                                     batch_index,
                                                                     len(train_dataset) // opt.batch_size,
                                                                     train_loss.cpu().detach().numpy()))
            train_loss.backward()
            model.optimizer.step()
        with torch.no_grad():
            val_loss_total = torch.zeros(1)
            for input, target in val_dataset:
                model.optimizer.zero_grad()
                output = model.network(input)
                val_loss = loss_fn(output, target)
                val_loss_total += val_loss
            val_loss_total /= len(val_dataset)
            logging.info("validation: epoch: {}, loss: {}".format(epoch, val_loss_total.cpu().detach().numpy()))
        model.scheduler.step(val_loss)


def parse_arguments():
    parser = argparse.ArgumentParser(description='This script trains STORM-Net')
    parser.add_argument("model_name", help="The name to give the newly trained model (without extension).")
    parser.add_argument("data_path", help="The path to the folder containing the synthetic data")
    parser.add_argument("--output_path", default="models", help="The trained model will be saved to this folder")
    parser.add_argument("--pretrained_model_path", help="The path to the pretrained model file")
    parser.add_argument("--gpu_ids", type=int, default=-1, help="Which GPU to use (or -1 for cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--number_of_epochs", type=int, default=2000, help="Number of epochs for training loop")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for optimizer")
    parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument("--template",
                        help="The template file path (given in space delimited csv format of size nx3). Required if mode is auto")
    parser.add_argument("--network_input_size", type=int, default=10, help="Input layer size for STORM-Net")
    parser.add_argument("--network_output_size", type=int, default=3, help="Output layer size for STORM-Net")
    parser.add_argument("--num_threads", type=int, default=0, help="Number of worker threads for dataloader")
    parser.add_argument("--log", help="If present, writes training log to this path")
    parser.add_argument("--tensorboard",
                        help="If present, writes training stats to this path (readable with tensorboard)")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info", help="Selects verbosity level")
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    cmd = "test_torch cache/renders/telaviv_model --template C:/src/UnityCap/example_models/example_model.txt".split()
    args = parser.parse_args(cmd)
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
    args.is_train = True
    if args.gpu_ids == -1:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:{}'.format(args.gpu_ids))
    return args


if __name__ == "__main__":
    opt = parse_arguments()
    if opt.log:
        opt.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=opt.log, filemode='w', level=opt.verbosity.upper())
    else:
        logging.basicConfig(level=opt.verbosity.upper())
    logging.info("starting training loop.")
    train_loop(opt)
    logging.info("finished training.")
