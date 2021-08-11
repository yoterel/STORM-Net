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


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.transforms = opt.transforms
        pickle_file_path = opt.data_path / "serialized.pickle"
        if not pickle_file_path.is_file():
            logging.info("loading raw data")
            X, Y = file_io.load_raw_json_db(opt.data_path, False, False)
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
        if self.opt.is_train:
            self.data = x_train
            self.labels = y_train
        else:
            self.data = x_val
            self.labels = y_val
        self.labels = self.transform_labels_to_point_cloud(True)

    def __getitem__(self, idx):
        x = self.data[idx]
        self.shuffle_timeseries(x)
        self.shuffle_data(x)
        self.mask_data(x)
        self.center_data(x)
        y = self.labels[idx]

        return x, y

    def __len__(self):
        return len(self.data)

    def transform_labels_to_point_cloud(self, save_result):
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
        all_data = np.empty(self.labels.shape[0], data.shape[0], data.shape[1])
        data_origin = data[:names.index(0), :]  # non numbered optodes are not calibrated
        data_others = data[names.index(0):, :]  # selects optodes for applying calibration
        transformed_data = []
        for i, (rot_mat, scale_mat) in enumerate(zip(rs, sc)):
            transformed_data_sim = rot_mat @ (scale_mat @ data_others.T)
            data_others = transformed_data_sim.T
            transformed_data.append([names, np.vstack((data_origin, data_others))])
        projected_data = geometry.project_sensors_to_MNI(transformed_data)
        return [x[1] for x in projected_data]


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
        shuf_shp = b.shape[-1]
        c = np.reshape(b, (shuf_shp // 2, 2))  # shuffles in groups of 2 since each sticker has 2 coordinates
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
    def forward(self, x):
        return self.net(x)


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
                                                                     train_loss.cpu().numpy()))
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
            logging.info("validation: epoch: {}, loss: {}".format(epoch, val_loss_total.cpu().numpy()))
        model.scheduler.step(val_loss)


def parse_arguments():
    parser = argparse.ArgumentParser(description='This script trains STORM-Net')
    parser.add_argument("model_name", help="The name to give the newly trained model (without extension).")
    parser.add_argument("data_path", help="The path to the folder containing the synthetic data")
    parser.add_argument("--output_path", default="models", help="The trained model will be saved to this folder")
    parser.add_argument("--pretrained_model_path", help="The path to the pretrained model file")
    parser.add_argument("--gpu_ids", type=int, default=-1, help="Which GPU to use (or -1 for cpu)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--number_of_epochs", type=int, default=2000, help="Number of epochs for training loop")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for optimizer")
    parser.add_argument("--network_input_size", type=int, default=10, help="Input layer size for STORM-Net")
    parser.add_argument("--network_output_size", type=int, default=3, help="Output layer size for STORM-Net")
    parser.add_argument("--log", help="If present, writes training log to this path")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info", help="Selects verbosity level")
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    cmd = "test_torch cache/renders/telaviv_model".split()
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
