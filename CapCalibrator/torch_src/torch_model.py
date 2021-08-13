import torch
import torch.nn as nn
import torch_src.MNI_torch as MNI_torch
import file_io
import geometry
import numpy as np
from pathlib import Path


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
            nn.Linear(32, self.opt.network_output_size),
        )
        # self.net = torch.nn.Sequential(
        #     nn.Conv1d(in_channels=opt.network_input_size, out_channels=64, kernel_size=2),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2),
        #     nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(128, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, opt.network_output_size),
        # ).to(self.opt.device)
        self.anchors_xyz, self.sensors_xyz, self.selected_indices = self.load_static_model()

    def forward(self, x):
        euler_out = self.naive_net(x)
        mat_out = self.euler_to_matrix(euler_out)

        # rots = np.empty((x.shape[0], 3, 3), dtype=float)
        # for i in range(x.shape[0]):
        #      rot = R.from_euler('xyz', list(out[i].cpu().detach().numpy()), degrees=True)
        #      rots[i] = rot.as_matrix()
        # if not torch.all(torch.isclose(mat_out, torch.from_numpy(rots).float())):
        #     logging.warning("matrix from euler different than scipy matrix!")
        transformed_sensors = torch.transpose(torch.bmm(mat_out, self.sensors_xyz.T.repeat(x.shape[0], 1, 1)), 1, 2)
        projected_out = MNI_torch.torch_project(self.anchors_xyz, transformed_sensors, self.selected_indices)
        return projected_out, euler_out

    def load_static_model(self):
        names, data, format, _ = file_io.read_template_file(self.opt.template)
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
        if opt.continue_train:
            self.load_network("latest")
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=opt.lr,
                                          betas=(opt.beta1, 0.999),
                                          weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '{}_net.pth'.format(str(which_epoch))
        load_path = Path.joinpath(self.opt.root, save_filename)
        net = self.network
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.opt.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '{}_net.pth'.format(str(which_epoch))
        save_path = Path.joinpath(self.opt.root, save_filename)
        if self.opt.gpu_ids >= 0 and torch.cuda.is_available():
            torch.save(self.network.module.cpu().state_dict(), save_path)
            self.network.cuda(self.device)
        else:
            torch.save(self.network.cpu().state_dict(), save_path)