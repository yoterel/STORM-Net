import torch
import torch.nn as nn
import torch_src.MNI_torch as MNI_torch
import file_io
import geometry
import numpy as np
from pathlib import Path
import copy


class Convd2d():
    def __init__(self, input_size, output_size):
        self.network = torch.nn.ModuleList([
            nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(131072, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
                ])


class Convd1d():
    def __init__(self, input_size, output_size):
        self.network = torch.nn.ModuleList([
            nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
                ])


class FullyConnected():
    def __init__(self, output_size):
        self.network = torch.nn.ModuleList([
            nn.Flatten(),
            nn.Linear(140, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        ])


class MyNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(MyNetwork, self).__init__()
        self.opt = copy.deepcopy(opt)
        if opt.architecture == "fc":
            fc_network = FullyConnected(self.opt.network_output_size)
            self.net = fc_network.network
        elif opt.architecture == "1dconv":
            conv1d_network = Convd1d(opt.network_input_size, opt.network_output_size)
            self.net = conv1d_network.network
        elif opt.architecture == "2dconv":
            conv2d_network = Convd2d(opt.network_input_size, opt.network_output_size)
            self.net = conv2d_network.network
        else:
            raise NotImplementedError
        if self.opt.loss == "l2+projection":
            self.anchors_xyz, self.sensors_xyz, self.selected_indices = self.load_static_model()

    def forward(self, x):
        if not self.opt.architecture == "2dconv":
            x = x.permute(0, 2, 1)
        for i, layer in enumerate(self.net):
            x = layer(x)
        projected_out = None
        if self.opt.loss == "l2+projection":
            mat_out = self.euler_to_matrix(x)
            transformed_sensors = torch.transpose(torch.bmm(mat_out, self.sensors_xyz.T.repeat(x.shape[0], 1, 1)), 1, 2)
            projected_out = MNI_torch.torch_project(self.anchors_xyz, transformed_sensors, self.selected_indices)
        return projected_out, x

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
        return torch.from_numpy(anchors_xyz).float().to(self.opt.device),\
               torch.from_numpy(data_sensors).float().to(self.opt.device),\
               selected_indices

    def euler_to_matrix(self, x):
        """
        converts euler angels to matrix (x is batched, b x 3)
        :param x:
        :return:
        """
        Rx = torch.zeros(x.shape[0], 3, 3).to(self.opt.device)
        Ry = torch.zeros(x.shape[0], 3, 3).to(self.opt.device)
        Rz = torch.zeros(x.shape[0], 3, 3).to(self.opt.device)
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


class MyModel:
    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)
        self.network = MyNetwork(opt)
        if opt.continue_train:
            self.load_network("latest")
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=opt.lr,
                                          betas=(opt.beta1, 0.999),
                                          weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, verbose=True, patience=3)
        self.network.to(self.opt.device)

    def load_network(self, file_name):
        """load model from disk"""
        save_filename = '{}.pth'.format(str(file_name))
        load_path = Path.joinpath(self.opt.root, save_filename)
        net = self.network
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from {}'.format(str(load_path)))
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.opt.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, file_name, use_models_folder=False):
        """save model to disk"""
        if use_models_folder:
            save_filename = '{}.pth'.format(str(file_name))
            save_path = Path("models", save_filename)
        else:
            save_filename = '{}.pth'.format(str(file_name))
            save_path = Path.joinpath(self.opt.root, save_filename)
        torch.save(self.network.cpu().state_dict(), save_path)
        self.network.to(self.opt.device)

    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)