from torch.utils.tensorboard import SummaryWriter

class Writer:
    def __init__(self, opt):
        self.opt = opt
        self.bridge = SummaryWriter(log_dir=opt.root)

    def write_scaler(self, category, name, scalar_value, iterations):
        final_name = category + "/" + name
        self.bridge.add_scalar(final_name, scalar_value, iterations)

    def close(self):
        if self.bridge is not None:
            self.bridge.close()