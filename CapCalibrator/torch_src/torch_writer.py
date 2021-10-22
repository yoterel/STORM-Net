from torch.utils.tensorboard import SummaryWriter


class Writer:
    def __init__(self, opt, queue):
        if queue:
            self.queue_bridge = queue
        else:
            self.queue_bridge = None
        self.opt = opt
        if self.opt.tensorboard:
            self.tensorboard_wrapper = SummaryWriter(log_dir=opt.root)
        else:
            self.tensorboard_wrapper = None

    def write_scaler(self, category, name, scalar_value, iterations):
        final_name = category + "/" + name
        if self.tensorboard_wrapper:
            self.tensorboard_wrapper.add_scalar(final_name, scalar_value, iterations)
        if self.queue_bridge:
            if "epoch" in final_name:
                self.queue_bridge.put(["training_data", final_name, scalar_value, iterations])

    def close(self):
        if self.tensorboard_wrapper is not None:
            self.tensorboard_wrapper.close()
        if self.queue_bridge:
            self.queue_bridge.put(["training_done"])
