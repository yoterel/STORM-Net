import torch
import argparse
from pathlib import Path
import logging
import time
import torch_src.torch_data as torch_data
import torch_src.torch_model as torch_model
import torch_src.torch_writer as torch_writer
import numpy as np


def train_loop(opt, sync=None):
    """
    trains a neural network
    :param opt: training options (from command line usually)
    :param sync: list of synchronization objects: 0-event signaling to end training, 1-queue to report results
    :return:
    """
    if sync:
        writer = torch_writer.Writer(opt, sync[1])
    else:
        writer = torch_writer.Writer(opt, None)
    opt.is_train = True
    train_dataset = torch_data.MyDataLoader(opt)
    opt.is_train = False
    val_dataset = torch_data.MyDataLoader(opt)
    model = torch_model.MyModel(opt)
    # loss_fn = torch.nn.MSELoss()
    model.optimizer.zero_grad()
    for epoch in range(opt.number_of_epochs):
        train_loss_total = []
        for batch_index, (input, target) in enumerate(train_dataset):
            output_sensors, output_euler = model.network(input)
            train_loss_euler = torch.mean((target["rot_and_scale"] - output_euler) ** 2)
            writer.write_scaler("batch", "train_loss_euler", train_loss_euler.cpu().detach().numpy(), epoch*(len(train_dataset) // opt.batch_size) + batch_index)
            if opt.loss == "l2+projection":
                train_loss_sensors = torch.mean(torch.linalg.norm(target["raw_projected_data"] - output_sensors, dim=2))
                writer.write_scaler("batch", "train_loss_projection", train_loss_sensors.cpu().detach().numpy(), epoch*(len(train_dataset) // opt.batch_size) + batch_index)
                train_loss = opt.loss_alpha*train_loss_sensors + (1 - opt.loss_alpha)*train_loss_euler
            else:
                train_loss = train_loss_euler
            if opt.batch_size == 1:
                #divide loss by accumulation steps to have equivilant performance of using batch size = "batch_accumulation"
                train_loss /= opt.batch_accumulation
            # train_loss = loss_fn(output, target["raw_projected_data"])
            train_loss.backward()
            if opt.batch_size == 1:
                if batch_index % opt.batch_accumulation == 0:
                    model.optimizer.step()
                    model.optimizer.zero_grad()
            else:
                model.optimizer.step()
                model.optimizer.zero_grad()
            train_loss_np = train_loss.cpu().detach().numpy()
            writer.write_scaler("batch", "train_loss_total", train_loss_np, epoch*(len(train_dataset) // opt.batch_size) + batch_index)
            logging.info("train: epoch: {}, batch {} / {}, loss: {}".format(epoch,
                                                                            batch_index,
                                                                            len(train_dataset) // opt.batch_size,
                                                                            train_loss_np))
            train_loss_total.append(train_loss_np)
            # stops training if signaled from different thread
            if sync:
                if sync[0].isSet():
                    break
        # stops training if signaled from different thread
        if sync:
            if sync[0].isSet():
                break
        train_loss_total = np.mean(np.array(train_loss_total))
        writer.write_scaler("epoch", "train_loss_total", train_loss_total, epoch)
        logging.info("train: epoch: {}, training loss: {}".format(epoch,
                                                                  train_loss_total))
        if opt.create_new_checkpoints_per_epoch:
            model.save_network(file_name=str(epoch))

        if sync:
            model.save_network(file_name=opt.experiment_name, use_models_folder=True)
        else:
            model.save_network(file_name="latest")
        with torch.no_grad():
            val_loss_total = []
            for input, target in val_dataset:
                model.optimizer.zero_grad()
                output_sensors, output_euler = model.network(input)
                val_loss_euler = torch.mean((target["rot_and_scale"] - output_euler) ** 2)
                if opt.loss == "l2+projection":
                    val_loss_sensors = torch.mean(torch.linalg.norm(target["raw_projected_data"] - output_sensors, dim=2))
                    val_loss = opt.loss_alpha * val_loss_sensors + (1 - opt.loss_alpha) * val_loss_euler
                else:
                    val_loss = val_loss_euler
                # val_loss = loss_fn(output, target)
                val_loss_total.append(val_loss.cpu().detach().numpy())
            val_loss_total = np.mean(np.array(val_loss_total))
            writer.write_scaler("epoch", "val_loss_total", val_loss_total, epoch)
            logging.info("validation: epoch: {}, loss: {}".format(epoch, val_loss_total))
        writer.write_scaler("epoch", "learning rate", model.optimizer.param_groups[0]['lr'], epoch)
        logging.info("lr: {}".format(model.optimizer.param_groups[0]['lr']))
        model.scheduler.step(val_loss_total)
    writer.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='This script trains STORM-Net')
    parser.add_argument("experiment_name", help="The name to give the experiment")
    parser.add_argument("data_path", help="The path to the folder containing the synthetic data")
    parser.add_argument("--architecture", type=str, choices=["fc", "1dconv", "2dconv"], default="fc", help="Selects architecture")
    parser.add_argument("--force_load_raw_data", action="store_true", help="Forces loading data from disk (recreates train/val/test splits)")
    parser.add_argument("--loss", type=str, choices=["l2", "l2+projection"], help="loss function to use")
    parser.add_argument("--scale_faces", type=str, choices=["x", "y", "z", "xy", "xz", "yz", "xyz"], help="Renderer will also apply different scales to the virtual head & mask")
    parser.add_argument("--dont_use_gmm", action="store_true", default=False, help="do not use gmm to create heatmaps")
    parser.add_argument('--loss_alpha', type=float, default=0.1, help='coefficient of projection loss if used')
    parser.add_argument("--gpu_ids", type=int, default=-1, help="Which GPU to use (or -1 for cpu)")
    parser.add_argument("--continue_train", action="store_true", help="continue from latest epoch")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--batch_accumulation", type=int, default=16, help="Batch accumulation (only valid if batch size=1)")
    parser.add_argument("--number_of_epochs", type=int, default=100, help="Number of epochs for training loop")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for optimizer")
    parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument("--template",
                        help="The template file path (given in space delimited csv format of size nx3)."
                             " Required if using projection loss")
    parser.add_argument("--network_input_size", type=int, default=14, help="Input layer size for STORM-Net")
    parser.add_argument("--num_threads", type=int, default=0, help="Number of worker threads for dataloader")
    parser.add_argument("--log", action="store_true", help="If present, writes training log")
    parser.add_argument("--tensorboard",
                        help="If present, writes training stats to this path (readable with tensorboard)")
    parser.add_argument("--create_new_checkpoints_per_epoch", action="store_true",
                        help="Creates checkpoints every epoch (besides latest version of model)")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info", help="Selects verbosity level")
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    # cmd = "test_torch ../../renders --template ../../example_models/example_model.txt".split()
    args = parser.parse_args()

    args.root = Path("models", args.experiment_name)
    args.root.mkdir(parents=True, exist_ok=True)

    if args.log:
        args.log = Path(args.root, "log_{}".format(str(time.time())))

    if args.tensorboard:
        args.tensorboard = Path(args.tensorboard)

    args.data_path = Path(args.data_path)

    args.is_train = True

    args.network_output_size = 3
    if args.scale_faces:
        args.network_output_size += len(args.scale_faces)

    if args.gpu_ids == -1:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:{}'.format(args.gpu_ids))
    return args


if __name__ == "__main__":
    opt = parse_arguments()
    if opt.log:
        logging.basicConfig(filename=opt.log, filemode='w', level=opt.verbosity.upper())
    else:
        logging.basicConfig(level=opt.verbosity.upper())
    logging.info("starting training loop.")
    train_loop(opt)
    logging.info("finished training.")
