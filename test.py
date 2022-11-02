import argparse
import collections
import torch
import numpy as np
from data_loader import embeddings_dataloader as module_data
from trainer import loss as module_loss
import model as module_arch
from utils import prepare_device
from trainer.trainer import Trainer

from utils.config_parser import ConfigParser

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    val_data_loader = data_loader.get_val_dataloader()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)

    # get function handles of loss and metrics
    #metrics = [getattr(module_metric, met) for met in config['metrics']]
    criterion = getattr(module_loss, config['loss'])

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    config.config['trainer']['save_dir'] = config.save_dir

    print("Training {} using device {}".format(config["name"], device))
    trainer = Trainer(model, criterion, optimizer,
                      config=config.config['trainer'],
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=val_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    #config = ConfigParser.from_args(args, options)
    #args = parser.parse_args()

    config = ConfigParser.from_args(args, options)
    main(config)
