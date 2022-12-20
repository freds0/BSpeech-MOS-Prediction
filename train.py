import argparse
import collections
import torch
import numpy as np
import data_loader as module_data
from trainer import loss as module_loss
from trainer import metric as module_metric
import model as module_arch
from utils import prepare_device
from trainer.trainer import Trainer
from utils.logger import logger
from utils.config_parser import ConfigParser

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'], logger)

    logger.info("Training {} using device {}".format(config["name"], device))
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    val_data_loader = data_loader.get_val_dataloader()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    model = model.to(device)

    # get function handles of loss and metrics
    metrics = config['metrics']
    criterion = getattr(module_loss, config['loss'])
    if config['loss'].startswith('weighted'):
        weighted = True
    else:
        weighted = False

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    config.config['trainer']['save_dir'] = config.save_dir

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config.config['trainer'],
                      resume=config.resume,
                      device=device,
                      data_loader=data_loader,
                      weighted=weighted,
                      valid_data_loader=val_data_loader,
                      lr_scheduler=lr_scheduler,
                      logger=logger
               )
    trainer.train()


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='MOS Prediction Model')
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

    config = ConfigParser.from_args(args, options)
    logger.config(folder=config._save_dir)
    main(config)
