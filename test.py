import argparse
import collections
import torch
import numpy as np
import data_loader as module_data
from trainer import loss as module_loss
import model as module_arch
from utils import prepare_device
from utils.logger import logger
from utils.config_parser import ConfigParser
from utils.metric_handler import MetricHandler

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def resume_checkpoint(model, resume_path, logger):
    """
    Resume from saved checkpoints
    :param resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])
    logger.info("Checkpoint loaded.")

def main(config):
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    test_data_loader = data_loader.get_val_dataloader()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    resume_checkpoint(model, config.resume, logger)

    # get function handles of loss and metrics
    metrics_handler = MetricHandler(config['metrics'])
    criterion = getattr(module_loss, config['loss'])

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'], logger)
    model = model.to(device)

    logger.info("Testing {} using device {}".format(config["name"], device))
    model.eval()

    outputs = np.array([])
    targets = np.array([])
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            outputs = np.concatenate([outputs, output.squeeze().detach().cpu().numpy()])
            targets = np.concatenate([targets, target.detach().cpu().numpy()])

    metrics_handler.add("loss", total_loss / len(test_data_loader))
    metrics_handler.update(outputs, targets)

    print(metrics_handler.get_data_with_pvalue())


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', required=True, default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config = ConfigParser.from_args(args, options, log_to_file=False)
    logger.config(folder=None)
    main(config)
