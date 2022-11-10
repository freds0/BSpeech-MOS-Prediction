import argparse
import collections
import torch
import numpy as np
import data_loader as module_data
from trainer import loss as module_loss
import model as module_arch
from utils import prepare_device
from scipy import stats

from utils.config_parser import ConfigParser

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def resume_checkpoint(model, resume_path):
    """
    Resume from saved checkpoints
    :param
    model: model to be loaded
    resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    print("Loading checkpoint: {} ...".format(resume_path))

    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['state_dict'])

    print("Checkpoint loaded.")


def main(config):
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    val_data_loader = data_loader.get_val_dataloader()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    resume_checkpoint(model, config.resume)

    # get function handles of loss and metrics
    #metrics = [getattr(module_metric, met) for met in config['metrics']]
    criterion = getattr(module_loss, config['loss'])

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)

    print("Testing {} using device {}".format(config["name"], device))
    model.eval()
    metrics = {
        'loss': 0,
        'pearson': 0,
        'spearman': 0
    }
    results = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            metrics['loss'] += loss.item()
            results = np.concatenate([results, output.squeeze().detach().cpu().numpy()])
            targets = np.concatenate([targets, target.detach().cpu().numpy()])

    metrics['loss'] = metrics['loss'] / len(val_data_loader)
    metrics['pearson'] = stats.spearmanr(results, targets, axis=None).correlation
    metrics['spearman'] = stats.pearsonr(results, targets)[0]

    print(metrics)


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

    config = ConfigParser.from_args(args, options)
    main(config)
