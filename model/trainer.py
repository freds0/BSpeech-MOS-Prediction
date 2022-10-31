import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
#from utils.util import MetricTracker
#from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef

def pearson_correlation_loss(output, target):
    '''
    Source-code: https://discuss.pytorch.org/t/use-pearson-correlation-coefficient-as-cost-function/8739/4
    '''
    vx = output - torch.mean(output)
    vy = target - torch.mean(target)
    return torch.sum(vx * vy) / (
                torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))  # use Pearson correlation


class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.epochs = config['epochs']
        self.data_loader = data_loader
        self.start_epoch = 1
        self.checkpoint_dir = config['save_dir']
        self.save_period = config['save_period']
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        #self.log_step = int(np.sqrt(data_loader.batch_size))
        self.log_step = config['log_step']
        # setup visualization writer instance
        self.writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, config['log_dir']))

        self.monitor = config.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.monitor_mode = 'off'
            self.monitor_best = 0
        else:
            #self.monitor_mode, self.monitor_metric = self.monitor.split()
            self.monitor_mode, self.monitor_metric = 'min', 'loss'
            assert self.monitor_mode in ['min', 'max']

            self.monitor_best = float('inf') if self.monitor_mode == 'min' else float('-inf')
            self.early_stop = config.get('early_stop', float('inf'))

            if self.early_stop <= 0:
                self.early_stop = float('inf')

        #self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        #self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        #self.train_metrics.reset()
        epoch_loss = 0

        metrics = {
            'loss': 0,
            'pearson': 0,
            'spearman': 0
        }

        total_steps = len(self.data_loader)
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device, dtype=torch.float32), target.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            #self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            #self.train_metrics.update('loss', loss.item())
            #for met in self.metric_ftns:
            #    self.train_metrics.update(met.__name__, met(output, target))
            metrics['loss'] += loss.item()
            metrics['pearson'] += stats.spearmanr(output.squeeze().detach().cpu().numpy(), target.cpu().detach().numpy(), axis=None).correlation
            metrics['spearman'] += stats.pearsonr(output.squeeze().detach().cpu().numpy(), target.cpu().detach().numpy())[0]

            if (batch_idx % self.log_step) == 0:
                print('Train Step: {} {} Loss: {:.6f}'.format(
                    batch_idx,
                    self._progress(batch_idx + 1),
                    loss.item())
                )
                '''
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                '''
            if batch_idx == self.len_epoch:
                break

        metrics['loss'] = metrics['loss'] / len(self.data_loader)
        metrics['pearson'] = metrics['pearson']  / len(self.data_loader)
        metrics['spearman'] = metrics['spearman'] / len(self.data_loader)

        #log = self.train_metrics.result()
        if self.do_validation:
            val_metrics = self._valid_epoch(epoch)
            #log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return metrics, val_metrics


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        metrics = {
            'loss': 0,
            'pearson': 0,
            'spearman': 0
        }
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                metrics['loss'] += loss.item()
                metrics['pearson'] += stats.spearmanr(output.squeeze().detach().cpu().numpy(),
                                                      target.cpu().detach().numpy(), axis=None).correlation
                metrics['spearman'] += \
                stats.pearsonr(output.squeeze().detach().cpu().numpy(), target.cpu().detach().numpy())[0]

                #self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                #self.valid_metrics.update('loss', loss.item())
                #for met in self.metric_ftns:
                #    self.valid_metrics.update(met.__name__, met(output, target))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        metrics['loss'] = metrics['loss'] / len(self.data_loader)
        metrics['pearson'] = metrics['pearson'] / len(self.data_loader)
        metrics['spearman'] = metrics['spearman'] / len(self.data_loader)

        return metrics


    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        saved_checkpoints = []
        for epoch in range(self.start_epoch, self.epochs + 1):
            print('Epoch: {}/{}'.format(epoch, self.epochs))
            metrics, val_metrics = self._train_epoch(epoch)
            print("\tTrain Loss: {:.6f} Pearson Corr: {:.4f} Spearman Corr: {:.4f}".format(
                metrics['loss'], metrics['pearson'], metrics['spearman'])
            )
            print("\tVal   Loss: {:.6f} Pearson Corr: {:.4f} Spearman Corr: {:.4f}".format(
                val_metrics['loss'], val_metrics['pearson'], val_metrics['spearman'])
            )
            # save tensorboard informations
            self.writer.add_scalar("Loss/train", metrics['loss'], epoch)
            self.writer.add_scalar("Pearson Correlation/train", metrics['pearson'], epoch)
            self.writer.add_scalar("Spearman Correlation/train", metrics['spearman'], epoch)
            self.writer.add_scalar("Learning Rate", self.lr_scheduler.get_last_lr()[0], epoch)

            self.writer.add_scalar("Loss/val", val_metrics['loss'], epoch)
            self.writer.add_scalar("Pearson Correlation/val", val_metrics['pearson'], epoch)
            self.writer.add_scalar("Spearman Correlation/val", val_metrics['spearman'], epoch)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.monitor_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.monitor_mode == 'min' and metrics[self.monitor_metric] <= self.monitor_best) or \
                               (self.monitor_mode == 'max' and metrics[self.monitor_metric] >= self.monitor_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " + \
                                        "Model performance monitoring is disabled.".format(self.monitor_metric))
                    #self.logger.warning("Warning: Metric '{}' is not found. "
                    #                    "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.monitor_mode = 'off'
                    improved = False

                if improved:
                    self.monitor_best = metrics[self.monitor_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                    #self.logger.info("Validation performance didn\'t improve for {} epochs. "
                    #                 "Training stops.".format(self.early_stop))
                    break


            if ((epoch % self.save_period) == 0):
                checkpoint_filepath = self._save_checkpoint(epoch, save_best=best)
                saved_checkpoints.append(checkpoint_filepath)
                self._maintain_checkpoints(saved_checkpoints, 2)


    def _maintain_checkpoints(self, checkpoints_list, n_checkpoints=5):
        if (len(checkpoints_list) > n_checkpoints):
            filepath = checkpoints_list.pop(0)
            os.remove(filepath)


    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        #self.logger.info("Saving checkpoint: {} ...".format(filename))
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            #self.logger.info("Saving current best: model_best.pth ...")
            print("Saving current best: model_best.pth ...")

        return filename


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        #self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        #if checkpoint['config']['arch'] != self.config['arch']:
        #    self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
        #                        "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        #if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        #    self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
        #                        "Optimizer parameters not being resumed.")
        #else:
        #    self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        #self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)