import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.metric_handler import MetricHandler

#from torchmetrics import PearsonCorrCoef, SpearmanCorrCoef

class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, criterion, metrics_names, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, logger=None):
        self.model = model
        self.criterion = criterion
        self.metrics_handler = MetricHandler(metrics_names)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.epochs = config['epochs']
        self.data_loader = data_loader
        self.start_epoch = 1
        self.checkpoint_dir = config['save_dir']
        self.n_checkpoints = config['n_checkpoints']
        self.save_period = config['save_period']
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.logger = logger
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
            self.monitor_mode, self.monitor_metric = self.monitor.split('|')
            assert self.monitor_mode in ['min', 'max']

            self.monitor_best = float('inf') if self.monitor_mode == 'min' else float('-inf')
            self.early_stop = config.get('early_stop', float('inf'))

            if self.early_stop <= 0:
                self.early_stop = float('inf')

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A dict with train and val metrics
        """
        self.model.train()
        #self.train_metrics.reset()

        total_steps = len(self.data_loader)

        outputs = np.array([])
        targets = np.array([])
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device, dtype=torch.float32), target.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            outputs = np.concatenate([outputs, output.squeeze().detach().cpu().numpy()])
            targets = np.concatenate([targets, target.detach().cpu().numpy()])

            if (batch_idx % self.log_step) == 0:
                self.logger.info('Train Step: {} {} Loss: {:.6f}'.format(batch_idx, self._progress(batch_idx + 1), loss.item()))

        self.metrics_handler.add("loss", total_loss / len(self.data_loader))
        self.metrics_handler.update(outputs, targets)
        train_metrics = self.metrics_handler.get_data()

        if self.do_validation:
            val_metrics = self._valid_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return train_metrics, val_metrics


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A dict validation metrics
        """
        self.model.eval()
        self.metrics_handler.reset()
        outputs = np.array([])
        targets = np.array([])
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                outputs = np.concatenate([outputs, output.squeeze().detach().cpu().numpy()])
                targets = np.concatenate([targets, target.detach().cpu().numpy()])

        self.metrics_handler.add('loss', total_loss / len(self.data_loader))
        self.metrics_handler.update(outputs, targets)
        return self.metrics_handler.get_data()


    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        saved_checkpoints = []
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.logger.info('Epoch: {}/{}'.format(epoch, self.epochs))
            train_metrics, val_metrics = self._train_epoch(epoch)

            self.logger.info("\tTrain Loss: {:.6f} Pearson Corr: {:.4f} Spearman Corr: {:.4f}".format(train_metrics['loss'], train_metrics['pearson_corr'], train_metrics['spearman_corr']))
            self.logger.info("\tVal   Loss: {:.6f} Pearson Corr: {:.4f} Spearman Corr: {:.4f}".format(val_metrics['loss'], val_metrics['pearson_corr'], val_metrics['spearman_corr']))

            # save tensorboard informations
            self.writer.add_scalar("Loss/train", train_metrics['loss'], epoch)
            self.writer.add_scalar("Pearson Correlation/train", train_metrics['pearson_corr'], epoch)
            self.writer.add_scalar("Spearman Correlation/train", train_metrics['spearman_corr'], epoch)
            self.writer.add_scalar("Learning Rate", self.lr_scheduler.get_last_lr()[0], epoch)

            self.writer.add_scalar("Loss/val", val_metrics['loss'], epoch)
            self.writer.add_scalar("Pearson Correlation/val", val_metrics['pearson_corr'], epoch)
            self.writer.add_scalar("Spearman Correlation/val", val_metrics['spearman_corr'], epoch)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.monitor_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.monitor_mode == 'min' and val_metrics[self.monitor_metric] <= self.monitor_best) or \
                               (self.monitor_mode == 'max' and val_metrics[self.monitor_metric] >= self.monitor_best)
                except KeyError:
                    self.logger.error("Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(self.monitor_metric), type='ERROR')
                    self.monitor_mode = 'off'
                    improved = False

                if improved:
                    self.monitor_best = val_metrics[self.monitor_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                    self.logger.info("Best Metric = {}".format(self.monitor_best))
                    break

            if ((epoch % self.save_period) == 0):
                checkpoint_filepath = self._save_checkpoint(epoch, save_best=best)
                saved_checkpoints.append(checkpoint_filepath)
                self._maintain_checkpoints(saved_checkpoints, self.n_checkpoints)


    def _maintain_checkpoints(self, checkpoints_list, n_checkpoints=2):
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
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
        return filename


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))

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
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)