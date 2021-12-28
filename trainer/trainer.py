import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from model.loss import *
from utils import inf_loop, MetricTracker
import time

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        # self.data_set = iter(self.data_loader)
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        unifiedName = type(self.data_loader).__name__
        if 'unified' in unifiedName or 'Unified' in unifiedName:
            self.isUnified = True
            self.valid_metrics = MetricTracker('torque error', 'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
            self.alpha = torch.tensor(config['alpha']).to(device)
            self.Vdot0 = torch.tensor(config['Vdot0']).to(device)
            self.logger.info(f'================= UNIFIED IDENTIFICATION: alpha = {self.alpha} =================')
        else:
            self.isUnified = False
            self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        if epoch > 0:
            self.model.poe.updateNominalTwist()

        self.model.train()
        self.train_metrics.reset()
        # torch.autograd.set_detect_anomaly(True) # for debugging only

        if self.isUnified:
            for batch_idx, (data, target, motorPos, motorVel, motorAcc, motorTorque) in enumerate(self.data_loader):
                # to GPU
                data, target = data.to(self.device), target.to(self.device)
                motorPos, motorVel = motorPos.to(self.device), motorVel.to(self.device)
                motorAcc, motorTorque = motorAcc.to(self.device), motorTorque.to(self.device)
                motorState = State(motorPos, motorVel, motorAcc, motorTorque)
                jointState = self.model.getJointStates(motorState)
                twists = self.model.poe.getJointTwist()
                jacobian_theta = self.model.getJacobian(motorPos)

                output = self.model(data)
                # unified error
                loss = self.criterion(output, target, self.alpha, jointState, motorState, self.Vdot0, twists, jacobian_theta)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output, target))

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))
                    self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break
        else:
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                train_time = 0
                time1 = time.time()  
                output = self.model(data)    
                time2 = time.time()
                train_time = time2 - time1
                # print(train_time)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item())
                # #### TODO
                # if batch_idx % 100 == 0:
                #     self.writer.add_image(key, value, batch_idx)
                #
                #
                # ####
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output, target))

                # if batch_idx % self.log_step == 0:
                #     self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                #         epoch,
                #         self._progress(batch_idx),
                #         loss.item()))
                #     self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                if batch_idx == self.len_epoch:
                    break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # torch.cuda.empty_cache()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        if self.isUnified:
            for batch_idx, (data, target, motorPos, motorVel, motorAcc, motorTorque) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                motorPos, motorVel = motorPos.to(self.device), motorVel.to(self.device)
                motorAcc, motorTorque = motorAcc.to(self.device), motorTorque.to(self.device)
                # unified error
                output = self.model(data)
                motorState = State(motorPos, motorVel, motorAcc, motorTorque)
                jointState = self.model.getJointStates(motorState)
                jacobian_theta = self.model.getJacobian(motorPos)
                twists = self.model.poe.getJointTwist()
                loss = self.criterion(output, target, self.alpha, jointState, motorState, self.Vdot0, twists, jacobian_theta)
                dynError = torqueError(jointState, motorState, self.Vdot0, twists, jacobian_theta)
                # kineError = SE3Error(output, target)
                # loss = dynError / self.alpha + kineError
                # loss = kineError
                # torque error
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('torque error', dynError.item())
                # pose error
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        else:
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    output = self.model(data)
                    loss = self.criterion(output, target)

                    # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                    self.valid_metrics.update('loss', loss.item())
                    for met in self.metric_ftns:
                        self.valid_metrics.update(met.__name__, met(output, target))
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)