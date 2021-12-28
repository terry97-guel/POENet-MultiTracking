import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from utils.Dynamics import *
from utils.LieGroup import *
import random
from pathlib import Path
import time

# fix random seeds for reproducibility
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    # device = torch.device('cuda:1')
    # torch.cuda.set_device(device)

    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    poelyr_params = list(map(lambda x: x[1], list(filter(lambda named_params: 'poe' in named_params[0], model.named_parameters()))))
    others_params = list(map(lambda x: x[1], list(filter(lambda named_params: 'poe' not in named_params[0], model.named_parameters()))))
    optimizer = config.init_obj('optimizer', torch.optim, [{'params': others_params, 'lr': 1e-2}, {'params': poelyr_params, 'lr': 1e-1}])
    # optimizer = config.init_obj('optimizer', torch.optim, [{'params': others_params, 'lr': 1e-3}, {'params': poelyr_params, 'lr': 1e-1}])
    # optimizer = config.init_obj('optimizer', torch.optim, [{'params': others_params, 'lr': 1e-3}, {'params': poelyr_params, 'lr': 1e-2}])
    # optimizer = config.init_obj('optimizer', torch.optim, [{'params': others_params, 'lr': 1e-4}, {'params': poelyr_params, 'lr': 1e-2}])

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()
    jointTwist = trainer.model.poe.getJointTwist()
    M_se3 = trainer.model.poe.M_se3
    train_x_gpu = data_loader.dataset.x.to(device)
    train_output = trainer.model(train_x_gpu).cpu()
    train_target = data_loader.dataset.y
    jointAngle = trainer.model.getJointAngle(train_x_gpu).cpu()

    pathname = "./output/4param/txtfile/"
    Path(pathname).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        np.savetxt(pathname+'jointAngle.txt', jointAngle, delimiter=',')
        np.savetxt(pathname+'jointTwist.txt', jointTwist.cpu(), delimiter=',')
        np.savetxt(pathname+'M_se3.txt', M_se3.cpu(), delimiter=',')
        np.savetxt(pathname+'nominalTwist.txt', trainer.model.poe.nominalTwist.cpu(), delimiter=',')
        np.savetxt(pathname+'outputPose.txt', skew_se3(logSE3(train_output)), delimiter=',')
        np.savetxt(pathname+'targetPose.txt', skew_se3(logSE3(train_target)), delimiter=',')

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config_euler.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='1', type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)

    main(config)
