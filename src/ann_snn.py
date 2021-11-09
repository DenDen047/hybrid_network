# -*- coding: utf-8 -*-

"""
# File Name : snn_mlp_1.py
# Author: Haowen Fang
# Email: hfang02@syr.edu
# Description: multi-layer snn for MNIST classification. Use dual exponential psp kernel.
"""

import argparse
import pandas as pd
import os
import time
import sys
import logging

import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from snn_lib.snn_layers import *
from snn_lib.optimizers import *
from snn_lib.schedulers import *
from snn_lib.data_loaders import *
import snn_lib.utilities

import omegaconf
from omegaconf import OmegaConf

import networks.mlp_networks
import networks.cnn_networks
import utils


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# arg parser
parser = argparse.ArgumentParser(description='ann_snn_{mlp|cnn}')
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--config_file', type=str, help='path to configuration file')
parser.add_argument('--rand_seed', type=int, default=42, help='the seed of random functions')
parser.add_argument('--without_reset', action='store_false', help='membrane potentials will be resetted only once')
parser.add_argument('--train', action='store_true', help='train model')
parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--logging', action='store_true', default=True, help='if true, output the all image/pdf files during the process')

args = parser.parse_args()

# setting of logging
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
if args.logging:
    log_fpath = f'/logs/{TIMESTAMP}.log'
    logging.basicConfig(
        filename=log_fpath,
        level=logging.DEBUG
    )
logger = logging.getLogger(__name__)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.DEBUG)
logger.addHandler(_console_handler)

# %% config file
if args.config_file is None:
    logger.info('No config file provided, use default config file')
else:
    logger.info(f'Config file provided: {args.config_file}')

conf = OmegaConf.load(args.config_file)

torch.manual_seed(args.rand_seed)
np.random.seed(args.rand_seed)

experiment_name = conf['experiment_name']

# %% checkpoint
save_checkpoint = conf['save_checkpoint']
checkpoint_base_name = conf['checkpoint_base_name']
checkpoint_base_path = conf['checkpoint_base_path']
test_checkpoint_path = conf['test_checkpoint_path']

# %% training parameters
hyperparam_conf = conf['hyperparameters']
length = hyperparam_conf['length']
batch_size = hyperparam_conf['batch_size']
synapse_type = hyperparam_conf['synapse_type']
epoch = hyperparam_conf['epoch']
tau_m = hyperparam_conf['tau_m']
tau_s = hyperparam_conf['tau_s']
filter_tau_m = hyperparam_conf['filter_tau_m']
filter_tau_s = hyperparam_conf['filter_tau_s']

membrane_filter = hyperparam_conf['membrane_filter']

train_bias = hyperparam_conf['train_bias']
train_coefficients = hyperparam_conf['train_coefficients']

# %% dataset config
dataset_config = conf['dataset_config']
dataset_name = dataset_config['name']
in_channels = dataset_config['in_channels']
max_rate = dataset_config['max_rate']
use_transform = dataset_config['use_transform']

# acc file name
acc_file_name = experiment_name + '_' + conf['acc_file_name']


if __name__ == "__main__":
    logger.debug(conf)
    logger.debug(args)

    model = eval(args.model)(
        batch_size,
        length,
        in_channels,
        train_coefficients,
        train_bias,
        membrane_filter,
        tau_m,
        tau_s,
        reset_state=args.without_reset
    ).to(device)

    writer_log_dir = f"/torch_logs/{TIMESTAMP}"
    os.makedirs(writer_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=writer_log_dir)

    params = list(model.parameters())

    optimizer = get_optimizer(params, conf)
    scheduler = get_scheduler(optimizer, conf)

    train_dataloader, val_dataloader, test_dataloader = utils.load_datasetloader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        length=length,
        flatten=in_channels == 0,
        transform=get_rand_transform(conf['transform'])
    )

    train_acc_list = []
    val_acc_list = []
    checkpoint_list = []

    if args.train == True:
        train_it = 0
        test_it = 0
        for j in range(epoch):

            epoch_time_stamp = time.strftime("%Y%m%d-%H%M%S")

            model.train()
            train_acc, train_loss = utils.train(model, optimizer, scheduler, train_dataloader, device, writer=None)
            train_acc_list.append(train_acc)

            logger.info('Train epoch: {}, acc: {}'.format(j, train_acc))

            # save every checkpoint
            if save_checkpoint == True:
                checkpoint_name = checkpoint_base_name + experiment_name + '_' + str(j) + '_' + epoch_time_stamp
                checkpoint_path = os.path.join(checkpoint_base_path, checkpoint_name)
                checkpoint_list.append(checkpoint_path)

                torch.save({
                    'epoch': j,
                    'snn_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, checkpoint_path)

            # test model
            model.eval()
            val_acc, val_loss = utils.evaluate(model, val_dataloader, device, writer=None)

            logger.info('Val epoch: {}, acc: {}'.format(j, val_acc))
            val_acc_list.append(val_acc)

            # recode the metrics
            writer.add_scalar('Loss/train', train_loss, j)
            writer.add_scalar('Loss/val', val_loss, j)
            writer.add_scalar('Accuracy/train', train_acc, j)
            writer.add_scalar('Accuracy/val', val_acc, j)

        # save result and get best epoch
        train_acc_list = np.array(train_acc_list)
        val_acc_list = np.array(val_acc_list)

        acc_df = pd.DataFrame(data={'train_acc': train_acc_list, 'val_acc': val_acc_list})

        acc_df.to_csv(acc_file_name)

        best_train_acc = np.max(train_acc_list)
        best_train_epoch = np.argmax(val_acc_list)

        best_val_epoch = np.argmax(val_acc_list)
        best_val_acc = np.max(val_acc_list)

        best_checkpoint = checkpoint_list[best_val_epoch]
        logger.info(f'best checkpoint: {best_checkpoint}')

        # test
        test_checkpoint = torch.load(best_checkpoint)
        model.load_state_dict(test_checkpoint["snn_state_dict"])

        test_acc, test_loss = utils.evaluate(model, test_dataloader, device)

        # show summary
        logger.info('Summary:')
        logger.info('Best train acc: {}, epoch: {}'.format(best_train_acc, best_train_epoch))
        logger.info('Best val acc: {}, epoch: {}'.format(best_val_acc, best_val_epoch))
        logger.info('Best test acc: {}, loss: {}'.format(test_acc, test_loss))

    elif args.test == True:

        test_checkpoint = torch.load(test_checkpoint_path)
        model.load_state_dict(test_checkpoint["snn_state_dict"])

        test_acc, test_loss = utils.evaluate(model, test_dataloader, device)

        logger.info('Test checkpoint: {}, acc: {}'.format(test_checkpoint_path, test_acc))

