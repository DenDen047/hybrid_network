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


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# arg parser
parser = argparse.ArgumentParser(description='mlp snn')
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--config_file', type=str, default='snn_mlp_1.yaml',
                    help='path to configuration file')
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

torch.manual_seed(conf['pytorch_seed'])
np.random.seed(conf['pytorch_seed'])

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

# %% mnist config
dataset_config = conf['dataset_config']
max_rate = dataset_config['max_rate']
use_transform = dataset_config['use_transform']

# %% transform config
if use_transform == True:
    rand_transform = get_rand_transform(conf['transform'])
else:
    rand_transform = None

# load mnist training dataset
trainset = datasets.MNIST(root='/dataset', train=True, download=True, transform=rand_transform)

# load mnist test dataset
testset = datasets.MNIST(root='/dataset', train=False, download=True, transform=None)

# acc file name
acc_file_name = experiment_name + '_' + conf['acc_file_name']


########################### train function ###################################
def train(model, optimizer, scheduler, train_data_loader, writer=None):
    eval_image_number = 0
    correct_total = 0
    wrong_total = 0

    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for i_batch, sample_batched in enumerate(train_data_loader):

        x_train = sample_batched[0]
        target = sample_batched[1].to(device)
        x_train = x_train.to(device)  # [batch_size, dim0, time_length]
        out_spike = model(x_train)

        spike_count = torch.sum(out_spike, dim=2)

        model.zero_grad()
        loss = criterion(spike_count, target.long())
        loss.backward()
        optimizer.step()

        # calculate acc
        _, idx = torch.max(spike_count, dim=1)

        eval_image_number += len(sample_batched[1])
        wrong = len(torch.where(idx != target)[0])

        correct = len(sample_batched[1]) - wrong
        wrong_total += len(torch.where(idx != target)[0])
        correct_total += correct
        acc = correct_total / eval_image_number

        # scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
            scheduler.step()

    # scheduler step
    if isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR):
        scheduler.step()

    acc = correct_total / eval_image_number

    return acc, loss


def test(model, test_data_loader, writer=None):
    eval_image_number = 0
    correct_total = 0
    wrong_total = 0

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    for i_batch, sample_batched in enumerate(test_data_loader):

        x_test = sample_batched[0]
        target = sample_batched[1].to(device)
        x_test = x_test.repeat(length, 1, 1).permute(1, 2, 0).to(device)
        out_spike = model(x_test)

        spike_count = torch.sum(out_spike, dim=2)

        loss = criterion(spike_count, target.long())

        # calculate acc
        _, idx = torch.max(spike_count, dim=1)

        eval_image_number += len(sample_batched[1])
        wrong = len(torch.where(idx != target)[0])

        correct = len(sample_batched[1]) - wrong
        wrong_total += len(torch.where(idx != target)[0])
        correct_total += correct
        acc = correct_total / eval_image_number

    acc = correct_total / eval_image_number

    return acc, loss


if __name__ == "__main__":

    model = eval(args.model)(
        batch_size,
        length,
        train_coefficients,
        train_bias,
        membrane_filter,
        tau_m,
        tau_s
    ).to(device)

    writer_log_dir = f"/torch_logs/{TIMESTAMP}"
    os.makedirs(writer_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=writer_log_dir)

    params = list(model.parameters())

    optimizer = get_optimizer(params, conf)

    scheduler = get_scheduler(optimizer, conf)

    train_data = TorchvisionDataset_Poisson_Spike(trainset, max_rate=1, length=length, flatten=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    test_data = TorchvisionDataset_Poisson_Spike(testset, max_rate=1, length=length, flatten=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    train_acc_list = []
    test_acc_list = []
    checkpoint_list = []

    if args.train == True:
        train_it = 0
        test_it = 0
        for j in range(epoch):

            epoch_time_stamp = time.strftime("%Y%m%d-%H%M%S")

            model.train()
            train_acc, train_loss = train(model, optimizer, scheduler, train_dataloader, writer=None)
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
            test_acc, test_loss = test(model, test_dataloader, writer=None)

            logger.info('Test epoch: {}, acc: {}'.format(j, test_acc))
            test_acc_list.append(test_acc)

            # recode the metrics
            writer.add_scalar('Loss/train', train_loss, j)
            writer.add_scalar('Loss/test', train_acc, j)
            writer.add_scalar('Accuracy/train', test_loss, j)
            writer.add_scalar('Accuracy/test', test_acc, j)

        # save result and get best epoch
        train_acc_list = np.array(train_acc_list)
        test_acc_list = np.array(test_acc_list)

        acc_df = pd.DataFrame(data={'train_acc': train_acc_list, 'test_acc': test_acc_list})

        acc_df.to_csv(acc_file_name)

        best_train_acc = np.max(train_acc_list)
        best_train_epoch = np.argmax(test_acc_list)

        best_test_epoch = np.argmax(test_acc_list)
        best_test_acc = np.max(test_acc_list)

        best_checkpoint = checkpoint_list[best_test_epoch]

        logger.info('Summary:')
        logger.info('Best train acc: {}, epoch: {}'.format(best_train_acc, best_train_epoch))
        logger.info('Best test acc: {}, epoch: {}'.format(best_test_acc, best_test_epoch))
        logger.info(f'best checkpoint: {best_checkpoint}')

    elif args.test == True:
        test_checkpoint = torch.load(test_checkpoint_path)
        model.load_state_dict(test_checkpoint["snn_state_dict"])

        test_acc, test_loss = test(model, test_dataloader)

        logger.info('Test checkpoint: {}, acc: {}'.format(test_checkpoint_path, test_acc))

